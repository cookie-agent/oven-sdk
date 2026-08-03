//! Vertex request validation, encoding, and private replay decoding.

use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use oven_sdk::{
    AssistantPart, Capability, ErrorStage, FilePart, FileSource, HistoryTurn, InputPart, JsonValue,
    LanguageModelDescriptor, ModelCapabilities, ModelError, NativeContextScope, ReplayDecision,
    ReplayDisposition, ReplayOutcome, ReplayPolicy, Request, ResponseFormat, SystemPart,
    ToolChoice, ToolContent, ToolResultPart,
};

use crate::{
    REPLAY_FORMAT,
    model::{GoogleVertexMediaSettings, GoogleVertexSettings, GoogleVertexThinkingMode},
    options::{GoogleVertexProviderTool, GoogleVertexRequestOptions, GoogleVertexToolOptions},
};

pub(crate) struct Encoded {
    pub(crate) body: JsonValue,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
    pub(crate) shared_request_type: Option<String>,
    pub(crate) request_type: Option<String>,
}

struct ToolIdentity {
    name: String,
    provider_id: Option<String>,
}

pub(crate) fn validate_request(
    request: &Request,
    capabilities: &ModelCapabilities,
    settings: &GoogleVertexSettings,
) -> Result<(), ModelError> {
    request.validate_for(capabilities)?;
    let options = options(request)?;
    if let Some(cached_content) = options.cached_content.as_deref() {
        validate_cached_content(
            cached_content,
            capabilities,
            &settings.project,
            &settings.location,
        )?;
    }
    validate_media(request, &settings.media)?;
    if options
        .presence_penalty
        .is_some_and(|value| !(-2.0..=2.0).contains(&value))
        || options
            .frequency_penalty
            .is_some_and(|value| !(-2.0..=2.0).contains(&value))
    {
        return Err(ModelError::invalid_request(
            "Vertex presence and frequency penalties must be between -2 and 2",
        ));
    }
    if matches!(
        request.response_format,
        ResponseFormat::Json { schema: None }
    ) {
        return Err(ModelError::invalid_request(
            "Vertex structured output requires a JSON schema",
        ));
    }
    if let ResponseFormat::Json {
        schema: Some(schema),
    } = &request.response_format
    {
        validate_schema(schema.as_value())?;
    }
    if (!request.tools.is_empty() || !options.provider_tools.is_empty())
        && !capabilities.features.contains(Capability::TOOL_CALLING)
    {
        return Err(ModelError::unsupported(
            "tool calling is not supported by this Vertex declaration",
        ));
    }
    if !options.provider_tools.is_empty()
        && !capabilities.features.contains(Capability::PROVIDER_TOOLS)
    {
        return Err(ModelError::unsupported(
            "provider tools are not supported by this Vertex declaration",
        ));
    }
    if !request.tools.is_empty()
        && !options.provider_tools.is_empty()
        && !settings.tools.mixed_client_and_provider_tools
    {
        return Err(ModelError::unsupported(
            "mixing client functions with provider tools is disabled by Vertex settings",
        ));
    }
    if request.tools.iter().any(strict_tool) && !settings.tools.strict_functions {
        return Err(ModelError::unsupported(
            "validated strict functions are disabled by Vertex settings",
        ));
    }
    for tool in &request.tools {
        validate_tool_schema(tool.input_schema.as_value())?;
    }
    if let Some(thinking) = &options.thinking_config {
        if settings.thinking == GoogleVertexThinkingMode::Unsupported {
            return Err(ModelError::unsupported(
                "thinking controls are disabled by Vertex settings",
            ));
        }
        if thinking.thinking_budget.is_some() && thinking.thinking_level.is_some() {
            return Err(ModelError::invalid_request(
                "Vertex thinking_budget and thinking_level cannot both be set",
            ));
        }
        match settings.thinking {
            GoogleVertexThinkingMode::Budget if thinking.thinking_level.is_some() => {
                return Err(ModelError::unsupported(
                    "this Vertex declaration uses a thinking budget, not a thinking level",
                ));
            }
            GoogleVertexThinkingMode::Level if thinking.thinking_budget.is_some() => {
                return Err(ModelError::unsupported(
                    "this Vertex declaration uses a thinking level, not a thinking budget",
                ));
            }
            _ => {}
        }
    }
    for turn in &request.history {
        match turn {
            HistoryTurn::System(message) => {
                if message
                    .content
                    .iter()
                    .any(|part| matches!(part, SystemPart::Custom(_)))
                {
                    return Err(ModelError::unsupported(
                        "Vertex system messages support text only",
                    ));
                }
            }
            HistoryTurn::User(message) => {
                for part in &message.content {
                    match part {
                        InputPart::File(_) => {}
                        InputPart::Custom(_) => {
                            return Err(ModelError::unsupported(
                                "Vertex user custom parts are not supported",
                            ));
                        }
                        InputPart::Text(_) => {}
                    }
                }
            }
            HistoryTurn::Tool(message) => {
                if message
                    .results
                    .iter()
                    .any(|result| matches!(result.content, ToolContent::Mixed(_)))
                {
                    return Err(ModelError::unsupported(
                        "multimodal Vertex function results are not enabled",
                    ));
                }
            }
            HistoryTurn::Assistant(_) => {}
        }
    }
    Ok(())
}

fn validate_cached_content(
    value: &str,
    capabilities: &ModelCapabilities,
    project: &str,
    location: &str,
) -> Result<(), ModelError> {
    if !capabilities.features.contains(Capability::PROMPT_CACHING) {
        return Err(ModelError::unsupported(
            "Vertex cached content is unsupported by this model declaration",
        )
        .with_stage(ErrorStage::RequestEncoding));
    }
    let segments = value.split('/').collect::<Vec<_>>();
    match segments.as_slice() {
        ["projects", found_project, "locations", found_location, "cachedContents", id]
            if !id.is_empty()
                && *found_project == project
                && *found_location == location =>
        {
            Ok(())
        }
        _ => Err(ModelError::invalid_request(
            "Vertex cached_content must be a full cached-content resource in the configured project and location",
        )
        .with_stage(ErrorStage::RequestEncoding)),
    }
}

pub(crate) fn encode_request(
    request: &Request,
    descriptor: &LanguageModelDescriptor,
    replay_policy: ReplayPolicy,
    stream_function_call_arguments: bool,
    native_context_scope: &NativeContextScope,
) -> Result<Encoded, ModelError> {
    let options = options(request)?;
    let mut replay = ReplayOutcome::default();
    let mut warnings = Vec::new();
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut tool_names = BTreeMap::<String, ToolIdentity>::new();
    let mut previous_tool_content = false;

    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
                previous_tool_content = false;
                for part in &message.content {
                    if let SystemPart::Text(text) = part
                        && !text.text.is_empty()
                    {
                        system_parts.push(serde_json::json!({"text":text.text}));
                    }
                }
            }
            HistoryTurn::User(message) => {
                previous_tool_content = false;
                let parts = message
                    .content
                    .iter()
                    .map(input_part)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                if !parts.is_empty() {
                    contents.push(serde_json::json!({"role":"user","parts":parts}));
                }
            }
            HistoryTurn::Assistant(turn) => {
                previous_tool_content = false;
                let mut native = None;
                if replay_policy == ReplayPolicy::Never {
                    replay.decisions.push(ReplayDecision {
                        history_index,
                        disposition: ReplayDisposition::ReconstructedNormalized,
                    });
                } else if let Some(artifact) = &turn.finish.native_replay {
                    if artifact.adapter_id() != &descriptor.adapter_id {
                        replay.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedForeignAdapter {
                                found: artifact.adapter_id().clone(),
                                expected: descriptor.adapter_id.clone(),
                            },
                        });
                    } else if artifact.scope() != native_context_scope {
                        replay.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedForeignScope {
                                found: artifact.scope().clone(),
                                expected: native_context_scope.clone(),
                            },
                        });
                    } else {
                        match decode_replay(artifact.payload(), &turn.message.content) {
                            Ok(parts) => {
                                replay.decisions.push(ReplayDecision {
                                    history_index,
                                    disposition: ReplayDisposition::Replayed,
                                });
                                native = Some(parts);
                            }
                            Err(reason) => replay.decisions.push(ReplayDecision {
                                history_index,
                                disposition: ReplayDisposition::DiscardedInvalidPayload {
                                    reason: reason.into(),
                                },
                            }),
                        }
                    }
                } else {
                    replay.decisions.push(ReplayDecision {
                        history_index,
                        disposition: ReplayDisposition::NoArtifact,
                    });
                }
                let parts = match native {
                    Some(parts) => parts,
                    None => {
                        if replay_policy != ReplayPolicy::Never {
                            replay.decisions.push(ReplayDecision {
                                history_index,
                                disposition: ReplayDisposition::ReconstructedNormalized,
                            });
                        }
                        if turn.message.content.iter().any(|part| {
                            matches!(
                                part,
                                AssistantPart::Reasoning(_)
                                    | AssistantPart::Custom(_)
                                    | AssistantPart::Source(_)
                            )
                        }) {
                            warnings.push(format!(
                                "Vertex provider-only assistant state omitted during normalized reconstruction at history index {history_index}"
                            ));
                        }
                        turn.message
                            .content
                            .iter()
                            .filter_map(assistant_part)
                            .collect()
                    }
                };
                for part in &turn.message.content {
                    if let AssistantPart::ToolCall(call) = part {
                        tool_names.insert(
                            call.id.clone(),
                            ToolIdentity {
                                name: call.name.clone(),
                                provider_id: call.provider_item_id.clone(),
                            },
                        );
                    }
                }
                if !parts.is_empty() {
                    contents.push(serde_json::json!({"role":"model","parts":parts}));
                }
            }
            HistoryTurn::Tool(message) => {
                let parts = message
                    .results
                    .iter()
                    .map(|result| function_response(result, &tool_names))
                    .collect::<Result<Vec<_>, _>>()?;
                if previous_tool_content {
                    contents
                        .last_mut()
                        .and_then(|content| content.get_mut("parts"))
                        .and_then(JsonValue::as_array_mut)
                        .expect("previous tool content has parts")
                        .extend(parts);
                } else if !parts.is_empty() {
                    contents.push(serde_json::json!({"role":"user","parts":parts}));
                }
                previous_tool_content = true;
            }
        }
    }

    let mut generation_config = serde_json::json!({});
    if let Some(value) = request.inference.max_output_tokens {
        generation_config["maxOutputTokens"] = JsonValue::from(value);
    }
    if let Some(value) = request.inference.temperature {
        generation_config["temperature"] = JsonValue::from(value);
    }
    if let Some(value) = request.inference.top_p {
        generation_config["topP"] = JsonValue::from(value);
    }
    if let Some(value) = options.top_k {
        generation_config["topK"] = JsonValue::from(value);
    }
    if !options.stop_sequences.is_empty() {
        generation_config["stopSequences"] =
            serde_json::to_value(options.stop_sequences).expect("stop sequences are serializable");
    }
    if let Some(value) = options.seed {
        generation_config["seed"] = JsonValue::from(value);
    }
    if let Some(value) = options.presence_penalty {
        generation_config["presencePenalty"] = JsonValue::from(value);
    }
    if let Some(value) = options.frequency_penalty {
        generation_config["frequencyPenalty"] = JsonValue::from(value);
    }
    if let Some(value) = options.thinking_config {
        generation_config["thinkingConfig"] =
            serde_json::to_value(value).expect("thinking config is serializable");
    }
    if let ResponseFormat::Json {
        schema: Some(schema),
    } = &request.response_format
    {
        generation_config["responseFormat"] = serde_json::json!([{
            "text":{"mimeType":"APPLICATION_JSON","schema":schema.as_value()}
        }]);
    }

    let mut root = serde_json::json!({"contents":contents});
    if !system_parts.is_empty() {
        root["systemInstruction"] = serde_json::json!({"parts":system_parts});
    }
    if generation_config
        .as_object()
        .is_some_and(|object| !object.is_empty())
    {
        root["generationConfig"] = generation_config;
    }
    let mut tools = provider_tools(&options.provider_tools);
    if !request.tools.is_empty() {
        tools.push(serde_json::json!({
            "functionDeclarations": request.tools.iter().map(|tool| serde_json::json!({
                "name":tool.name,
                "description":tool.description,
                "parametersJsonSchema":tool.input_schema.as_value(),
            })).collect::<Vec<_>>()
        }));
    }
    if !tools.is_empty() {
        root["tools"] = JsonValue::Array(tools);
    }
    if !request.tools.is_empty() || stream_function_call_arguments {
        let strict = request.tools.iter().any(strict_tool);
        let (mode, names) = match &request.tool_choice {
            ToolChoice::Auto => (if strict { "VALIDATED" } else { "AUTO" }, None),
            ToolChoice::Required => (if strict { "VALIDATED" } else { "ANY" }, None),
            ToolChoice::None => ("NONE", None),
            ToolChoice::Tool(name) => (
                if strict { "VALIDATED" } else { "ANY" },
                Some(vec![name.clone()]),
            ),
        };
        root["toolConfig"] = serde_json::json!({"functionCallingConfig":{"mode":mode}});
        if stream_function_call_arguments {
            root["toolConfig"]["functionCallingConfig"]["streamFunctionCallArguments"] =
                JsonValue::Bool(true);
        }
        if let Some(names) = names {
            root["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"] =
                serde_json::to_value(names).expect("tool names are serializable");
        }
    }
    if let Some(value) = options.cached_content {
        root["cachedContent"] = JsonValue::String(value);
    }
    if !options.safety_settings.is_empty() {
        root["safetySettings"] = serde_json::to_value(options.safety_settings)
            .expect("safety settings are serializable");
    }
    Ok(Encoded {
        body: root,
        replay,
        warnings,
        shared_request_type: options.shared_request_type,
        request_type: options.request_type,
    })
}

fn options(request: &Request) -> Result<GoogleVertexRequestOptions, ModelError> {
    request
        .provider_options
        .get("google_vertex")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| ModelError::invalid_request("invalid Google Vertex request options"))
        })
        .transpose()
        .map(|value| value.unwrap_or_default())
}

fn strict_tool(tool: &oven_sdk::ToolDefinition) -> bool {
    tool.provider_options
        .get("google_vertex")
        .and_then(|value| serde_json::from_value::<GoogleVertexToolOptions>(value.clone()).ok())
        .is_some_and(|options| options.strict)
}

fn provider_tools(tools: &[GoogleVertexProviderTool]) -> Vec<JsonValue> {
    tools
        .iter()
        .map(|tool| match tool {
            GoogleVertexProviderTool::GoogleSearch => serde_json::json!({"googleSearch":{}}),
            GoogleVertexProviderTool::UrlContext => serde_json::json!({"urlContext":{}}),
            GoogleVertexProviderTool::CodeExecution => serde_json::json!({"codeExecution":{}}),
            GoogleVertexProviderTool::VertexRagStore { rag_corpus, top_k } => {
                let mut value = serde_json::json!({"retrieval":{"vertexRagStore":{"ragResources":{"ragCorpus":rag_corpus}}}});
                if let Some(top_k) = top_k {
                    value["retrieval"]["vertexRagStore"]["similarityTopK"] = JsonValue::from(*top_k);
                }
                value
            }
            GoogleVertexProviderTool::GoogleMaps => serde_json::json!({"googleMaps":{}}),
        })
        .collect()
}

fn input_part(part: &InputPart) -> Result<Option<JsonValue>, ModelError> {
    match part {
        InputPart::Text(text) => {
            Ok((!text.text.is_empty()).then(|| serde_json::json!({"text":text.text})))
        }
        InputPart::File(file) => file_part(file).map(Some),
        InputPart::Custom(_) => Err(ModelError::unsupported(
            "Vertex custom input parts are not supported",
        )),
    }
}

fn file_part(file: &FilePart) -> Result<JsonValue, ModelError> {
    match &file.source {
        FileSource::Bytes(bytes) => Ok(serde_json::json!({
            "inlineData":{"mimeType":file.media_type,"data":STANDARD.encode(bytes)}
        })),
        FileSource::Text(text) => Ok(serde_json::json!({
            "inlineData":{"mimeType":file.media_type,"data":STANDARD.encode(text.as_bytes())}
        })),
        FileSource::Url(url) => Ok(serde_json::json!({
            "fileData":{"mimeType":file.media_type,"fileUri":url.as_str()}
        })),
        FileSource::ProviderReference { .. } => Err(ModelError::unsupported(
            "Vertex Gemini does not accept Gemini Files provider references",
        )
        .with_stage(ErrorStage::RequestEncoding)),
    }
}

fn assistant_part(part: &AssistantPart) -> Option<JsonValue> {
    match part {
        AssistantPart::Text(text) if !text.text.is_empty() => {
            Some(serde_json::json!({"text":text.text}))
        }
        AssistantPart::ToolCall(call) => {
            let mut function_call = serde_json::json!({"name":call.name,"args":call.input});
            if let Some(id) = &call.provider_item_id {
                function_call["id"] = JsonValue::String(id.clone());
            }
            Some(serde_json::json!({"functionCall":function_call}))
        }
        _ => None,
    }
}

fn function_response(
    result: &ToolResultPart,
    identities: &BTreeMap<String, ToolIdentity>,
) -> Result<JsonValue, ModelError> {
    let identity = identities.get(&result.tool_call_id).ok_or_else(|| {
        ModelError::invalid_request("Vertex tool result could not resolve its function name")
    })?;
    let output = match &result.content {
        ToolContent::Text(value) => JsonValue::String(value.clone()),
        ToolContent::Json(value) => value.clone(),
        ToolContent::Denied { reason } => JsonValue::String(
            reason
                .clone()
                .unwrap_or_else(|| "Tool call execution denied.".into()),
        ),
        ToolContent::Mixed(_) => {
            return Err(ModelError::unsupported(
                "multimodal Vertex function results are not enabled",
            ));
        }
    };
    let response = if result.is_error {
        serde_json::json!({"error":output})
    } else {
        serde_json::json!({"output":output})
    };
    let mut function_response = serde_json::json!({"name":identity.name,"response":response});
    if let Some(id) = &identity.provider_id {
        function_response["id"] = JsonValue::String(id.clone());
    }
    Ok(serde_json::json!({"functionResponse":function_response}))
}

fn validate_media(
    request: &Request,
    settings: &GoogleVertexMediaSettings,
) -> Result<(), ModelError> {
    let mut counts = MediaCounts::default();
    for turn in &request.history {
        let HistoryTurn::User(message) = turn else {
            continue;
        };
        for part in &message.content {
            if let InputPart::File(file) = part {
                validate_file(file, settings, &mut counts)?;
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct MediaCounts {
    images: usize,
    https_images: usize,
    documents: usize,
    audio: usize,
    videos: usize,
    https_videos: usize,
}

fn validate_file(
    file: &FilePart,
    settings: &GoogleVertexMediaSettings,
    counts: &mut MediaCounts,
) -> Result<(), ModelError> {
    if image_mime(&file.media_type) {
        counts.images += 1;
        if counts.images > settings.max_images {
            return Err(media_limit("Vertex image count exceeds configured limit"));
        }
    } else if document_mime(&file.media_type) {
        counts.documents += 1;
        if counts.documents > settings.max_documents {
            return Err(media_limit(
                "Vertex document count exceeds configured limit",
            ));
        }
    } else if audio_mime(&file.media_type) {
        counts.audio += 1;
        if counts.audio > settings.max_audio {
            return Err(media_limit("Vertex audio count exceeds configured limit"));
        }
    } else if video_mime(&file.media_type) {
        counts.videos += 1;
        if counts.videos > settings.max_videos {
            return Err(media_limit("Vertex video count exceeds configured limit"));
        }
    } else {
        return Err(ModelError::unsupported("unsupported Vertex media modality")
            .with_stage(ErrorStage::RequestEncoding));
    }
    match &file.source {
        FileSource::Bytes(bytes) => validate_inline_size(&file.media_type, bytes.len(), settings)?,
        FileSource::Text(text) if document_mime(&file.media_type) => {
            validate_inline_size(&file.media_type, text.len(), settings)?;
        }
        FileSource::Text(_) => {
            return Err(
                ModelError::unsupported("binary Vertex media cannot use FileSource::Text")
                    .with_stage(ErrorStage::RequestEncoding),
            );
        }
        FileSource::Url(url)
            if settings
                .url_schemes
                .iter()
                .any(|scheme| scheme == url.scheme()) =>
        {
            if url.scheme() != "https" {
                return Ok(());
            }
            if image_mime(&file.media_type) {
                counts.https_images += 1;
                if counts.https_images > settings.max_https_images {
                    return Err(media_limit(
                        "Vertex HTTPS image count exceeds configured limit",
                    ));
                }
            }
            if video_mime(&file.media_type) {
                counts.https_videos += 1;
                if counts.https_videos > settings.max_https_videos {
                    return Err(media_limit(
                        "Vertex HTTPS video count exceeds configured limit",
                    ));
                }
            }
        }
        FileSource::Url(_) => {
            return Err(
                ModelError::unsupported("Vertex media URL scheme is not declared")
                    .with_stage(ErrorStage::RequestEncoding),
            );
        }
        FileSource::ProviderReference { .. } => {
            return Err(ModelError::unsupported(
                "Vertex Gemini does not accept Gemini Files references",
            )
            .with_stage(ErrorStage::RequestEncoding));
        }
    }
    Ok(())
}

fn validate_inline_size(
    media_type: &str,
    bytes: usize,
    settings: &GoogleVertexMediaSettings,
) -> Result<(), ModelError> {
    let limit = if image_mime(media_type) {
        Some((
            settings.max_inline_image_bytes,
            "Vertex inline image exceeds configured limit",
        ))
    } else if media_type == "application/pdf" {
        Some((
            settings.max_inline_pdf_bytes,
            "Vertex inline PDF exceeds configured limit",
        ))
    } else if media_type.starts_with("text/") {
        Some((
            settings.max_inline_text_bytes,
            "Vertex inline text exceeds configured limit",
        ))
    } else {
        None
    };
    if let Some((limit, message)) = limit
        && bytes > limit
    {
        return Err(media_limit(message));
    }
    Ok(())
}

fn media_limit(message: &str) -> ModelError {
    ModelError::unsupported(message).with_stage(ErrorStage::RequestEncoding)
}

fn image_mime(value: &str) -> bool {
    value.starts_with("image/")
}
fn document_mime(value: &str) -> bool {
    value == "application/pdf" || value.starts_with("text/")
}
fn audio_mime(value: &str) -> bool {
    value.starts_with("audio/")
}
fn video_mime(value: &str) -> bool {
    value.starts_with("video/")
}

fn validate_schema(schema: &JsonValue) -> Result<(), ModelError> {
    validate_json_schema(schema, "responseFormat")
}

fn validate_tool_schema(schema: &JsonValue) -> Result<(), ModelError> {
    let Some(root) = schema.as_object() else {
        return Err(ModelError::unsupported(
            "Vertex function parametersJsonSchema requires an object-only root",
        )
        .with_stage(ErrorStage::RequestEncoding));
    };
    if root.get("type").and_then(JsonValue::as_str) != Some("object")
        || root.contains_key("$ref")
        || root.contains_key("anyOf")
        || root.contains_key("oneOf")
    {
        return Err(ModelError::unsupported(
            "Vertex function parametersJsonSchema root must use exactly type object without a root reference or union",
        )
        .with_stage(ErrorStage::RequestEncoding));
    }
    validate_json_schema(schema, "function parametersJsonSchema")?;
    validate_tool_schema_refs(schema, schema)
}

fn validate_tool_schema_refs(root: &JsonValue, schema: &JsonValue) -> Result<(), ModelError> {
    if schema.is_boolean() {
        return Ok(());
    }
    let object = schema
        .as_object()
        .expect("validated JSON Schema nodes are objects or booleans");
    if let Some(reference) = object.get("$ref").and_then(JsonValue::as_str) {
        let Some(pointer) = reference.strip_prefix("#/$defs/") else {
            return Err(schema_error(
                "function parametersJsonSchema",
                "requires references to use a local #/$defs/ JSON pointer",
            ));
        };
        let target = root.pointer(reference.strip_prefix('#').expect("local reference"));
        if pointer.is_empty()
            || !target.is_some_and(|value| value.is_object() || value.is_boolean())
        {
            return Err(schema_error(
                "function parametersJsonSchema",
                "contains an unresolved local reference",
            ));
        }
    }
    for key in ["properties", "$defs"] {
        if let Some(values) = object.get(key).and_then(JsonValue::as_object) {
            for value in values.values() {
                validate_tool_schema_refs(root, value)?;
            }
        }
    }
    for key in ["items", "additionalProperties"] {
        if let Some(value) = object.get(key) {
            validate_tool_schema_refs(root, value)?;
        }
    }
    for key in ["anyOf", "oneOf", "prefixItems"] {
        if let Some(values) = object.get(key).and_then(JsonValue::as_array) {
            for value in values {
                validate_tool_schema_refs(root, value)?;
            }
        }
    }
    Ok(())
}

fn validate_json_schema(schema: &JsonValue, context: &str) -> Result<(), ModelError> {
    if schema.is_boolean() {
        return Ok(());
    }
    let JsonValue::Object(object) = schema else {
        return Err(ModelError::unsupported(format!(
            "Vertex {context} contains a non-schema value"
        ))
        .with_stage(ErrorStage::RequestEncoding));
    };
    const ALLOWED: &[&str] = &[
        "$id",
        "$defs",
        "$ref",
        "$anchor",
        "type",
        "format",
        "title",
        "description",
        "enum",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "prefixItems",
        "anyOf",
        "oneOf",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "propertyOrdering",
    ];
    if object.keys().any(|key| !ALLOWED.contains(&key.as_str())) {
        return Err(schema_error(
            context,
            "uses a keyword unsupported by current Vertex",
        ));
    }
    if object.contains_key("$ref")
        && object
            .keys()
            .any(|key| key != "$ref" && !key.starts_with('$'))
    {
        return Err(schema_error(
            context,
            "does not allow non-$ siblings beside $ref",
        ));
    }
    for key in ["$id", "$ref", "$anchor", "format", "title", "description"] {
        if object.get(key).is_some_and(|value| !value.is_string()) {
            return Err(schema_error(
                context,
                &format!("requires {key} to be a string"),
            ));
        }
    }
    if let Some(value) = object.get("type") {
        let valid_type = |value: &JsonValue| {
            value.as_str().is_some_and(|value| {
                matches!(
                    value,
                    "null" | "boolean" | "object" | "array" | "number" | "integer" | "string"
                )
            })
        };
        let valid = valid_type(value)
            || value
                .as_array()
                .is_some_and(|values| !values.is_empty() && values.iter().all(valid_type));
        if !valid {
            return Err(schema_error(
                context,
                "requires type to be a supported string or non-empty string array",
            ));
        }
    }
    if let Some(value) = object.get("enum") {
        let Some(values) = value.as_array() else {
            return Err(schema_error(
                context,
                "requires enum to be a non-empty array",
            ));
        };
        if values.is_empty() {
            return Err(schema_error(
                context,
                "requires enum to be a non-empty array",
            ));
        }
    }
    for key in ["required", "propertyOrdering"] {
        if let Some(value) = object.get(key) {
            let valid = value.as_array().is_some_and(|values| {
                let strings = values
                    .iter()
                    .filter_map(JsonValue::as_str)
                    .collect::<std::collections::BTreeSet<_>>();
                strings.len() == values.len()
            });
            if !valid {
                return Err(schema_error(
                    context,
                    &format!("requires {key} to be an array of unique strings"),
                ));
            }
        }
    }
    for key in ["minimum", "maximum"] {
        if object.get(key).is_some_and(|value| !value.is_number()) {
            return Err(schema_error(
                context,
                &format!("requires {key} to be a number"),
            ));
        }
    }
    for key in ["minItems", "maxItems"] {
        if object
            .get(key)
            .is_some_and(|value| value.as_u64().is_none())
        {
            return Err(schema_error(
                context,
                &format!("requires {key} to be a non-negative integer"),
            ));
        }
    }
    for key in ["properties", "$defs"] {
        if let Some(value) = object.get(key) {
            let JsonValue::Object(values) = value else {
                return Err(schema_error(
                    context,
                    &format!("requires {key} to be an object"),
                ));
            };
            for value in values.values() {
                validate_json_schema(value, context)?;
            }
        }
    }
    if let Some(value) = object.get("items") {
        validate_json_schema(value, context)?;
    }
    if let Some(value) = object.get("additionalProperties") {
        validate_json_schema(value, context)?;
    }
    for key in ["anyOf", "oneOf", "prefixItems"] {
        if let Some(value) = object.get(key) {
            let JsonValue::Array(values) = value else {
                return Err(schema_error(
                    context,
                    &format!("requires {key} to be an array"),
                ));
            };
            if matches!(key, "anyOf" | "oneOf") && values.is_empty() {
                return Err(schema_error(
                    context,
                    &format!("requires {key} to be non-empty"),
                ));
            }
            for value in values {
                validate_json_schema(value, context)?;
            }
        }
    }
    Ok(())
}

fn schema_error(context: &str, detail: &str) -> ModelError {
    ModelError::unsupported(format!("Vertex {context} {detail}"))
        .with_stage(ErrorStage::RequestEncoding)
}

fn decode_replay(
    payload: &JsonValue,
    normalized: &[AssistantPart],
) -> Result<Vec<JsonValue>, &'static str> {
    if payload.get("format").and_then(JsonValue::as_str) != Some(REPLAY_FORMAT) {
        return Err("Vertex replay format is invalid");
    }
    if payload.pointer("/content/role").and_then(JsonValue::as_str) != Some("model") {
        return Err("Vertex replay content role is invalid");
    }
    let parts = payload
        .pointer("/content/parts")
        .and_then(JsonValue::as_array)
        .ok_or("Vertex replay parts are invalid")?
        .clone();
    if parts.iter().any(|part| !valid_native_part(part)) {
        return Err("Vertex replay contains an unsupported native part");
    }
    if native_semantics(&parts) != normalized_semantics(normalized) {
        return Err("Vertex replay payload did not match normalized content");
    }
    Ok(parts)
}

fn valid_native_part(part: &JsonValue) -> bool {
    let Some(object) = part.as_object() else {
        return false;
    };
    object.keys().all(|key| {
        matches!(
            key.as_str(),
            "text"
                | "thought"
                | "thoughtSignature"
                | "functionCall"
                | "toolCall"
                | "toolResponse"
                | "executableCode"
                | "codeExecutionResult"
                | "inlineData"
                | "fileData"
                | "partMetadata"
        )
    })
}

fn native_semantics(parts: &[JsonValue]) -> Vec<JsonValue> {
    parts.iter().filter_map(|part| {
        if let Some(text) = part.get("text").and_then(JsonValue::as_str) {
            if text.is_empty() && part.get("thoughtSignature").is_some() {
                return None;
            }
            return Some(serde_json::json!({
                "type":if part.get("thought").and_then(JsonValue::as_bool) == Some(true) {"reasoning"} else {"text"},
                "text":text,
            }));
        }
        part.get("functionCall").map(|call| serde_json::json!({
            "type":"tool_call",
            "provider_id":call.get("id").cloned().unwrap_or(JsonValue::Null),
            "name":call.get("name").and_then(JsonValue::as_str).unwrap_or(""),
            "args":call.get("args").cloned().unwrap_or_else(|| serde_json::json!({})),
        }))
    }).collect()
}

fn normalized_semantics(parts: &[AssistantPart]) -> Vec<JsonValue> {
    parts
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Text(text) => Some(serde_json::json!({"type":"text","text":text.text})),
            AssistantPart::Reasoning(reasoning) => {
                Some(serde_json::json!({"type":"reasoning","text":reasoning.text}))
            }
            AssistantPart::ToolCall(call) => Some(serde_json::json!({
                "type":"tool_call",
                "provider_id":call.provider_item_id,
                "name":call.name,
                "args":call.input,
            })),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn media() -> GoogleVertexMediaSettings {
        GoogleVertexMediaSettings {
            max_images: 3,
            max_https_images: 2,
            max_documents: 3,
            max_audio: 1,
            max_videos: 2,
            max_https_videos: 1,
            max_inline_image_bytes: 7,
            max_inline_pdf_bytes: 50,
            max_inline_text_bytes: 7,
            url_schemes: vec!["https".into(), "gs".into()],
        }
    }

    #[test]
    fn explicit_media_settings_accept_gcs_and_reject_provider_references() {
        let settings = media();
        let mut counts = MediaCounts::default();
        let gcs = FilePart::video(
            "video/mp4",
            FileSource::Url("gs://bucket/video.mp4".parse().unwrap()),
        );
        assert!(validate_file(&gcs, &settings, &mut counts).is_ok());
        let reference = FilePart::image(
            "image/png",
            FileSource::ProviderReference {
                provider: oven_sdk::ProviderId::new("google"),
                id: "files/x".into(),
            },
        );
        assert!(validate_file(&reference, &settings, &mut counts).is_err());
    }

    #[test]
    fn inline_media_size_boundaries_are_exact() {
        let settings = media();
        for (media_type, limit) in [
            ("image/png", settings.max_inline_image_bytes),
            ("application/pdf", settings.max_inline_pdf_bytes),
            ("text/plain", settings.max_inline_text_bytes),
        ] {
            assert!(validate_inline_size(media_type, limit, &settings).is_ok());
            assert!(validate_inline_size(media_type, limit + 1, &settings).is_err());
        }
    }

    #[test]
    fn media_file_counts_and_https_source_counts_are_enforced() {
        let settings = media();
        let mut audio_counts = MediaCounts::default();
        let audio = FilePart::audio("audio/mpeg", FileSource::Bytes(Vec::new().into()));
        assert!(validate_file(&audio, &settings, &mut audio_counts).is_ok());
        assert!(validate_file(&audio, &settings, &mut audio_counts).is_err());

        let mut image_counts = MediaCounts::default();
        for index in 0..2 {
            let image = FilePart::image(
                "image/jpeg",
                FileSource::Url(format!("https://example.com/{index}.jpg").parse().unwrap()),
            );
            validate_file(&image, &settings, &mut image_counts).unwrap();
        }
        let third = FilePart::image(
            "image/jpeg",
            FileSource::Url("https://example.com/2.jpg".parse().unwrap()),
        );
        assert!(validate_file(&third, &settings, &mut image_counts).is_err());
    }

    #[test]
    fn current_json_schema_tuple_and_additional_property_schemas_are_validated() {
        validate_schema(&serde_json::json!({
            "type":"array",
            "prefixItems":[{"type":"string"}],
            "items":{"type":"number"}
        }))
        .unwrap();
        assert!(
            validate_schema(&serde_json::json!({
                "type":"object",
                "additionalProperties":{"unsupportedKeyword":true}
            }))
            .is_err()
        );
    }

    #[test]
    fn signature_only_empty_text_is_preserved_but_not_compared_as_semantic_text() {
        let payload = serde_json::json!({
            "format":REPLAY_FORMAT,
            "content":{"role":"model","parts":[{"text":"","thoughtSignature":"signature"}]}
        });
        let parts = decode_replay(&payload, &[]).unwrap();
        assert_eq!(parts[0]["thoughtSignature"], "signature");
    }

    #[test]
    fn replay_accepts_only_the_current_v4_private_format() {
        let normalized = vec![AssistantPart::Text(oven_sdk::TextPart::new("answer"))];
        let current = serde_json::json!({
            "format":REPLAY_FORMAT,
            "content":{"role":"model","parts":[{"text":"answer"}]}
        });
        assert!(decode_replay(&current, &normalized).is_ok());

        let mut stale = current;
        stale["format"] =
            JsonValue::String("oven.google.vertex.generate-content.assistant.v3".into());
        assert!(decode_replay(&stale, &normalized).is_err());
    }

    #[test]
    fn replay_semantics_require_the_same_provider_function_id() {
        let mut call = oven_sdk::ToolCallPart::new("local-call", "lookup", serde_json::json!({}));
        call.provider_item_id = Some("provider-call".into());
        let normalized = vec![AssistantPart::ToolCall(call)];
        let payload = serde_json::json!({
            "format":REPLAY_FORMAT,
            "content":{"role":"model","parts":[{"functionCall":{
                "id":"provider-call","name":"lookup","args":{}
            }}]}
        });
        assert!(decode_replay(&payload, &normalized).is_ok());

        let mut altered = payload.clone();
        altered["content"]["parts"][0]["functionCall"]["id"] =
            JsonValue::String("other-call".into());
        assert!(decode_replay(&altered, &normalized).is_err());

        let mut missing = payload;
        missing["content"]["parts"][0]["functionCall"]
            .as_object_mut()
            .unwrap()
            .remove("id");
        assert!(decode_replay(&missing, &normalized).is_err());
    }
}
