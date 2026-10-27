//! Vertex request validation, encoding, and private replay decoding.

use std::collections::BTreeMap;

use oven_sdk::{
    AssistantPart, Capability, ErrorStage, FilePart, FileSource, HistoryTurn, InputPart, JsonValue,
    LanguageModelDescriptor, ModelCapabilities, ModelError, NativeReplayArtifact, ReplayDecision,
    ReplayDisposition, ReplayOutcome, ReplayPolicy, Request, ResponseFormat, SystemPart,
    ToolChoice, ToolContent, ToolResultPart,
};
use serde::Deserialize as _;

use crate::{
    REPLAY_FORMAT,
    body::{
        self, Body, Content, Empty, FunctionCallingConfig, FunctionDeclaration, GenerationConfig,
        Part, Parts, RagResources, ResponseFormatText, Tool, ToolConfig, ToolOutput,
        VertexRagStore,
    },
    model::{GoogleVertexMediaSettings, GoogleVertexSettings, GoogleVertexThinkingMode},
    options::{GoogleVertexProviderTool, GoogleVertexRequestOptions, GoogleVertexToolOptions},
};

pub(crate) struct Encoded {
    /// Serialized JSON request body.
    pub(crate) body: Vec<u8>,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
    pub(crate) shared_request_type: Option<String>,
    pub(crate) request_type: Option<String>,
}

pub(crate) struct ParsedOptions {
    request: GoogleVertexRequestOptions,
    strict_tools: Vec<bool>,
}

struct ToolIdentity<'a> {
    name: &'a str,
    provider_id: Option<&'a str>,
}

pub(crate) fn validate_request(
    request: &Request,
    parsed: &ParsedOptions,
    capabilities: &ModelCapabilities,
    settings: &GoogleVertexSettings,
) -> Result<(), ModelError> {
    let options = &parsed.request;
    request.validate_for(capabilities)?;
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
    if parsed.strict_tools.iter().any(|strict| *strict) && !settings.tools.strict_functions {
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
                for result in &message.results {
                    validate_tool_result_files(result)?;
                }
            }
            HistoryTurn::Assistant(turn) => {
                for part in &turn.message.content {
                    if let AssistantPart::ToolResult(result) = part {
                        validate_tool_result_files(result)?;
                    }
                }
            }
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

/// Where a replayed assistant turn's validated native parts live.
enum ReplaySource<'a> {
    /// Borrowed in place from the artifact payload.
    Borrowed(&'a [JsonValue]),
    /// Index into the owned parts produced by the shared cross-adapter codec.
    Owned(usize),
}

pub(crate) fn encode_request(
    request: &Request,
    parsed: &ParsedOptions,
    descriptor: &LanguageModelDescriptor,
    replay_policy: ReplayPolicy,
    stream_function_call_arguments: bool,
    wire_model_id: &oven_sdk::ModelId,
) -> Result<Encoded, ModelError> {
    let options = &parsed.request;
    let provider_tools_supported = descriptor
        .capabilities
        .features
        .contains(Capability::PROVIDER_TOOLS);
    let mut replay = ReplayOutcome::default();
    let mut warnings = Vec::new();
    warnings.extend(oven_sdk::DeferredTools::of(request).upfront_warning("google-vertex"));

    // Validate native payloads first. Only the shared cross-adapter codec yields owned parts;
    // they are stored here so every body part can borrow for the whole encoding.
    let mut fallback_parts = Vec::<Vec<JsonValue>>::new();
    let sources = if replay_policy == ReplayPolicy::Never {
        Vec::new()
    } else {
        request
            .history
            .iter()
            .map(|turn| {
                let HistoryTurn::Assistant(turn) = turn else {
                    return None;
                };
                let artifact = turn.finish.native_replay.as_ref()?;
                Some(
                    match decode_replay(artifact.payload(), &turn.message.content) {
                        Ok(parts) => Ok(ReplaySource::Borrowed(parts)),
                        Err(reason) => oven_sdk::replay::generate_content_parts(
                            artifact,
                            &turn.message.content,
                        )
                        .map(|parts| {
                            fallback_parts.push(parts);
                            ReplaySource::Owned(fallback_parts.len() - 1)
                        })
                        .ok_or(reason),
                    },
                )
            })
            .collect::<Vec<_>>()
    };

    let mut system_parts = Vec::new();
    let mut contents = Vec::<Content<'_>>::with_capacity(request.history.len());
    let mut tool_names = BTreeMap::<&str, ToolIdentity<'_>>::new();
    let mut previous_tool_content = false;

    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
                previous_tool_content = false;
                for part in &message.content {
                    if let SystemPart::Text(text) = part
                        && !text.text.is_empty()
                    {
                        system_parts.push(Part::Text(&text.text));
                    }
                }
            }
            HistoryTurn::User(message) => {
                previous_tool_content = false;
                let parts = message
                    .content
                    .iter()
                    .map(input_part)
                    .filter_map(Result::transpose)
                    .collect::<Result<Vec<_>, _>>()?;
                if !parts.is_empty() {
                    contents.push(Content {
                        role: "user",
                        parts,
                    });
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
                    match sources[history_index]
                        .as_ref()
                        .expect("assistant replay source was validated")
                    {
                        Ok(source) => {
                            replay.decisions.push(ReplayDecision {
                                history_index,
                                disposition: ReplayDisposition::Replayed,
                            });
                            let parts = match source {
                                ReplaySource::Borrowed(parts) => *parts,
                                ReplaySource::Owned(index) => fallback_parts[*index].as_slice(),
                            };
                            native = Some(eligible_parts(
                                artifact,
                                wire_model_id,
                                provider_tools_supported,
                                descriptor.capabilities.replay.reasoning,
                                parts,
                                &mut warnings,
                            )?);
                        }
                        Err(reason) => replay.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedInvalidPayload {
                                reason: (*reason).into(),
                            },
                        }),
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
                        if oven_sdk::replay::has_required_vertex_signature(&turn.message.content) {
                            return Err(ModelError::replay(
                                "Vertex tool continuation requires its captured native thought signature",
                            ));
                        }
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
                let mut wire_calls = parts.iter().filter_map(Part::function_call_id);
                for part in &turn.message.content {
                    if let AssistantPart::ToolCall(call) = part {
                        tool_names.insert(
                            call.id.as_str(),
                            ToolIdentity {
                                name: &call.name,
                                provider_id: wire_calls.next().flatten(),
                            },
                        );
                    }
                }
                if !parts.is_empty() {
                    contents.push(Content {
                        role: "model",
                        parts,
                    });
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
                        .expect("previous tool content has parts")
                        .parts
                        .extend(parts);
                } else if !parts.is_empty() {
                    contents.push(Content {
                        role: "user",
                        parts,
                    });
                }
                previous_tool_content = true;
            }
        }
    }

    let generation_config = GenerationConfig {
        max_output_tokens: request.inference.max_output_tokens,
        temperature: request.inference.temperature,
        top_p: request.inference.top_p,
        top_k: options.top_k,
        stop_sequences: &options.stop_sequences,
        seed: options.seed,
        presence_penalty: options.presence_penalty,
        frequency_penalty: options.frequency_penalty,
        thinking_config: options.thinking_config.as_ref(),
        response_format: match &request.response_format {
            ResponseFormat::Json {
                schema: Some(schema),
            } => Some([body::ResponseFormat {
                text: ResponseFormatText {
                    mime_type: "APPLICATION_JSON",
                    schema: schema.as_value(),
                },
            }]),
            _ => None,
        },
    };

    let mut tools = provider_tools(&options.provider_tools);
    if !request.tools.is_empty() {
        tools.push(Tool::FunctionDeclarations(
            request
                .tools
                .iter()
                .map(|tool| FunctionDeclaration {
                    name: &tool.name,
                    description: &tool.description,
                    parameters_json_schema: tool.input_schema.as_value(),
                })
                .collect(),
        ));
    }
    let tool_config = (!request.tools.is_empty() || stream_function_call_arguments).then(|| {
        let strict = parsed.strict_tools.iter().any(|strict| *strict);
        let (mode, names) = match &request.tool_choice {
            ToolChoice::Auto => (if strict { "VALIDATED" } else { "AUTO" }, None),
            ToolChoice::Required => (if strict { "VALIDATED" } else { "ANY" }, None),
            ToolChoice::None => ("NONE", None),
            ToolChoice::Tool(name) => (
                if strict { "VALIDATED" } else { "ANY" },
                Some([name.as_str()]),
            ),
        };
        ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode,
                stream_function_call_arguments,
                allowed_function_names: names,
            },
        }
    });
    let body = Body {
        contents,
        system_instruction: (!system_parts.is_empty()).then_some(Parts {
            parts: system_parts,
        }),
        generation_config,
        tools,
        tool_config,
        cached_content: options.cached_content.as_deref(),
        safety_settings: &options.safety_settings,
    }
    .to_bytes()
    .map_err(|_| {
        ModelError::invalid_request("could not serialize Vertex Gemini request")
            .with_stage(ErrorStage::RequestEncoding)
    })?;
    Ok(Encoded {
        body,
        replay,
        warnings,
        shared_request_type: options.shared_request_type.clone(),
        request_type: options.request_type.clone(),
    })
}

pub(crate) fn options(request: &Request) -> Result<ParsedOptions, ModelError> {
    let request_options = request
        .provider_options
        .get("google_vertex")
        .map(|value| {
            GoogleVertexRequestOptions::deserialize(value)
                .map_err(|_| ModelError::invalid_request("invalid Google Vertex request options"))
        })
        .transpose()?
        .unwrap_or_default();
    let strict_tools = request
        .tools
        .iter()
        .map(|tool| {
            tool.provider_options
                .get("google_vertex")
                .and_then(|value| GoogleVertexToolOptions::deserialize(value).ok())
                .is_some_and(|options| options.strict)
        })
        .collect();
    Ok(ParsedOptions {
        request: request_options,
        strict_tools,
    })
}

fn provider_tools(tools: &[GoogleVertexProviderTool]) -> Vec<Tool<'_>> {
    let mut output = Vec::with_capacity(tools.len() + 1);
    output.extend(tools.iter().map(|tool| match tool {
        GoogleVertexProviderTool::GoogleSearch => Tool::GoogleSearch(Empty {}),
        GoogleVertexProviderTool::UrlContext => Tool::UrlContext(Empty {}),
        GoogleVertexProviderTool::CodeExecution => Tool::CodeExecution(Empty {}),
        GoogleVertexProviderTool::VertexRagStore { rag_corpus, top_k } => Tool::Retrieval {
            store: VertexRagStore {
                rag_resources: RagResources { rag_corpus },
                similarity_top_k: *top_k,
            },
        },
        GoogleVertexProviderTool::GoogleMaps => Tool::GoogleMaps(Empty {}),
    }));
    output
}

fn input_part(part: &InputPart) -> Result<Option<Part<'_>>, ModelError> {
    match part {
        InputPart::Text(text) => Ok((!text.text.is_empty()).then_some(Part::Text(&text.text))),
        InputPart::File(file) => file_part(file).map(Some),
        InputPart::Custom(_) => Err(ModelError::unsupported(
            "Vertex custom input parts are not supported",
        )),
    }
}

fn file_part(file: &FilePart) -> Result<Part<'_>, ModelError> {
    let mime_type = file.media_type.as_str();
    match &file.source {
        FileSource::Bytes(bytes) => Ok(Part::InlineData {
            mime_type,
            data: bytes,
        }),
        FileSource::Text(text) => Ok(Part::InlineData {
            mime_type,
            data: text.as_bytes(),
        }),
        FileSource::Url(url) => Ok(Part::FileData {
            mime_type,
            file_uri: url.as_str(),
        }),
        FileSource::ProviderReference { .. } => Err(ModelError::unsupported(
            "Vertex Gemini does not accept Gemini Files provider references",
        )
        .with_stage(ErrorStage::RequestEncoding)),
    }
}

fn assistant_part(part: &AssistantPart) -> Option<Part<'_>> {
    match part {
        AssistantPart::Text(text) if !text.text.is_empty() => Some(Part::Text(&text.text)),
        AssistantPart::ToolCall(call) => Some(Part::FunctionCall {
            id: call.provider_item_id.as_deref(),
            name: &call.name,
            args: &call.input,
        }),
        _ => None,
    }
}

fn function_response<'a>(
    result: &'a ToolResultPart,
    identities: &BTreeMap<&str, ToolIdentity<'a>>,
) -> Result<Part<'a>, ModelError> {
    let identity = identities
        .get(result.tool_call_id.as_str())
        .ok_or_else(|| {
            ModelError::invalid_request("Vertex tool result could not resolve its function name")
        })?;
    let output = match &result.content {
        ToolContent::Text(value) => ToolOutput::Str(value),
        ToolContent::LoadTools { text: value, .. } => ToolOutput::Str(value),
        ToolContent::Json(value) => ToolOutput::Json(value),
        ToolContent::Denied { reason } => {
            ToolOutput::Str(reason.as_deref().unwrap_or("Tool call execution denied."))
        }
        ToolContent::Mixed(values) => {
            if values
                .iter()
                .any(|value| matches!(value, oven_sdk::ContentValue::File(_)))
            {
                return Err(ModelError::unsupported(
                    "files in tool results are not deliverable via google-vertex",
                ));
            }
            ToolOutput::Mixed(values)
        }
    };
    Ok(Part::FunctionResponse {
        id: identity.provider_id,
        name: identity.name,
        is_error: result.is_error,
        output,
    })
}

fn validate_tool_result_files(result: &ToolResultPart) -> Result<(), ModelError> {
    if let ToolContent::Mixed(values) = &result.content
        && values
            .iter()
            .any(|value| matches!(value, oven_sdk::ContentValue::File(_)))
    {
        return Err(ModelError::unsupported(
            "files in tool results are not deliverable via google-vertex",
        ));
    }
    Ok(())
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

fn decode_replay<'a>(
    payload: &'a JsonValue,
    normalized: &[AssistantPart],
) -> Result<&'a [JsonValue], &'static str> {
    if payload.get("format").and_then(JsonValue::as_str) != Some(REPLAY_FORMAT) {
        return Err("Vertex replay format is invalid");
    }
    if payload.pointer("/content/role").and_then(JsonValue::as_str) != Some("model") {
        return Err("Vertex replay content role is invalid");
    }
    let parts = payload
        .pointer("/content/parts")
        .and_then(JsonValue::as_array)
        .ok_or("Vertex replay parts are invalid")?;
    if parts.iter().any(|part| !valid_native_part(part)) {
        return Err("Vertex replay contains an unsupported native part");
    }
    if !semantics_match(parts, normalized) {
        return Err("Vertex replay payload did not match normalized content");
    }
    if !oven_sdk::replay::vertex_signatures_match(parts, normalized) {
        return Err("Vertex required thought-signature witness does not match native state");
    }
    Ok(parts)
}

/// Applies `generateContent` part eligibility to validated replay parts without copying them.
///
/// Mirrors `oven_sdk::replay::eligible_blocks` with `BlockFormat::GenerateContent`: thought
/// signatures need target reasoning support and the same known wire model, thought parts need
/// target reasoning support, and a tool continuation cannot silently lose reasoning state.
fn eligible_parts<'a>(
    artifact: &NativeReplayArtifact,
    target: &oven_sdk::ModelId,
    provider_tools: bool,
    reasoning_supported: bool,
    parts: &'a [JsonValue],
    warnings: &mut Vec<String>,
) -> Result<Vec<Part<'a>>, ModelError> {
    const EXCLUSIVE: [&str; 6] = [
        "text",
        "functionCall",
        "toolCall",
        "toolResponse",
        "executableCode",
        "codeExecutionResult",
    ];
    const SERVER: [&str; 4] = [
        "toolCall",
        "toolResponse",
        "executableCode",
        "codeExecutionResult",
    ];
    let encrypted_allowed = reasoning_supported && artifact.source_wire_model_id() == Some(target);
    let mut omitted = false;
    let mut has_calls = false;
    let mut eligible = Vec::with_capacity(parts.len());
    for part in parts {
        let object = part
            .as_object()
            .ok_or_else(|| ModelError::replay("native replay block must be an object"))?;
        if EXCLUSIVE
            .iter()
            .filter(|key| object.contains_key(**key))
            .count()
            != 1
        {
            return Err(ModelError::replay("ambiguous generateContent native part"));
        }
        let server_part = SERVER.iter().any(|key| object.contains_key(*key));
        if !object.keys().all(|key| {
            matches!(
                key.as_str(),
                "text" | "thought" | "thoughtSignature" | "functionCall"
            ) || provider_tools && SERVER.contains(&key.as_str())
        }) {
            return Err(ModelError::replay(
                "target has no explicit support for these custom generateContent replay fields",
            ));
        }
        if object.get("text").is_some_and(|value| !value.is_string())
            || object
                .get("thought")
                .is_some_and(|value| !value.is_boolean())
            || object
                .get("thoughtSignature")
                .is_some_and(|value| !value.is_string())
            || object.contains_key("text") && object.contains_key("functionCall")
        {
            return Err(ModelError::replay("invalid generateContent replay part"));
        }
        if let Some(call) = object.get("functionCall") {
            let call = call
                .as_object()
                .ok_or_else(|| ModelError::replay("invalid native functionCall"))?;
            if !call
                .keys()
                .all(|key| matches!(key.as_str(), "id" | "name" | "args"))
                || !call.get("name").is_some_and(JsonValue::is_string)
                || call.get("id").is_some_and(|value| !value.is_string())
                || call.get("args").is_some_and(|value| !value.is_object())
            {
                return Err(ModelError::replay("invalid native functionCall fields"));
            }
        }
        has_calls |= object.contains_key("functionCall") || server_part;
        if !(object.contains_key("text")
            || object.contains_key("functionCall")
            || (provider_tools && server_part))
        {
            return Err(ModelError::replay(
                "target has no explicit support for this custom generateContent replay block",
            ));
        }
        let strip_signature = !encrypted_allowed && object.contains_key("thoughtSignature");
        omitted |= strip_signature;
        let reasoning = object.get("thought").and_then(JsonValue::as_bool) == Some(true);
        if reasoning && !reasoning_supported {
            omitted = true;
        } else {
            eligible.push(Part::Native {
                value: object,
                strip_signature,
            });
        }
    }
    if omitted {
        if has_calls {
            return Err(ModelError::replay(
                "native tool continuation cannot omit ineligible reasoning state",
            ));
        }
        warnings.push("native reasoning excluded: target support or equal known wire model_id is required for encrypted reasoning".into());
    }
    Ok(eligible)
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

/// The replay-relevant meaning of one native part, borrowed from the payload.
enum NativeSemantic<'a> {
    Text {
        reasoning: bool,
        text: &'a str,
    },
    ToolCall {
        /// `None` compares as JSON null.
        provider_id: Option<&'a JsonValue>,
        name: &'a str,
        /// `None` compares as an empty object.
        args: Option<&'a JsonValue>,
    },
}

impl<'a> NativeSemantic<'a> {
    fn from_part(part: &'a JsonValue) -> Option<Self> {
        if let Some(text) = part.get("text").and_then(JsonValue::as_str) {
            if text.is_empty() && part.get("thoughtSignature").is_some() {
                return None;
            }
            return Some(Self::Text {
                reasoning: part.get("thought").and_then(JsonValue::as_bool) == Some(true),
                text,
            });
        }
        part.get("functionCall").map(|call| Self::ToolCall {
            provider_id: call.get("id"),
            name: call.get("name").and_then(JsonValue::as_str).unwrap_or(""),
            args: call.get("args"),
        })
    }

    /// Whether this native part carries the same meaning as one normalized part.
    fn matches(&self, expected: &AssistantPart) -> bool {
        match (self, expected) {
            (Self::Text { reasoning, text }, AssistantPart::Text(expected)) => {
                !reasoning && *text == expected.text
            }
            (Self::Text { reasoning, text }, AssistantPart::Reasoning(expected)) => {
                *reasoning && *text == expected.text
            }
            (
                Self::ToolCall {
                    provider_id,
                    name,
                    args,
                },
                AssistantPart::ToolCall(call),
            ) => {
                let id_matches = match call.provider_item_id.as_deref() {
                    Some(expected) => provider_id.and_then(JsonValue::as_str) == Some(expected),
                    None => provider_id.is_none_or(JsonValue::is_null),
                };
                id_matches
                    && *name == call.name
                    && match args {
                        Some(args) => **args == call.input,
                        None => call
                            .input
                            .as_object()
                            .is_some_and(serde_json::Map::is_empty),
                    }
            }
            _ => false,
        }
    }
}

/// Compares native and normalized semantics pairwise, without materializing either side.
fn semantics_match(parts: &[JsonValue], normalized: &[AssistantPart]) -> bool {
    let mut native = parts.iter().filter_map(NativeSemantic::from_part);
    for expected in normalized.iter().filter(|part| {
        matches!(
            part,
            AssistantPart::Text(_) | AssistantPart::Reasoning(_) | AssistantPart::ToolCall(_)
        )
    }) {
        if !native.next().is_some_and(|part| part.matches(expected)) {
            return false;
        }
    }
    native.next().is_none()
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

    fn artifact(source_model: Option<&str>) -> NativeReplayArtifact {
        let artifact = NativeReplayArtifact::new(
            oven_sdk::AdapterId::new("oven.google.vertex.generate-content"),
            oven_sdk::NativeContextScope::new(
                oven_sdk::ProviderId::new("google.vertex"),
                oven_sdk::ModelId::new("target"),
                oven_sdk::ResourceId::new("scope").unwrap(),
            )
            .unwrap(),
            serde_json::json!({"role":"model","parts":[]}),
        )
        .unwrap();
        match source_model {
            Some(model) => artifact
                .with_source_wire_model_id(oven_sdk::ModelId::new(model))
                .unwrap(),
            None => artifact,
        }
    }

    #[test]
    fn borrowed_eligibility_matches_the_shared_generate_content_codec() {
        let cases = [
            serde_json::json!([
                {"text":"think","thought":true,"thoughtSignature":"sig-1"},
                {"text":"answer","thoughtSignature":"sig-2"}
            ]),
            serde_json::json!([
                {"text":"think","thought":true,"thoughtSignature":"sig-1"},
                {"functionCall":{"id":"c","name":"lookup","args":{}},"thoughtSignature":"sig-2"}
            ]),
            serde_json::json!([
                {"toolCall":{"toolType":"URL_CONTEXT","args":{}},"thoughtSignature":"sig"},
                {"text":"done"}
            ]),
            serde_json::json!([{"text":"a","functionCall":{"name":"x"}}]),
            serde_json::json!([{"thought":true}]),
            serde_json::json!([{"text":7}]),
            serde_json::json!([{"text":"a","partMetadata":{}}]),
            serde_json::json!([{"functionCall":{"name":"x","extra":1}}]),
            serde_json::json!([{"functionCall":"x"}]),
            serde_json::json!(["not an object"]),
        ];
        for parts in cases {
            let parts = parts.as_array().unwrap();
            for source in [Some("target"), Some("other"), None] {
                for provider_tools in [false, true] {
                    for reasoning in [false, true] {
                        let artifact = artifact(source);
                        let mut core_warnings = Vec::new();
                        let core = oven_sdk::replay::eligible_blocks(
                            &artifact,
                            &oven_sdk::ModelId::new("target"),
                            oven_sdk::replay::BlockFormat::GenerateContent { provider_tools },
                            reasoning,
                            parts.clone(),
                            &mut core_warnings,
                        );
                        let mut warnings = Vec::new();
                        let borrowed = eligible_parts(
                            &artifact,
                            &oven_sdk::ModelId::new("target"),
                            provider_tools,
                            reasoning,
                            parts,
                            &mut warnings,
                        )
                        .map(|parts| {
                            parts
                                .iter()
                                .map(|part| serde_json::to_value(part).unwrap())
                                .collect::<Vec<_>>()
                        });
                        let context = format!("{parts:?} {source:?} {provider_tools} {reasoning}");
                        match (core, borrowed) {
                            (Ok(core), Ok(borrowed)) => assert_eq!(core, borrowed, "{context}"),
                            (Err(core), Err(borrowed)) => {
                                assert_eq!(core.to_string(), borrowed.to_string(), "{context}");
                            }
                            (core, borrowed) => panic!("{context}: {core:?} vs {borrowed:?}"),
                        }
                        assert_eq!(core_warnings, warnings, "{context}");
                    }
                }
            }
        }
    }

    #[test]
    fn in_place_semantics_match_text_reasoning_and_provider_call_ids_exactly() {
        let mut with_id = oven_sdk::ToolCallPart::new("c1", "lookup", serde_json::json!({}));
        with_id.provider_item_id = Some("p1".into());
        let normalized = vec![
            AssistantPart::Reasoning(oven_sdk::ReasoningPart::new("think")),
            AssistantPart::Text(oven_sdk::TextPart::new("")),
            AssistantPart::ToolCall(with_id),
            AssistantPart::ToolCall(oven_sdk::ToolCallPart::new(
                "c2",
                "lookup",
                serde_json::json!({"q":1}),
            )),
            AssistantPart::Custom(oven_sdk::CustomPart::new("ignored", serde_json::json!({}))),
        ];
        let parts = serde_json::json!([
            {"text":"think","thought":true,"thoughtSignature":"sig"},
            {"text":""},
            {"functionCall":{"id":"p1","name":"lookup"}},
            {"functionCall":{"id":null,"name":"lookup","args":{"q":1}}},
            {"text":"","thoughtSignature":"tail"}
        ]);
        assert!(semantics_match(parts.as_array().unwrap(), &normalized));
        let mut missing_id = parts.clone();
        missing_id[3]["functionCall"]
            .as_object_mut()
            .unwrap()
            .remove("id");
        assert!(semantics_match(missing_id.as_array().unwrap(), &normalized));
        let mutations: [fn(&mut JsonValue); 8] = [
            |parts| parts[0]["thought"] = JsonValue::Bool(false),
            |parts| parts[1]["text"] = JsonValue::String(" ".into()),
            |parts| parts[2]["functionCall"]["args"] = serde_json::json!({"x":1}),
            |parts| parts[2]["functionCall"]["id"] = JsonValue::String("other".into()),
            |parts| parts[3]["functionCall"]["id"] = JsonValue::String("p2".into()),
            |parts| parts[3]["functionCall"]["name"] = JsonValue::String("other".into()),
            |parts| parts[4]["text"] = JsonValue::String("visible".into()),
            |parts| {
                parts.as_array_mut().unwrap().remove(1);
            },
        ];
        for mutate in mutations {
            let mut changed = parts.clone();
            mutate(&mut changed);
            assert!(!semantics_match(changed.as_array().unwrap(), &normalized));
        }
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
