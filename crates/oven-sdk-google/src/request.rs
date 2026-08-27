//! Request validation, encoding, and scoped private replay validation.

use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use oven_sdk::{
    AssistantPart, Capability, ContentValue, ErrorStage, FilePart, FileSource, HistoryTurn,
    InputPart, JsonValue, LanguageModelDescriptor, ModelCapabilities, ModelError,
    NativeContextScope, ProviderId, ReplayDecision, ReplayDisposition, ReplayOutcome, ReplayPolicy,
    Request, ResponseFormat, SystemPart, ToolChoice, ToolContent, ToolResultPart,
};

use crate::{
    model::{GoogleGenerateContentSettings, GoogleThinkingSettings},
    options::{GoogleProviderTool, GoogleRequestOptions, GoogleThinkingConfig, GoogleToolOptions},
};

const MIB: usize = 1024 * 1024;
pub(crate) const GOOGLE_REQUEST_MAX_BYTES: usize = 20 * MIB;
const GOOGLE_MAX_IMAGES: usize = 3_600;
const GOOGLE_MAX_VIDEOS: usize = 10;
const SKIP_THOUGHT_SIGNATURE_VALIDATOR: &str = "skip_thought_signature_validator";

pub(crate) struct Encoded {
    pub(crate) body: JsonValue,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
}

pub(crate) struct ParsedOptions {
    request: GoogleRequestOptions,
    strict_tools: Vec<bool>,
}

pub(crate) fn validate_request(
    request: &Request,
    parsed: &ParsedOptions,
    capabilities: &ModelCapabilities,
    provider_id: &ProviderId,
    settings: &GoogleGenerateContentSettings,
) -> Result<(), ModelError> {
    let options = &parsed.request;
    request.validate_for(capabilities)?;
    if request.inference.max_output_tokens == Some(0) {
        return Err(ModelError::invalid_request(
            "Google max_output_tokens must be at least 1",
        ));
    }
    if let (Some(requested), Some(maximum)) = (
        request.inference.max_output_tokens,
        capabilities.limits.output,
    ) && requested > maximum
    {
        return Err(ModelError::invalid_request(
            "max_output_tokens exceeds the selected Gemini model limit",
        ));
    }
    if request
        .inference
        .temperature
        .is_some_and(|value| !(0.0..=2.0).contains(&value))
    {
        return Err(ModelError::invalid_request(
            "Google temperature must be between 0 and 2",
        ));
    }
    if request
        .inference
        .top_p
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(ModelError::invalid_request(
            "Google top_p must be between 0 and 1",
        ));
    }
    if options
        .presence_penalty
        .is_some_and(|value| !(-2.0..=2.0).contains(&value))
        || options
            .frequency_penalty
            .is_some_and(|value| !(-2.0..=2.0).contains(&value))
    {
        return Err(ModelError::invalid_request(
            "Google presence and frequency penalties must be between -2 and 2",
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
            "tool calling is not supported by this explicit Google model declaration",
        ));
    }
    if !options.provider_tools.is_empty()
        && !capabilities.features.contains(Capability::PROVIDER_TOOLS)
    {
        return Err(ModelError::unsupported(
            "provider tools are not supported by this explicit Google model declaration",
        ));
    }
    if let Some(cached_content) = &options.cached_content {
        if !capabilities.features.contains(Capability::PROMPT_CACHING) {
            return Err(ModelError::unsupported(
                "prompt caching is not supported by this explicit Google model declaration",
            ));
        }
        let Some(id) = cached_content.strip_prefix("cachedContents/") else {
            return Err(ModelError::invalid_request(
                "Google cached_content must use `cachedContents/{id}`",
            ));
        };
        if id.is_empty() || id.contains('/') || id.chars().any(char::is_whitespace) {
            return Err(ModelError::invalid_request(
                "Google cached_content must use `cachedContents/{id}`",
            ));
        }
    }
    if !request.tools.is_empty()
        && !options.provider_tools.is_empty()
        && !settings.tools.mixed_client_and_provider_tools
    {
        return Err(ModelError::unsupported(
            "mixing client functions with provider tools is disabled by Google tool settings",
        ));
    }
    let has_strict = parsed.strict_tools.iter().any(|strict| *strict);
    if has_strict && !settings.tools.strict_functions {
        return Err(ModelError::unsupported(
            "validated strict functions are disabled by Google tool settings",
        ));
    }
    resolved_thinking_config(request, options, &settings.thinking)?;
    let mut media_counts = MediaCounts::default();
    for turn in &request.history {
        match turn {
            HistoryTurn::System(message) => {
                if message
                    .content
                    .iter()
                    .any(|part| matches!(part, SystemPart::Custom(_)))
                {
                    return Err(ModelError::unsupported(
                        "Google system messages support text only",
                    ));
                }
            }
            HistoryTurn::User(message) => {
                for part in &message.content {
                    match part {
                        InputPart::File(file) => {
                            validate_file(file, provider_id)?;
                            media_counts.observe(file)?;
                        }
                        InputPart::Custom(_) => {
                            return Err(ModelError::unsupported(
                                "Google user custom parts are not supported",
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

pub(crate) fn encode_request(
    request: &Request,
    parsed: &ParsedOptions,
    descriptor: &LanguageModelDescriptor,
    settings: &GoogleGenerateContentSettings,
    native_context_scope: &NativeContextScope,
) -> Result<Encoded, ModelError> {
    let options = &parsed.request;
    let thinking_config = resolved_thinking_config(request, options, &settings.thinking)?;
    let replay_policy = descriptor.capabilities.replay.policy;
    let mut replay = ReplayOutcome::default();
    let mut warnings = Vec::new();
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut tool_names = BTreeMap::<String, String>::new();
    let mut previous_tool_content = false;
    let current_turn_start = request
        .history
        .iter()
        .rposition(|turn| matches!(turn, HistoryTurn::User(_)))
        .map_or(0, |index| index + 1);

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
                    .map(|part| input_part(part, &descriptor.identity.provider_id))
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
                        match validated_replay_parts(artifact.payload(), &turn.message.content) {
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
                                    | AssistantPart::Source(_)
                                    | AssistantPart::File(_)
                            ) || matches!(part, AssistantPart::Custom(custom) if server_tool_context_field(custom).is_none())
                        }) {
                            warnings.push(format!(
                                "Google provider-only assistant state omitted during normalized reconstruction at history index {history_index}"
                            ));
                        }
                        let (parts, injected_sentinel, omitted_server_tools) =
                            reconstructed_assistant_parts(
                                &turn.message.content,
                                settings.tools.current_turn_signature_sentinel,
                                history_index >= current_turn_start,
                            );
                        if injected_sentinel {
                            warnings.push(format!(
                                "Google reconstructed Gemini 3 current-turn tool history without an opaque thought signature at history index {history_index}; injected the documented `skip_thought_signature_validator` sentinel"
                            ));
                        }
                        let mut omitted_counts = BTreeMap::<&str, usize>::new();
                        for field in omitted_server_tools {
                            *omitted_counts.entry(field).or_default() += 1;
                        }
                        for (field, count) in omitted_counts {
                            warnings.push(format!(
                                "Google omitted {count} normalized server-tool `{field}` part(s) during reconstruction at history index {history_index} because safe replay requires the original opaque thought signature"
                            ));
                        }
                        parts
                    }
                };
                for part in &turn.message.content {
                    if let AssistantPart::ToolCall(call) = part {
                        tool_names.insert(call.id.clone(), call.name.clone());
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
            serde_json::to_value(&options.stop_sequences).expect("stop sequences are serializable");
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
    if let Some(value) = thinking_config {
        generation_config["thinkingConfig"] =
            serde_json::to_value(value).expect("thinking config is serializable");
    }
    if let ResponseFormat::Json { schema } = &request.response_format {
        let mut text = serde_json::json!({"mimeType":"APPLICATION_JSON"});
        if let Some(schema) = schema {
            text["schema"] = schema.as_value().clone();
        }
        generation_config["responseFormat"] = serde_json::json!({"text":text});
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
                "parameters":tool.input_schema.as_value(),
            })).collect::<Vec<_>>()
        }));
    }
    if !tools.is_empty() {
        root["tools"] = JsonValue::Array(tools);
    }
    if !request.tools.is_empty() {
        let strict = parsed.strict_tools.iter().any(|strict| *strict);
        let mixed = !options.provider_tools.is_empty();
        let (mode, names) = match &request.tool_choice {
            ToolChoice::Auto if mixed => ("VALIDATED", None),
            ToolChoice::Auto => (if strict { "VALIDATED" } else { "AUTO" }, None),
            ToolChoice::Required if mixed => ("ANY", None),
            ToolChoice::Required => (if strict { "VALIDATED" } else { "ANY" }, None),
            ToolChoice::None => ("NONE", None),
            ToolChoice::Tool(name) if mixed => ("ANY", Some(vec![name.clone()])),
            ToolChoice::Tool(name) => (
                if strict { "VALIDATED" } else { "ANY" },
                Some(vec![name.clone()]),
            ),
        };
        root["toolConfig"] = serde_json::json!({"functionCallingConfig":{"mode":mode}});
        if let Some(names) = names {
            root["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"] =
                serde_json::to_value(names).expect("tool names are serializable");
        }
        if mixed {
            root["toolConfig"]["includeServerSideToolInvocations"] = JsonValue::Bool(true);
        }
    }
    if let Some(value) = &options.service_tier {
        root["serviceTier"] = JsonValue::String(value.clone());
    }
    if let Some(value) = &options.cached_content {
        root["cachedContent"] = JsonValue::String(value.clone());
    }
    if !options.safety_settings.is_empty() {
        root["safetySettings"] = serde_json::to_value(&options.safety_settings)
            .expect("safety settings are serializable");
    }
    Ok(Encoded {
        body: root,
        replay,
        warnings,
    })
}

pub(crate) fn options(request: &Request) -> Result<ParsedOptions, ModelError> {
    let request_options = request
        .provider_options
        .get("google")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| ModelError::invalid_request("invalid Google request options"))
        })
        .transpose()?
        .unwrap_or_default();
    let strict_tools = request
        .tools
        .iter()
        .map(|tool| {
            tool.provider_options
                .get("google")
                .and_then(|value| serde_json::from_value(value.clone()).ok())
                .is_some_and(|options: GoogleToolOptions| options.strict)
        })
        .collect();
    Ok(ParsedOptions {
        request: request_options,
        strict_tools,
    })
}

fn resolved_thinking_config(
    request: &Request,
    options: &GoogleRequestOptions,
    settings: &GoogleThinkingSettings,
) -> Result<Option<GoogleThinkingConfig>, ModelError> {
    if let Some(effort) = request.inference.reasoning_effort.as_deref() {
        if options.thinking_config.is_some() {
            return Err(ModelError::invalid_request(
                "normalized reasoning_effort cannot be combined with Google thinking_config",
            ));
        }
        return match settings {
            GoogleThinkingSettings::Unsupported => Err(ModelError::unsupported(
                "normalized reasoning_effort is disabled by Google thinking settings",
            )),
            GoogleThinkingSettings::Budget { effort_budgets } => effort_budgets
                .get(effort)
                .copied()
                .map(|thinking_budget| GoogleThinkingConfig {
                    thinking_budget: Some(thinking_budget),
                    ..Default::default()
                })
                .map(Some)
                .ok_or_else(|| {
                    ModelError::unsupported(
                        "normalized reasoning_effort has no configured Google budget mapping",
                    )
                }),
            GoogleThinkingSettings::Level { effort_levels } => effort_levels
                .get(effort)
                .cloned()
                .map(|thinking_level| GoogleThinkingConfig {
                    thinking_level: Some(thinking_level),
                    ..Default::default()
                })
                .map(Some)
                .ok_or_else(|| {
                    ModelError::unsupported(
                        "normalized reasoning_effort has no configured Google level mapping",
                    )
                }),
        };
    }

    let Some(thinking) = options.thinking_config.clone() else {
        return Ok(None);
    };
    if thinking.thinking_budget.is_some() && thinking.thinking_level.is_some() {
        return Err(ModelError::invalid_request(
            "Google thinking_budget and thinking_level cannot both be set",
        ));
    }
    match settings {
        GoogleThinkingSettings::Unsupported => Err(ModelError::unsupported(
            "thinking controls are disabled by Google thinking settings",
        )),
        GoogleThinkingSettings::Budget { .. } if thinking.thinking_level.is_some() => {
            Err(ModelError::unsupported(
                "this Google configuration accepts a thinking budget, not a thinking level",
            ))
        }
        GoogleThinkingSettings::Level { .. } if thinking.thinking_budget.is_some() => {
            Err(ModelError::unsupported(
                "this Google configuration accepts a thinking level, not a thinking budget",
            ))
        }
        _ => Ok(Some(thinking)),
    }
}

fn provider_tools(tools: &[GoogleProviderTool]) -> Vec<JsonValue> {
    tools
        .iter()
        .map(|tool| match tool {
            GoogleProviderTool::GoogleSearch => serde_json::json!({"googleSearch":{}}),
            GoogleProviderTool::UrlContext => serde_json::json!({"urlContext":{}}),
            GoogleProviderTool::CodeExecution => serde_json::json!({"codeExecution":{}}),
            GoogleProviderTool::FileSearch { stores } => {
                serde_json::json!({"fileSearch":{"fileSearchStoreNames":stores}})
            }
            GoogleProviderTool::GoogleMaps => serde_json::json!({"googleMaps":{}}),
        })
        .collect()
}

fn input_part(part: &InputPart, provider_id: &ProviderId) -> Result<Option<JsonValue>, ModelError> {
    match part {
        InputPart::Text(text) => {
            Ok((!text.text.is_empty()).then(|| serde_json::json!({"text":text.text})))
        }
        InputPart::File(file) => file_part(file, provider_id).map(Some),
        InputPart::Custom(_) => Err(ModelError::unsupported(
            "Google custom input parts are not supported",
        )),
    }
}

fn file_part(file: &FilePart, provider_id: &ProviderId) -> Result<JsonValue, ModelError> {
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
        FileSource::ProviderReference { provider, id } if provider == provider_id => {
            Ok(serde_json::json!({"fileData":{"mimeType":file.media_type,"fileUri":id}}))
        }
        FileSource::ProviderReference { .. } => Err(ModelError::unsupported(
            "Gemini Files references must use provider `google`",
        )
        .with_stage(ErrorStage::RequestEncoding)),
    }
}

fn reconstructed_assistant_parts(
    parts: &[AssistantPart],
    gemini3: bool,
    current_turn: bool,
) -> (Vec<JsonValue>, bool, Vec<&'static str>) {
    let mut output = Vec::new();
    let mut saw_function_call = false;
    let mut injected_sentinel = false;
    let mut omitted_server_tools = Vec::new();
    for part in parts {
        match part {
            AssistantPart::Text(text) if !text.text.is_empty() => {
                output.push(serde_json::json!({"text":text.text}));
            }
            AssistantPart::ToolCall(call) => {
                let mut part = serde_json::json!({
                    "functionCall":{"id":call.id,"name":call.name,"args":call.input}
                });
                if gemini3 && current_turn && !saw_function_call {
                    part["thoughtSignature"] =
                        JsonValue::String(SKIP_THOUGHT_SIGNATURE_VALIDATOR.into());
                    injected_sentinel = true;
                }
                saw_function_call = true;
                output.push(part);
            }
            AssistantPart::Custom(custom) => {
                if let Some(field) = server_tool_context_field(custom) {
                    omitted_server_tools.push(field);
                }
            }
            _ => {}
        }
    }
    (output, injected_sentinel, omitted_server_tools)
}

fn server_tool_context_field(part: &oven_sdk::CustomPart) -> Option<&'static str> {
    match part.kind.as_str() {
        "google.server_tool_call" if part.data.get("toolCall").is_some() => Some("toolCall"),
        "google.server_tool_call" if part.data.get("executableCode").is_some() => {
            Some("executableCode")
        }
        "google.server_tool_result" if part.data.get("toolResponse").is_some() => {
            Some("toolResponse")
        }
        "google.server_tool_result" if part.data.get("codeExecutionResult").is_some() => {
            Some("codeExecutionResult")
        }
        _ => None,
    }
}

fn function_response(
    result: &ToolResultPart,
    names: &BTreeMap<String, String>,
) -> Result<JsonValue, ModelError> {
    let name = names.get(&result.tool_call_id).ok_or_else(|| {
        ModelError::invalid_request("Google tool result could not resolve its function name")
    })?;
    let response = if result.is_error {
        serde_json::json!({"error":tool_content(&result.content)?})
    } else {
        serde_json::json!({"output":tool_content(&result.content)?})
    };
    Ok(serde_json::json!({
        "functionResponse":{"id":result.tool_call_id,"name":name,"response":response}
    }))
}

fn tool_content(content: &ToolContent) -> Result<JsonValue, ModelError> {
    match content {
        ToolContent::Text(value) => Ok(JsonValue::String(value.clone())),
        ToolContent::Json(value) => Ok(value.clone()),
        ToolContent::Denied { reason } => Ok(JsonValue::String(
            reason
                .clone()
                .unwrap_or_else(|| "Tool call execution denied.".into()),
        )),
        ToolContent::Mixed(values) => {
            let mut output = Vec::new();
            for value in values {
                match value {
                    ContentValue::Text(value) => output.push(JsonValue::String(value.clone())),
                    ContentValue::Json(value) => output.push(value.clone()),
                    ContentValue::File(_) => {
                        return Err(ModelError::unsupported(
                            "multimodal Gemini function results are not enabled",
                        ));
                    }
                }
            }
            Ok(JsonValue::Array(output))
        }
    }
}

fn validate_tool_result_files(result: &ToolResultPart) -> Result<(), ModelError> {
    if let ToolContent::Mixed(values) = &result.content
        && values
            .iter()
            .any(|value| matches!(value, ContentValue::File(_)))
    {
        return Err(ModelError::unsupported(
            "files in tool results are not deliverable via google-gemini",
        ));
    }
    Ok(())
}

fn validate_file(file: &FilePart, provider_id: &ProviderId) -> Result<(), ModelError> {
    match &file.source {
        FileSource::Bytes(_) | FileSource::Text(_) => {}
        FileSource::Url(url) if url.scheme() == "https" => {}
        FileSource::Url(_) => {
            return Err(
                ModelError::unsupported("Google Gemini media URLs must use HTTPS")
                    .with_stage(ErrorStage::RequestEncoding),
            );
        }
        FileSource::ProviderReference { provider, id }
            if provider == provider_id && !id.trim().is_empty() => {}
        FileSource::ProviderReference { .. } => {
            return Err(ModelError::unsupported(
                "Gemini Files references must use provider `google`",
            )
            .with_stage(ErrorStage::RequestEncoding));
        }
    }
    Ok(())
}

#[derive(Default)]
struct MediaCounts {
    images: usize,
    videos: usize,
}

impl MediaCounts {
    fn observe(&mut self, file: &FilePart) -> Result<(), ModelError> {
        if image_mime(&file.media_type) {
            self.images += 1;
            if self.images > GOOGLE_MAX_IMAGES {
                return Err(ModelError::invalid_request(
                    "Google Gemini requests support at most 3,600 images",
                )
                .with_stage(ErrorStage::RequestEncoding));
            }
        } else if video_mime(&file.media_type) {
            self.videos += 1;
            if self.videos > GOOGLE_MAX_VIDEOS {
                return Err(ModelError::invalid_request(
                    "Google Gemini requests support at most 10 videos",
                )
                .with_stage(ErrorStage::RequestEncoding));
            }
        }
        Ok(())
    }
}

fn image_mime(value: &str) -> bool {
    value.starts_with("image/")
}
fn video_mime(value: &str) -> bool {
    value.starts_with("video/")
}

fn validate_schema(schema: &JsonValue) -> Result<(), ModelError> {
    validate_schema_keywords(schema)?;
    let mut identifiers = BTreeMap::new();
    collect_schema_identifiers(schema, None, &mut identifiers)?;
    validate_schema_recursion(schema, schema, &identifiers, false, &mut Vec::new())
}

fn validate_schema_keywords(schema: &JsonValue) -> Result<(), ModelError> {
    let JsonValue::Object(object) = schema else {
        return Ok(());
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
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "anyOf",
        "oneOf",
        "properties",
        "additionalProperties",
        "required",
        "propertyOrdering",
    ];
    if object.keys().any(|key| !ALLOWED.contains(&key.as_str())) {
        return Err(ModelError::unsupported(
            "JSON Schema uses a keyword unsupported by current Gemini responseFormat",
        )
        .with_stage(ErrorStage::RequestEncoding));
    }
    if object.contains_key("$ref")
        && object
            .keys()
            .any(|key| key != "$ref" && !key.starts_with('$'))
    {
        return Err(ModelError::unsupported(
            "Gemini JSON Schema `$ref` subschemas may contain only `$`-prefixed siblings",
        )
        .with_stage(ErrorStage::RequestEncoding));
    }
    for key in ["properties", "$defs"] {
        if let Some(JsonValue::Object(values)) = object.get(key) {
            for value in values.values() {
                validate_schema_keywords(value)?;
            }
        }
    }
    if let Some(value) = object.get("items") {
        validate_schema_keywords(value)?;
    }
    if let Some(JsonValue::Object(_)) = object.get("additionalProperties") {
        validate_schema_keywords(&object["additionalProperties"])?;
    }
    for key in ["anyOf", "oneOf", "prefixItems"] {
        if let Some(JsonValue::Array(values)) = object.get(key) {
            for value in values {
                validate_schema_keywords(value)?;
            }
        }
    }
    Ok(())
}

fn collect_schema_identifiers<'a>(
    schema: &'a JsonValue,
    base_id: Option<&str>,
    identifiers: &mut BTreeMap<String, &'a JsonValue>,
) -> Result<(), ModelError> {
    let JsonValue::Object(object) = schema else {
        return Ok(());
    };
    let declared_id = object
        .get("$id")
        .map(|value| {
            value.as_str().ok_or_else(|| {
                ModelError::invalid_request("Gemini JSON Schema `$id` must be a string")
                    .with_stage(ErrorStage::RequestEncoding)
            })
        })
        .transpose()?;
    let current_id = declared_id.or(base_id);
    if let Some(id) = object.get("$id").and_then(JsonValue::as_str) {
        identifiers.insert(id.to_owned(), schema);
    }
    if let Some(anchor) = object.get("$anchor") {
        let Some(anchor) = anchor.as_str().filter(|anchor| !anchor.is_empty()) else {
            return Err(ModelError::invalid_request(
                "Gemini JSON Schema `$anchor` must be a non-empty string",
            )
            .with_stage(ErrorStage::RequestEncoding));
        };
        identifiers.insert(format!("#{anchor}"), schema);
        if let Some(base) = current_id {
            identifiers.insert(
                format!("{}#{anchor}", base.split('#').next().unwrap_or(base)),
                schema,
            );
        }
    }
    for key in ["properties", "$defs"] {
        if let Some(JsonValue::Object(values)) = object.get(key) {
            for value in values.values() {
                collect_schema_identifiers(value, current_id, identifiers)?;
            }
        }
    }
    for key in ["items", "additionalProperties"] {
        if let Some(value @ JsonValue::Object(_)) = object.get(key) {
            collect_schema_identifiers(value, current_id, identifiers)?;
        }
    }
    for key in ["anyOf", "oneOf", "prefixItems"] {
        if let Some(JsonValue::Array(values)) = object.get(key) {
            for value in values {
                collect_schema_identifiers(value, current_id, identifiers)?;
            }
        }
    }
    Ok(())
}

fn validate_schema_recursion<'a>(
    schema: &'a JsonValue,
    root: &'a JsonValue,
    identifiers: &BTreeMap<String, &'a JsonValue>,
    optional_property: bool,
    stack: &mut Vec<usize>,
) -> Result<(), ModelError> {
    let JsonValue::Object(object) = schema else {
        return Ok(());
    };
    let address = std::ptr::from_ref(schema).addr();
    stack.push(address);
    if let Some(reference) = object.get("$ref") {
        let Some(reference) = reference.as_str() else {
            return Err(
                ModelError::invalid_request("Gemini JSON Schema `$ref` must be a string")
                    .with_stage(ErrorStage::RequestEncoding),
            );
        };
        let target =
            if let Some(pointer) = reference.strip_prefix('#').filter(|v| v.starts_with('/')) {
                root.pointer(pointer)
            } else {
                identifiers.get(reference).copied()
            }
            .ok_or_else(|| {
                ModelError::invalid_request("Gemini JSON Schema contains an unresolved `$ref`")
                    .with_stage(ErrorStage::RequestEncoding)
            })?;
        let target_address = std::ptr::from_ref(target).addr();
        if stack.contains(&target_address) {
            if !optional_property {
                return Err(ModelError::unsupported(
                    "Gemini cyclic JSON Schema references are allowed only within non-required properties",
                )
                .with_stage(ErrorStage::RequestEncoding));
            }
        } else {
            validate_schema_recursion(target, root, identifiers, optional_property, stack)?;
        }
        stack.pop();
        return Ok(());
    }
    let required = object
        .get("required")
        .and_then(JsonValue::as_array)
        .into_iter()
        .flatten()
        .filter_map(JsonValue::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    if let Some(JsonValue::Object(properties)) = object.get("properties") {
        for (name, value) in properties {
            validate_schema_recursion(
                value,
                root,
                identifiers,
                !required.contains(name.as_str()),
                stack,
            )?;
        }
    }
    for key in ["items", "additionalProperties"] {
        if let Some(value @ JsonValue::Object(_)) = object.get(key) {
            validate_schema_recursion(value, root, identifiers, optional_property, stack)?;
        }
    }
    for key in ["anyOf", "oneOf", "prefixItems"] {
        if let Some(JsonValue::Array(values)) = object.get(key) {
            for value in values {
                validate_schema_recursion(value, root, identifiers, optional_property, stack)?;
            }
        }
    }
    stack.pop();
    Ok(())
}

pub(crate) fn validate_request_body_size(bytes: usize) -> Result<(), ModelError> {
    if bytes > GOOGLE_REQUEST_MAX_BYTES {
        return Err(ModelError::invalid_request(
            "Google Gemini request exceeds the 20 MiB AI Studio limit",
        )
        .with_stage(ErrorStage::RequestEncoding));
    }
    Ok(())
}

fn validated_replay_parts(
    payload: &JsonValue,
    normalized: &[AssistantPart],
) -> Result<Vec<JsonValue>, &'static str> {
    if payload.get("role").and_then(JsonValue::as_str) != Some("model") {
        return Err("Google replay content role is invalid");
    }
    let parts = payload
        .get("parts")
        .and_then(JsonValue::as_array)
        .ok_or("Google replay parts are invalid")?
        .clone();
    if parts.iter().any(|part| !valid_native_part(part)) {
        return Err("Google replay contains an unsupported native part");
    }
    if native_semantics(&parts) != normalized_semantics(normalized) {
        return Err("Google replay payload did not match normalized content");
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
    }) && ["toolCall", "toolResponse"].into_iter().all(|field| {
        object
            .get(field)
            .and_then(JsonValue::as_object)
            .and_then(|value| value.get("id"))
            .is_none_or(JsonValue::is_string)
    })
}

fn native_semantics(parts: &[JsonValue]) -> Vec<JsonValue> {
    parts
        .iter()
        .filter_map(|part| {
            if let Some(text) = part.get("text").and_then(JsonValue::as_str) {
                return Some(serde_json::json!({
                    "type":if part.get("thought").and_then(JsonValue::as_bool) == Some(true) {"reasoning"} else {"text"},
                    "text":text,
                }));
            }
            if let Some(call) = part.get("functionCall") {
                return Some(serde_json::json!({
                    "type":"tool_call",
                    "id":call.get("id").and_then(JsonValue::as_str).unwrap_or(""),
                    "name":call.get("name").and_then(JsonValue::as_str).unwrap_or(""),
                    "args":call.get("args").cloned().unwrap_or_else(|| serde_json::json!({})),
                }));
            }
            native_server_tool_semantic(part)
        })
        .collect()
}

fn native_server_tool_semantic(part: &JsonValue) -> Option<JsonValue> {
    if let Some(call) = part.get("toolCall") {
        let mut semantic = serde_json::json!({
            "type":"custom",
            "kind":"google.server_tool_call",
            "data":{"toolCall":{
                "toolType":call.get("toolType").cloned().unwrap_or(JsonValue::Null),
                "args":call.get("args").cloned().unwrap_or_else(|| serde_json::json!({}))
            }}
        });
        insert_nonempty_id(&mut semantic["data"]["toolCall"], call);
        if let Some(tool_name) = call.get("toolName").cloned() {
            semantic["data"]["toolCall"]["toolName"] = tool_name;
        }
        return Some(semantic);
    }
    if let Some(code) = part.get("executableCode") {
        let mut semantic = serde_json::json!({
            "type":"custom",
            "kind":"google.server_tool_call",
            "data":{"executableCode":{
                "language":code.get("language").cloned().unwrap_or(JsonValue::Null),
                "code":code.get("code").cloned().unwrap_or(JsonValue::Null)
            }}
        });
        if let Some(id) = code.get("id").cloned() {
            semantic["data"]["executableCode"]["id"] = id;
        }
        return Some(semantic);
    }
    if let Some(response) = part.get("toolResponse") {
        let mut semantic = serde_json::json!({
            "type":"custom",
            "kind":"google.server_tool_result",
            "data":{"toolResponse":{
                "toolType":response.get("toolType").cloned().unwrap_or(JsonValue::Null),
                "response":response.get("response").cloned().unwrap_or_else(|| serde_json::json!({}))
            }}
        });
        insert_nonempty_id(&mut semantic["data"]["toolResponse"], response);
        return Some(semantic);
    }
    if let Some(result) = part.get("codeExecutionResult") {
        let mut semantic = serde_json::json!({
            "type":"custom",
            "kind":"google.server_tool_result",
            "data":{"codeExecutionResult":{
                "outcome":result.get("outcome").cloned().unwrap_or(JsonValue::Null),
                "output":result.get("output").cloned().unwrap_or_else(|| JsonValue::String(String::new()))
            }}
        });
        if let Some(id) = result.get("id").cloned() {
            semantic["data"]["codeExecutionResult"]["id"] = id;
        }
        return Some(semantic);
    }
    None
}

fn insert_nonempty_id(target: &mut JsonValue, source: &JsonValue) {
    if let Some(id) = source
        .get("id")
        .and_then(JsonValue::as_str)
        .filter(|id| !id.is_empty())
    {
        target["id"] = JsonValue::String(id.to_owned());
    }
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
                "type":"tool_call","id":call.id,"name":call.name,"args":call.input
            })),
            AssistantPart::Custom(custom)
                if matches!(
                    custom.kind.as_str(),
                    "google.server_tool_call" | "google.server_tool_result"
                ) =>
            {
                let mut data = custom.data.clone();
                for field in ["toolCall", "toolResponse"] {
                    if let Some(value) = data.get_mut(field)
                        && value.get("id").and_then(JsonValue::as_str) == Some("")
                        && let Some(object) = value.as_object_mut()
                    {
                        object.remove("id");
                    }
                }
                Some(serde_json::json!({
                    "type":"custom","kind":custom.kind,"data":data
                }))
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oven_sdk::{FilePart, ProviderId};

    #[test]
    fn media_constructors_share_the_same_validation_path() {
        let provider_id = ProviderId::new("google");
        for file in [
            FilePart::image("image/png", FileSource::Bytes(vec![1].into())),
            FilePart::document("application/pdf", FileSource::Bytes(vec![1].into())),
            FilePart::audio("audio/mp3", FileSource::Bytes(vec![1].into())),
            FilePart::video("video/mp4", FileSource::Bytes(vec![1].into())),
        ] {
            validate_file(&file, &provider_id).unwrap();
        }
    }

    #[test]
    fn foreign_provider_reference_is_rejected() {
        let file = FilePart::image(
            "image/png",
            FileSource::ProviderReference {
                provider: ProviderId::new("other"),
                id: "files/x".into(),
            },
        );
        assert!(validate_file(&file, &ProviderId::new("google")).is_err());
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
    fn current_json_schema_allowlist_and_optional_recursion_are_supported() {
        validate_schema(&serde_json::json!({
            "$id":"https://example.test/tree",
            "$defs":{
                "node":{
                    "$anchor":"node",
                    "type":"object",
                    "properties":{
                        "name":{"type":"string"},
                        "child":{"$ref":"#node"}
                    },
                    "propertyOrdering":["name","child"]
                }
            },
            "type":"object",
            "properties":{"root":{"$ref":"#/$defs/node"}}
        }))
        .unwrap();
    }

    #[test]
    fn required_recursion_and_removed_keywords_are_rejected() {
        assert!(
            validate_schema(&serde_json::json!({
                "$defs":{
                    "node":{
                        "type":"object",
                        "properties":{"child":{"$ref":"#/$defs/node"}},
                        "required":["child"]
                    }
                },
                "$ref":"#/$defs/node"
            }))
            .is_err()
        );
        for keyword in ["const", "nullable", "pattern"] {
            assert!(validate_schema(&serde_json::json!({(keyword):true})).is_err());
        }
    }

    #[test]
    fn request_size_boundary_is_exact() {
        validate_request_body_size(GOOGLE_REQUEST_MAX_BYTES).unwrap();
        assert!(validate_request_body_size(GOOGLE_REQUEST_MAX_BYTES + 1).is_err());
    }

    #[test]
    fn server_tool_replay_semantics_ignore_signatures_but_reject_changed_context() {
        let normalized = vec![
            AssistantPart::Custom(oven_sdk::CustomPart::new(
                "google.server_tool_call",
                serde_json::json!({"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","toolName":"google_search","args":{"q":"rust"},"id":"s1"}}),
            )),
            AssistantPart::Custom(oven_sdk::CustomPart::new(
                "google.server_tool_call",
                serde_json::json!({"executableCode":{"language":"PYTHON","code":"print(1)","id":"code-1"}}),
            )),
            AssistantPart::Custom(oven_sdk::CustomPart::new(
                "google.server_tool_result",
                serde_json::json!({"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1","id":"code-1"}}),
            )),
            AssistantPart::Custom(oven_sdk::CustomPart::new(
                "google.server_tool_result",
                serde_json::json!({"toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"items":[]},"id":"s1"}}),
            )),
        ];
        let payload = serde_json::json!({
            "role":"model","parts":[
                {
                    "toolCall":{"toolType":"GOOGLE_SEARCH_WEB","toolName":"google_search","args":{"q":"rust"},"id":"s1"},
                    "thoughtSignature":"opaque-call"
                },
                {
                    "executableCode":{"language":"PYTHON","code":"print(1)","id":"code-1"},
                    "thoughtSignature":"opaque-code"
                },
                {
                    "codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1","id":"code-1"},
                    "thoughtSignature":"opaque-result"
                },
                {
                    "toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"items":[]},"id":"s1"},
                    "thoughtSignature":"opaque-response"
                }
            ]
        });
        let replayed = validated_replay_parts(&payload, &normalized).unwrap();
        assert_eq!(replayed[1]["thoughtSignature"], "opaque-code");
        assert_eq!(replayed[3]["thoughtSignature"], "opaque-response");
        let mut changed = payload;
        changed["parts"][0]["toolCall"]["args"]["q"] = JsonValue::String("changed".into());
        assert!(validated_replay_parts(&changed, &normalized).is_err());
    }

    #[test]
    fn optional_server_tool_ids_have_consistent_replay_semantics() {
        let custom = |kind, data| AssistantPart::Custom(oven_sdk::CustomPart::new(kind, data));
        let normalized = vec![
            custom(
                "google.server_tool_call",
                serde_json::json!({"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"q":"missing"}}}),
            ),
            custom(
                "google.server_tool_result",
                serde_json::json!({"toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"ok":true}}}),
            ),
            custom(
                "google.server_tool_call",
                serde_json::json!({"toolCall":{"toolType":"URL_CONTEXT","args":{},"id":""}}),
            ),
            custom(
                "google.server_tool_result",
                serde_json::json!({"toolResponse":{"toolType":"URL_CONTEXT","response":{},"id":""}}),
            ),
            custom(
                "google.server_tool_call",
                serde_json::json!({"toolCall":{"toolType":"FILE_SEARCH","args":{"query":"guide"},"id":"call-exact"}}),
            ),
            custom(
                "google.server_tool_result",
                serde_json::json!({"toolResponse":{"toolType":"FILE_SEARCH","response":{"matches":1},"id":"response-exact"}}),
            ),
        ];
        let payload = serde_json::json!({
            "role":"model","parts":[
                {"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"q":"missing"}}},
                {"toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"ok":true}}},
                {"toolCall":{"toolType":"URL_CONTEXT","args":{},"id":""}},
                {"toolResponse":{"toolType":"URL_CONTEXT","response":{},"id":""}},
                {"toolCall":{"toolType":"FILE_SEARCH","args":{"query":"guide"},"id":"call-exact"}},
                {"toolResponse":{"toolType":"FILE_SEARCH","response":{"matches":1},"id":"response-exact"}}
            ]
        });
        let replayed = validated_replay_parts(&payload, &normalized).unwrap();
        assert!(replayed[0]["toolCall"].get("id").is_none());
        assert_eq!(replayed[2]["toolCall"]["id"], "");
        assert_eq!(replayed[4]["toolCall"]["id"], "call-exact");
        assert_eq!(replayed[5]["toolResponse"]["id"], "response-exact");

        let mut changed = payload.clone();
        changed["parts"][4]["toolCall"]["id"] = JsonValue::String("changed".into());
        assert!(validated_replay_parts(&changed, &normalized).is_err());

        let mut invalid = payload;
        invalid["parts"][0]["toolCall"]["id"] = JsonValue::from(7);
        assert!(validated_replay_parts(&invalid, &normalized).is_err());
    }

    #[test]
    fn aggregate_image_and_video_count_boundaries_are_exact() {
        for (mime, accepted, rejected) in [
            ("image/png", GOOGLE_MAX_IMAGES, GOOGLE_MAX_IMAGES + 1),
            ("video/mp4", GOOGLE_MAX_VIDEOS, GOOGLE_MAX_VIDEOS + 1),
        ] {
            let file = FilePart::new(mime, FileSource::Bytes(Vec::new().into()));
            let mut counts = MediaCounts::default();
            for _ in 0..accepted {
                counts.observe(&file).unwrap();
            }
            let mut rejected_counts = MediaCounts::default();
            let error = (0..rejected)
                .find_map(|_| rejected_counts.observe(&file).err())
                .unwrap();
            assert_eq!(error.kind, oven_sdk::ModelErrorKind::InvalidRequest);
            assert_eq!(error.diagnostics.stage, ErrorStage::RequestEncoding);
        }
    }
}
