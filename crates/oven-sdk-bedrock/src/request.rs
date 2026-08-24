//! Bedrock request validation, encoding, media mapping, and replay decoding.

use std::collections::BTreeSet;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use oven_sdk::{
    AssistantPart, Capability, ContentValue, ErrorStage, FilePart, FileSource, HistoryTurn,
    InputPart, JsonValue, LanguageModelDescriptor, ModelCapabilities, ModelError,
    NativeContextScope, ReplayDecision, ReplayDisposition, ReplayOutcome, ReplayPolicy, Request,
    ResponseFormat, SystemPart, ToolChoice, ToolContent, ToolResultPart,
};

use crate::{
    BedrockReasoningWireFormat, BedrockStructuredOutput, REPLAY_FORMAT,
    options::BedrockRequestOptions,
};

pub(crate) struct Encoded {
    pub(crate) body: JsonValue,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
}

pub(crate) struct EncodeSettings {
    pub(crate) reasoning_wire_format: BedrockReasoningWireFormat,
    pub(crate) signed_reasoning: bool,
    pub(crate) structured_output: BedrockStructuredOutput,
    pub(crate) streaming: bool,
}

pub(crate) fn validate_request(
    request: &Request,
    options: &BedrockRequestOptions,
    capabilities: &ModelCapabilities,
    reasoning_wire_format: BedrockReasoningWireFormat,
    signed_reasoning: bool,
    structured_output: BedrockStructuredOutput,
) -> Result<(), ModelError> {
    request.validate_for(capabilities)?;
    validate_bucket_owner(options)?;
    if request.inference.max_output_tokens == Some(0) {
        return Err(ModelError::invalid_request(
            "Bedrock max_output_tokens must be at least 1",
        ));
    }
    if let (Some(requested), Some(maximum)) = (
        request.inference.max_output_tokens,
        capabilities.limits.output,
    ) && requested > maximum
    {
        return Err(ModelError::invalid_request(
            "max_output_tokens exceeds the selected Bedrock model limit",
        ));
    }
    if request
        .inference
        .temperature
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
        || request
            .inference
            .top_p
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(ModelError::invalid_request(
            "Bedrock temperature and top_p must be between 0 and 1",
        ));
    }
    if options.additional_model_response_field_paths.len() > 10
        || options
            .additional_model_response_field_paths
            .iter()
            .any(|value| value.is_empty() || value.len() > 256 || !value.starts_with('/'))
    {
        return Err(ModelError::invalid_request(
            "Bedrock additional response field paths must be 1-10 non-empty JSON Pointers",
        ));
    }
    if options.request_metadata.len() > 16
        || options
            .request_metadata
            .iter()
            .any(|(key, value)| key.is_empty() || key.len() > 256 || value.len() > 256)
    {
        return Err(ModelError::invalid_request(
            "Bedrock request metadata exceeds documented limits",
        ));
    }
    if let Some(fields) = &options.additional_model_request_fields {
        let fields = fields.as_object().ok_or_else(|| {
            ModelError::invalid_request(
                "Bedrock additional_model_request_fields must be a JSON object",
            )
        })?;
        if let Some(key) = fields.keys().find(|key| reserved_additional_key(key)) {
            return Err(ModelError::invalid_request(format!(
                "Bedrock additional_model_request_fields cannot set reserved key `{key}`; use typed reasoning or output controls"
            )));
        }
    }
    if matches!(
        request.response_format,
        ResponseFormat::Json { schema: None }
    ) {
        return Err(ModelError::invalid_request(
            "Bedrock structured output requires a JSON schema",
        ));
    }
    if !request.tools.is_empty() && request.tool_choice == ToolChoice::None {
        return Err(ModelError::unsupported(
            "Bedrock has no tool-choice none shape; omit tools to disable tool calling",
        ));
    }
    if matches!(request.response_format, ResponseFormat::Json { .. })
        && structured_output != BedrockStructuredOutput::JsonSchema
    {
        return Err(ModelError::unsupported(
            "native structured output is unsupported by this Bedrock configuration",
        ));
    }
    validate_reasoning_options(request, options, reasoning_wire_format, signed_reasoning)?;
    validate_media(request)?;
    let mut seen_messages = false;
    for turn in &request.history {
        match turn {
            HistoryTurn::System(message) => {
                if seen_messages {
                    return Err(ModelError::unsupported(
                        "Bedrock system messages must precede all conversation messages",
                    ));
                }
                if message
                    .content
                    .iter()
                    .any(|part| matches!(part, SystemPart::Custom(_)))
                {
                    return Err(ModelError::unsupported(
                        "Bedrock system messages support text only",
                    ));
                }
            }
            HistoryTurn::User(message) => {
                seen_messages = true;
                for part in &message.content {
                    match part {
                        InputPart::Text(_) => {}
                        InputPart::File(_) => {}
                        InputPart::Custom(_) => {
                            return Err(ModelError::unsupported(
                                "Bedrock custom input parts are unsupported",
                            ));
                        }
                    }
                }
            }
            HistoryTurn::Assistant(turn) => {
                seen_messages = true;
                for part in &turn.message.content {
                    match part {
                        AssistantPart::Text(_)
                        | AssistantPart::Reasoning(_)
                        | AssistantPart::ToolCall(_)
                        | AssistantPart::ToolResult(_) => {}
                        _ => {
                            return Err(ModelError::unsupported(
                                "Bedrock assistant history contains an unsupported part",
                            ));
                        }
                    }
                }
            }
            HistoryTurn::Tool(message) => {
                seen_messages = true;
                for result in &message.results {
                    validate_tool_content(&result.content)?;
                }
            }
        }
    }
    Ok(())
}

fn reserved_additional_key(key: &str) -> bool {
    let normalized = key
        .bytes()
        .filter(u8::is_ascii_alphanumeric)
        .map(|byte| byte.to_ascii_lowercase())
        .collect::<Vec<_>>();
    matches!(
        normalized.as_slice(),
        b"thinking" | b"outputconfig" | b"reasoningeffort" | b"reasoningconfig"
    )
}

pub(crate) fn encode_request(
    request: &Request,
    options: &BedrockRequestOptions,
    descriptor: &LanguageModelDescriptor,
    native_context_scope: &NativeContextScope,
    settings: EncodeSettings,
) -> Result<Encoded, ModelError> {
    let replay_policy = descriptor.capabilities.replay.policy;
    let mut replay = ReplayOutcome::default();
    let mut warnings = Vec::new();
    let mut system = Vec::new();
    let mut messages = Vec::<JsonValue>::new();
    let mut document_counter = 0_u64;

    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
                for part in &message.content {
                    if let SystemPart::Text(text) = part
                        && !text.text.is_empty()
                    {
                        system.push(serde_json::json!({"text":text.text}));
                    }
                }
            }
            HistoryTurn::User(message) => {
                let mut content = Vec::new();
                for part in &message.content {
                    match part {
                        InputPart::Text(text) if !text.text.is_empty() => {
                            content.push(serde_json::json!({"text":text.text}));
                        }
                        InputPart::File(file) => {
                            content.push(file_content(file, options, &mut document_counter)?)
                        }
                        _ => {}
                    }
                }
                push_message(&mut messages, "user", content);
            }
            HistoryTurn::Assistant(turn) => {
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
                        match decode_replay(
                            artifact.payload(),
                            &turn.message.content,
                            &descriptor.capabilities,
                            settings.signed_reasoning,
                        ) {
                            Ok(content) => {
                                replay.decisions.push(ReplayDecision {
                                    history_index,
                                    disposition: ReplayDisposition::Replayed,
                                });
                                native = Some(content);
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
                let inline_results = turn
                    .message
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        AssistantPart::ToolResult(result) => Some(result.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let content = match native {
                    Some(content) => content,
                    None => {
                        if replay_policy != ReplayPolicy::Never {
                            replay.decisions.push(ReplayDecision {
                                history_index,
                                disposition: ReplayDisposition::ReconstructedNormalized,
                            });
                        }
                        if turn
                            .message
                            .content
                            .iter()
                            .any(|part| matches!(part, AssistantPart::Reasoning(_)))
                        {
                            warnings.push(format!(
                                "Bedrock signed/redacted reasoning omitted during normalized reconstruction at history index {history_index}"
                            ));
                        }
                        turn.message
                            .content
                            .iter()
                            .filter_map(|part| match part {
                                AssistantPart::Text(text) if !text.text.is_empty() => {
                                    Some(serde_json::json!({"text":text.text}))
                                }
                                AssistantPart::ToolCall(call) => Some(serde_json::json!({
                                    "toolUse":{"toolUseId":call.id,"name":call.name,"input":call.input}
                                })),
                                AssistantPart::ToolResult(_) => None,
                                _ => None,
                            })
                            .collect()
                    }
                };
                push_message(&mut messages, "assistant", content);
                if !inline_results.is_empty() {
                    let content = inline_results
                        .iter()
                        .map(|result| tool_result(result, options, &mut document_counter))
                        .collect::<Result<Vec<_>, _>>()?;
                    push_message(&mut messages, "user", content);
                }
            }
            HistoryTurn::Tool(message) => {
                let content = message
                    .results
                    .iter()
                    .map(|result| tool_result(result, options, &mut document_counter))
                    .collect::<Result<Vec<_>, _>>()?;
                push_message(&mut messages, "user", content);
            }
        }
    }

    let mut root = serde_json::json!({"messages":messages});
    if !system.is_empty() {
        root["system"] = JsonValue::Array(system);
    }
    let mut inference = serde_json::json!({});
    if let Some(value) = request.inference.max_output_tokens {
        inference["maxTokens"] = JsonValue::from(value);
    }
    if let Some(value) = request.inference.temperature {
        inference["temperature"] = JsonValue::from(value);
    }
    if let Some(value) = request.inference.top_p {
        inference["topP"] = JsonValue::from(value);
    }
    if !options.stop_sequences.is_empty() {
        inference["stopSequences"] =
            serde_json::to_value(&options.stop_sequences).expect("stop sequences are serializable");
    }
    if inference
        .as_object()
        .is_some_and(|object| !object.is_empty())
    {
        root["inferenceConfig"] = inference;
    }
    if !request.tools.is_empty() {
        root["toolConfig"] = serde_json::json!({
            "tools":request.tools.iter().map(|tool| serde_json::json!({
                "toolSpec":{"name":tool.name,"description":tool.description,"inputSchema":{"json":tool.input_schema.as_value()}}
            })).collect::<Vec<_>>()
        });
        root["toolConfig"]["toolChoice"] = match &request.tool_choice {
            ToolChoice::Auto => serde_json::json!({"auto":{}}),
            ToolChoice::Required => serde_json::json!({"any":{}}),
            ToolChoice::None => JsonValue::Null,
            ToolChoice::Tool(name) => serde_json::json!({"tool":{"name":name}}),
        };
        if request.tool_choice == ToolChoice::None {
            root["toolConfig"]
                .as_object_mut()
                .expect("object")
                .remove("toolChoice");
        }
    }
    if let ResponseFormat::Json {
        schema: Some(schema),
    } = &request.response_format
    {
        if settings.structured_output != BedrockStructuredOutput::JsonSchema {
            return Err(ModelError::unsupported(
                "native structured output is unsupported by this Bedrock configuration",
            ));
        }
        let schema = serde_json::to_string(schema.as_value()).map_err(|_| {
            ModelError::invalid_request("could not serialize Bedrock structured-output schema")
                .with_stage(ErrorStage::RequestEncoding)
        })?;
        root["outputConfig"] = serde_json::json!({
            "textFormat":{"type":"json_schema","structure":{"jsonSchema":{"schema":schema}}}
        });
    }
    let mut additional = options
        .additional_model_request_fields
        .clone()
        .unwrap_or_else(|| serde_json::json!({}));
    encode_reasoning(
        &mut additional,
        request,
        options,
        settings.reasoning_wire_format,
    )?;
    if additional
        .as_object()
        .is_some_and(|object| !object.is_empty())
    {
        root["additionalModelRequestFields"] = additional;
    }
    if !options.additional_model_response_field_paths.is_empty() {
        root["additionalModelResponseFieldPaths"] =
            serde_json::to_value(&options.additional_model_response_field_paths)
                .expect("paths are serializable");
    }
    if let Some(value) = &options.service_tier {
        root["serviceTier"] = serde_json::json!({"type":value});
    }
    if let Some(value) = &options.performance_latency {
        root["performanceConfig"] = serde_json::json!({"latency":value});
    }
    if !options.request_metadata.is_empty() {
        root["requestMetadata"] = serde_json::to_value(&options.request_metadata)
            .expect("request metadata is serializable");
    }
    if let Some(guardrail) = &options.guardrail {
        if !settings.streaming && guardrail.stream_processing_mode.is_some() {
            return Err(ModelError::invalid_request(
                "Bedrock streamProcessingMode is valid only for ConverseStream",
            )
            .with_stage(ErrorStage::RequestEncoding));
        }
        let mut encoded = serde_json::json!({
            "guardrailIdentifier":guardrail.guardrail_identifier,
            "guardrailVersion":guardrail.guardrail_version,
        });
        if let Some(trace) = &guardrail.trace {
            encoded["trace"] = JsonValue::String(trace.clone());
        }
        if settings.streaming
            && let Some(mode) = &guardrail.stream_processing_mode
        {
            encoded["streamProcessingMode"] = JsonValue::String(mode.clone());
        }
        root["guardrailConfig"] = encoded;
    }
    Ok(Encoded {
        body: root,
        replay,
        warnings,
    })
}

pub(crate) fn validate_serialized_body(request: &Request, bytes: usize) -> Result<(), ModelError> {
    if bytes >= MAX_VIDEO_BASE64_BYTES && request_has_inline_video(request) {
        return Err(ModelError::invalid_request(
            "Bedrock request containing inline video must be smaller than 25 MiB",
        )
        .with_stage(ErrorStage::RequestEncoding));
    }
    Ok(())
}

fn encode_reasoning(
    additional: &mut JsonValue,
    request: &Request,
    options: &BedrockRequestOptions,
    wire_format: BedrockReasoningWireFormat,
) -> Result<(), ModelError> {
    let effort = options
        .max_reasoning_effort
        .as_ref()
        .or(request.inference.reasoning_effort.as_ref());
    if options.reasoning_type.is_none()
        && options.reasoning_budget_tokens.is_none()
        && options.reasoning_display.is_none()
        && effort.is_none()
    {
        return Ok(());
    }
    let object = additional.as_object_mut().ok_or_else(|| {
        ModelError::invalid_request("Bedrock additional model fields must be an object")
    })?;
    match wire_format {
        BedrockReasoningWireFormat::Unsupported => {
            return Err(ModelError::unsupported(
                "Bedrock reasoning controls are unsupported by this configuration",
            ));
        }
        BedrockReasoningWireFormat::AnthropicThinking => {
            if options.reasoning_type.as_deref() == Some("adaptive") {
                let mut thinking = serde_json::Map::new();
                thinking.insert("type".into(), JsonValue::String("adaptive".into()));
                if let Some(display) = &options.reasoning_display {
                    thinking.insert("display".into(), JsonValue::String(display.clone()));
                }
                object.insert("thinking".into(), JsonValue::Object(thinking));
            } else if let Some(budget) = options.reasoning_budget_tokens {
                object.insert(
                    "thinking".into(),
                    serde_json::json!({"type":options.reasoning_type.as_deref().unwrap_or("enabled"),"budget_tokens":budget}),
                );
            } else if let Some(kind) = &options.reasoning_type {
                object.insert("thinking".into(), serde_json::json!({"type":kind}));
            }
            if let Some(effort) = effort {
                object.insert("output_config".into(), serde_json::json!({"effort":effort}));
            }
        }
        BedrockReasoningWireFormat::OpenAiReasoningEffort => {
            if let Some(effort) = effort {
                object.insert("reasoning_effort".into(), JsonValue::String(effort.clone()));
            }
        }
        BedrockReasoningWireFormat::BedrockReasoningConfig => {
            let mut reasoning = serde_json::Map::new();
            if let Some(kind) = &options.reasoning_type {
                reasoning.insert("type".into(), JsonValue::String(kind.clone()));
            }
            if let Some(budget) = options.reasoning_budget_tokens {
                reasoning.insert("budgetTokens".into(), JsonValue::from(budget));
            }
            if let Some(effort) = effort {
                reasoning.insert(
                    "maxReasoningEffort".into(),
                    JsonValue::String(effort.clone()),
                );
            }
            object.insert("reasoningConfig".into(), JsonValue::Object(reasoning));
        }
    }
    Ok(())
}

fn validate_reasoning_options(
    request: &Request,
    options: &BedrockRequestOptions,
    wire_format: BedrockReasoningWireFormat,
    signed_reasoning: bool,
) -> Result<(), ModelError> {
    if options.max_reasoning_effort.is_some() && request.inference.reasoning_effort.is_some() {
        return Err(ModelError::invalid_request(
            "configure reasoning effort in either normalized inference or Bedrock options, not both",
        ));
    }
    for (name, value) in [
        ("reasoning type", options.reasoning_type.as_deref()),
        ("reasoning display", options.reasoning_display.as_deref()),
        (
            "reasoning effort",
            options
                .max_reasoning_effort
                .as_deref()
                .or(request.inference.reasoning_effort.as_deref()),
        ),
    ] {
        if value.is_some_and(|value| value.trim().is_empty()) {
            return Err(ModelError::invalid_request(format!(
                "Bedrock {name} must not be empty"
            )));
        }
    }
    if options.reasoning_budget_tokens == Some(0) {
        return Err(ModelError::invalid_request(
            "Bedrock reasoning budget must be positive",
        ));
    }
    let effort = options
        .max_reasoning_effort
        .as_deref()
        .or(request.inference.reasoning_effort.as_deref());
    let has_effort = effort.is_some();
    let has_bedrock_options = options.reasoning_type.is_some()
        || options.reasoning_budget_tokens.is_some()
        || options.reasoning_display.is_some();
    let kind = options.reasoning_type.as_deref();
    let budget = options.reasoning_budget_tokens;
    let display = options.reasoning_display.as_deref();
    match wire_format {
        BedrockReasoningWireFormat::Unsupported if has_effort || has_bedrock_options => {
            Err(ModelError::unsupported(
                "Bedrock reasoning controls are unsupported by this configuration",
            ))
        }
        BedrockReasoningWireFormat::OpenAiReasoningEffort if has_bedrock_options => {
            Err(ModelError::unsupported(
                "OpenAI Bedrock reasoning accepts only reasoning effort controls",
            ))
        }
        BedrockReasoningWireFormat::AnthropicThinking => {
            if kind == Some("adaptive") && budget.is_some() {
                return Err(ModelError::invalid_request(
                    "Anthropic adaptive reasoning cannot include a token budget",
                ));
            }
            if kind == Some("enabled") && budget.is_none() {
                return Err(ModelError::invalid_request(
                    "enabled Anthropic reasoning requires a token budget",
                ));
            }
            if kind != Some("adaptive") && display.is_some() {
                return Err(ModelError::invalid_request(
                    "Anthropic reasoning display requires adaptive reasoning",
                ));
            }
            if kind == Some("disabled") && (budget.is_some() || display.is_some() || has_effort) {
                return Err(ModelError::invalid_request(
                    "disabled Anthropic reasoning cannot include budget, display, or effort",
                ));
            }
            if budget.is_some() && kind.is_some_and(|kind| kind != "enabled") {
                return Err(ModelError::invalid_request(
                    "Anthropic reasoning budgets require type enabled or no explicit type",
                ));
            }
            let thinking_enabled =
                kind == Some("adaptive") || kind == Some("enabled") || budget.is_some();
            if thinking_enabled
                && (request.inference.temperature.is_some() || request.inference.top_p.is_some())
            {
                return Err(ModelError::invalid_request(
                    "Anthropic reasoning cannot combine with temperature or top_p",
                ));
            }
            Ok(())
        }
        BedrockReasoningWireFormat::BedrockReasoningConfig => {
            if display.is_some() {
                return Err(ModelError::unsupported(
                    "Bedrock reasoningConfig does not encode reasoning display",
                ));
            }
            if kind == Some("adaptive") {
                return Err(ModelError::invalid_request(
                    "adaptive reasoning requires the Anthropic thinking wire format",
                ));
            }
            if kind == Some("disabled") && (budget.is_some() || has_effort) {
                return Err(ModelError::invalid_request(
                    "disabled Bedrock reasoning cannot include budget or effort",
                ));
            }
            if budget.is_some() && kind.is_some_and(|kind| kind != "enabled") {
                return Err(ModelError::invalid_request(
                    "Bedrock reasoning budgets require type enabled or no explicit type",
                ));
            }
            Ok(())
        }
        _ => {
            if signed_reasoning && wire_format != BedrockReasoningWireFormat::AnthropicThinking {
                return Err(ModelError::invalid_request(
                    "signed Bedrock reasoning requires the Anthropic thinking wire format",
                ));
            }
            Ok(())
        }
    }
}

pub(crate) fn options(request: &Request) -> Result<BedrockRequestOptions, ModelError> {
    request
        .provider_options
        .get("bedrock")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| ModelError::invalid_request("invalid Bedrock request options"))
        })
        .transpose()
        .map(|value| value.unwrap_or_default())
}

fn push_message(messages: &mut Vec<JsonValue>, role: &str, content: Vec<JsonValue>) {
    if content.is_empty() {
        return;
    }
    if messages
        .last()
        .and_then(|value| value.get("role"))
        .and_then(JsonValue::as_str)
        == Some(role)
    {
        messages
            .last_mut()
            .and_then(|value| value.get_mut("content"))
            .and_then(JsonValue::as_array_mut)
            .expect("existing message content is an array")
            .extend(content);
    } else {
        messages.push(serde_json::json!({"role":role,"content":content}));
    }
}

fn validate_file(file: &FilePart, has_accompanying_text: bool) -> Result<(), ModelError> {
    let format = media_format(&file.media_type).ok_or_else(|| {
        ModelError::unsupported("unsupported Bedrock media MIME type")
            .with_stage(ErrorStage::RequestEncoding)
    })?;
    if format.kind == MediaKind::Document && !has_accompanying_text {
        return Err(ModelError::invalid_request(
            "Bedrock document messages require accompanying text",
        )
        .with_stage(ErrorStage::RequestEncoding));
    }
    match &file.source {
        FileSource::Bytes(bytes) => validate_inline_bytes(format.kind, bytes),
        FileSource::Text(text)
            if format.kind == MediaKind::Document && textual_document_mime(&file.media_type) =>
        {
            validate_inline_document(text.as_bytes())
        }
        FileSource::Text(_) => Err(ModelError::unsupported(
            "Bedrock FileSource::Text requires a textual document MIME type",
        )
        .with_stage(ErrorStage::RequestEncoding)),
        FileSource::Url(url) if url.scheme() == "s3" => validate_s3_uri(url.as_str()),
        FileSource::Url(_) => Err(ModelError::unsupported(
            "Bedrock media URLs must use s3://; the adapter never downloads URLs",
        )
        .with_stage(ErrorStage::RequestEncoding)),
        FileSource::ProviderReference { .. } => Err(ModelError::unsupported(
            "Bedrock provider file references are unsupported",
        )
        .with_stage(ErrorStage::RequestEncoding)),
    }
}

fn validate_tool_content(content: &ToolContent) -> Result<(), ModelError> {
    if let ToolContent::Mixed(values) = content
        && values.is_empty()
    {
        return Err(ModelError::invalid_request(
            "Bedrock mixed tool content must not be empty",
        ));
    }
    Ok(())
}

const MAX_IMAGES: usize = 20;
const MAX_DOCUMENTS: usize = 5;
const MAX_VIDEOS: usize = 1;
const MAX_IMAGE_BYTES: usize = 15 * 1024 * 1024 / 4;
const MAX_DOCUMENT_BYTES: usize = 9 * 1024 * 1024 / 2;
const MAX_VIDEO_BASE64_BYTES: usize = 25 * 1024 * 1024;

#[derive(Clone, Copy, Default)]
struct MediaCounts {
    images: usize,
    documents: usize,
    videos: usize,
}

impl MediaCounts {
    fn add_file(&mut self, file: &FilePart) -> Result<(), ModelError> {
        let kind = media_format(&file.media_type)
            .ok_or_else(|| ModelError::unsupported("unsupported Bedrock media MIME type"))?
            .kind;
        let counter = match kind {
            MediaKind::Image => &mut self.images,
            MediaKind::Document => &mut self.documents,
            MediaKind::Video => &mut self.videos,
        };
        *counter = counter.checked_add(1).ok_or_else(|| {
            ModelError::invalid_request("Bedrock media count overflowed")
                .with_stage(ErrorStage::RequestEncoding)
        })?;
        Ok(())
    }

    fn add(&mut self, other: Self) -> Result<(), ModelError> {
        self.images = self
            .images
            .checked_add(other.images)
            .ok_or_else(|| ModelError::invalid_request("Bedrock image count overflowed"))?;
        self.documents = self
            .documents
            .checked_add(other.documents)
            .ok_or_else(|| ModelError::invalid_request("Bedrock document count overflowed"))?;
        self.videos = self
            .videos
            .checked_add(other.videos)
            .ok_or_else(|| ModelError::invalid_request("Bedrock video count overflowed"))?;
        Ok(())
    }

    fn validate(self, scope: &str) -> Result<(), ModelError> {
        if self.images > MAX_IMAGES || self.documents > MAX_DOCUMENTS || self.videos > MAX_VIDEOS {
            return Err(ModelError::invalid_request(format!(
                "Bedrock {scope} exceeds 20 images, 5 documents, or 1 video"
            ))
            .with_stage(ErrorStage::RequestEncoding));
        }
        Ok(())
    }
}

fn validate_media(request: &Request) -> Result<(), ModelError> {
    let mut messages = Vec::<(&'static str, MediaCounts)>::new();
    for turn in &request.history {
        match turn {
            HistoryTurn::System(_) => {}
            HistoryTurn::User(message) => {
                let has_text = message
                    .content
                    .iter()
                    .any(|part| matches!(part, InputPart::Text(text) if !text.text.is_empty()));
                let mut counts = MediaCounts::default();
                let mut has_content = false;
                for part in &message.content {
                    match part {
                        InputPart::Text(text) => has_content |= !text.text.is_empty(),
                        InputPart::File(file) => {
                            validate_file(file, has_text)?;
                            counts.add_file(file)?;
                            has_content = true;
                        }
                        InputPart::Custom(_) => {}
                    }
                }
                push_media_message(&mut messages, "user", counts, has_content)?;
            }
            HistoryTurn::Assistant(turn) => {
                let has_assistant_content = turn.message.content.iter().any(|part| {
                    matches!(
                        part,
                        AssistantPart::Text(text) if !text.text.is_empty()
                    ) || matches!(
                        part,
                        AssistantPart::Reasoning(_) | AssistantPart::ToolCall(_)
                    )
                });
                push_media_message(
                    &mut messages,
                    "assistant",
                    MediaCounts::default(),
                    has_assistant_content,
                )?;
                let mut counts = MediaCounts::default();
                let mut has_results = false;
                for part in &turn.message.content {
                    if let AssistantPart::ToolResult(result) = part {
                        validate_tool_content(&result.content)?;
                        add_tool_media(&result.content, &mut counts)?;
                        has_results = true;
                    }
                }
                push_media_message(&mut messages, "user", counts, has_results)?;
            }
            HistoryTurn::Tool(message) => {
                let mut counts = MediaCounts::default();
                for result in &message.results {
                    validate_tool_content(&result.content)?;
                    add_tool_media(&result.content, &mut counts)?;
                }
                push_media_message(&mut messages, "user", counts, !message.results.is_empty())?;
            }
        }
    }
    let mut request_counts = MediaCounts::default();
    for (_, counts) in messages {
        counts.validate("message")?;
        request_counts.add(counts)?;
    }
    request_counts.validate("request")?;
    Ok(())
}

fn request_has_inline_video(request: &Request) -> bool {
    request.history.iter().any(|turn| match turn {
        HistoryTurn::System(_) => false,
        HistoryTurn::User(message) => message.content.iter().any(|part| {
            matches!(part, InputPart::File(file) if is_inline_video(file))
        }),
        HistoryTurn::Assistant(turn) => turn.message.content.iter().any(|part| {
            matches!(part, AssistantPart::ToolResult(result) if tool_has_inline_video(&result.content))
        }),
        HistoryTurn::Tool(message) => message
            .results
            .iter()
            .any(|result| tool_has_inline_video(&result.content)),
    })
}

fn tool_has_inline_video(content: &ToolContent) -> bool {
    matches!(
        content,
        ToolContent::Mixed(values)
            if values.iter().any(|value| matches!(value, ContentValue::File(file) if is_inline_video(file)))
    )
}

fn is_inline_video(file: &FilePart) -> bool {
    media_format(&file.media_type).is_some_and(|format| format.kind == MediaKind::Video)
        && matches!(&file.source, FileSource::Bytes(_))
}

fn push_media_message(
    messages: &mut Vec<(&'static str, MediaCounts)>,
    role: &'static str,
    counts: MediaCounts,
    has_content: bool,
) -> Result<(), ModelError> {
    if !has_content {
        return Ok(());
    }
    if let Some((last_role, last_counts)) = messages.last_mut()
        && *last_role == role
    {
        last_counts.add(counts)
    } else {
        messages.push((role, counts));
        Ok(())
    }
}

fn add_tool_media(content: &ToolContent, counts: &mut MediaCounts) -> Result<(), ModelError> {
    if let ToolContent::Mixed(values) = content {
        for value in values {
            if let ContentValue::File(file) = value {
                validate_file(file, true)?;
                counts.add_file(file)?;
            }
        }
    }
    Ok(())
}

fn validate_inline_bytes(kind: MediaKind, bytes: &[u8]) -> Result<(), ModelError> {
    if bytes.is_empty() {
        return Err(
            ModelError::invalid_request("Bedrock inline media bytes must not be empty")
                .with_stage(ErrorStage::RequestEncoding),
        );
    }
    match kind {
        MediaKind::Image if bytes.len() > MAX_IMAGE_BYTES => Err(ModelError::invalid_request(
            "Bedrock inline image exceeds 3.75 MiB",
        )
        .with_stage(ErrorStage::RequestEncoding)),
        MediaKind::Document => validate_inline_document(bytes),
        MediaKind::Video => {
            let encoded = base64_encoded_len(bytes.len()).ok_or_else(|| {
                ModelError::invalid_request("Bedrock inline video size overflowed")
                    .with_stage(ErrorStage::RequestEncoding)
            })?;
            if encoded >= MAX_VIDEO_BASE64_BYTES {
                Err(ModelError::invalid_request(
                    "Bedrock inline video base64 payload must be smaller than 25 MiB",
                )
                .with_stage(ErrorStage::RequestEncoding))
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
    }
}

fn validate_inline_document(bytes: &[u8]) -> Result<(), ModelError> {
    if bytes.is_empty() {
        return Err(ModelError::invalid_request(
            "Bedrock inline document content must not be empty",
        )
        .with_stage(ErrorStage::RequestEncoding));
    }
    if bytes.len() > MAX_DOCUMENT_BYTES {
        return Err(
            ModelError::invalid_request("Bedrock inline document exceeds 4.5 MiB")
                .with_stage(ErrorStage::RequestEncoding),
        );
    }
    Ok(())
}

fn base64_encoded_len(len: usize) -> Option<usize> {
    len.checked_add(2)?.checked_div(3)?.checked_mul(4)
}

fn textual_document_mime(mime: &str) -> bool {
    matches!(
        mime,
        "text/csv" | "text/html" | "text/plain" | "text/markdown"
    )
}

fn validate_s3_uri(uri: &str) -> Result<(), ModelError> {
    if uri.len() > 1024 || !uri.starts_with("s3://") {
        return Err(invalid_s3_uri());
    }
    let remainder = &uri[5..];
    let bucket = remainder
        .split_once('/')
        .map_or(remainder, |(bucket, _)| bucket);
    let valid_bucket = (3..=63).contains(&bucket.len())
        && bucket
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && bucket
            .bytes()
            .last()
            .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && bucket.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-')
        });
    if !valid_bucket {
        return Err(invalid_s3_uri());
    }
    Ok(())
}

fn invalid_s3_uri() -> ModelError {
    ModelError::invalid_request("Bedrock S3 URI does not match the documented pattern")
        .with_stage(ErrorStage::RequestEncoding)
}

fn validate_bucket_owner(options: &BedrockRequestOptions) -> Result<(), ModelError> {
    if let Some(owner) = options
        .s3
        .as_ref()
        .and_then(|options| options.bucket_owner.as_deref())
        && (owner.len() != 12 || !owner.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err(ModelError::invalid_request(
            "Bedrock S3 bucket owner must be exactly 12 digits",
        )
        .with_stage(ErrorStage::RequestEncoding));
    }
    Ok(())
}

fn file_content(
    file: &FilePart,
    options: &BedrockRequestOptions,
    document_counter: &mut u64,
) -> Result<JsonValue, ModelError> {
    let format = media_format(&file.media_type).ok_or_else(|| {
        ModelError::unsupported("unsupported Bedrock media MIME type")
            .with_stage(ErrorStage::RequestEncoding)
    })?;
    let source = media_source(file, options)?;
    Ok(match format.kind {
        MediaKind::Image => serde_json::json!({"image":{"format":format.format,"source":source}}),
        MediaKind::Video => serde_json::json!({"video":{"format":format.format,"source":source}}),
        MediaKind::Document => {
            *document_counter = document_counter.checked_add(1).ok_or_else(|| {
                ModelError::invalid_request("Bedrock document counter overflowed")
                    .with_stage(ErrorStage::RequestEncoding)
            })?;
            let name = document_name(file.filename.as_deref(), *document_counter)?;
            serde_json::json!({"document":{"format":format.format,"name":name,"source":source}})
        }
    })
}

fn media_source(file: &FilePart, options: &BedrockRequestOptions) -> Result<JsonValue, ModelError> {
    match &file.source {
        FileSource::Bytes(bytes) => Ok(serde_json::json!({"bytes":STANDARD.encode(bytes)})),
        FileSource::Text(text) => Ok(serde_json::json!({"text":text})),
        FileSource::Url(url) => {
            let mut location = serde_json::json!({"uri":url.as_str()});
            if let Some(owner) = options
                .s3
                .as_ref()
                .and_then(|options| options.bucket_owner.as_ref())
            {
                location["bucketOwner"] = JsonValue::String(owner.clone());
            }
            Ok(serde_json::json!({"s3Location":location}))
        }
        FileSource::ProviderReference { .. } => Err(ModelError::unsupported(
            "Bedrock provider file references are unsupported",
        )),
    }
}

fn tool_result(
    result: &ToolResultPart,
    options: &BedrockRequestOptions,
    document_counter: &mut u64,
) -> Result<JsonValue, ModelError> {
    let content = match &result.content {
        ToolContent::Text(value) => vec![serde_json::json!({"text":value})],
        ToolContent::Json(value) => vec![serde_json::json!({"json":value})],
        ToolContent::Denied { reason } => vec![serde_json::json!({
            "text":reason.as_deref().unwrap_or("Tool call execution denied.")
        })],
        ToolContent::Mixed(values) => values
            .iter()
            .map(|value| match value {
                ContentValue::Text(value) => Ok(serde_json::json!({"text":value})),
                ContentValue::Json(value) => Ok(serde_json::json!({"json":value})),
                ContentValue::File(file) => {
                    validate_file(file, true)?;
                    file_content(file, options, document_counter)
                }
            })
            .collect::<Result<Vec<_>, ModelError>>()?,
    };
    Ok(serde_json::json!({
        "toolResult":{
            "toolUseId":result.tool_call_id,
            "content":content,
            "status":if result.is_error {"error"} else {"success"}
        }
    }))
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum MediaKind {
    Image,
    Document,
    Video,
}

struct MediaFormat {
    kind: MediaKind,
    format: &'static str,
}

fn media_format(mime: &str) -> Option<MediaFormat> {
    let (kind, format) = match mime {
        "image/png" => (MediaKind::Image, "png"),
        "image/jpeg" => (MediaKind::Image, "jpeg"),
        "image/gif" => (MediaKind::Image, "gif"),
        "image/webp" => (MediaKind::Image, "webp"),
        "application/pdf" => (MediaKind::Document, "pdf"),
        "text/csv" => (MediaKind::Document, "csv"),
        "application/msword" => (MediaKind::Document, "doc"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            (MediaKind::Document, "docx")
        }
        "application/vnd.ms-excel" => (MediaKind::Document, "xls"),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => {
            (MediaKind::Document, "xlsx")
        }
        "text/html" => (MediaKind::Document, "html"),
        "text/plain" => (MediaKind::Document, "txt"),
        "text/markdown" => (MediaKind::Document, "md"),
        "video/x-matroska" => (MediaKind::Video, "mkv"),
        "video/quicktime" => (MediaKind::Video, "mov"),
        "video/mp4" => (MediaKind::Video, "mp4"),
        "video/webm" => (MediaKind::Video, "webm"),
        "video/x-flv" => (MediaKind::Video, "flv"),
        "video/mpeg" => (MediaKind::Video, "mpeg"),
        "video/mpg" => (MediaKind::Video, "mpg"),
        "video/wmv" => (MediaKind::Video, "wmv"),
        "video/3gpp" => (MediaKind::Video, "three_gp"),
        _ => return None,
    };
    Some(MediaFormat { kind, format })
}

fn document_name(filename: Option<&str>, counter: u64) -> Result<String, ModelError> {
    let raw = filename
        .and_then(|value| {
            value
                .rsplit_once('.')
                .map_or(Some(value), |(stem, _)| Some(stem))
        })
        .unwrap_or("document");
    let mut name = raw
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric()
                || matches!(character, ' ' | '-' | '(' | ')' | '[' | ']')
            {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    name = name.split_whitespace().collect::<Vec<_>>().join(" ");
    if name.is_empty() {
        name = format!("document-{counter}");
    }
    if name.len() > 200 {
        return Err(
            ModelError::invalid_request("Bedrock document name exceeds 200 bytes")
                .with_stage(ErrorStage::RequestEncoding),
        );
    }
    Ok(name)
}

fn decode_replay(
    payload: &JsonValue,
    normalized: &[AssistantPart],
    capabilities: &ModelCapabilities,
    signed_reasoning: bool,
) -> Result<Vec<JsonValue>, &'static str> {
    let root = payload
        .as_object()
        .ok_or("Bedrock replay payload is not an object")?;
    if !exact_keys(root, &["assistant_content", "format"]) {
        return Err("Bedrock replay payload shape is invalid");
    }
    if root.get("format").and_then(JsonValue::as_str) != Some(REPLAY_FORMAT) {
        return Err("Bedrock replay format is invalid");
    }
    let content = root
        .get("assistant_content")
        .and_then(JsonValue::as_array)
        .ok_or("Bedrock replay assistant content is invalid")?
        .clone();
    let semantics = validate_native_content(&content, capabilities, signed_reasoning)?;
    if semantics != normalized_semantics(normalized) {
        return Err("Bedrock replay payload did not match normalized content");
    }
    Ok(content)
}

fn validate_native_content(
    parts: &[JsonValue],
    capabilities: &ModelCapabilities,
    signed_reasoning: bool,
) -> Result<Vec<JsonValue>, &'static str> {
    let mut semantics = Vec::new();
    let mut tool_ids = BTreeSet::new();
    for part in parts {
        let object = part
            .as_object()
            .ok_or("Bedrock replay content block is not an object")?;
        if object.len() != 1 {
            return Err("Bedrock replay content block union is ambiguous");
        }
        if let Some(value) = object.get("text") {
            let text = value
                .as_str()
                .ok_or("Bedrock replay text block is invalid")?;
            semantics.push(serde_json::json!({"type":"text","text":text}));
            continue;
        }
        if let Some(value) = object.get("toolUse") {
            if !capabilities.features.contains(Capability::TOOL_CALLING) {
                return Err("Bedrock replay tool use is unsupported by the declaration");
            }
            let tool = value
                .as_object()
                .ok_or("Bedrock replay toolUse block is invalid")?;
            if !exact_keys(tool, &["input", "name", "toolUseId"]) {
                return Err("Bedrock replay toolUse shape is invalid");
            }
            let id = nonempty_string(tool.get("toolUseId"))
                .ok_or("Bedrock replay toolUseId is invalid")?;
            let name =
                nonempty_string(tool.get("name")).ok_or("Bedrock replay tool name is invalid")?;
            let input = tool
                .get("input")
                .filter(|value| value.is_object())
                .ok_or("Bedrock replay tool input must be an object")?;
            if !tool_ids.insert(id) {
                return Err("Bedrock replay contains duplicate toolUse IDs");
            }
            semantics.push(serde_json::json!({
                "type":"tool_call","id":id,"name":name,"input":input
            }));
            continue;
        }
        if let Some(value) = object.get("reasoningContent") {
            if !capabilities.features.contains(Capability::REASONING) {
                return Err("Bedrock replay reasoning is unsupported by the declaration");
            }
            let reasoning = value
                .as_object()
                .ok_or("Bedrock replay reasoningContent block is invalid")?;
            if reasoning.len() != 1 {
                return Err("Bedrock replay reasoningContent union is ambiguous");
            }
            if let Some(value) = reasoning.get("reasoningText") {
                let text = value
                    .as_object()
                    .ok_or("Bedrock replay reasoningText block is invalid")?;
                let expected = if signed_reasoning {
                    &["signature", "text"][..]
                } else {
                    &["text"][..]
                };
                if !exact_keys(text, expected) {
                    return Err("Bedrock replay reasoningText shape is invalid");
                }
                let visible = text
                    .get("text")
                    .and_then(JsonValue::as_str)
                    .ok_or("Bedrock replay reasoning text is invalid")?;
                if signed_reasoning && nonempty_string(text.get("signature")).is_none() {
                    return Err("Bedrock replay signed reasoning signature is invalid");
                }
                semantics.push(serde_json::json!({"type":"reasoning","text":visible}));
                continue;
            }
            if let Some(value) = reasoning.get("redactedContent") {
                if !signed_reasoning || value.as_str().is_none_or(str::is_empty) {
                    return Err(
                        "Bedrock replay redacted reasoning is invalid for the configuration",
                    );
                }
                continue;
            }
            return Err("Bedrock replay reasoningContent union member is unsupported");
        }
        return Err("Bedrock replay contains an unsupported native part");
    }
    Ok(semantics)
}

fn exact_keys(object: &serde_json::Map<String, JsonValue>, expected: &[&str]) -> bool {
    object.len() == expected.len() && expected.iter().all(|key| object.contains_key(*key))
}

fn nonempty_string(value: Option<&JsonValue>) -> Option<&str> {
    value
        .and_then(JsonValue::as_str)
        .filter(|value| !value.trim().is_empty())
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
                "type":"tool_call","id":call.id,"name":call.name,"input":call.input
            })),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oven_sdk::FileSource;

    fn replay_capabilities() -> ModelCapabilities {
        let mut capabilities = ModelCapabilities::conservative();
        capabilities.features = Capability::TOOL_CALLING | Capability::REASONING;
        capabilities
    }

    #[test]
    fn exact_media_mimes_s3_and_no_download_are_enforced() {
        let s3 = FilePart::video(
            "video/3gpp",
            FileSource::Url("s3://bucket/video.3gp".parse().unwrap()),
        );
        assert!(validate_file(&s3, true).is_ok());
        let https = FilePart::image(
            "image/png",
            FileSource::Url("https://example.com/image.png".parse().unwrap()),
        );
        assert!(validate_file(&https, true).is_err());
        let audio = FilePart::audio("audio/mpeg", FileSource::Bytes(Vec::new().into()));
        assert!(validate_file(&audio, true).is_err());
    }

    #[test]
    fn replay_is_semantically_exact() {
        let capabilities = replay_capabilities();
        let payload = serde_json::json!({
            "format":REPLAY_FORMAT,
            "assistant_content":[{"reasoningContent":{"reasoningText":{"text":"think","signature":"sig"}}},{"text":"answer"}]
        });
        let normalized = vec![
            AssistantPart::Reasoning(oven_sdk::ReasoningPart::new("think")),
            AssistantPart::Text(oven_sdk::TextPart::new("answer")),
        ];
        assert!(decode_replay(&payload, &normalized, &capabilities, true).is_ok());
    }

    #[test]
    fn adversarial_replay_blocks_fail_closed() {
        let capabilities = replay_capabilities();
        let tool = vec![AssistantPart::ToolCall(oven_sdk::ToolCallPart::new(
            "call",
            "lookup",
            serde_json::json!({"x":1}),
        ))];
        let reasoning = vec![AssistantPart::Reasoning(oven_sdk::ReasoningPart::new(
            "think",
        ))];
        let cases = [
            serde_json::json!([{"text":"ok","toolUse":{"toolUseId":"call","name":"lookup","input":{}}}]),
            serde_json::json!([{"toolUse":{"toolUseId":"","name":"lookup","input":{}}}]),
            serde_json::json!([{"toolUse":{"toolUseId":"call","name":"","input":{}}}]),
            serde_json::json!([{"toolUse":{"toolUseId":"call","name":"lookup","input":[]}}]),
            serde_json::json!([{"toolUse":{"toolUseId":"call","name":"lookup","input":{},"extra":true}}]),
            serde_json::json!([{"reasoningContent":{"reasoningText":{"text":"think"}}}]),
            serde_json::json!([{"reasoningContent":{"reasoningText":{"text":"think","signature":""}}}]),
            serde_json::json!([{"reasoningContent":{"reasoningText":{"text":"think","signature":"sig","extra":true}}}]),
            serde_json::json!([{"reasoningContent":{"reasoningText":{"text":"think","signature":"sig"},"redactedContent":"opaque"}}]),
            serde_json::json!([{"reasoningContent":{"redactedContent":""}}]),
            serde_json::json!([{"reasoningContent":{"unknown":"value"}}]),
        ];
        for content in cases {
            let normalized = if content
                .as_array()
                .is_some_and(|parts| parts.iter().any(|part| part.get("toolUse").is_some()))
            {
                &tool
            } else {
                &reasoning
            };
            let payload = serde_json::json!({
                "format":REPLAY_FORMAT,
                "assistant_content":content,
            });
            assert!(decode_replay(&payload, normalized, &capabilities, true).is_err());
        }

        let valid_redacted = serde_json::json!({
            "format":REPLAY_FORMAT,
            "assistant_content":[{"reasoningContent":{"redactedContent":"opaque"}}],
        });
        assert!(decode_replay(&valid_redacted, &[], &capabilities, true).is_ok());

        let unsigned_reasoning = vec![AssistantPart::Reasoning(oven_sdk::ReasoningPart::new(
            "think",
        ))];
        for content in [
            serde_json::json!([{"reasoningContent":{"reasoningText":{"text":"think","signature":"unexpected"}}}]),
            serde_json::json!([{"reasoningContent":{"redactedContent":"opaque"}}]),
        ] {
            let payload = serde_json::json!({
                "format":REPLAY_FORMAT,
                "assistant_content":content,
            });
            assert!(decode_replay(&payload, &unsigned_reasoning, &capabilities, false).is_err());
        }
        let valid_unsigned = serde_json::json!({
            "format":REPLAY_FORMAT,
            "assistant_content":[{"reasoningContent":{"reasoningText":{"text":"think"}}}],
        });
        assert!(decode_replay(&valid_unsigned, &unsigned_reasoning, &capabilities, false).is_ok());

        for mutation in [
            serde_json::json!({
                "format":"oven.bedrock.converse.assistant.v1",
                "source_model":"legacy-model",
                "assistant_content":[],
            }),
            serde_json::json!({
                "format":"oven.bedrock.converse.assistant.scoped.v1",
                "assistant_content":[],
            }),
            serde_json::json!({
                "format":REPLAY_FORMAT,
                "assistant_content":[],
                "extra":true,
            }),
            serde_json::json!({
                "format":REPLAY_FORMAT,
                "assistant_content":"not-an-array",
            }),
        ] {
            assert!(decode_replay(&mutation, &[], &capabilities, true).is_err());
        }
    }

    #[test]
    fn media_sizes_sources_s3_and_owner_validate_at_exact_boundaries() {
        assert!(validate_inline_bytes(MediaKind::Image, &vec![0; MAX_IMAGE_BYTES]).is_ok());
        assert!(validate_inline_bytes(MediaKind::Image, &vec![0; MAX_IMAGE_BYTES + 1]).is_err());
        assert!(validate_inline_document(&vec![0; MAX_DOCUMENT_BYTES]).is_ok());
        assert!(validate_inline_document(&vec![0; MAX_DOCUMENT_BYTES + 1]).is_err());
        let max_inline_video = MAX_VIDEO_BASE64_BYTES / 4 * 3 - 3;
        assert!(validate_inline_bytes(MediaKind::Video, &vec![0; max_inline_video]).is_ok());
        assert!(validate_inline_bytes(MediaKind::Video, &vec![0; max_inline_video + 1]).is_err());
        assert!(validate_inline_bytes(MediaKind::Image, &[]).is_err());

        assert!(validate_s3_uri("s3://abc").is_ok());
        assert!(validate_s3_uri("s3://abc/path/to/object").is_ok());
        assert!(validate_s3_uri("s3://ab/object").is_err());
        assert!(validate_s3_uri("s3://-bad/object").is_err());
        assert!(validate_s3_uri(&format!("s3://abc/{}", "x".repeat(1015))).is_ok());
        assert!(validate_s3_uri(&format!("s3://abc/{}", "x".repeat(1017))).is_err());

        let good = BedrockRequestOptions {
            s3: Some(crate::BedrockS3LocationOptions {
                bucket_owner: Some("123456789012".into()),
            }),
            ..Default::default()
        };
        assert!(validate_bucket_owner(&good).is_ok());
        let bad = BedrockRequestOptions {
            s3: Some(crate::BedrockS3LocationOptions {
                bucket_owner: Some("123".into()),
            }),
            ..Default::default()
        };
        assert!(validate_bucket_owner(&bad).is_err());
    }

    #[test]
    fn media_counts_and_exact_mime_source_combinations_are_enforced() {
        let images = (0..=MAX_IMAGES)
            .map(|_| {
                InputPart::File(FilePart::image(
                    "image/png",
                    FileSource::Bytes(vec![1].into()),
                ))
            })
            .collect();
        let request = Request::new(vec![HistoryTurn::user(oven_sdk::UserMessage::new(images))]);
        assert!(validate_media(&request).is_err());

        let pdf_text = FilePart::document(
            "application/pdf",
            FileSource::Text("not a PDF byte stream".into()),
        );
        assert!(validate_file(&pdf_text, true).is_err());
        let markdown = FilePart::document("text/markdown", FileSource::Text("# valid".into()));
        assert!(validate_file(&markdown, true).is_ok());
        assert_eq!(
            media_source(&markdown, &BedrockRequestOptions::default()).unwrap(),
            serde_json::json!({"text":"# valid"})
        );
        let wmv = FilePart::video(
            "video/wmv",
            FileSource::Url("s3://bucket/video.wmv".parse().unwrap()),
        );
        assert!(validate_file(&wmv, true).is_ok());
        let legacy_wmv = FilePart::video(
            "video/x-ms-wmv",
            FileSource::Url("s3://bucket/video.wmv".parse().unwrap()),
        );
        assert!(validate_file(&legacy_wmv, true).is_err());

        let video_request =
            Request::new(vec![HistoryTurn::user(oven_sdk::UserMessage::new(vec![
                InputPart::File(FilePart::video(
                    "video/mp4",
                    FileSource::Bytes(vec![1].into()),
                )),
            ]))]);
        assert!(validate_serialized_body(&video_request, MAX_VIDEO_BASE64_BYTES - 1).is_ok());
        assert!(validate_serialized_body(&video_request, MAX_VIDEO_BASE64_BYTES).is_err());
    }
}
