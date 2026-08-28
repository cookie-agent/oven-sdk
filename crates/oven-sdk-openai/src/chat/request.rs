//! Chat Completions request validation and translation.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use oven_sdk::{
    AssistantPart, ContentValue, ErrorStage, FilePart, FileSource, HistoryTurn, InputPart,
    JsonValue, LanguageModelDescriptor, ModelCapabilities, ModelError, NativeContextScope,
    ReplayDecision, ReplayDisposition, ReplayOutcome, ReplayPolicy, Request, ResponseFormat,
    ToolChoice, ToolContent, ToolResultPart,
};

use crate::{
    chat::replay,
    configuration::{MaxTokensField, ReasoningField, StructuredOutputSupport, SystemMessageRole},
    options::{
        CompatibleChatOptions, OpenAiChatOptions, chat_options, compatible_options,
        prompt_cache_breakpoint, validate_prompt_cache_key,
    },
};

pub(crate) struct ChatWireProfile {
    pub(crate) compatible: bool,
    pub(crate) system_role: SystemMessageRole,
    pub(crate) max_tokens_field: MaxTokensField,
    pub(crate) stream_usage: bool,
    pub(crate) structured_output: StructuredOutputSupport,
    pub(crate) reasoning_field: ReasoningField,
}

pub(crate) struct Encoded {
    pub(crate) body: JsonValue,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
}

pub(crate) struct ParsedOptions {
    official: OpenAiChatOptions,
    compatible: CompatibleChatOptions,
}

pub(crate) fn parse_options(request: &Request) -> Result<ParsedOptions, ModelError> {
    Ok(ParsedOptions {
        official: chat_options(request)?,
        compatible: compatible_options(request)?,
    })
}

pub(crate) fn validate_request(
    request: &Request,
    options: &ParsedOptions,
    capabilities: &ModelCapabilities,
    profile: &ChatWireProfile,
) -> Result<(), ModelError> {
    validate_prompt_cache_key(options.official.prompt_cache_key.as_deref())?;
    validate_prompt_cache_breakpoints(request, profile)?;
    request.validate_for(capabilities)?;
    validate_media(request, profile)?;
    Ok(())
}

fn validate_media(request: &Request, profile: &ChatWireProfile) -> Result<(), ModelError> {
    for turn in &request.history {
        match turn {
            HistoryTurn::User(message) => {
                for part in &message.content {
                    let InputPart::File(file) = part else {
                        continue;
                    };
                    match (&*file.media_type, &file.source) {
                        (media_type, FileSource::Bytes(_) | FileSource::Text(_))
                            if media_type.starts_with("image/")
                                || media_type == "application/pdf"
                                || (profile.compatible && media_type.starts_with("video/")) => {}
                        (media_type, FileSource::Url(_)) if media_type.starts_with("image/") => {}
                        (media_type, _)
                            if !media_type.starts_with("image/")
                                && media_type != "application/pdf" =>
                        {
                            return Err(ModelError::unsupported("unsupported Chat media type")
                                .with_stage(ErrorStage::RequestEncoding));
                        }
                        _ => {
                            return Err(ModelError::unsupported("unsupported Chat media source")
                                .with_stage(ErrorStage::RequestEncoding));
                        }
                    }
                }
            }
            HistoryTurn::Assistant(turn) => {
                for part in &turn.message.content {
                    if let AssistantPart::ToolResult(result) = part {
                        reject_tool_result_files(result)?;
                    }
                }
                if turn
                    .message
                    .content
                    .iter()
                    .any(|part| matches!(part, AssistantPart::File(_)))
                {
                    return Err(ModelError::unsupported(
                        "Chat assistant-history media is not supported",
                    )
                    .with_stage(ErrorStage::RequestEncoding));
                }
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    reject_tool_result_files(result)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn encode_request(
    request: &Request,
    options: &ParsedOptions,
    descriptor: &LanguageModelDescriptor,
    scope: &NativeContextScope,
    policy: ReplayPolicy,
    profile: &ChatWireProfile,
) -> Result<Encoded, ModelError> {
    let official_options = &options.official;
    let compatible = &options.compatible;
    let mut replay_outcome = ReplayOutcome::default();
    let mut messages = Vec::new();
    let mut warnings = Vec::new();
    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
                let role = match profile.system_role {
                    SystemMessageRole::System => "system",
                    SystemMessageRole::Developer => "developer",
                    SystemMessageRole::Omit => {
                        warnings.push("Chat profile omitted a system message".into());
                        continue;
                    }
                };
                let has_breakpoint = message.content.iter().any(|part| {
                    matches!(part, oven_sdk::SystemPart::Text(text) if prompt_cache_breakpoint(&text.metadata).unwrap_or(false))
                });
                if has_breakpoint {
                    let content = message
                        .content
                        .iter()
                        .filter_map(|part| match part {
                            oven_sdk::SystemPart::Text(text) if !text.text.is_empty() => {
                                Some(attach_prompt_cache_breakpoint(
                                    serde_json::json!({"type":"text","text":text.text}),
                                    &text.metadata,
                                ))
                            }
                            _ => None,
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if !content.is_empty() {
                        messages.push(serde_json::json!({"role":role,"content":content}));
                    }
                } else {
                    let content = message
                        .content
                        .iter()
                        .filter_map(|part| match part {
                            oven_sdk::SystemPart::Text(text) => Some(text.text.as_str()),
                            oven_sdk::SystemPart::Custom(_) => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !content.is_empty() {
                        messages.push(serde_json::json!({"role":role,"content":content}));
                    }
                }
            }
            HistoryTurn::User(message) => {
                let content = user_content(&message.content, profile)?;
                if !content.as_array().is_some_and(Vec::is_empty)
                    && content.as_str().is_none_or(|value| !value.is_empty())
                {
                    messages.push(serde_json::json!({"role":"user","content":content}));
                }
            }
            HistoryTurn::Tool(message) => {
                messages.extend(
                    message
                        .results
                        .iter()
                        .map(tool_result_message)
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            HistoryTurn::Assistant(turn) => {
                let mut replayed = None;
                if policy == ReplayPolicy::Never {
                    replay_outcome.decisions.push(ReplayDecision {
                        history_index,
                        disposition: ReplayDisposition::ReconstructedNormalized,
                    });
                } else if let Some(artifact) = &turn.finish.native_replay {
                    if artifact.adapter_id() != &descriptor.adapter_id {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedForeignAdapter {
                                found: artifact.adapter_id().clone(),
                                expected: descriptor.adapter_id.clone(),
                            },
                        });
                    } else if artifact.scope() != scope {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedForeignScope {
                                found: artifact.scope().clone(),
                                expected: scope.clone(),
                            },
                        });
                    } else if let Some(message) =
                        replay::decode(artifact, &turn.message.content, profile.reasoning_field)
                    {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::Replayed,
                        });
                        replayed = Some(message);
                    } else {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedInvalidPayload {
                                reason: "Chat replay payload did not match normalized content"
                                    .into(),
                            },
                        });
                    }
                } else {
                    replay_outcome.decisions.push(ReplayDecision {
                        history_index,
                        disposition: ReplayDisposition::NoArtifact,
                    });
                }
                if let Some(message) = replayed {
                    messages.push(message);
                } else {
                    if policy != ReplayPolicy::Never {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::ReconstructedNormalized,
                        });
                    }
                    messages.push(assistant_message(
                        &turn.message.content,
                        profile.reasoning_field,
                    )?);
                }
            }
        }
    }
    let mut body = serde_json::json!({
        "model": descriptor.identity.model_id.as_str(),
        "messages": messages,
        "stream": true
    });
    if profile.stream_usage {
        body["stream_options"] = serde_json::json!({"include_usage":true});
    }
    if let Some(maximum) = request.inference.max_output_tokens {
        match profile.max_tokens_field {
            MaxTokensField::MaxTokens => body["max_tokens"] = maximum.into(),
            MaxTokensField::MaxCompletionTokens => {
                body["max_completion_tokens"] = maximum.into();
            }
            MaxTokensField::Omit => warnings.push("Chat profile omitted max output tokens".into()),
        }
    }
    if let Some(value) = request.inference.temperature {
        body["temperature"] = value.into();
    }
    if let Some(value) = request.inference.top_p {
        body["top_p"] = value.into();
    }
    if let Some(effort) = official_options
        .reasoning_effort
        .clone()
        .or_else(|| request.inference.reasoning_effort.clone())
    {
        body["reasoning_effort"] = effort.into();
    }
    if let Some(user) = &official_options.user {
        body["user"] = user.clone().into();
    }
    if let Some(service_tier) = &official_options.service_tier {
        body["service_tier"] = service_tier.clone().into();
    }
    if let Some(verbosity) = &official_options.verbosity {
        body["verbosity"] = verbosity.clone().into();
    }
    if let Some(parallel) = official_options.parallel_tool_calls {
        body["parallel_tool_calls"] = parallel.into();
    }
    if let Some(key) = &official_options.prompt_cache_key {
        body["prompt_cache_key"] = key.clone().into();
    }
    if let Some(retention) = official_options.prompt_cache_retention {
        body["prompt_cache_retention"] =
            serde_json::to_value(retention).expect("prompt-cache retention is serializable");
    }
    if let Some(cache_options) = &official_options.prompt_cache_options {
        body["prompt_cache_options"] =
            serde_json::to_value(cache_options).expect("prompt-cache options are serializable");
    }
    if !request.tools.is_empty() && !matches!(request.tool_choice, ToolChoice::None) {
        body["tools"] = JsonValue::Array(
            request
                .tools
                .iter()
                .map(|tool| {
                    let strict = tool
                        .provider_options
                        .get("openai")
                        .and_then(|value| value.get("strict"))
                        .and_then(JsonValue::as_bool)
                        .unwrap_or(false);
                    serde_json::json!({"type":"function","function":{"name":tool.name,"description":tool.description,"parameters":tool.input_schema.as_value(),"strict":strict}})
                })
                .collect(),
        );
    }
    body["tool_choice"] = match &request.tool_choice {
        ToolChoice::Auto => JsonValue::String("auto".into()),
        ToolChoice::Required => JsonValue::String("required".into()),
        ToolChoice::None => JsonValue::String("none".into()),
        ToolChoice::Tool(name) => {
            serde_json::json!({"type":"function","function":{"name":name}})
        }
    };
    match &request.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::Json { schema: None } => {
            body["response_format"] = serde_json::json!({"type":"json_object"});
        }
        ResponseFormat::Json {
            schema: Some(schema),
        } => match profile.structured_output {
            StructuredOutputSupport::JsonSchema => {
                body["response_format"] = serde_json::json!({"type":"json_schema","json_schema":{"name":"response","strict":true,"schema":schema.as_value()}});
            }
            StructuredOutputSupport::JsonObject => {
                warnings.push("Chat profile downgraded JSON Schema to JSON object mode".into());
                body["response_format"] = serde_json::json!({"type":"json_object"});
            }
            StructuredOutputSupport::Unsupported => {
                return Err(ModelError::unsupported(
                    "structured output is not supported by this Chat profile",
                ));
            }
        },
    }
    if profile.compatible
        && let Some(object) = body.as_object_mut()
    {
        object.extend(compatible.extra_body.clone());
    }
    Ok(Encoded {
        body,
        replay: replay_outcome,
        warnings,
    })
}

fn user_content(parts: &[InputPart], profile: &ChatWireProfile) -> Result<JsonValue, ModelError> {
    if parts.iter().all(|part| matches!(part, InputPart::Text(_)))
        && !parts.iter().any(|part| {
            matches!(part, InputPart::Text(text) if prompt_cache_breakpoint(&text.metadata).unwrap_or(false))
        })
    {
        return Ok(JsonValue::String(
            parts
                .iter()
                .filter_map(|part| match part {
                    InputPart::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        ));
    }
    parts
        .iter()
        .filter_map(|part| match part {
            InputPart::Text(text) => Some(attach_prompt_cache_breakpoint(
                serde_json::json!({"type":"text","text":text.text}),
                &text.metadata,
            )),
            InputPart::File(file) => Some(file_content(file, profile)),
            InputPart::Custom(_) => None,
        })
        .collect::<Result<Vec<_>, _>>()
        .map(JsonValue::Array)
}

fn file_content(file: &FilePart, profile: &ChatWireProfile) -> Result<JsonValue, ModelError> {
    let data = match &file.source {
        FileSource::Bytes(bytes) => {
            format!("data:{};base64,{}", file.media_type, STANDARD.encode(bytes))
        }
        FileSource::Text(text) => {
            format!("data:{};base64,{}", file.media_type, STANDARD.encode(text))
        }
        FileSource::Url(url) if file.media_type.starts_with("image/") => url.to_string(),
        _ => {
            return Err(ModelError::unsupported("unsupported Chat media source")
                .with_stage(ErrorStage::RequestEncoding));
        }
    };
    let value = if file.media_type.starts_with("image/") {
        serde_json::json!({"type":"image_url","image_url":{"url":data}})
    } else if profile.compatible && file.media_type.starts_with("video/") {
        serde_json::json!({"type":"video_url","video_url":{"url":data}})
    } else if file.media_type == "application/pdf" {
        serde_json::json!({"type":"file","file":{"filename":file.filename.clone().unwrap_or_else(|| "document.pdf".into()),"file_data":data}})
    } else {
        return Err(ModelError::unsupported("unsupported Chat media type")
            .with_stage(ErrorStage::RequestEncoding));
    };
    attach_prompt_cache_breakpoint(value, &file.metadata)
}

fn validate_prompt_cache_breakpoints(
    request: &Request,
    profile: &ChatWireProfile,
) -> Result<(), ModelError> {
    let mut count = 0_usize;
    for turn in &request.history {
        match turn {
            HistoryTurn::System(message) => {
                for part in &message.content {
                    if let oven_sdk::SystemPart::Text(text) = part
                        && prompt_cache_breakpoint(&text.metadata)?
                    {
                        if text.text.is_empty() || profile.system_role == SystemMessageRole::Omit {
                            return Err(ModelError::invalid_request(
                                "OpenAI prompt-cache breakpoint has no encodable Chat text block",
                            ));
                        }
                        count += 1;
                    }
                }
            }
            HistoryTurn::User(message) => {
                for part in &message.content {
                    match part {
                        InputPart::Text(text) if prompt_cache_breakpoint(&text.metadata)? => {
                            if text.text.is_empty() {
                                return Err(ModelError::invalid_request(
                                    "OpenAI prompt-cache breakpoint has no encodable Chat text block",
                                ));
                            }
                            count += 1;
                        }
                        InputPart::File(file) if prompt_cache_breakpoint(&file.metadata)? => {
                            count += 1
                        }
                        _ => {}
                    }
                }
            }
            HistoryTurn::Assistant(turn) => {
                for part in &turn.message.content {
                    let marked = match part {
                        AssistantPart::Text(text) => prompt_cache_breakpoint(&text.metadata)?,
                        AssistantPart::File(file) => prompt_cache_breakpoint(&file.metadata)?,
                        AssistantPart::ToolResult(result) => {
                            tool_content_has_prompt_cache_breakpoint(&result.content)?
                        }
                        _ => false,
                    };
                    if marked {
                        return Err(ModelError::invalid_request(
                            "OpenAI Chat prompt-cache breakpoints are valid only on system or user content",
                        ));
                    }
                }
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    if tool_content_has_prompt_cache_breakpoint(&result.content)? {
                        return Err(ModelError::invalid_request(
                            "OpenAI Chat prompt-cache breakpoints are not valid on tool results",
                        ));
                    }
                }
            }
        }
    }
    if count > 4 {
        return Err(ModelError::invalid_request(
            "OpenAI allows at most four prompt-cache breakpoints per request",
        ));
    }
    Ok(())
}

fn tool_content_has_prompt_cache_breakpoint(content: &ToolContent) -> Result<bool, ModelError> {
    match content {
        ToolContent::Mixed(values) => values.iter().try_fold(false, |marked, value| {
            Ok(marked
                || matches!(value, ContentValue::File(file) if prompt_cache_breakpoint(&file.metadata)?))
        }),
        _ => Ok(false),
    }
}

fn attach_prompt_cache_breakpoint(
    mut block: JsonValue,
    metadata: &oven_sdk::PartMetadata,
) -> Result<JsonValue, ModelError> {
    if prompt_cache_breakpoint(metadata)? {
        block["prompt_cache_breakpoint"] = serde_json::json!({"mode":"explicit"});
    }
    Ok(block)
}

pub(crate) fn assistant_message(
    parts: &[AssistantPart],
    reasoning_field: ReasoningField,
) -> Result<JsonValue, ModelError> {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut refusal = String::new();
    let mut calls = Vec::new();
    for part in parts {
        match part {
            AssistantPart::Text(part) => text.push_str(&part.text),
            AssistantPart::Reasoning(part) => reasoning.push_str(&part.text),
            AssistantPart::ToolCall(call) => calls.push(serde_json::json!({"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.raw_input.clone().unwrap_or_else(|| call.input.to_string())}})),
            AssistantPart::Custom(part) if part.kind == "openai.refusal" => {
                if let Some(value) = part.data.as_str() { refusal.push_str(value); }
            }
            AssistantPart::File(_) => {
                return Err(ModelError::unsupported(
                    "Chat assistant-history media is not supported",
                )
                .with_stage(ErrorStage::RequestEncoding));
            }
            _ => {}
        }
    }
    let mut message = serde_json::json!({"role":"assistant","content":if text.is_empty(){JsonValue::Null}else{JsonValue::String(text)}});
    if !calls.is_empty() {
        message["tool_calls"] = JsonValue::Array(calls);
    }
    if !reasoning.is_empty() {
        match reasoning_field {
            ReasoningField::ReasoningContent => message["reasoning_content"] = reasoning.into(),
            ReasoningField::Reasoning => message["reasoning"] = reasoning.into(),
            ReasoningField::None => {}
        }
    }
    if !refusal.is_empty() {
        message["refusal"] = refusal.into();
    }
    Ok(message)
}

fn tool_result_message(result: &ToolResultPart) -> Result<JsonValue, ModelError> {
    let content = match &result.content {
        ToolContent::Text(value) => value.clone(),
        ToolContent::Json(value) => value.to_string(),
        ToolContent::Mixed(values)
            if values
                .iter()
                .any(|value| matches!(value, ContentValue::File(_))) =>
        {
            return Err(ModelError::unsupported(
                "files in tool results are not deliverable via openai-chat",
            ));
        }
        ToolContent::Mixed(values) => serde_json::to_string(values).unwrap_or_default(),
        ToolContent::Denied { reason } => reason.clone().unwrap_or_default(),
    };
    Ok(serde_json::json!({"role":"tool","tool_call_id":result.tool_call_id,"content":content}))
}

fn reject_tool_result_files(result: &ToolResultPart) -> Result<(), ModelError> {
    if let ToolContent::Mixed(values) = &result.content
        && values
            .iter()
            .any(|value| matches!(value, ContentValue::File(_)))
    {
        return Err(ModelError::unsupported(
            "files in tool results are not deliverable via openai-chat",
        ));
    }
    Ok(())
}
