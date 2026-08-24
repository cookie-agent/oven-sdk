//! Chat Completions request validation and translation.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use oven_sdk::{
    AssistantPart, Capability, ErrorStage, FilePart, FileSource, HistoryTurn, InputPart, JsonValue,
    LanguageModelDescriptor, ModelCapabilities, ModelError, NativeContextScope, ReplayDecision,
    ReplayDisposition, ReplayOutcome, ReplayPolicy, Request, ResponseFormat, ToolChoice,
    ToolContent, ToolResultPart,
};

use crate::{
    chat::replay,
    configuration::{
        AzureMaxTokensField, AzureReasoningField, AzureStructuredOutputSupport,
        AzureSystemMessageRole,
    },
    media,
    options::AzureOpenAiChatOptions,
    schema,
};

pub(crate) struct ChatWireProfile {
    pub(crate) system_role: AzureSystemMessageRole,
    pub(crate) max_tokens_field: AzureMaxTokensField,
    pub(crate) stream_usage: bool,
    pub(crate) structured_output: AzureStructuredOutputSupport,
    pub(crate) reasoning_field: AzureReasoningField,
    pub(crate) omit_reasoning_sampling: bool,
}

pub(crate) struct Encoded {
    pub(crate) body: JsonValue,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
}

pub(crate) fn validate_request(
    request: &Request,
    options: &AzureOpenAiChatOptions,
    capabilities: &ModelCapabilities,
    _profile: &ChatWireProfile,
) -> Result<(), ModelError> {
    request.validate_for(capabilities)?;
    if !request.tools.is_empty() && !capabilities.features.contains(Capability::TOOL_CALLING) {
        return Err(ModelError::unsupported(
            "tool calling is not enabled by this Chat configuration",
        ));
    }
    schema::validate_request_schemas(request, options.parallel_tool_calls)?;
    media::validate_request_media(request, media::Surface::Chat)?;
    Ok(())
}

pub(crate) fn encode_request(
    request: &Request,
    official_options: &AzureOpenAiChatOptions,
    descriptor: &LanguageModelDescriptor,
    policy: ReplayPolicy,
    profile: &ChatWireProfile,
    replay_binding: &JsonValue,
    replay_scope: &NativeContextScope,
) -> Result<Encoded, ModelError> {
    let mut replay_outcome = ReplayOutcome::default();
    let mut messages = Vec::new();
    let mut warnings = Vec::new();
    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
                let role = match profile.system_role {
                    AzureSystemMessageRole::System => "system",
                    AzureSystemMessageRole::Developer => "developer",
                    AzureSystemMessageRole::Omit => {
                        warnings.push("Chat configuration omitted a system message".into());
                        continue;
                    }
                };
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
            HistoryTurn::User(message) => {
                let content = user_content(&message.content)?;
                if !content.as_array().is_some_and(Vec::is_empty)
                    && content.as_str().is_none_or(|value| !value.is_empty())
                {
                    messages.push(serde_json::json!({"role":"user","content":content}));
                }
            }
            HistoryTurn::Tool(message) => {
                messages.extend(message.results.iter().map(tool_result_message));
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
                    } else if artifact.scope() != replay_scope {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedForeignScope {
                                found: artifact.scope().clone(),
                                expected: replay_scope.clone(),
                            },
                        });
                    } else if let Some(message) = replay::decode(
                        artifact,
                        &turn.message.content,
                        profile.reasoning_field,
                        replay_binding,
                    ) {
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
                    ));
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
            AzureMaxTokensField::MaxTokens => body["max_tokens"] = maximum.into(),
            AzureMaxTokensField::MaxCompletionTokens => {
                body["max_completion_tokens"] = maximum.into();
            }
            AzureMaxTokensField::Omit => {
                warnings.push("Chat configuration omitted max output tokens".into())
            }
        }
    }
    let omit_sampling = profile.omit_reasoning_sampling;
    if let Some(value) = request.inference.temperature {
        if omit_sampling {
            warnings.push("Azure reasoning configuration omitted incompatible temperature".into());
        } else {
            body["temperature"] = value.into();
        }
    }
    if let Some(value) = request.inference.top_p {
        if omit_sampling {
            warnings.push("Azure reasoning configuration omitted incompatible top_p".into());
        } else {
            body["top_p"] = value.into();
        }
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
    if !request.tools.is_empty() && !matches!(request.tool_choice, ToolChoice::None) {
        body["tools"] = JsonValue::Array(
            request
                .tools
                .iter()
                .map(|tool| {
                    let strict = schema::strict_tool(tool)?;
                    Ok(serde_json::json!({"type":"function","function":{"name":tool.name,"description":tool.description,"parameters":tool.input_schema.as_value(),"strict":strict}}))
                })
                .collect::<Result<Vec<_>, ModelError>>()?,
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
            AzureStructuredOutputSupport::JsonSchema => {
                body["response_format"] = serde_json::json!({"type":"json_schema","json_schema":{"name":"response","strict":true,"schema":schema.as_value()}});
            }
            AzureStructuredOutputSupport::JsonObject => {
                warnings
                    .push("Chat configuration downgraded JSON Schema to JSON object mode".into());
                body["response_format"] = serde_json::json!({"type":"json_object"});
            }
            AzureStructuredOutputSupport::Unsupported => {
                return Err(ModelError::unsupported(
                    "structured output is not enabled by this Chat configuration",
                ));
            }
        },
    }
    Ok(Encoded {
        body,
        replay: replay_outcome,
        warnings,
    })
}

fn user_content(parts: &[InputPart]) -> Result<JsonValue, ModelError> {
    if parts.iter().all(|part| matches!(part, InputPart::Text(_))) {
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
            InputPart::Text(text) => Some(Ok(serde_json::json!({"type":"text","text":text.text}))),
            InputPart::File(file) => Some(file_content(file)),
            InputPart::Custom(_) => None,
        })
        .collect::<Result<Vec<_>, _>>()
        .map(JsonValue::Array)
}

fn file_content(file: &FilePart) -> Result<JsonValue, ModelError> {
    let data = match &file.source {
        FileSource::Bytes(bytes) => {
            format!("data:{};base64,{}", file.media_type, STANDARD.encode(bytes))
        }
        FileSource::Text(text) => {
            format!("data:{};base64,{}", file.media_type, STANDARD.encode(text))
        }
        FileSource::Url(url) => url.to_string(),
        _ => {
            return Err(ModelError::unsupported("unsupported Chat media source")
                .with_stage(ErrorStage::RequestEncoding));
        }
    };
    if media::is_image(&file.media_type) {
        Ok(serde_json::json!({"type":"image_url","image_url":{"url":data}}))
    } else if file.media_type == "application/pdf" {
        Ok(
            serde_json::json!({"type":"file","file":{"filename":file.filename.clone().unwrap_or_else(|| "document.pdf".into()),"file_data":data}}),
        )
    } else {
        Err(ModelError::unsupported("unsupported Chat media type")
            .with_stage(ErrorStage::RequestEncoding))
    }
}

pub(crate) fn assistant_message(
    parts: &[AssistantPart],
    reasoning_field: AzureReasoningField,
) -> JsonValue {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut refusal = String::new();
    let mut calls = Vec::new();
    for part in parts {
        match part {
            AssistantPart::Text(part) => text.push_str(&part.text),
            AssistantPart::Reasoning(part) => reasoning.push_str(&part.text),
            AssistantPart::ToolCall(call) => calls.push(serde_json::json!({"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.raw_input.clone().unwrap_or_else(|| call.input.to_string())}})),
            AssistantPart::Custom(part) if part.kind == "azure.openai.refusal" => {
                if let Some(value) = part.data.as_str() { refusal.push_str(value); }
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
            AzureReasoningField::ReasoningContent => {
                message["reasoning_content"] = reasoning.into()
            }
            AzureReasoningField::Reasoning => message["reasoning"] = reasoning.into(),
            AzureReasoningField::None => {}
        }
    }
    if !refusal.is_empty() {
        message["refusal"] = refusal.into();
    }
    message
}

fn tool_result_message(result: &ToolResultPart) -> JsonValue {
    let content = match &result.content {
        ToolContent::Text(value) => value.clone(),
        ToolContent::Json(value) => value.to_string(),
        ToolContent::Mixed(value) => serde_json::to_string(value).unwrap_or_default(),
        ToolContent::Denied { reason } => reason.clone().unwrap_or_default(),
    };
    serde_json::json!({"role":"tool","tool_call_id":result.tool_call_id,"content":content})
}
