//! OpenAI Responses request validation and translation.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use oven_sdk::{
    AssistantPart, CompactionRequest, ContentValue, ErrorStage, FilePart, FileSource, HistoryTurn,
    InputPart, JsonValue, LanguageModelDescriptor, ModelCapabilities, ModelError,
    NativeContextScope, ReplayDecision, ReplayDisposition, ReplayOutcome, ReplayPolicy, Request,
    ResponseFormat, ToolChoice, ToolContent, ToolResultPart,
};

use crate::{
    options::{
        OpenAiResponsesCompactionOptions, OpenAiResponsesOptions, prompt_cache_breakpoint,
        validate_prompt_cache_key,
    },
    responses::{compaction, replay},
};

pub(crate) struct Encoded {
    pub(crate) body: JsonValue,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
}

pub(crate) fn validate_request(
    request: &Request,
    options: &OpenAiResponsesOptions,
    capabilities: &ModelCapabilities,
    descriptor: &LanguageModelDescriptor,
    scope: &NativeContextScope,
) -> Result<(), ModelError> {
    validate_prompt_cache_key(options.prompt_cache_key.as_deref())?;
    let new_breakpoints = validate_prompt_cache_breakpoints(request)?;
    request.validate_for(capabilities)?;
    let retained_breakpoints = validate_native_context(request, descriptor, scope)?;
    // Native windows are authoritative retained state. Fail closed instead of
    // silently deleting their oldest markers when the aggregate wire request
    // would exceed OpenAI's four-write limit.
    if retained_breakpoints.saturating_add(new_breakpoints) > 4 {
        return Err(ModelError::invalid_request(
            "OpenAI allows at most four aggregate prompt-cache breakpoints per request",
        ));
    }
    if (request.inference.reasoning_effort.is_some()
        || options.reasoning_summary.is_some()
        || options.reasoning_mode.is_some())
        && !capabilities
            .features
            .contains(oven_sdk::Capability::REASONING)
    {
        return Err(ModelError::unsupported(
            "reasoning options are not supported by this Responses profile",
        ));
    }
    for turn in &request.history {
        match turn {
            HistoryTurn::User(message) => {
                for part in &message.content {
                    let InputPart::File(file) = part else {
                        continue;
                    };
                    if !file.media_type.starts_with("image/")
                        && file.media_type != "application/pdf"
                    {
                        return Err(ModelError::unsupported("unsupported Responses media type")
                            .with_stage(ErrorStage::RequestEncoding));
                    }
                    if !matches!(
                        file.source,
                        FileSource::Bytes(_) | FileSource::Text(_) | FileSource::Url(_)
                    ) {
                        return Err(
                            ModelError::unsupported("unsupported Responses media source")
                                .with_stage(ErrorStage::RequestEncoding),
                        );
                    }
                }
            }
            HistoryTurn::Assistant(turn)
                if turn
                    .message
                    .content
                    .iter()
                    .any(|part| matches!(part, AssistantPart::File(_))) =>
            {
                return Err(ModelError::unsupported(
                    "Responses assistant-history media is not supported",
                )
                .with_stage(ErrorStage::RequestEncoding));
            }
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn encode_request(
    request: &Request,
    options: &OpenAiResponsesOptions,
    descriptor: &LanguageModelDescriptor,
    scope: &NativeContextScope,
    policy: ReplayPolicy,
) -> Result<Encoded, ModelError> {
    let verbosity = options.verbosity.clone();
    let (input, replay_outcome, warnings) = encode_input(request, descriptor, scope, policy)?;
    let mut include = options.include.clone();
    if !include
        .iter()
        .any(|value| value == "reasoning.encrypted_content")
    {
        include.push("reasoning.encrypted_content".into());
    }
    let mut body = serde_json::json!({
        "model":descriptor.identity.model_id.as_str(),
        "input":input,
        "stream":true,
        "store":false,
        "include":include
    });
    if let Some(maximum) = request.inference.max_output_tokens {
        body["max_output_tokens"] = maximum.into();
    }
    if let Some(value) = request.inference.temperature {
        body["temperature"] = value.into();
    }
    if let Some(value) = request.inference.top_p {
        body["top_p"] = value.into();
    }
    if request.inference.reasoning_effort.is_some()
        || options.reasoning_summary.is_some()
        || options.reasoning_mode.is_some()
    {
        let mut reasoning = serde_json::Map::new();
        if let Some(effort) = &request.inference.reasoning_effort {
            reasoning.insert("effort".into(), effort.clone().into());
        }
        if let Some(summary) = &options.reasoning_summary {
            reasoning.insert("summary".into(), summary.clone().into());
        }
        if let Some(mode) = &options.reasoning_mode {
            reasoning.insert("mode".into(), mode.clone().into());
        }
        body["reasoning"] = JsonValue::Object(reasoning);
    }
    if let Some(user) = &options.user {
        body["user"] = user.clone().into();
    }
    if let Some(service_tier) = &options.service_tier {
        body["service_tier"] = service_tier.clone().into();
    }
    if let Some(truncation) = &options.truncation {
        body["truncation"] = truncation.clone().into();
    }
    if let Some(parallel) = options.parallel_tool_calls {
        body["parallel_tool_calls"] = parallel.into();
    }
    if let Some(key) = &options.prompt_cache_key {
        body["prompt_cache_key"] = key.clone().into();
    }
    if let Some(retention) = options.prompt_cache_retention {
        body["prompt_cache_retention"] =
            serde_json::to_value(retention).expect("prompt-cache retention is serializable");
    }
    if let Some(cache_options) = &options.prompt_cache_options {
        body["prompt_cache_options"] =
            serde_json::to_value(cache_options).expect("prompt-cache options are serializable");
    }
    add_tools_and_output(request, verbosity.as_deref(), &mut body);
    Ok(Encoded {
        body,
        replay: replay_outcome,
        warnings,
    })
}

pub(crate) fn encode_compaction_request(
    request: &CompactionRequest,
    options: &OpenAiResponsesCompactionOptions,
    descriptor: &LanguageModelDescriptor,
    scope: &NativeContextScope,
    policy: ReplayPolicy,
) -> Result<Encoded, ModelError> {
    let (input, replay, warnings) = encode_input(&request.request, descriptor, scope, policy)?;
    let mut body = serde_json::json!({
        "model": descriptor.identity.model_id.as_str(),
        "input": input,
    });
    if let Some(value) = &options.instructions {
        body["instructions"] = value.clone().into();
    }
    if let Some(value) = &options.prompt_cache_key {
        body["prompt_cache_key"] = value.clone().into();
    }
    if let Some(value) = &options.prompt_cache_options {
        body["prompt_cache_options"] = serde_json::to_value(value).map_err(|_| {
            ModelError::invalid_request("could not encode OpenAI compaction cache options")
                .with_stage(ErrorStage::NativeContextEncode)
        })?;
    }
    if let Some(value) = &options.prompt_cache_retention {
        body["prompt_cache_retention"] = value.clone().into();
    }
    if let Some(value) = &options.service_tier {
        body["service_tier"] = value.clone().into();
    }
    Ok(Encoded {
        body,
        replay,
        warnings,
    })
}

fn encode_input(
    request: &Request,
    descriptor: &LanguageModelDescriptor,
    scope: &NativeContextScope,
    policy: ReplayPolicy,
) -> Result<(Vec<JsonValue>, ReplayOutcome, Vec<String>), ModelError> {
    let mut input = request
        .native_context
        .as_ref()
        .map(compaction::native_input)
        .transpose()?
        .unwrap_or_default();
    let mut replay_outcome = ReplayOutcome::default();
    let mut warnings = Vec::new();
    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
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
                                    serde_json::json!({"type":"input_text","text":text.text}),
                                    &text.metadata,
                                ))
                            }
                            _ => None,
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if !content.is_empty() {
                        input.push(serde_json::json!({"type":"message","role":"developer","content":content}));
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
                        input.push(serde_json::json!({"type":"message","role":"developer","content":[{"type":"input_text","text":content}]}));
                    }
                }
            }
            HistoryTurn::User(message) => {
                let content = message
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        InputPart::Text(text) => Some(attach_prompt_cache_breakpoint(
                            serde_json::json!({"type":"input_text","text":text.text}),
                            &text.metadata,
                        )),
                        InputPart::File(file) => Some(input_file(file)),
                        InputPart::Custom(_) => None,
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if !content.is_empty() {
                    input.push(
                        serde_json::json!({"type":"message","role":"user","content":content}),
                    );
                }
            }
            HistoryTurn::Tool(message) => {
                input.extend(message.results.iter().map(function_output));
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
                    } else if let Some(items) = replay::decode(artifact, &turn.message.content) {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::Replayed,
                        });
                        replayed = Some(items);
                    } else {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedInvalidPayload {
                                reason: "Responses replay payload did not match normalized content"
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
                if let Some(items) = replayed {
                    input.extend(items);
                } else {
                    if policy != ReplayPolicy::Never {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::ReconstructedNormalized,
                        });
                    }
                    input.extend(normalized_assistant(&turn.message.content, &mut warnings)?);
                }
            }
        }
    }
    Ok((input, replay_outcome, warnings))
}

fn add_tools_and_output(request: &Request, verbosity: Option<&str>, body: &mut JsonValue) {
    if !request.tools.is_empty() && !matches!(request.tool_choice, ToolChoice::None) {
        body["tools"] = JsonValue::Array(request.tools.iter().map(|tool| {
            let strict = tool.provider_options.get("openai").and_then(|value| value.get("strict")).and_then(JsonValue::as_bool).unwrap_or(false);
            serde_json::json!({"type":"function","name":tool.name,"description":tool.description,"parameters":tool.input_schema.as_value(),"strict":strict})
        }).collect());
    }
    body["tool_choice"] = match &request.tool_choice {
        ToolChoice::Auto => JsonValue::String("auto".into()),
        ToolChoice::Required => JsonValue::String("required".into()),
        ToolChoice::None => JsonValue::String("none".into()),
        ToolChoice::Tool(name) => serde_json::json!({"type":"function","name":name}),
    };
    match &request.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::Json { schema: None } => {
            body["text"] = serde_json::json!({"format":{"type":"json_object"}});
        }
        ResponseFormat::Json {
            schema: Some(schema),
        } => {
            body["text"] = serde_json::json!({"format":{"type":"json_schema","name":"response","strict":true,"schema":schema.as_value()}});
        }
    }
    if let Some(verbosity) = verbosity {
        if body.get("text").is_none() {
            body["text"] = serde_json::json!({});
        }
        body["text"]["verbosity"] = verbosity.into();
    }
}

fn validate_native_context(
    request: &Request,
    descriptor: &LanguageModelDescriptor,
    scope: &NativeContextScope,
) -> Result<usize, ModelError> {
    let Some(window) = &request.native_context else {
        return Ok(0);
    };
    if window.adapter_id() != &descriptor.adapter_id {
        return Err(ModelError::native_context(
            "OpenAI Responses native context belongs to a different adapter",
        )
        .with_stage(ErrorStage::NativeContextDecode));
    }
    if window.scope() != scope {
        return Err(ModelError::native_context(
            "OpenAI Responses native context belongs to a different scope",
        )
        .with_stage(ErrorStage::NativeContextDecode));
    }
    let input = compaction::native_input(window)?;
    Ok(input.iter().map(count_wire_prompt_cache_breakpoints).sum())
}

fn count_wire_prompt_cache_breakpoints(value: &JsonValue) -> usize {
    match value {
        JsonValue::Array(values) => values.iter().map(count_wire_prompt_cache_breakpoints).sum(),
        JsonValue::Object(object) => {
            usize::from(object.contains_key("prompt_cache_breakpoint"))
                + object
                    .values()
                    .map(count_wire_prompt_cache_breakpoints)
                    .sum::<usize>()
        }
        _ => 0,
    }
}

fn input_file(file: &FilePart) -> Result<JsonValue, ModelError> {
    let data = match &file.source {
        FileSource::Bytes(bytes) => {
            format!("data:{};base64,{}", file.media_type, STANDARD.encode(bytes))
        }
        FileSource::Text(text) => {
            format!("data:{};base64,{}", file.media_type, STANDARD.encode(text))
        }
        FileSource::Url(url) => {
            if file.media_type.starts_with("image/") {
                return attach_prompt_cache_breakpoint(
                    serde_json::json!({"type":"input_image","image_url":url}),
                    &file.metadata,
                );
            }
            return attach_prompt_cache_breakpoint(
                serde_json::json!({
                    "type":"input_file",
                    "filename":file.filename.clone().unwrap_or_else(|| "document.pdf".into()),
                    "file_url":url
                }),
                &file.metadata,
            );
        }
        FileSource::ProviderReference { .. } => {
            return Err(
                ModelError::unsupported("provider file references are unsupported")
                    .with_stage(ErrorStage::RequestEncoding),
            );
        }
    };
    let value = if file.media_type.starts_with("image/") {
        serde_json::json!({"type":"input_image","image_url":data})
    } else {
        serde_json::json!({"type":"input_file","filename":file.filename.clone().unwrap_or_else(|| "document.pdf".into()),"file_data":data})
    };
    attach_prompt_cache_breakpoint(value, &file.metadata)
}

fn validate_prompt_cache_breakpoints(request: &Request) -> Result<usize, ModelError> {
    let mut count = 0_usize;
    for turn in &request.history {
        match turn {
            HistoryTurn::System(message) => {
                for part in &message.content {
                    if let oven_sdk::SystemPart::Text(text) = part
                        && prompt_cache_breakpoint(&text.metadata)?
                    {
                        if text.text.is_empty() {
                            return Err(ModelError::invalid_request(
                                "OpenAI prompt-cache breakpoint has no encodable Responses text block",
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
                                    "OpenAI prompt-cache breakpoint has no encodable Responses text block",
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
                            "OpenAI Responses prompt-cache breakpoints are valid only on system or user input content",
                        ));
                    }
                }
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    if tool_content_has_prompt_cache_breakpoint(&result.content)? {
                        return Err(ModelError::invalid_request(
                            "OpenAI Responses prompt-cache breakpoints are not valid on function outputs",
                        ));
                    }
                }
            }
        }
    }
    Ok(count)
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

fn function_output(result: &ToolResultPart) -> JsonValue {
    let output = match &result.content {
        ToolContent::Text(value) => value.clone(),
        ToolContent::Json(value) => value.to_string(),
        ToolContent::Mixed(value) => serde_json::to_string(value).unwrap_or_default(),
        ToolContent::Denied { reason } => reason.clone().unwrap_or_default(),
    };
    serde_json::json!({"type":"function_call_output","call_id":result.tool_call_id,"output":output})
}

fn normalized_assistant(
    parts: &[AssistantPart],
    warnings: &mut Vec<String>,
) -> Result<Vec<JsonValue>, ModelError> {
    let mut output = Vec::new();
    let text = parts
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Text(part) => Some(part.text.as_str()),
            _ => None,
        })
        .collect::<String>();
    if !text.is_empty() {
        output.push(serde_json::json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":text}]}));
    }
    for part in parts {
        match part {
            AssistantPart::ToolCall(call) => {
                let mut item = serde_json::json!({
                    "type":"function_call",
                    "call_id":call.id,
                    "name":call.name,
                    "arguments":call.raw_input.clone().unwrap_or_else(|| call.input.to_string())
                });
                if let Some(provider_item_id) = &call.provider_item_id {
                    item["id"] = provider_item_id.clone().into();
                }
                output.push(item);
            }
            AssistantPart::ToolResult(result) => output.push(function_output(result)),
            AssistantPart::Reasoning(_) => warnings.push(
                "Responses normalized fallback omitted reasoning without encrypted replay state"
                    .into(),
            ),
            AssistantPart::File(_) => {
                return Err(ModelError::unsupported(
                    "Responses assistant-history media is not supported",
                )
                .with_stage(ErrorStage::RequestEncoding));
            }
            _ => {}
        }
    }
    Ok(output)
}
