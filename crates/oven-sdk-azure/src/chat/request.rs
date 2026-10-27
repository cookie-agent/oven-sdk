//! Chat Completions request validation and translation.

use std::borrow::Cow;

use oven_sdk::{
    AssistantPart, Capability, ContentValue, ErrorStage, FilePart, HistoryTurn, InputPart,
    JsonValue, LanguageModelDescriptor, ModelCapabilities, ModelError, NativeContextScope,
    ReplayDecision, ReplayDisposition, ReplayOutcome, ReplayPolicy, Request, ResponseFormat,
    ToolChoice, ToolContent, ToolResultPart,
};

use crate::{
    chat::{
        body::{self, Content, Message, Part, TypedMessage},
        replay,
    },
    configuration::{
        AzureMaxTokensField, AzureReasoningField, AzureStructuredOutputSupport,
        AzureSystemMessageRole,
    },
    encode::{FileRef, Text, arguments, breakpoint, encoding_error, join, push_text, to_body},
    media,
    options::{AzureOpenAiChatOptions, prompt_cache_breakpoint, validate_prompt_cache_key},
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
    /// Serialized JSON request body.
    pub(crate) body: Vec<u8>,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
}

pub(crate) fn validate_request(
    request: &Request,
    options: &AzureOpenAiChatOptions,
    capabilities: &ModelCapabilities,
    profile: &ChatWireProfile,
) -> Result<(), ModelError> {
    validate_prompt_cache_key(options.prompt_cache_key.as_deref())?;
    crate::options::validate_prompt_cache_controls(
        options.prompt_cache_retention.is_some(),
        options.prompt_cache_options.as_ref(),
    )?;
    validate_prompt_cache_breakpoints(request, profile)?;
    request.validate_for(capabilities)?;
    if !request.tools.is_empty() && !capabilities.features.contains(Capability::TOOL_CALLING) {
        return Err(ModelError::unsupported(
            "tool calling is not enabled by this Chat configuration",
        ));
    }
    schema::validate_request_schemas(request, options.parallel_tool_calls)?;
    media::validate_request_media(request, media::Surface::Chat)?;
    validate_tool_result_files(request)?;
    Ok(())
}

pub(crate) fn encode_request(
    request: &Request,
    official_options: &AzureOpenAiChatOptions,
    descriptor: &LanguageModelDescriptor,
    policy: ReplayPolicy,
    profile: &ChatWireProfile,
    replay_binding: &JsonValue,
    _replay_scope: &NativeContextScope,
) -> Result<Encoded, ModelError> {
    let mut replay_outcome = ReplayOutcome::default();
    let mut messages = Vec::new();
    let mut warnings = Vec::new();
    warnings.extend(oven_sdk::DeferredTools::of(request).upfront_warning("azure-openai-chat"));
    let mut size_hint = 0_usize;
    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
                let role = match profile.system_role {
                    AzureSystemMessageRole::System => SystemRole::System,
                    AzureSystemMessageRole::Developer => SystemRole::Developer,
                    AzureSystemMessageRole::Omit => {
                        warnings.push("Chat configuration omitted a system message".into());
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
                                Some(text_part(&text.text, &text.metadata))
                            }
                            _ => None,
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if !content.is_empty() {
                        messages.push(role.message(Content::Parts(content)));
                    }
                } else {
                    let content = join(
                        message.content.iter().filter_map(|part| match part {
                            oven_sdk::SystemPart::Text(text) => Some(text.text.as_str()),
                            oven_sdk::SystemPart::Custom(_) => None,
                        }),
                        "\n",
                    );
                    if !content.is_empty() {
                        messages.push(role.message(Content::Text(content)));
                    }
                }
            }
            HistoryTurn::User(message) => {
                let content = user_content(&message.content)?;
                if !matches!(&content, Content::Parts(parts) if parts.is_empty())
                    && !matches!(&content, Content::Text(text) if text.is_empty())
                {
                    messages.push(Message::Typed(TypedMessage::User { content }));
                }
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    messages.push(tool_result_message(result)?);
                }
            }
            HistoryTurn::Assistant(turn) => {
                let mut replayed = None;
                if policy == ReplayPolicy::Never {
                    replay_outcome.decisions.push(ReplayDecision {
                        history_index,
                        disposition: ReplayDisposition::ReconstructedNormalized,
                    });
                } else if let Some(artifact) = &turn.finish.native_replay {
                    if let Some(message) = replay::decode(
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
                    size_hint += replayed_size_hint(&turn.message.content);
                    messages.push(Message::Replayed(message));
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
    size_hint += messages.iter().map(Message::size_hint).sum::<usize>();
    let (mut max_tokens, mut max_completion_tokens) = (None, None);
    if let Some(maximum) = request.inference.max_output_tokens {
        match profile.max_tokens_field {
            AzureMaxTokensField::MaxTokens => max_tokens = Some(maximum),
            AzureMaxTokensField::MaxCompletionTokens => max_completion_tokens = Some(maximum),
            AzureMaxTokensField::Omit => {
                warnings.push("Chat configuration omitted max output tokens".into())
            }
        }
    }
    let omit_sampling = profile.omit_reasoning_sampling;
    let mut temperature = request.inference.temperature;
    if temperature.is_some() && omit_sampling {
        warnings.push("Azure reasoning configuration omitted incompatible temperature".into());
        temperature = None;
    }
    let mut top_p = request.inference.top_p;
    if top_p.is_some() && omit_sampling {
        warnings.push("Azure reasoning configuration omitted incompatible top_p".into());
        top_p = None;
    }
    let (tools, tool_choice) = tools(request)?;
    size_hint += body::tools_size_hint(tools.as_ref());
    let response_format = match &request.response_format {
        ResponseFormat::Text => None,
        ResponseFormat::Json { schema: None } => Some(body::ResponseFormat::JsonObject),
        ResponseFormat::Json {
            schema: Some(schema),
        } => match profile.structured_output {
            AzureStructuredOutputSupport::JsonSchema => Some(body::ResponseFormat::JsonSchema {
                json_schema: body::JsonSchemaFormat {
                    name: "response",
                    strict: true,
                    schema: schema.as_value(),
                },
            }),
            AzureStructuredOutputSupport::JsonObject => {
                warnings
                    .push("Chat configuration downgraded JSON Schema to JSON object mode".into());
                Some(body::ResponseFormat::JsonObject)
            }
            AzureStructuredOutputSupport::Unsupported => {
                return Err(ModelError::unsupported(
                    "structured output is not enabled by this Chat configuration",
                ));
            }
        },
    };
    let body = body::Body {
        model: descriptor.identity.model_id.as_str(),
        messages,
        stream: true,
        stream_options: profile.stream_usage.then_some(body::StreamOptions {
            include_usage: true,
        }),
        max_tokens,
        max_completion_tokens,
        temperature,
        top_p,
        reasoning_effort: official_options
            .reasoning_effort
            .as_deref()
            .or(request.inference.reasoning_effort.as_deref()),
        user: official_options.user.as_deref(),
        service_tier: official_options.service_tier.as_deref(),
        verbosity: official_options.verbosity.as_deref(),
        parallel_tool_calls: official_options.parallel_tool_calls,
        prompt_cache_key: official_options.prompt_cache_key.as_deref(),
        prompt_cache_retention: official_options.prompt_cache_retention,
        prompt_cache_options: official_options.prompt_cache_options.as_ref(),
        tools,
        tool_choice,
        response_format,
    };
    Ok(Encoded {
        body: to_body(
            &body,
            size_hint,
            encoding_error("could not encode Azure OpenAI Chat request"),
        )?,
        replay: replay_outcome,
        warnings,
    })
}

#[derive(Clone, Copy)]
enum SystemRole {
    System,
    Developer,
}

impl SystemRole {
    fn message(self, content: Content<'_>) -> Message<'_> {
        Message::Typed(match self {
            Self::System => TypedMessage::System { content },
            Self::Developer => TypedMessage::Developer { content },
        })
    }
}

/// Approximates a replayed message from the normalized parts it matched.
fn replayed_size_hint(parts: &[AssistantPart]) -> usize {
    96 + parts
        .iter()
        .map(|part| match part {
            AssistantPart::Text(part) => part.text.len(),
            AssistantPart::Reasoning(part) => part.text.len(),
            AssistantPart::ToolCall(call) => {
                96 + call.id.len()
                    + call.name.len()
                    + call.raw_input.as_ref().map_or(64, String::len) * 9 / 8
            }
            _ => 16,
        })
        .sum::<usize>()
}

type Tools<'a> = (Option<Vec<body::Tool<'a>>>, Option<body::ToolChoice<'a>>);

fn tools(request: &Request) -> Result<Tools<'_>, ModelError> {
    if request.tools.is_empty() || matches!(request.tool_choice, ToolChoice::None) {
        return Ok((None, None));
    }
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            Ok(body::Tool {
                kind: "function",
                function: body::Function {
                    name: &tool.name,
                    description: &tool.description,
                    parameters: tool.input_schema.as_value(),
                    strict: schema::strict_tool(tool)?,
                },
            })
        })
        .collect::<Result<Vec<_>, ModelError>>()?;
    let choice = match &request.tool_choice {
        ToolChoice::Auto => body::ToolChoice::Mode("auto"),
        ToolChoice::Required => body::ToolChoice::Mode("required"),
        ToolChoice::None => body::ToolChoice::Mode("none"),
        ToolChoice::Tool(name) => body::ToolChoice::Function {
            kind: "function",
            function: body::Named { name },
        },
    };
    Ok((Some(tools), Some(choice)))
}

fn text_part<'a>(text: &'a str, metadata: &oven_sdk::PartMetadata) -> Result<Part<'a>, ModelError> {
    Ok(Part::Text {
        text,
        prompt_cache_breakpoint: breakpoint(metadata)?,
    })
}

fn user_content(parts: &[InputPart]) -> Result<Content<'_>, ModelError> {
    if parts.iter().all(|part| matches!(part, InputPart::Text(_)))
        && !parts.iter().any(|part| {
            matches!(part, InputPart::Text(text) if prompt_cache_breakpoint(&text.metadata).unwrap_or(false))
        })
    {
        return Ok(Content::Text(join(
            parts.iter().filter_map(|part| match part {
                InputPart::Text(text) => Some(text.text.as_str()),
                _ => None,
            }),
            "",
        )));
    }
    parts
        .iter()
        .filter_map(|part| match part {
            InputPart::Text(text) => Some(text_part(&text.text, &text.metadata)),
            InputPart::File(file) => Some(file_content(file)),
            InputPart::Custom(_) => None,
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Content::Parts)
}

fn file_content(file: &FilePart) -> Result<Part<'_>, ModelError> {
    let Some(data) = FileRef::of(file) else {
        return Err(ModelError::unsupported("unsupported Chat media source")
            .with_stage(ErrorStage::RequestEncoding));
    };
    let prompt_cache_breakpoint = breakpoint(&file.metadata)?;
    if media::is_image(&file.media_type) {
        Ok(Part::ImageUrl {
            image_url: body::Url { url: data },
            prompt_cache_breakpoint,
        })
    } else if file.media_type == "application/pdf" {
        Ok(Part::File {
            file: body::File {
                filename: file.filename.as_deref().unwrap_or("document.pdf"),
                file_data: data,
            },
            prompt_cache_breakpoint,
        })
    } else {
        Err(ModelError::unsupported("unsupported Chat media type")
            .with_stage(ErrorStage::RequestEncoding))
    }
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
                        if text.text.is_empty()
                            || profile.system_role == AzureSystemMessageRole::Omit
                        {
                            return Err(ModelError::invalid_request(
                                "Azure OpenAI prompt-cache breakpoint has no encodable Chat text block",
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
                                    "Azure OpenAI prompt-cache breakpoint has no encodable Chat text block",
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
                            "Azure OpenAI Chat prompt-cache breakpoints are valid only on system or user content",
                        ));
                    }
                }
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    if tool_content_has_prompt_cache_breakpoint(&result.content)? {
                        return Err(ModelError::invalid_request(
                            "Azure OpenAI Chat prompt-cache breakpoints are not valid on tool results",
                        ));
                    }
                }
            }
        }
    }
    if count > 4 {
        return Err(ModelError::invalid_request(
            "Azure OpenAI allows at most four prompt-cache breakpoints per request",
        ));
    }
    Ok(())
}

fn tool_content_has_prompt_cache_breakpoint(content: &ToolContent) -> Result<bool, ModelError> {
    match content {
        ToolContent::Mixed(values) => values.iter().try_fold(false, |marked, value| {
            Ok(marked || matches!(value, ContentValue::File(file) if prompt_cache_breakpoint(&file.metadata)?))
        }),
        _ => Ok(false),
    }
}

fn assistant_message(parts: &[AssistantPart], reasoning_field: AzureReasoningField) -> Message<'_> {
    let mut text = Cow::Borrowed("");
    let mut reasoning = Cow::Borrowed("");
    let mut refusal = Cow::Borrowed("");
    let mut calls = Vec::new();
    for part in parts {
        match part {
            AssistantPart::Text(part) => push_text(&mut text, &part.text),
            AssistantPart::Reasoning(part) => push_text(&mut reasoning, &part.text),
            AssistantPart::ToolCall(call) => calls.push(body::ToolCall {
                id: &call.id,
                kind: "function",
                function: body::CallFunction {
                    name: &call.name,
                    arguments: arguments(call),
                },
            }),
            AssistantPart::Custom(part) if part.kind == "azure.openai.refusal" => {
                if let Some(value) = part.data.as_str() {
                    push_text(&mut refusal, value);
                }
            }
            _ => {}
        }
    }
    let reasoning = (!reasoning.is_empty()).then_some(reasoning);
    let (reasoning_content, reasoning) = match reasoning_field {
        AzureReasoningField::ReasoningContent => (reasoning, None),
        AzureReasoningField::Reasoning => (None, reasoning),
        AzureReasoningField::None => (None, None),
    };
    Message::Typed(TypedMessage::Assistant {
        content: (!text.is_empty()).then_some(text),
        tool_calls: (!calls.is_empty()).then_some(calls),
        reasoning_content,
        reasoning,
        refusal: (!refusal.is_empty()).then_some(refusal),
    })
}

fn tool_result_message(result: &ToolResultPart) -> Result<Message<'_>, ModelError> {
    let content = match &result.content {
        ToolContent::Text(value) => Text::Borrowed(value),
        ToolContent::LoadTools { text: value, .. } => Text::Borrowed(value),
        ToolContent::Json(value) => Text::Json(value),
        ToolContent::Mixed(values)
            if values
                .iter()
                .any(|value| matches!(value, ContentValue::File(_))) =>
        {
            return Err(ModelError::unsupported(
                "files in tool results are not deliverable via azure-openai-chat",
            ));
        }
        ToolContent::Mixed(values) => {
            Text::Owned(serde_json::to_string(values).unwrap_or_default())
        }
        ToolContent::Denied { reason } => Text::Borrowed(reason.as_deref().unwrap_or_default()),
    };
    Ok(Message::Typed(TypedMessage::Tool {
        tool_call_id: &result.tool_call_id,
        content,
    }))
}

fn validate_tool_result_files(request: &Request) -> Result<(), ModelError> {
    for turn in &request.history {
        match turn {
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    reject_tool_result_files(result)?;
                }
            }
            HistoryTurn::Assistant(turn) => {
                for part in &turn.message.content {
                    if let AssistantPart::ToolResult(result) = part {
                        reject_tool_result_files(result)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn reject_tool_result_files(result: &ToolResultPart) -> Result<(), ModelError> {
    if let ToolContent::Mixed(values) = &result.content
        && values
            .iter()
            .any(|value| matches!(value, ContentValue::File(_)))
    {
        return Err(ModelError::unsupported(
            "files in tool results are not deliverable via azure-openai-chat",
        ));
    }
    Ok(())
}
