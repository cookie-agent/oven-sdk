//! Chat Completions request validation and translation.

use std::borrow::Cow;

use oven_sdk::{
    AssistantPart, ContentValue, DeferredTools, DeferredToolsCapability, ErrorStage, FilePart,
    FileSource, HistoryTurn, InputPart, JsonValue, LanguageModelDescriptor, ModelCapabilities,
    ModelError, NativeContextScope, ReplayDecision, ReplayDisposition, ReplayOutcome, ReplayPolicy,
    Request, ResponseFormat, ToolChoice, ToolContent, ToolDefinition, ToolResultPart,
};

use crate::{
    chat::{
        body::{self, Content, Message, Part, TypedMessage},
        replay,
    },
    configuration::{
        CompatibleChatDeferredTools, MaxTokensField, ReasoningField, StructuredOutputSupport,
        SystemMessageRole,
    },
    options::{
        CompatibleChatOptions, OpenAiChatOptions, OpenAiPromptCacheTtl, PromptCacheSurface,
        chat_options, compatible_options, prompt_cache_breakpoint, prompt_cache_breakpoint_ttl,
        validate_prompt_cache_controls, validate_prompt_cache_key,
    },
    wire::encode::{
        FileRef, Text, arguments, breakpoint, encoding_error, join, json_size_hint, push_text,
        to_body,
    },
};

pub(crate) struct ChatWireProfile {
    pub(crate) compatible: bool,
    pub(crate) deferred_tools: CompatibleChatDeferredTools,
    pub(crate) system_role: SystemMessageRole,
    pub(crate) max_tokens_field: MaxTokensField,
    pub(crate) stream_usage: bool,
    pub(crate) structured_output: StructuredOutputSupport,
    pub(crate) reasoning_field: ReasoningField,
}

pub(crate) struct Encoded {
    /// Serialized JSON request body.
    pub(crate) body: Vec<u8>,
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
    validate_prompt_cache_controls(
        PromptCacheSurface::Chat,
        options.official.prompt_cache_retention.is_some(),
        options.official.prompt_cache_options.as_ref(),
    )?;
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
    _scope: &NativeContextScope,
    policy: ReplayPolicy,
    profile: &ChatWireProfile,
) -> Result<Encoded, ModelError> {
    let official_options = &options.official;
    let compatible = &options.compatible;
    let mut replay_outcome = ReplayOutcome::default();
    let mut messages = Vec::new();
    let mut warnings = Vec::new();
    let mut size_hint = 0_usize;
    let deferred = DeferredTools::of(request);
    let native_deferral = native_deferral(request, descriptor, profile);
    if native_deferral {
        warnings.extend(deferred.warnings().iter().cloned());
    } else {
        warnings.extend(deferred.upfront_warning(if profile.compatible {
            "openai-compatible-chat"
        } else {
            "openai-chat"
        }));
    }
    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
                let role = match profile.system_role {
                    SystemMessageRole::System => SystemRole::System,
                    SystemMessageRole::Developer => SystemRole::Developer,
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
                let content = user_content(&message.content, profile)?;
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
                if native_deferral {
                    let tools = message
                        .results
                        .iter()
                        .flat_map(|result| deferred.newly_loaded(&result.tool_call_id))
                        .filter_map(|name| request.tools.iter().find(|tool| tool.name == *name))
                        .map(function_tool)
                        .collect::<Vec<_>>();
                    if !tools.is_empty() {
                        messages.push(Message::Typed(TypedMessage::SystemTools { tools }));
                    }
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
                    if let Some(message) =
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
                // Interleaved-thinking compatible providers (DeepSeek in thinking
                // mode) reject a tool loop whose assistant messages omit
                // `reasoning_content`, including turns that produced no reasoning
                // or came from another model. An empty field satisfies them.
                let empty_reasoning_content = profile.compatible
                    && profile.reasoning_field == ReasoningField::ReasoningContent;
                let message = if let Some(mut message) = replayed {
                    if empty_reasoning_content {
                        message = message.with_empty_reasoning_content();
                    }
                    size_hint += replayed_size_hint(&turn.message.content);
                    Message::Replayed(message)
                } else {
                    if policy != ReplayPolicy::Never {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::ReconstructedNormalized,
                        });
                    }
                    assistant_message(
                        &turn.message.content,
                        profile.reasoning_field,
                        empty_reasoning_content,
                    )?
                };
                messages.push(message);
            }
        }
    }
    size_hint += messages.iter().map(Message::size_hint).sum::<usize>();
    let (mut max_tokens, mut max_completion_tokens) = (None, None);
    if let Some(maximum) = request.inference.max_output_tokens {
        match profile.max_tokens_field {
            MaxTokensField::MaxTokens => max_tokens = Some(maximum),
            MaxTokensField::MaxCompletionTokens => max_completion_tokens = Some(maximum),
            MaxTokensField::Omit => warnings.push("Chat profile omitted max output tokens".into()),
        }
    }
    let (tools, tool_choice) = tools(request, native_deferral);
    size_hint += body::tools_size_hint(tools.as_ref());
    let response_format = match &request.response_format {
        ResponseFormat::Text => None,
        ResponseFormat::Json { schema: None } => Some(body::ResponseFormat::JsonObject),
        ResponseFormat::Json {
            schema: Some(schema),
        } => match profile.structured_output {
            StructuredOutputSupport::JsonSchema => Some(body::ResponseFormat::JsonSchema {
                json_schema: body::JsonSchemaFormat {
                    name: "response",
                    strict: true,
                    schema: schema.as_value(),
                },
            }),
            StructuredOutputSupport::JsonObject => {
                warnings.push("Chat profile downgraded JSON Schema to JSON object mode".into());
                Some(body::ResponseFormat::JsonObject)
            }
            StructuredOutputSupport::Unsupported => {
                return Err(ModelError::unsupported(
                    "structured output is not supported by this Chat profile",
                ));
            }
        },
    };
    let extra =
        (profile.compatible && !compatible.extra_body.is_empty()).then_some(&compatible.extra_body);
    size_hint += extra.map_or(0, |extra| {
        extra
            .iter()
            .map(|(key, value)| key.len() + 4 + json_size_hint(value))
            .sum()
    });
    let body = body::Body {
        model: descriptor.identity.model_id.as_str(),
        messages,
        stream: true,
        stream_options: profile.stream_usage.then_some(body::StreamOptions {
            include_usage: true,
        }),
        max_tokens,
        max_completion_tokens,
        temperature: request.inference.temperature,
        top_p: request.inference.top_p,
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
        extra,
    };
    Ok(Encoded {
        body: to_body(
            &body,
            size_hint,
            encoding_error("could not encode OpenAI Chat request"),
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

/// Whether `request` loads deferred tools natively: the model declares it,
/// the endpoint has a dialect, and the request sends tools at all.
fn native_deferral(
    request: &Request,
    descriptor: &LanguageModelDescriptor,
    profile: &ChatWireProfile,
) -> bool {
    descriptor.capabilities.deferred_tools == DeferredToolsCapability::Native
        && profile.deferred_tools == CompatibleChatDeferredTools::SystemMessage
        && !request.tools.is_empty()
        && !matches!(request.tool_choice, ToolChoice::None)
}

fn function_tool(tool: &ToolDefinition) -> body::Tool<'_> {
    body::Tool {
        kind: "function",
        function: body::Function {
            name: &tool.name,
            description: &tool.description,
            parameters: tool.input_schema.as_value(),
            strict: tool
                .provider_options
                .get("openai")
                .and_then(|value| value.get("strict"))
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
        },
    }
}

fn tools(request: &Request, native_deferral: bool) -> Tools<'_> {
    if request.tools.is_empty() || matches!(request.tool_choice, ToolChoice::None) {
        return (None, None);
    }
    // Natively deferred tools are declared by the system message that loads
    // them, never up front.
    let tools = request
        .tools
        .iter()
        .filter(|tool| !(native_deferral && tool.is_deferred()))
        .map(function_tool)
        .collect();
    let choice = match &request.tool_choice {
        ToolChoice::Auto => body::ToolChoice::Mode("auto"),
        ToolChoice::Required => body::ToolChoice::Mode("required"),
        ToolChoice::None => body::ToolChoice::Mode("none"),
        ToolChoice::Tool(name) => body::ToolChoice::Function {
            kind: "function",
            function: body::Named { name },
        },
    };
    (Some(tools), Some(choice))
}

fn text_part<'a>(text: &'a str, metadata: &oven_sdk::PartMetadata) -> Result<Part<'a>, ModelError> {
    Ok(Part::Text {
        text,
        prompt_cache_breakpoint: breakpoint(metadata)?,
    })
}

fn user_content<'a>(
    parts: &'a [InputPart],
    profile: &ChatWireProfile,
) -> Result<Content<'a>, ModelError> {
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
            InputPart::File(file) => Some(file_content(file, profile)),
            InputPart::Custom(_) => None,
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Content::Parts)
}

fn file_content<'a>(file: &'a FilePart, profile: &ChatWireProfile) -> Result<Part<'a>, ModelError> {
    let image = file.media_type.starts_with("image/");
    let data = match FileRef::of(file) {
        Some(data @ FileRef::Data { .. }) => data,
        Some(url @ FileRef::Url(_)) if image => url,
        _ => {
            return Err(ModelError::unsupported("unsupported Chat media source")
                .with_stage(ErrorStage::RequestEncoding));
        }
    };
    let prompt_cache_breakpoint = breakpoint(&file.metadata)?;
    if image {
        Ok(Part::ImageUrl {
            image_url: body::Url { url: data },
            prompt_cache_breakpoint,
        })
    } else if profile.compatible && file.media_type.starts_with("video/") {
        Ok(Part::VideoUrl {
            video_url: body::Url { url: data },
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
                        && chat_breakpoint(&text.metadata)?
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
                        InputPart::Text(text) if chat_breakpoint(&text.metadata)? => {
                            if text.text.is_empty() {
                                return Err(ModelError::invalid_request(
                                    "OpenAI prompt-cache breakpoint has no encodable Chat text block",
                                ));
                            }
                            count += 1;
                        }
                        InputPart::File(file) if chat_breakpoint(&file.metadata)? => count += 1,
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

/// A Chat content part's breakpoint flag; Chat accepts only the `30m` lifetime.
fn chat_breakpoint(metadata: &oven_sdk::PartMetadata) -> Result<bool, ModelError> {
    match prompt_cache_breakpoint_ttl(metadata)? {
        None => Ok(false),
        Some(Some(OpenAiPromptCacheTtl::TwelveHours)) => Err(ModelError::unsupported(
            "OpenAI Chat Completions prompt-cache breakpoints support only the 30m ttl; 12h requires Responses",
        )),
        Some(_) => Ok(true),
    }
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

fn assistant_message(
    parts: &[AssistantPart],
    reasoning_field: ReasoningField,
    empty_reasoning_content: bool,
) -> Result<Message<'_>, ModelError> {
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
            AssistantPart::Custom(part) if part.kind == "openai.refusal" => {
                if let Some(value) = part.data.as_str() {
                    push_text(&mut refusal, value);
                }
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
    let reasoning = (!reasoning.is_empty()).then_some(reasoning);
    let (mut reasoning_content, reasoning) = match reasoning_field {
        ReasoningField::ReasoningContent => (reasoning, None),
        ReasoningField::Reasoning => (None, reasoning),
        ReasoningField::None => (None, None),
    };
    if empty_reasoning_content && reasoning_content.is_none() {
        reasoning_content = Some(Cow::Borrowed(""));
    }
    Ok(Message::Typed(TypedMessage::Assistant {
        content: (!text.is_empty()).then_some(text),
        tool_calls: (!calls.is_empty()).then_some(calls),
        reasoning_content,
        reasoning,
        refusal: (!refusal.is_empty()).then_some(refusal),
    }))
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
                "files in tool results are not deliverable via openai-chat",
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
