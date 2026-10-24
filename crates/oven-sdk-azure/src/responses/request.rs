//! Azure OpenAI Responses request validation and translation.

use std::borrow::Cow;

use oven_sdk::{
    AssistantPart, Capability, ContentValue, ErrorStage, FilePart, HistoryTurn, InputPart,
    JsonValue, LanguageModelDescriptor, ModelCapabilities, ModelError, NativeContextScope,
    ReplayDecision, ReplayDisposition, ReplayOutcome, ReplayPolicy, Request, ResponseFormat,
    ToolChoice, ToolContent, ToolResultPart,
};

use crate::{
    encode::{
        FileRef, Text, arguments, breakpoint, encoding_error, join, json_size_hint, push_text,
        to_body,
    },
    media,
    options::{AzureOpenAiResponsesOptions, prompt_cache_breakpoint, validate_prompt_cache_key},
    responses::{
        body::{self, Content, Item, Output, TypedItem},
        replay,
    },
    schema,
};

pub(crate) struct Encoded {
    /// Serialized JSON request body.
    pub(crate) body: Vec<u8>,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
}

pub(crate) struct EncodedInput<'a> {
    pub(crate) input: Vec<Item<'a>>,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
}

pub(crate) fn validate_request(
    request: &Request,
    options: &AzureOpenAiResponsesOptions,
    capabilities: &ModelCapabilities,
    descriptor: &LanguageModelDescriptor,
    native_scope: Option<&NativeContextScope>,
) -> Result<(), ModelError> {
    validate_prompt_cache_key(options.prompt_cache_key.as_deref())?;
    crate::options::validate_prompt_cache_controls(
        options.prompt_cache_retention.is_some(),
        options.prompt_cache_options.as_ref(),
    )?;
    validate_prompt_cache_breakpoints(request)?;
    request.validate_for(capabilities)?;
    if !request.tools.is_empty() && !capabilities.features.contains(Capability::TOOL_CALLING) {
        return Err(ModelError::unsupported(
            "tool calling is not enabled by this Responses configuration",
        ));
    }
    if (request.inference.reasoning_effort.is_some()
        || options.reasoning_summary.is_some()
        || options.reasoning_mode.is_some())
        && !capabilities.features.contains(Capability::REASONING)
    {
        return Err(ModelError::unsupported(
            "reasoning options are not enabled by this Responses configuration",
        ));
    }
    schema::validate_request_schemas(request, options.parallel_tool_calls)?;
    media::validate_request_media(request, media::Surface::Responses)?;
    validate_tool_result_files(request)?;
    validate_native_context(request, descriptor, native_scope)?;
    Ok(())
}

fn validate_native_context(
    request: &Request,
    descriptor: &LanguageModelDescriptor,
    native_scope: Option<&NativeContextScope>,
) -> Result<(), ModelError> {
    let Some(window) = &request.native_context else {
        return Ok(());
    };
    let expected = native_scope.ok_or_else(|| {
        ModelError::native_context("Azure Responses native context is not configured")
    })?;
    if window.adapter_id() != &descriptor.adapter_id {
        return Err(ModelError::native_context(
            "Azure Responses native context belongs to another adapter",
        ));
    }
    if window.scope() != expected {
        return Err(ModelError::native_context(
            "Azure Responses native context belongs to another configured resource",
        ));
    }
    super::compaction::decode_window(window)?;
    Ok(())
}

pub(crate) fn encode_input<'a>(
    request: &'a Request,
    descriptor: &LanguageModelDescriptor,
    policy: ReplayPolicy,
    replay_binding: &JsonValue,
    _replay_scope: &NativeContextScope,
) -> Result<EncodedInput<'a>, ModelError> {
    let mut input = if let Some(window) = &request.native_context {
        super::compaction::decode_window(window)?
            .iter()
            .map(Item::Native)
            .collect()
    } else {
        Vec::new()
    };
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
                                Some(input_text(Text::Borrowed(&text.text), &text.metadata))
                            }
                            _ => None,
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if !content.is_empty() {
                        input.push(Item::message("developer", content));
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
                        input.push(Item::message(
                            "developer",
                            vec![Content::InputText {
                                text: cow_text(content),
                                prompt_cache_breakpoint: None,
                            }],
                        ));
                    }
                }
            }
            HistoryTurn::User(message) => {
                let content = message
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        InputPart::Text(text) => {
                            Some(input_text(Text::Borrowed(&text.text), &text.metadata))
                        }
                        InputPart::File(file) => Some(input_file(file)),
                        InputPart::Custom(_) => None,
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if !content.is_empty() {
                    input.push(Item::message("user", content));
                }
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    input.push(function_output(result)?);
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
                    if let Some(items) =
                        replay::decode(artifact, &turn.message.content, replay_binding)
                    {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::Replayed,
                        });
                        replayed = Some(oven_sdk::replay::eligible_responses_items(
                            artifact,
                            &descriptor.identity.model_id,
                            descriptor.capabilities.replay.reasoning,
                            items,
                            &mut warnings,
                        )?);
                    } else {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedInvalidPayload {
                                reason: discard_reason(artifact),
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
                    input.extend(items.into_iter().map(Item::Native));
                } else {
                    if policy != ReplayPolicy::Never {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::ReconstructedNormalized,
                        });
                    }
                    normalized_assistant(&turn.message.content, &mut input, &mut warnings)?;
                }
            }
        }
    }
    Ok(EncodedInput {
        input,
        replay: replay_outcome,
        warnings,
    })
}

pub(crate) fn encode_request(
    request: &Request,
    options: &AzureOpenAiResponsesOptions,
    descriptor: &LanguageModelDescriptor,
    policy: ReplayPolicy,
    replay_binding: &JsonValue,
    replay_scope: &NativeContextScope,
) -> Result<Encoded, ModelError> {
    let EncodedInput {
        input,
        replay,
        mut warnings,
    } = encode_input(request, descriptor, policy, replay_binding, replay_scope)?;
    let mut size_hint = input.iter().map(Item::size_hint).sum::<usize>();
    let mut include = options
        .include
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if descriptor.capabilities.replay.reasoning && !include.contains(&"reasoning.encrypted_content")
    {
        include.push("reasoning.encrypted_content");
    }
    let reasoning_model = descriptor
        .capabilities
        .features
        .contains(Capability::REASONING);
    let mut temperature = request.inference.temperature;
    if temperature.is_some() && reasoning_model {
        warnings.push("Azure reasoning configuration omitted incompatible temperature".into());
        temperature = None;
    }
    let mut top_p = request.inference.top_p;
    if top_p.is_some() && reasoning_model {
        warnings.push("Azure reasoning configuration omitted incompatible top_p".into());
        top_p = None;
    }
    let reasoning = (request.inference.reasoning_effort.is_some()
        || options.reasoning_summary.is_some()
        || options.reasoning_mode.is_some())
    .then_some(body::Reasoning {
        effort: request.inference.reasoning_effort.as_deref(),
        summary: options.reasoning_summary.as_deref(),
        mode: options.reasoning_mode.as_deref(),
    });
    let (tools, tool_choice) = tools(request)?;
    size_hint += tools.as_ref().map_or(0, |tools| {
        tools
            .iter()
            .map(|tool| {
                tool.name.len() + tool.description.len() + json_size_hint(tool.parameters) + 64
            })
            .sum()
    });
    let body = body::Body {
        model: descriptor.identity.model_id.as_str(),
        input,
        stream: true,
        store: false,
        include,
        max_output_tokens: request.inference.max_output_tokens,
        temperature,
        top_p,
        reasoning,
        user: options.user.as_deref(),
        service_tier: options.service_tier.as_deref(),
        truncation: options.truncation.as_deref(),
        parallel_tool_calls: options.parallel_tool_calls,
        prompt_cache_key: options.prompt_cache_key.as_deref(),
        prompt_cache_retention: options.prompt_cache_retention,
        prompt_cache_options: options.prompt_cache_options.as_ref(),
        tools,
        tool_choice,
        text: text_config(request, options.verbosity.as_deref()),
    };
    Ok(Encoded {
        body: to_body(
            &body,
            size_hint,
            encoding_error("could not encode Azure OpenAI Responses request"),
        )?,
        replay,
        warnings,
    })
}

fn discard_reason(artifact: &oven_sdk::NativeReplayArtifact) -> String {
    match artifact.payload().get("format").and_then(JsonValue::as_str) {
        Some("oven.openai.responses.output.v1" | "oven.azure.openai.responses.output.v4") => {
            "Responses replay payload does not match normalized content".to_owned()
        }
        _ => "replay artifact format is not supported by the Responses target".to_owned(),
    }
}

type Tools<'a> = (
    Option<Vec<body::FunctionTool<'a>>>,
    Option<body::ToolChoice<'a>>,
);

fn tools(request: &Request) -> Result<Tools<'_>, ModelError> {
    if request.tools.is_empty() || matches!(request.tool_choice, ToolChoice::None) {
        return Ok((None, None));
    }
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            Ok(body::FunctionTool {
                kind: "function",
                name: &tool.name,
                description: &tool.description,
                parameters: tool.input_schema.as_value(),
                strict: schema::strict_tool(tool)?,
            })
        })
        .collect::<Result<Vec<_>, ModelError>>()?;
    let choice = match &request.tool_choice {
        ToolChoice::Auto => body::ToolChoice::Mode("auto"),
        ToolChoice::Required => body::ToolChoice::Mode("required"),
        ToolChoice::None => body::ToolChoice::Mode("none"),
        ToolChoice::Tool(name) => body::ToolChoice::Function {
            kind: "function",
            name,
        },
    };
    Ok((Some(tools), Some(choice)))
}

fn text_config<'a>(
    request: &'a Request,
    verbosity: Option<&'a str>,
) -> Option<body::TextConfig<'a>> {
    let format = match &request.response_format {
        ResponseFormat::Text => None,
        ResponseFormat::Json { schema: None } => Some(body::TextFormat::JsonObject),
        ResponseFormat::Json {
            schema: Some(schema),
        } => Some(body::TextFormat::JsonSchema {
            name: "response",
            strict: true,
            schema: schema.as_value(),
        }),
    };
    (format.is_some() || verbosity.is_some()).then_some(body::TextConfig { format, verbosity })
}

fn cow_text(text: Cow<'_, str>) -> Text<'_> {
    match text {
        Cow::Borrowed(text) => Text::Borrowed(text),
        Cow::Owned(text) => Text::Owned(text),
    }
}

fn input_text<'a>(
    text: Text<'a>,
    metadata: &oven_sdk::PartMetadata,
) -> Result<Content<'a>, ModelError> {
    Ok(Content::InputText {
        text,
        prompt_cache_breakpoint: breakpoint(metadata)?,
    })
}

fn input_file(file: &FilePart) -> Result<Content<'_>, ModelError> {
    let Some(source) = FileRef::of(file) else {
        return Err(
            ModelError::unsupported("provider file references are unsupported")
                .with_stage(ErrorStage::RequestEncoding),
        );
    };
    let prompt_cache_breakpoint = breakpoint(&file.metadata)?;
    let filename = file.filename.as_deref().unwrap_or("document.pdf");
    Ok(if media::is_image(&file.media_type) {
        Content::InputImage {
            image_url: source,
            prompt_cache_breakpoint,
        }
    } else if let FileRef::Url(url) = source {
        Content::InputFile {
            filename,
            file_data: None,
            file_url: Some(url),
            prompt_cache_breakpoint,
        }
    } else {
        Content::InputFile {
            filename,
            file_data: Some(source),
            file_url: None,
            prompt_cache_breakpoint,
        }
    })
}

fn validate_prompt_cache_breakpoints(request: &Request) -> Result<(), ModelError> {
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
                                "Azure OpenAI prompt-cache breakpoint has no encodable Responses text block",
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
                                    "Azure OpenAI prompt-cache breakpoint has no encodable Responses text block",
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
                            "Azure OpenAI Responses prompt-cache breakpoints are valid only on system or user input content",
                        ));
                    }
                }
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    if tool_content_has_prompt_cache_breakpoint(&result.content)? {
                        return Err(ModelError::invalid_request(
                            "Azure OpenAI Responses prompt-cache breakpoints are not valid on function outputs",
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

fn function_output(result: &ToolResultPart) -> Result<Item<'_>, ModelError> {
    let output = match &result.content {
        ToolContent::Text(value) => Output::Text(Text::Borrowed(value)),
        ToolContent::Json(value) => Output::Text(Text::Json(value)),
        ToolContent::Mixed(values)
            if values
                .iter()
                .any(|value| matches!(value, ContentValue::File(_))) =>
        {
            Output::Parts(
                values
                    .iter()
                    .map(|value| match value {
                        ContentValue::Text(value) => Ok(Content::InputText {
                            text: Text::Borrowed(value),
                            prompt_cache_breakpoint: None,
                        }),
                        ContentValue::Json(value) => Ok(Content::InputText {
                            text: Text::Json(value),
                            prompt_cache_breakpoint: None,
                        }),
                        ContentValue::File(file) if tool_result_file_supported(file) => {
                            input_file(file)
                        }
                        ContentValue::File(_) => Err(ModelError::unsupported(
                            "only image and PDF files in tool results are deliverable via azure-openai-responses",
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        ToolContent::Mixed(values) => {
            Output::Text(Text::Owned(serde_json::to_string(values).unwrap_or_default()))
        }
        ToolContent::Denied { reason } => {
            Output::Text(Text::Borrowed(reason.as_deref().unwrap_or_default()))
        }
    };
    Ok(Item::Typed(TypedItem::FunctionCallOutput {
        call_id: &result.tool_call_id,
        output,
    }))
}

fn validate_tool_result_files(request: &Request) -> Result<(), ModelError> {
    for turn in &request.history {
        match turn {
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    reject_unsupported_tool_result_files(result)?;
                }
            }
            HistoryTurn::Assistant(turn) => {
                for part in &turn.message.content {
                    if let AssistantPart::ToolResult(result) = part {
                        reject_unsupported_tool_result_files(result)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn reject_unsupported_tool_result_files(result: &ToolResultPart) -> Result<(), ModelError> {
    if let ToolContent::Mixed(values) = &result.content
        && values.iter().any(
            |value| matches!(value, ContentValue::File(file) if !tool_result_file_supported(file)),
        )
    {
        return Err(ModelError::unsupported(
            "only image and PDF files in tool results are deliverable via azure-openai-responses",
        ));
    }
    Ok(())
}

/// `function_call_output` accepts `input_image` and `input_file` parts.
fn tool_result_file_supported(file: &FilePart) -> bool {
    media::is_image(&file.media_type) || file.media_type == "application/pdf"
}

fn normalized_assistant<'a>(
    parts: &'a [AssistantPart],
    output: &mut Vec<Item<'a>>,
    warnings: &mut Vec<String>,
) -> Result<(), ModelError> {
    if parts.iter().any(|part| matches!(part, AssistantPart::ToolCall(_)))
        && parts.iter().any(|part| matches!(part, AssistantPart::Custom(custom) if matches!(custom.kind.as_str(), "openai.responses.reasoning_continuation" | "azure.openai.responses.reasoning_continuation")))
    {
        return Err(ModelError::replay("Responses tool continuation requires valid native reasoning state"));
    }
    // Contiguous text runs stay in position relative to tool items so the
    // reconstructed input preserves the original turn's interleaving instead
    // of hoisting all text ahead of the tool calls.
    let mut text_run = Cow::Borrowed("");
    let refusal = parts
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Custom(part) if part.kind == "azure.openai.refusal" => {
                part.data.as_str()
            }
            _ => None,
        })
        .collect::<String>();
    for part in parts {
        match part {
            AssistantPart::Text(part) => push_text(&mut text_run, &part.text),
            AssistantPart::ToolCall(call) => {
                push_text_run(output, &mut text_run);
                output.push(Item::Typed(TypedItem::FunctionCall {
                    call_id: &call.id,
                    name: &call.name,
                    arguments: arguments(call),
                    id: call.provider_item_id.as_deref(),
                }));
            }
            AssistantPart::ToolResult(result) => {
                push_text_run(output, &mut text_run);
                output.push(function_output(result)?);
            }
            AssistantPart::Reasoning(_) => warnings.push(
                "Responses normalized fallback omitted reasoning without encrypted replay state"
                    .into(),
            ),
            _ => {}
        }
    }
    push_text_run(output, &mut text_run);
    if !refusal.is_empty() {
        output.push(Item::message(
            "assistant",
            vec![Content::Refusal { refusal }],
        ));
    }
    Ok(())
}

fn push_text_run<'a>(output: &mut Vec<Item<'a>>, text_run: &mut Cow<'a, str>) {
    if !text_run.is_empty() {
        output.push(Item::message(
            "assistant",
            vec![Content::OutputText {
                text: std::mem::take(text_run),
            }],
        ));
    }
}
