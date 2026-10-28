//! OpenAI Responses request validation and translation.

use std::borrow::Cow;

use oven_sdk::{
    AssistantPart, CompactionRequest, ContentValue, DeferredTools, DeferredToolsCapability,
    ErrorStage, FilePart, FileSource, HistoryTurn, InputPart, JsonValue, LanguageModelDescriptor,
    ModelCapabilities, ModelError, NativeContextScope, ReplayDecision, ReplayDisposition,
    ReplayOutcome, ReplayPolicy, Request, ResponseFormat, ToolCallPart, ToolChoice, ToolContent,
    ToolDefinition, ToolResultPart,
};

/// Metadata key of the namespace a streamed call to a loaded tool came back in.
pub(crate) const NAMESPACE_METADATA_KEY: &str = "openai.namespace";

/// Whether `request` loads deferred tools natively: the model declares it and
/// the request sends tools at all.
fn native_deferral(request: &Request, descriptor: &LanguageModelDescriptor) -> bool {
    descriptor.capabilities.deferred_tools == DeferredToolsCapability::Native
        && !request.tools.is_empty()
        && !matches!(request.tool_choice, ToolChoice::None)
}

use crate::{
    configuration::OpenAiResponsesWireProfile,
    options::{
        OpenAiResponsesCompactionOptions, OpenAiResponsesOptions, PromptCacheSurface,
        prompt_cache_breakpoint, validate_prompt_cache_controls, validate_prompt_cache_key,
    },
    responses::{
        body::{self, Content, Item, Output, TypedItem},
        compaction, replay,
    },
    wire::encode::{
        FileRef, Text, arguments, breakpoint, encoding_error, join, push_text, to_body,
    },
};

pub(crate) struct Encoded {
    /// Serialized JSON request body.
    pub(crate) body: Vec<u8>,
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
    validate_prompt_cache_controls(
        PromptCacheSurface::Responses,
        options.prompt_cache_retention.is_some(),
        options.prompt_cache_options.as_ref(),
    )?;
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
                if turn
                    .message
                    .content
                    .iter()
                    .any(|part| matches!(part, AssistantPart::File(_)))
                {
                    return Err(ModelError::unsupported(
                        "Responses assistant-history media is not supported",
                    )
                    .with_stage(ErrorStage::RequestEncoding));
                }
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
    profile: OpenAiResponsesWireProfile,
) -> Result<Encoded, ModelError> {
    let (input, replay_outcome, mut warnings) = encode_input(request, descriptor, scope, policy)?;
    let mut size_hint = input.iter().map(Item::size_hint).sum::<usize>();
    let mut include = options
        .include
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if !include.contains(&"reasoning.encrypted_content") {
        include.push("reasoning.encrypted_content");
    }
    let reasoning = (request.inference.reasoning_effort.is_some()
        || options.reasoning_summary.is_some()
        || options.reasoning_mode.is_some())
    .then_some(body::Reasoning {
        effort: request.inference.reasoning_effort.as_deref(),
        summary: options.reasoning_summary.as_deref(),
        mode: options.reasoning_mode.as_deref(),
    });
    let (tools, tool_choice) = tools(request, native_deferral(request, descriptor));
    size_hint += tools.as_ref().map_or(0, |tools| {
        tools.iter().map(body::FunctionTool::size_hint).sum()
    });
    let mut body = body::Body {
        model: descriptor.identity.model_id.as_str(),
        input,
        stream: true,
        store: false,
        include,
        max_output_tokens: request.inference.max_output_tokens,
        temperature: request.inference.temperature,
        top_p: request.inference.top_p,
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
    if profile == OpenAiResponsesWireProfile::ChatGptTokenSharing {
        // The plan manages output length; every request carries a limit, so
        // dropping it is not worth a warning.
        body.max_output_tokens = None;
        let mut dropped = Vec::new();
        if body.temperature.take().is_some() {
            dropped.push("temperature");
        }
        if body.top_p.take().is_some() {
            dropped.push("top_p");
        }
        if body.user.take().is_some() {
            dropped.push("user");
        }
        if body.service_tier.take().is_some() {
            dropped.push("service_tier");
        }
        if body.truncation.take().is_some() {
            dropped.push("truncation");
        }
        if body.prompt_cache_retention.take().is_some() {
            dropped.push("prompt_cache_retention");
        }
        if body.prompt_cache_options.take().is_some() {
            dropped.push("prompt_cache_options");
        }
        warnings.extend(
            dropped.into_iter().map(|field| {
                format!("`{field}` dropped: ChatGPT token sharing does not accept it")
            }),
        );
    }
    Ok(Encoded {
        body: to_body(
            &body,
            size_hint,
            encoding_error("could not encode OpenAI Responses request"),
        )?,
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
    profile: OpenAiResponsesWireProfile,
) -> Result<Encoded, ModelError> {
    let (input, replay, warnings) = encode_input(&request.request, descriptor, scope, policy)?;
    let size_hint = input.iter().map(Item::size_hint).sum::<usize>();
    let sharing = profile == OpenAiResponsesWireProfile::ChatGptTokenSharing;
    let body = body::CompactionBody {
        model: descriptor.identity.model_id.as_str(),
        input,
        instructions: options.instructions.as_deref(),
        prompt_cache_key: options.prompt_cache_key.as_deref(),
        prompt_cache_options: options.prompt_cache_options.as_ref().filter(|_| !sharing),
        prompt_cache_retention: options
            .prompt_cache_retention
            .as_deref()
            .filter(|_| !sharing),
        service_tier: options.service_tier.as_deref().filter(|_| !sharing),
    };
    Ok(Encoded {
        body: to_body(&body, size_hint, || {
            ModelError::invalid_request("could not encode OpenAI compaction request")
                .with_stage(ErrorStage::NativeContextEncode)
        })?,
        replay,
        warnings,
    })
}

type EncodedInput<'a> = (Vec<Item<'a>>, ReplayOutcome, Vec<String>);

fn encode_input<'a>(
    request: &'a Request,
    descriptor: &LanguageModelDescriptor,
    _scope: &NativeContextScope,
    policy: ReplayPolicy,
) -> Result<EncodedInput<'a>, ModelError> {
    let mut input = request
        .native_context
        .as_ref()
        .map(compaction::native_input)
        .transpose()?
        .unwrap_or_default()
        .iter()
        .map(Item::Native)
        .collect::<Vec<_>>();
    let mut replay_outcome = ReplayOutcome::default();
    let mut warnings = Vec::new();
    let deferred = DeferredTools::of(request);
    let native_deferral = native_deferral(request, descriptor);
    if native_deferral {
        warnings.extend(deferred.warnings().iter().cloned());
    } else {
        warnings.extend(deferred.upfront_warning("openai-responses"));
    }
    // A call keeps its namespace only while its tool is loaded natively.
    let loaded = |name: &str| native_deferral && deferred.is_loaded(name);
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
                if native_deferral {
                    let tools = message
                        .results
                        .iter()
                        .flat_map(|result| deferred.newly_loaded(&result.tool_call_id))
                        .filter_map(|name| request.tools.iter().find(|tool| tool.name == *name))
                        .map(function_tool)
                        .collect::<Vec<_>>();
                    if !tools.is_empty() {
                        input.push(Item::Typed(TypedItem::AdditionalTools {
                            role: "developer",
                            tools,
                        }));
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
                    if let Some(items) = replay::decode(artifact, &turn.message.content) {
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
                    input.extend(items.into_iter().map(|item| native_item(item, &loaded)));
                } else {
                    if policy != ReplayPolicy::Never {
                        replay_outcome.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::ReconstructedNormalized,
                        });
                    }
                    normalized_assistant(
                        &turn.message.content,
                        &mut input,
                        &mut warnings,
                        &loaded,
                    )?;
                }
            }
        }
    }
    Ok((input, replay_outcome, warnings))
}

fn discard_reason(artifact: &oven_sdk::NativeReplayArtifact) -> String {
    match artifact.payload().get("format").and_then(JsonValue::as_str) {
        Some("oven.openai.responses.output.v1" | "oven.azure.openai.responses.output.v4") => {
            "Responses replay payload does not match normalized content".to_owned()
        }
        _ => "replay artifact format is not supported by the Responses target".to_owned(),
    }
}

/// A replayed native item, without the namespace of a call whose tool is not
/// loaded in this request: an eager tool lives in the default namespace.
fn native_item<'a>(item: &'a JsonValue, loaded: &dyn Fn(&str) -> bool) -> Item<'a> {
    let strip = item.get("type").and_then(JsonValue::as_str) == Some("function_call")
        && item.get("namespace").is_some()
        && !item
            .get("name")
            .and_then(JsonValue::as_str)
            .is_some_and(loaded);
    if strip {
        let mut owned = item.clone();
        if let Some(object) = owned.as_object_mut() {
            object.remove("namespace");
        }
        Item::NativeOwned(owned)
    } else {
        Item::Native(item)
    }
}

/// The namespace a call is sent with: the one it came back in, or, for a
/// call recorded without one, the tool's own name, which is where an
/// `additional_tools` item puts a function.
fn call_namespace<'a>(call: &'a ToolCallPart, loaded: &dyn Fn(&str) -> bool) -> Option<&'a str> {
    if !loaded(&call.name) {
        return None;
    }
    Some(
        call.metadata
            .as_ref()
            .and_then(|metadata| metadata.get(NAMESPACE_METADATA_KEY))
            .and_then(JsonValue::as_str)
            .unwrap_or(&call.name),
    )
}

fn function_tool(tool: &ToolDefinition) -> body::FunctionTool<'_> {
    body::FunctionTool {
        kind: "function",
        name: &tool.name,
        description: &tool.description,
        parameters: tool.input_schema.as_value(),
        strict: tool
            .provider_options
            .get("openai")
            .and_then(|value| value.get("strict"))
            .and_then(JsonValue::as_bool)
            .unwrap_or(false),
    }
}

fn tools(
    request: &Request,
    native_deferral: bool,
) -> (
    Option<Vec<body::FunctionTool<'_>>>,
    Option<body::ToolChoice<'_>>,
) {
    if request.tools.is_empty() || matches!(request.tool_choice, ToolChoice::None) {
        return (None, None);
    }
    // Natively deferred tools are declared by the additional_tools item that
    // loads them, never up front.
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
            name,
        },
    };
    (Some(tools), Some(choice))
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
    Ok(compaction::native_input(window)?
        .iter()
        .map(count_wire_prompt_cache_breakpoints)
        .sum())
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
    Ok(if file.media_type.starts_with("image/") {
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

fn function_output(result: &ToolResultPart) -> Result<Item<'_>, ModelError> {
    let output = match &result.content {
        ToolContent::Text(value) => Output::Text(Text::Borrowed(value)),
        ToolContent::LoadTools { text: value, .. } => Output::Text(Text::Borrowed(value)),
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
                            "only image and PDF files in tool results are deliverable via openai-responses",
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

fn validate_tool_result_files(result: &ToolResultPart) -> Result<(), ModelError> {
    if let ToolContent::Mixed(values) = &result.content
        && values.iter().any(
            |value| matches!(value, ContentValue::File(file) if !tool_result_file_supported(file)),
        )
    {
        return Err(ModelError::unsupported(
            "only image and PDF files in tool results are deliverable via openai-responses",
        ));
    }
    Ok(())
}

/// `function_call_output` accepts `input_image` and `input_file` parts.
fn tool_result_file_supported(file: &FilePart) -> bool {
    file.media_type.starts_with("image/") || file.media_type == "application/pdf"
}

fn normalized_assistant<'a>(
    parts: &'a [AssistantPart],
    output: &mut Vec<Item<'a>>,
    warnings: &mut Vec<String>,
    loaded: &dyn Fn(&str) -> bool,
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
                    namespace: call_namespace(call, loaded),
                }));
            }
            AssistantPart::ToolResult(result) => {
                push_text_run(output, &mut text_run);
                output.push(function_output(result)?);
            }
            AssistantPart::Custom(part)
                if part.kind == "openai.responses.reasoning_continuation" =>
            {
                warnings.push(
                    "encrypted reasoning is unavailable for normalized reconstruction".into(),
                );
            }
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
    push_text_run(output, &mut text_run);
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
