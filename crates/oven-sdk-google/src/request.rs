//! Request validation, encoding, and scoped private replay validation.

use std::collections::BTreeMap;

use oven_sdk::{
    AssistantPart, Capability, ContentValue, ErrorStage, FilePart, FileSource, HistoryTurn,
    InputPart, JsonSchema, JsonValue, LanguageModelDescriptor, ModelCapabilities, ModelError,
    NativeContextScope, NativeReplayArtifact, ProviderId, ReplayDecision, ReplayDisposition,
    ReplayOutcome, ReplayPolicy, Request, ResponseFormat, SystemPart, ToolCallPart, ToolChoice,
    ToolContent, ToolResultPart,
};
use serde::Deserialize as _;

use crate::{
    body::{
        self, Body, Content, Empty, FunctionCallingConfig, FunctionDeclaration, GenerationConfig,
        JsonMap, Part, Parts, ResponseFormatText, Tool, ToolConfig, ToolOutput,
    },
    model::{GoogleGenerateContentSettings, GoogleThinkingSettings},
    options::{GoogleProviderTool, GoogleRequestOptions, GoogleThinkingConfig, GoogleToolOptions},
};

const MIB: usize = 1024 * 1024;
pub(crate) const GOOGLE_REQUEST_MAX_BYTES: usize = 20 * MIB;
const GOOGLE_MAX_IMAGES: usize = 3_600;
const GOOGLE_MAX_VIDEOS: usize = 10;
const SKIP_THOUGHT_SIGNATURE_VALIDATOR: &str = "skip_thought_signature_validator";

pub(crate) struct Encoded {
    /// Serialized JSON request body.
    pub(crate) body: Vec<u8>,
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
    settings: &GoogleGenerateContentSettings,
    _native_context_scope: &NativeContextScope,
) -> Result<Encoded, ModelError> {
    let options = &parsed.request;
    let thinking_config = resolved_thinking_config(request, options, &settings.thinking)?;
    let replay_policy = descriptor.capabilities.replay.policy;
    let provider_tools_supported = descriptor
        .capabilities
        .features
        .contains(Capability::PROVIDER_TOOLS);
    let target_model = settings
        .model_resource
        .strip_prefix("models/")
        .expect("validated model resource");
    let mut replay = ReplayOutcome::default();
    let mut warnings = Vec::new();
    warnings.extend(oven_sdk::DeferredTools::of(request).upfront_warning("google-gemini"));

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
                    match validated_replay_parts(artifact.payload(), &turn.message.content) {
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
    let mut tool_names = BTreeMap::<&str, (&str, Option<&str>)>::new();
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
                        system_parts.push(Part::Text(&text.text));
                    }
                }
            }
            HistoryTurn::User(message) => {
                previous_tool_content = false;
                let parts = message
                    .content
                    .iter()
                    .map(|part| input_part(part, &descriptor.identity.provider_id))
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
                                target_model,
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
                let mut wire_calls = parts.iter().filter_map(Part::function_call_id);
                for part in &turn.message.content {
                    if let AssistantPart::ToolCall(call) = part {
                        let wire_id = wire_calls.next().flatten();
                        tool_names.insert(call.id.as_str(), (call.name.as_str(), wire_id));
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
        thinking_config,
        response_format: match &request.response_format {
            ResponseFormat::Json { schema } => Some(body::ResponseFormat {
                text: ResponseFormatText {
                    mime_type: "APPLICATION_JSON",
                    schema: schema.as_ref().map(JsonSchema::as_value),
                },
            }),
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
                    parameters: tool.input_schema.as_value(),
                })
                .collect(),
        ));
    }
    let tool_config = (!request.tools.is_empty()).then(|| {
        let strict = parsed.strict_tools.iter().any(|strict| *strict);
        let mixed = !options.provider_tools.is_empty();
        let (mode, names) = match &request.tool_choice {
            ToolChoice::Auto if mixed => ("VALIDATED", None),
            ToolChoice::Auto => (if strict { "VALIDATED" } else { "AUTO" }, None),
            ToolChoice::Required if mixed => ("ANY", None),
            ToolChoice::Required => (if strict { "VALIDATED" } else { "ANY" }, None),
            ToolChoice::None => ("NONE", None),
            ToolChoice::Tool(name) if mixed => ("ANY", Some([name.as_str()])),
            ToolChoice::Tool(name) => (
                if strict { "VALIDATED" } else { "ANY" },
                Some([name.as_str()]),
            ),
        };
        ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode,
                allowed_function_names: names,
            },
            include_server_side_tool_invocations: mixed,
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
        service_tier: options.service_tier.as_deref(),
        cached_content: options.cached_content.as_deref(),
        safety_settings: &options.safety_settings,
    }
    .to_bytes()
    .map_err(|_| {
        ModelError::invalid_request("could not serialize Google Gemini request")
            .with_stage(ErrorStage::RequestEncoding)
    })?;
    Ok(Encoded {
        body,
        replay,
        warnings,
    })
}

pub(crate) fn options(request: &Request) -> Result<ParsedOptions, ModelError> {
    let request_options = request
        .provider_options
        .get("google")
        .map(|value| {
            GoogleRequestOptions::deserialize(value)
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
                .and_then(|value| GoogleToolOptions::deserialize(value).ok())
                .is_some_and(|options| options.strict)
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

fn provider_tools(tools: &[GoogleProviderTool]) -> Vec<Tool<'_>> {
    let mut output = Vec::with_capacity(tools.len() + 1);
    output.extend(tools.iter().map(|tool| match tool {
        GoogleProviderTool::GoogleSearch => Tool::GoogleSearch(Empty {}),
        GoogleProviderTool::UrlContext => Tool::UrlContext(Empty {}),
        GoogleProviderTool::CodeExecution => Tool::CodeExecution(Empty {}),
        GoogleProviderTool::FileSearch { stores } => Tool::FileSearch { stores },
        GoogleProviderTool::GoogleMaps => Tool::GoogleMaps(Empty {}),
    }));
    output
}

fn input_part<'a>(
    part: &'a InputPart,
    provider_id: &ProviderId,
) -> Result<Option<Part<'a>>, ModelError> {
    match part {
        InputPart::Text(text) => Ok((!text.text.is_empty()).then_some(Part::Text(&text.text))),
        InputPart::File(file) => file_part(file, provider_id).map(Some),
        InputPart::Custom(_) => Err(ModelError::unsupported(
            "Google custom input parts are not supported",
        )),
    }
}

fn file_part<'a>(file: &'a FilePart, provider_id: &ProviderId) -> Result<Part<'a>, ModelError> {
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
        FileSource::ProviderReference { provider, id } if provider == provider_id => {
            Ok(Part::FileData {
                mime_type,
                file_uri: id,
            })
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
) -> (Vec<Part<'_>>, bool, Vec<&'static str>) {
    let mut output = Vec::with_capacity(parts.len());
    let mut saw_function_call = false;
    let mut injected_sentinel = false;
    let mut omitted_server_tools = Vec::new();
    for part in parts {
        match part {
            AssistantPart::Text(text) if !text.text.is_empty() => {
                output.push(Part::Text(&text.text));
            }
            AssistantPart::ToolCall(call) => {
                let thought_signature = (gemini3 && current_turn && !saw_function_call)
                    .then_some(SKIP_THOUGHT_SIGNATURE_VALIDATOR);
                injected_sentinel |= thought_signature.is_some();
                saw_function_call = true;
                output.push(Part::FunctionCall {
                    id: &call.id,
                    name: &call.name,
                    args: &call.input,
                    thought_signature,
                });
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

fn function_response<'a>(
    result: &'a ToolResultPart,
    names: &BTreeMap<&str, (&'a str, Option<&'a str>)>,
) -> Result<Part<'a>, ModelError> {
    let (name, wire_id) = names.get(result.tool_call_id.as_str()).ok_or_else(|| {
        ModelError::invalid_request("Google tool result could not resolve its function name")
    })?;
    Ok(Part::FunctionResponse {
        id: *wire_id,
        name,
        is_error: result.is_error,
        output: tool_content(&result.content)?,
    })
}

fn tool_content(content: &ToolContent) -> Result<ToolOutput<'_>, ModelError> {
    match content {
        ToolContent::Text(value) => Ok(ToolOutput::Str(value)),
        ToolContent::LoadTools { text: value, .. } => Ok(ToolOutput::Str(value)),
        ToolContent::Json(value) => Ok(ToolOutput::Json(value)),
        ToolContent::Denied { reason } => Ok(ToolOutput::Str(
            reason.as_deref().unwrap_or("Tool call execution denied."),
        )),
        ToolContent::Mixed(values) => {
            if values
                .iter()
                .any(|value| matches!(value, ContentValue::File(_)))
            {
                return Err(ModelError::unsupported(
                    "multimodal Gemini function results are not enabled",
                ));
            }
            Ok(ToolOutput::Mixed(values))
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

fn validated_replay_parts<'a>(
    payload: &'a JsonValue,
    normalized: &[AssistantPart],
) -> Result<&'a [JsonValue], &'static str> {
    if payload.get("role").and_then(JsonValue::as_str) != Some("model") {
        return Err("Google replay content role is invalid");
    }
    let parts = payload
        .get("parts")
        .and_then(JsonValue::as_array)
        .ok_or("Google replay parts are invalid")?;
    if parts.iter().any(|part| !valid_native_part(part)) {
        return Err("Google replay contains an unsupported native part");
    }
    if !semantics_match(parts, normalized) {
        return Err("Google replay payload did not match normalized content");
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
    target: &str,
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
    let encrypted_allowed = reasoning_supported
        && artifact
            .source_wire_model_id()
            .is_some_and(|source| source.as_str() == target);
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
    }) && ["toolCall", "toolResponse"].into_iter().all(|field| {
        object
            .get(field)
            .and_then(JsonValue::as_object)
            .and_then(|value| value.get("id"))
            .is_none_or(JsonValue::is_string)
    })
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

/// The replay-relevant meaning of one native part, borrowed where possible.
enum NativeSemantic<'a> {
    Text {
        reasoning: bool,
        text: &'a str,
    },
    ToolCall {
        id: &'a str,
        name: &'a str,
        /// `None` compares as an empty object.
        args: Option<&'a JsonValue>,
    },
    /// Server-tool context; rare, so it keeps the owned comparison shape.
    Custom(JsonValue),
}

impl<'a> NativeSemantic<'a> {
    fn from_part(part: &'a JsonValue) -> Option<Self> {
        if let Some(text) = part.get("text").and_then(JsonValue::as_str) {
            return Some(Self::Text {
                reasoning: part.get("thought").and_then(JsonValue::as_bool) == Some(true),
                text,
            });
        }
        if let Some(call) = part.get("functionCall") {
            return Some(Self::ToolCall {
                id: call.get("id").and_then(JsonValue::as_str).unwrap_or(""),
                name: call.get("name").and_then(JsonValue::as_str).unwrap_or(""),
                args: call.get("args"),
            });
        }
        native_server_tool_semantic(part).map(Self::Custom)
    }

    /// Whether this native part carries the same meaning as one normalized part.
    fn matches(&self, expected: &NormalizedSemantic<'_>) -> bool {
        match (self, expected) {
            (
                Self::Text { reasoning, text },
                NormalizedSemantic::Text {
                    reasoning: expected_reasoning,
                    text: expected_text,
                },
            ) => reasoning == expected_reasoning && text == expected_text,
            (Self::ToolCall { id, name, args }, NormalizedSemantic::ToolCall(call)) => {
                *id == call.id
                    && *name == call.name
                    && match args {
                        Some(args) => **args == call.input,
                        None => call.input.as_object().is_some_and(JsonMap::is_empty),
                    }
            }
            (Self::Custom(native), NormalizedSemantic::Custom(expected)) => native == expected,
            _ => false,
        }
    }
}

/// The replay-relevant meaning of one normalized assistant part.
enum NormalizedSemantic<'a> {
    Text { reasoning: bool, text: &'a str },
    ToolCall(&'a ToolCallPart),
    Custom(JsonValue),
}

impl<'a> NormalizedSemantic<'a> {
    fn from_part(part: &'a AssistantPart) -> Option<Self> {
        match part {
            AssistantPart::Text(text) => Some(Self::Text {
                reasoning: false,
                text: &text.text,
            }),
            AssistantPart::Reasoning(reasoning) => Some(Self::Text {
                reasoning: true,
                text: &reasoning.text,
            }),
            AssistantPart::ToolCall(call) => Some(Self::ToolCall(call)),
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
                Some(Self::Custom(serde_json::json!({
                    "type":"custom","kind":custom.kind,"data":data
                })))
            }
            _ => None,
        }
    }
}

/// Compares native and normalized semantics pairwise, without materializing either side.
fn semantics_match(parts: &[JsonValue], normalized: &[AssistantPart]) -> bool {
    let mut native = parts.iter().filter_map(NativeSemantic::from_part);
    for expected in normalized.iter().filter_map(NormalizedSemantic::from_part) {
        if !native.next().is_some_and(|part| part.matches(&expected)) {
            return false;
        }
    }
    native.next().is_none()
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

    fn artifact(source_model: Option<&str>) -> NativeReplayArtifact {
        let artifact = NativeReplayArtifact::new(
            oven_sdk::AdapterId::new("oven.google.generate-content"),
            NativeContextScope::new(
                ProviderId::new("google"),
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
                            "target",
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
    fn in_place_semantics_match_text_reasoning_and_calls_exactly() {
        let normalized = vec![
            AssistantPart::Reasoning(oven_sdk::ReasoningPart::new("think")),
            AssistantPart::Text(oven_sdk::TextPart::new("")),
            AssistantPart::ToolCall(ToolCallPart::new("c1", "lookup", serde_json::json!({}))),
            AssistantPart::ToolCall(ToolCallPart::new(
                "c2",
                "lookup",
                serde_json::json!({"q":1}),
            )),
            AssistantPart::Source(oven_sdk::SourcePart::new()),
        ];
        let parts = serde_json::json!([
            {"text":"think","thought":true,"thoughtSignature":"sig"},
            {"text":""},
            {"functionCall":{"id":"c1","name":"lookup"}},
            {"functionCall":{"id":"c2","name":"lookup","args":{"q":1}}},
            {"inlineData":{"mimeType":"image/png","data":""}}
        ]);
        assert!(semantics_match(parts.as_array().unwrap(), &normalized));
        let mutations: [fn(&mut JsonValue); 7] = [
            |parts| parts[0]["thought"] = JsonValue::Bool(false),
            |parts| parts[1]["text"] = JsonValue::String(" ".into()),
            |parts| parts[2]["functionCall"]["args"] = serde_json::json!({"x":1}),
            |parts| parts[3]["functionCall"]["id"] = JsonValue::String("other".into()),
            |parts| parts[3]["functionCall"]["name"] = JsonValue::String("other".into()),
            |parts| {
                parts.as_array_mut().unwrap().remove(1);
            },
            |parts| {
                parts
                    .as_array_mut()
                    .unwrap()
                    .push(serde_json::json!({"text":"x"}))
            },
        ];
        for mutate in mutations {
            let mut changed = parts.clone();
            mutate(&mut changed);
            assert!(!semantics_match(changed.as_array().unwrap(), &normalized));
        }
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
