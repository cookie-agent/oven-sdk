//! Request validation and Anthropic Messages encoding.

use std::{borrow::Cow, collections::BTreeSet};

use bytes::Bytes;
use oven_sdk::{
    AssistantPart, Capability, ContentValue, ErrorStage, FileSource, HistoryTurn, InputPart,
    JsonValue, LanguageModelDescriptor, ModelCapabilities, ModelError, NativeContextScope,
    ReplayDecision, ReplayDisposition, ReplayOutcome, ReplayPolicy, Request, ResponseFormat,
    ToolChoice, ToolContent, ToolResultPart,
};

use crate::{
    body::{
        Block, CacheControl, Message, MessagesBody, Metadata, NativeBlock, OutputConfig,
        OutputFormat, Source, TextValue, Thinking, Tool, ToolChoice as WireToolChoice, ToolInput,
        ToolResultContent, TypedBlock,
    },
    config::{AnthropicProtocolSettings, AnthropicThinkingSupport, MiniMaxProtocolSettings},
    model::ProtocolSettings,
    options::{
        AnthropicAwsRequestOptions, AnthropicCacheControl, AnthropicCacheTtl,
        AnthropicRequestOptions, AnthropicThinking, AnthropicToolOptions, MiniMaxMediaOptions,
        MiniMaxRequestOptions, MiniMaxThinking,
    },
    replay,
    wire::Protocol,
};

const MIB: usize = 1024 * 1024;
pub(crate) const ANTHROPIC_REQUEST_MAX_BYTES: usize = 32 * MIB;
pub(crate) const MINIMAX_REQUEST_MAX_BYTES: usize = 64 * MIB;
const ANTHROPIC_IMAGE_MAX_BASE64_BYTES: usize = 10 * MIB;
const MINIMAX_IMAGE_MAX_BYTES: usize = 10 * MIB;
const MINIMAX_VIDEO_MAX_BYTES: usize = 50 * MIB;

fn anthropic_protocol(
    settings: &ProtocolSettings,
) -> Result<&AnthropicProtocolSettings, ModelError> {
    match settings {
        ProtocolSettings::Anthropic(settings) => Ok(settings),
        ProtocolSettings::MiniMax(_) => Err(ModelError::invalid_request(
            "Anthropic model has MiniMax protocol settings",
        )),
    }
}

fn minimax_protocol(settings: &ProtocolSettings) -> Result<&MiniMaxProtocolSettings, ModelError> {
    match settings {
        ProtocolSettings::MiniMax(settings) => Ok(settings),
        ProtocolSettings::Anthropic(_) => Err(ModelError::invalid_request(
            "MiniMax model has Anthropic protocol settings",
        )),
    }
}

pub(crate) fn validate_request(
    request: &Request,
    parsed: &ParsedOptions,
    capabilities: &ModelCapabilities,
    protocol: Protocol,
    protocol_settings: &ProtocolSettings,
    compatible: bool,
) -> Result<(), ModelError> {
    request.validate_for(capabilities)?;
    if protocol.is_first_party()
        && matches!(
            request.response_format,
            ResponseFormat::Json { schema: None }
        )
    {
        return Err(ModelError::invalid_request(
            "Anthropic structured output requires a JSON schema",
        ));
    }
    if protocol.is_first_party()
        && let ResponseFormat::Json {
            schema: Some(schema),
        } = &request.response_format
    {
        validate_schema(schema.as_value())?;
        if matches!(request.history.last(), Some(HistoryTurn::Assistant(_))) {
            return Err(ModelError::invalid_request(
                "Anthropic JSON output cannot use an assistant prefill",
            ));
        }
    }
    validate_inference(
        request,
        capabilities,
        protocol,
        protocol_settings,
        &parsed.anthropic,
        &parsed.minimax,
    )?;
    if protocol.is_first_party()
        && matches!(request.history.last(), Some(HistoryTurn::Assistant(_)))
        && !anthropic_protocol(protocol_settings)?.assistant_prefill
    {
        return Err(ModelError::unsupported(
            "this Anthropic configuration does not support assistant message prefill",
        ));
    }
    if protocol == Protocol::MiniMax
        && matches!(
            request.tool_choice,
            ToolChoice::Required | ToolChoice::Tool(_)
        )
    {
        return Err(ModelError::unsupported(
            "MiniMax Messages supports only automatic or disabled tool choice",
        ));
    }
    let cache_plan = &parsed.cache_plan;
    if !cache_plan.markers.is_empty() && !capabilities.features.contains(Capability::PROMPT_CACHING)
    {
        return Err(ModelError::unsupported(
            "prompt caching is not supported by this Anthropic configuration",
        ));
    }
    validate_cache_plan(cache_plan)?;
    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::User(message) => {
                for (part_index, part) in message.content.iter().enumerate() {
                    if let InputPart::File(file) = part {
                        validate_file(
                            file,
                            protocol,
                            parsed.minimax_media[history_index][part_index].as_ref(),
                            compatible,
                        )?;
                    }
                }
            }
            HistoryTurn::Assistant(message)
                if message
                    .message
                    .content
                    .iter()
                    .any(|part| matches!(part, AssistantPart::File(_))) =>
            {
                return Err(ModelError::unsupported(
                    "Messages media input is supported only in user turns",
                )
                .with_stage(ErrorStage::RequestEncoding));
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    if let ToolContent::Mixed(values) = &result.content {
                        for value in values {
                            if let ContentValue::File(file) = value {
                                if protocol.is_first_party() {
                                    validate_file(file, protocol, None, false)?;
                                } else {
                                    return Err(ModelError::unsupported(
                                        "files in tool results are not deliverable via minimax-messages",
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_inference(
    request: &Request,
    capabilities: &ModelCapabilities,
    protocol: Protocol,
    protocol_settings: &ProtocolSettings,
    anthropic: &AnthropicRequestOptions,
    minimax: &MiniMaxRequestOptions,
) -> Result<(), ModelError> {
    let requested = request
        .inference
        .max_output_tokens
        .or(capabilities.limits.output)
        .unwrap_or(4096);
    if protocol.is_first_party() && requested == 0 {
        return Err(ModelError::invalid_request(
            "Anthropic max_output_tokens must be at least 1",
        ));
    }
    if let Some(maximum) = capabilities.limits.output
        && requested > maximum
    {
        return Err(ModelError::invalid_request(
            "max_output_tokens exceeds this model's output limit",
        ));
    }
    if protocol == Protocol::MiniMax {
        let settings = minimax_protocol(protocol_settings)?;
        if minimax.thinking.is_some() && !settings.thinking {
            return Err(ModelError::unsupported(
                "thinking is not supported by this MiniMax configuration",
            ));
        }
        if requested == 0 {
            return Err(ModelError::invalid_request(
                "MiniMax max_output_tokens must be at least 1",
            ));
        }
        if !settings.thinking_disable_allowed
            && matches!(minimax.thinking.as_ref(), Some(MiniMaxThinking::Disabled))
        {
            return Err(ModelError::unsupported(
                "thinking cannot be disabled by this MiniMax configuration",
            ));
        }
        return Ok(());
    }

    if let Some(temperature) = request.inference.temperature
        && temperature > 1.0
    {
        return Err(ModelError::invalid_request(
            "Anthropic temperature must be between 0 and 1",
        ));
    }
    let settings = anthropic_protocol(protocol_settings)?;
    if anthropic.effort.is_some() && !settings.effort {
        return Err(ModelError::unsupported(
            "effort is not supported by this Anthropic configuration",
        ));
    }
    let active = matches!(
        anthropic.thinking.as_ref(),
        Some(AnthropicThinking::Enabled { .. } | AnthropicThinking::Adaptive { .. })
    ) || (anthropic.thinking.is_none() && settings.thinking_default_active);
    if settings.reject_non_default_sampling {
        if request
            .inference
            .temperature
            .is_some_and(|temperature| temperature != 1.0)
            || request.inference.top_p.is_some_and(|top_p| top_p != 1.0)
        {
            return Err(ModelError::invalid_request(
                "this Anthropic configuration rejects non-default sampling parameters",
            ));
        }
    } else if active {
        if request.inference.temperature.is_some() {
            return Err(ModelError::invalid_request(
                "active Anthropic thinking cannot be combined with temperature",
            ));
        }
        if let Some(top_p) = request.inference.top_p
            && !(0.95..=1.0).contains(&top_p)
        {
            return Err(ModelError::invalid_request(
                "active Anthropic thinking requires top_p between 0.95 and 1",
            ));
        }
    } else if request.inference.temperature.is_some() && request.inference.top_p.is_some() {
        return Err(ModelError::invalid_request(
            "Anthropic temperature and top_p cannot be set together",
        ));
    }
    if active && matches!(request.history.last(), Some(HistoryTurn::Assistant(_))) {
        return Err(ModelError::invalid_request(
            "active Anthropic thinking cannot be combined with assistant prefill",
        ));
    }
    match &anthropic.thinking {
        None => {}
        Some(AnthropicThinking::Disabled) => {
            if settings.thinking == AnthropicThinkingSupport::None {
                return Err(ModelError::unsupported(
                    "thinking controls are not supported by this Anthropic configuration",
                ));
            }
            if !settings.thinking_disable_allowed {
                return Err(ModelError::unsupported(
                    "thinking cannot be disabled by this Anthropic configuration",
                ));
            }
            if anthropic
                .effort
                .as_ref()
                .is_some_and(|effort| settings.thinking_disable_forbidden_efforts.contains(effort))
            {
                return Err(ModelError::invalid_request(
                    "disabled thinking cannot use the configured effort label",
                ));
            }
        }
        Some(AnthropicThinking::Adaptive { .. }) => {
            if !matches!(
                settings.thinking,
                AnthropicThinkingSupport::Adaptive | AnthropicThinkingSupport::Both
            ) {
                return Err(ModelError::unsupported(
                    "adaptive thinking is not supported by this Anthropic configuration",
                ));
            }
        }
        Some(AnthropicThinking::Enabled {
            budget_tokens,
            display: _,
        }) => {
            if !matches!(
                settings.thinking,
                AnthropicThinkingSupport::Extended | AnthropicThinkingSupport::Both
            ) {
                return Err(ModelError::unsupported(
                    "manual extended thinking is not supported by this Anthropic configuration",
                ));
            }
            if *budget_tokens < 1024 {
                return Err(ModelError::invalid_request(
                    "Anthropic thinking budget_tokens must be at least 1024",
                ));
            }
            if matches!(
                request.tool_choice,
                ToolChoice::Required | ToolChoice::Tool(_)
            ) {
                return Err(ModelError::invalid_request(
                    "manual Anthropic thinking cannot force a tool choice",
                ));
            }
            let total = if request.inference.max_output_tokens.is_some() {
                requested.checked_add(*budget_tokens).ok_or_else(|| {
                    ModelError::invalid_request("Anthropic max_tokens arithmetic overflowed")
                })?
            } else {
                requested
            };
            if *budget_tokens >= total {
                return Err(ModelError::invalid_request(
                    "Anthropic thinking budget_tokens must be less than max_tokens",
                ));
            }
            if let Some(maximum) = capabilities.limits.output
                && total > maximum
            {
                return Err(ModelError::invalid_request(
                    "output tokens plus thinking budget exceed this model's output limit",
                ));
            }
        }
    }
    Ok(())
}

fn validate_file(
    file: &oven_sdk::FilePart,
    protocol: Protocol,
    minimax_options: Option<&MiniMaxMediaOptions>,
    compatible: bool,
) -> Result<(), ModelError> {
    if protocol.is_first_party() && !(compatible && file.media_type.starts_with("video/")) {
        let image = matches!(
            file.media_type.as_str(),
            "image/jpeg" | "image/png" | "image/gif" | "image/webp"
        );
        let document = matches!(file.media_type.as_str(), "application/pdf" | "text/plain");
        if !image && !document {
            return Err(ModelError::unsupported("unsupported Anthropic media type")
                .with_stage(ErrorStage::RequestEncoding));
        }
        match (&file.source, file.media_type.as_str()) {
            (FileSource::Bytes(bytes), media_type) if media_type.starts_with("image/") => {
                if bytes.len().div_ceil(3).saturating_mul(4) > ANTHROPIC_IMAGE_MAX_BASE64_BYTES {
                    return Err(ModelError::invalid_request(
                        "Anthropic base64 image exceeds 10 MiB",
                    )
                    .with_stage(ErrorStage::RequestEncoding));
                }
            }
            (FileSource::Url(url), media_type)
                if url.scheme() == "https"
                    && (media_type.starts_with("image/") || media_type == "application/pdf") => {}
            (FileSource::Bytes(_), "application/pdf") | (FileSource::Text(_), "text/plain") => {}
            (FileSource::Text(_), _) => {
                return Err(ModelError::unsupported(
                    "binary Anthropic media cannot use FileSource::Text",
                )
                .with_stage(ErrorStage::RequestEncoding));
            }
            _ => {
                return Err(ModelError::unsupported(
                    "unsupported Anthropic media source for this MIME type",
                )
                .with_stage(ErrorStage::RequestEncoding));
            }
        }
        return Ok(());
    }
    let image = matches!(
        file.media_type.as_str(),
        "image/jpeg" | "image/png" | "image/gif" | "image/webp"
    );
    let video = matches!(
        file.media_type.as_str(),
        "video/mp4"
            | "video/avi"
            | "video/x-msvideo"
            | "video/quicktime"
            | "video/mov"
            | "video/x-matroska"
    );
    if !image && !video {
        return Err(ModelError::unsupported("unsupported MiniMax media type")
            .with_stage(ErrorStage::RequestEncoding));
    }
    match &file.source {
        FileSource::Bytes(bytes) => {
            if file.media_type == "video/quicktime" {
                return Err(
                    ModelError::unsupported("MiniMax base64 MOV input must use video/mov")
                        .with_stage(ErrorStage::RequestEncoding),
                );
            }
            let limit = if image {
                MINIMAX_IMAGE_MAX_BYTES
            } else {
                MINIMAX_VIDEO_MAX_BYTES
            };
            if bytes.len() > limit {
                return Err(ModelError::invalid_request(if image {
                    "MiniMax image exceeds 10 MiB"
                } else {
                    "MiniMax video exceeds 50 MiB"
                })
                .with_stage(ErrorStage::RequestEncoding));
            }
        }
        FileSource::Url(url) => {
            if url.scheme() != "https" {
                return Err(ModelError::unsupported("MiniMax media URLs must use HTTPS")
                    .with_stage(ErrorStage::RequestEncoding));
            }
            if file.media_type == "video/mov" {
                return Err(ModelError::unsupported(
                    "MiniMax URL MOV input must use video/quicktime",
                )
                .with_stage(ErrorStage::RequestEncoding));
            }
        }
        FileSource::ProviderReference { provider, .. }
            if provider.as_str() == "minimax" && video => {}
        FileSource::Text(_) => {
            return Err(ModelError::unsupported(
                "binary MiniMax media cannot use FileSource::Text",
            )
            .with_stage(ErrorStage::RequestEncoding));
        }
        _ => {
            return Err(ModelError::unsupported(
                "unsupported MiniMax media source for this MIME type",
            )
            .with_stage(ErrorStage::RequestEncoding));
        }
    }
    if let Some(options) = minimax_options {
        if image && options.fps.is_some() {
            return Err(ModelError::invalid_request(
                "MiniMax fps is valid only for video input",
            ));
        }
        if options.fps.is_some_and(|fps| !(0.2..=5.0).contains(&fps)) {
            return Err(ModelError::invalid_request(
                "MiniMax video fps must be between 0.2 and 5",
            ));
        }
        if options.max_long_side_pixel == Some(0) {
            return Err(ModelError::invalid_request(
                "MiniMax max_long_side_pixel must be positive",
            ));
        }
    }
    Ok(())
}

struct CachePlan {
    request: Option<AnthropicCacheControl>,
    tools: Vec<Option<AnthropicCacheControl>>,
    history: Vec<TurnCachePlan>,
    markers: Vec<AnthropicCacheControl>,
    warnings: Vec<String>,
}

fn empty_cache_plan(request: &Request) -> CachePlan {
    CachePlan {
        request: None,
        tools: vec![None; request.tools.len()],
        history: request
            .history
            .iter()
            .map(|turn| TurnCachePlan {
                parts: vec![
                    None;
                    match turn {
                        HistoryTurn::System(message) => message.content.len(),
                        HistoryTurn::User(message) => message.content.len(),
                        HistoryTurn::Assistant(turn) => turn.message.content.len(),
                        HistoryTurn::Tool(message) => message.results.len(),
                    }
                ],
                message: None,
            })
            .collect(),
        markers: Vec::new(),
        warnings: Vec::new(),
    }
}

#[derive(Default)]
struct TurnCachePlan {
    parts: Vec<Option<AnthropicCacheControl>>,
    message: Option<(usize, AnthropicCacheControl)>,
}

impl TurnCachePlan {
    fn marker_for(&self, part_index: usize) -> Option<AnthropicCacheControl> {
        self.parts.get(part_index).cloned().flatten().or_else(|| {
            self.message
                .as_ref()
                .filter(|(target, _)| *target == part_index)
                .map(|(_, marker)| marker.clone())
        })
    }

    fn markers_in_part_order(&self) -> Vec<AnthropicCacheControl> {
        let mut markers = Vec::new();
        for (part_index, part_marker) in self.parts.iter().enumerate() {
            if let Some(marker) = part_marker {
                markers.push(marker.clone());
            }
            if let Some((target, marker)) = &self.message
                && *target == part_index
            {
                markers.push(marker.clone());
            }
        }
        markers
    }
}

fn cache_plan(
    request: &Request,
    options: &AnthropicRequestOptions,
) -> Result<CachePlan, ModelError> {
    let request_marker = options.cache_control.clone();

    let emit_tools = !request.tools.is_empty() && !matches!(request.tool_choice, ToolChoice::None);
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            emit_tools
                .then(|| cache_control_from_options(&tool.provider_options))
                .flatten()
        })
        .collect::<Vec<_>>();

    let mut history = Vec::with_capacity(request.history.len());
    let mut warnings = Vec::new();
    for turn in &request.history {
        let (parts, last_eligible, message_marker, omitted_warning) = match turn {
            HistoryTurn::System(message) => {
                let mut parts = vec![None; message.content.len()];
                let mut last_eligible = None;
                for (part_index, part) in message.content.iter().enumerate() {
                    if let oven_sdk::SystemPart::Text(text) = part {
                        let marker = cache_control_from_metadata(&text.metadata);
                        if text.text.is_empty() {
                            reject_marked_empty_text(marker.as_ref())?;
                        } else {
                            last_eligible = Some(part_index);
                            parts[part_index] = marker;
                        }
                    }
                }
                (
                    parts,
                    last_eligible,
                    cache_control_from_options(&message.provider_options),
                    "Anthropic message cache marker omitted: no eligible block",
                )
            }
            HistoryTurn::User(message) => {
                let mut parts = vec![None; message.content.len()];
                let mut last_eligible = None;
                for (part_index, part) in message.content.iter().enumerate() {
                    match part {
                        InputPart::Text(text) => {
                            let marker = cache_control_from_metadata(&text.metadata);
                            if text.text.is_empty() {
                                reject_marked_empty_text(marker.as_ref())?;
                            } else {
                                last_eligible = Some(part_index);
                                parts[part_index] = marker;
                            }
                        }
                        InputPart::File(file) => {
                            last_eligible = Some(part_index);
                            parts[part_index] = cache_control_from_metadata(&file.metadata);
                        }
                        InputPart::Custom(_) => {}
                    }
                }
                (
                    parts,
                    last_eligible,
                    cache_control_from_options(&message.provider_options),
                    "Anthropic message cache marker omitted: no eligible block",
                )
            }
            HistoryTurn::Assistant(turn) => {
                let mut parts = vec![None; turn.message.content.len()];
                let mut last_eligible = None;
                for (part_index, part) in turn.message.content.iter().enumerate() {
                    match part {
                        AssistantPart::Text(text) => {
                            let marker = cache_control_from_metadata(&text.metadata);
                            if text.text.is_empty() {
                                reject_marked_empty_text(marker.as_ref())?;
                            } else {
                                last_eligible = Some(part_index);
                                parts[part_index] = marker;
                            }
                        }
                        AssistantPart::ToolCall(call) => {
                            last_eligible = Some(part_index);
                            parts[part_index] = cache_control_from_metadata(&call.metadata);
                        }
                        AssistantPart::Reasoning(reasoning)
                            if cache_control_from_metadata(&reasoning.metadata).is_some() =>
                        {
                            return Err(ModelError::invalid_request(
                                "Anthropic thinking blocks cannot carry cache markers",
                            ));
                        }
                        _ => {}
                    }
                }
                (
                    parts,
                    last_eligible,
                    cache_control_from_options(&turn.message.provider_options),
                    "Anthropic assistant cache marker omitted: no eligible block",
                )
            }
            HistoryTurn::Tool(message) => {
                let parts = message
                    .results
                    .iter()
                    .map(|result| cache_control_from_metadata(&result.metadata))
                    .collect::<Vec<_>>();
                (
                    parts,
                    (!message.results.is_empty()).then(|| message.results.len() - 1),
                    cache_control_from_options(&message.provider_options),
                    "Anthropic assistant cache marker omitted: no eligible block",
                )
            }
        };

        let message = match (message_marker, last_eligible) {
            (Some(_), Some(target)) if parts[target].is_some() => {
                warnings.push(
                    "Anthropic part cache marker takes precedence over message marker".into(),
                );
                None
            }
            (Some(marker), Some(target)) => Some((target, marker)),
            (Some(_), None) => {
                warnings.push(omitted_warning.into());
                None
            }
            (None, _) => None,
        };
        history.push(TurnCachePlan { parts, message });
    }

    let mut markers = tools.iter().flatten().cloned().collect::<Vec<_>>();
    for (history_index, turn) in request.history.iter().enumerate() {
        if matches!(turn, HistoryTurn::System(_)) {
            markers.extend(history[history_index].markers_in_part_order());
        }
    }
    for (history_index, turn) in request.history.iter().enumerate() {
        if !matches!(turn, HistoryTurn::System(_)) {
            markers.extend(history[history_index].markers_in_part_order());
        }
    }
    markers.extend(request_marker.iter().cloned());

    Ok(CachePlan {
        request: request_marker,
        tools,
        history,
        markers,
        warnings,
    })
}

fn reject_marked_empty_text(marker: Option<&AnthropicCacheControl>) -> Result<(), ModelError> {
    if marker.is_some() {
        Err(ModelError::invalid_request(
            "empty text cannot carry an Anthropic cache marker",
        ))
    } else {
        Ok(())
    }
}

fn validate_cache_plan(plan: &CachePlan) -> Result<(), ModelError> {
    if plan.markers.len() > 4 {
        return Err(ModelError::invalid_request(
            "Anthropic requests allow at most four cache breakpoints",
        ));
    }
    let mut saw_five_minutes = false;
    for cache in &plan.markers {
        match cache.ttl {
            AnthropicCacheTtl::FiveMinutes => saw_five_minutes = true,
            AnthropicCacheTtl::OneHour if saw_five_minutes => {
                return Err(ModelError::invalid_request(
                    "one-hour Anthropic cache markers must precede five-minute markers",
                ));
            }
            AnthropicCacheTtl::OneHour => {}
        }
    }
    Ok(())
}

fn validate_schema(schema: &JsonValue) -> Result<(), ModelError> {
    let JsonValue::Object(object) = schema else {
        return Ok(());
    };
    const ALLOWED: &[&str] = &[
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "description",
        "title",
        "default",
    ];
    if object.keys().any(|key| !ALLOWED.contains(&key.as_str())) {
        return Err(ModelError::invalid_request(
            "JSON Schema uses an unsupported Anthropic keyword",
        ));
    }
    if let Some(JsonValue::Object(properties)) = object.get("properties") {
        for value in properties.values() {
            validate_schema(value)?;
        }
    }
    if let Some(items) = object.get("items") {
        validate_schema(items)?;
    }
    Ok(())
}

fn cache_control_from_options(
    options: &oven_sdk::ProviderOptions,
) -> Option<AnthropicCacheControl> {
    serde_json::from_value::<AnthropicRequestOptions>(options.get("anthropic")?.clone())
        .ok()?
        .cache_control
}
fn cache_control_from_metadata(metadata: &oven_sdk::PartMetadata) -> Option<AnthropicCacheControl> {
    cache_control_from_options(metadata.as_ref()?)
}
fn wire_cache(marker: Option<AnthropicCacheControl>) -> Option<CacheControl> {
    marker.as_ref().map(CacheControl::from)
}

pub(crate) struct Encoded {
    /// Serialized request body; reused verbatim for signing and sending.
    pub(crate) body: Bytes,
    pub(crate) replay: ReplayOutcome,
    pub(crate) warnings: Vec<String>,
    pub(crate) betas: Vec<String>,
}
pub(crate) struct ParsedOptions {
    anthropic: AnthropicRequestOptions,
    minimax: MiniMaxRequestOptions,
    aws: AnthropicAwsRequestOptions,
    cache_plan: CachePlan,
    strict_tools: Vec<bool>,
    minimax_media: Vec<Vec<Option<MiniMaxMediaOptions>>>,
}

pub(crate) fn parse_options(
    request: &Request,
    protocol: Protocol,
    compatible: bool,
) -> Result<ParsedOptions, ModelError> {
    let anthropic = if protocol.is_first_party() {
        options(request)?
    } else {
        AnthropicRequestOptions::default()
    };
    let minimax = if protocol == Protocol::MiniMax {
        minimax_options(request)?
    } else {
        MiniMaxRequestOptions::default()
    };
    let aws = if protocol == Protocol::AnthropicAws {
        aws_options(request)?
    } else {
        AnthropicAwsRequestOptions::default()
    };
    let cache_plan = if protocol.is_first_party() {
        cache_plan(request, &anthropic)?
    } else {
        empty_cache_plan(request)
    };
    let strict_tools = request
        .tools
        .iter()
        .map(|tool| {
            protocol.is_first_party()
                && tool
                    .provider_options
                    .get("anthropic")
                    .and_then(|value| {
                        serde_json::from_value::<AnthropicToolOptions>(value.clone()).ok()
                    })
                    .is_some_and(|options| options.strict)
        })
        .collect();
    let minimax_media = request
        .history
        .iter()
        .map(|turn| match turn {
            HistoryTurn::User(message) if protocol == Protocol::MiniMax || compatible => message
                .content
                .iter()
                .map(|part| match part {
                    InputPart::File(file) => minimax_media_options(file),
                    _ => Ok(None),
                })
                .collect(),
            HistoryTurn::User(message) => Ok(vec![None; message.content.len()]),
            _ => Ok(Vec::new()),
        })
        .collect::<Result<Vec<_>, ModelError>>()?;
    Ok(ParsedOptions {
        anthropic,
        minimax,
        aws,
        cache_plan,
        strict_tools,
        minimax_media,
    })
}

fn options(request: &Request) -> Result<AnthropicRequestOptions, ModelError> {
    request
        .provider_options
        .get("anthropic")
        .map(|v| {
            serde_json::from_value(v.clone())
                .map_err(|_| ModelError::invalid_request("invalid Anthropic request options"))
        })
        .transpose()
        .map(|v| v.unwrap_or_default())
}

fn minimax_options(request: &Request) -> Result<MiniMaxRequestOptions, ModelError> {
    request
        .provider_options
        .get("minimax")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| ModelError::invalid_request("invalid MiniMax request options"))
        })
        .transpose()
        .map(|value| value.unwrap_or_default())
}

fn aws_options(request: &Request) -> Result<AnthropicAwsRequestOptions, ModelError> {
    request
        .provider_options
        .get("anthropic_aws")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| ModelError::invalid_request("invalid Anthropic AWS request options"))
        })
        .transpose()
        .map(|value| value.unwrap_or_default())
}

pub(crate) fn encode_request(
    request: &Request,
    parsed: &ParsedOptions,
    descriptor: &LanguageModelDescriptor,
    _native_context_scope: &NativeContextScope,
    policy: ReplayPolicy,
    protocol: Protocol,
    compatible: bool,
) -> Result<Encoded, ModelError> {
    let opts = &parsed.anthropic;
    let minimax_opts = &parsed.minimax;
    let aws_opts = &parsed.aws;
    let cache_plan = &parsed.cache_plan;
    validate_cache_plan(cache_plan)?;
    let mut replay = ReplayOutcome::default();
    let mut messages: Vec<Message<'_>> = Vec::with_capacity(request.history.len());
    let mut warnings = cache_plan.warnings.clone();
    let mut system = Vec::new();
    let mut previous_was_tool = false;
    for (index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
                previous_was_tool = false;
                for (part_index, part) in message.content.iter().enumerate() {
                    if let oven_sdk::SystemPart::Text(text) = part
                        && !text.text.is_empty()
                    {
                        system.push(Block::Typed(TypedBlock::Text {
                            text: TextValue::Str(&text.text),
                            cache_control: wire_cache(
                                cache_plan.history[index].marker_for(part_index),
                            ),
                        }));
                    }
                }
            }
            HistoryTurn::User(message) => {
                previous_was_tool = false;
                let mut content = Vec::with_capacity(message.content.len());
                for (part_index, part) in message.content.iter().enumerate() {
                    if let Some(block) = input_block(
                        part,
                        wire_cache(cache_plan.history[index].marker_for(part_index)),
                        protocol,
                        parsed.minimax_media[index][part_index].as_ref(),
                        compatible,
                    )? {
                        content.push(block);
                    }
                }
                if !content.is_empty() {
                    messages.push(Message {
                        role: "user",
                        content,
                    });
                }
            }
            HistoryTurn::Tool(message) => {
                let content = message
                    .results
                    .iter()
                    .enumerate()
                    .map(|(part_index, result)| {
                        tool_result_block(
                            result,
                            wire_cache(cache_plan.history[index].marker_for(part_index)),
                            protocol,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if !content.is_empty() {
                    if previous_was_tool {
                        messages
                            .last_mut()
                            .expect("previous tool message has content")
                            .content
                            .extend(content);
                    } else {
                        messages.push(Message {
                            role: "user",
                            content,
                        });
                    }
                }
                previous_was_tool = true;
            }
            HistoryTurn::Assistant(turn) => {
                previous_was_tool = false;
                let mut content = None;
                if policy == ReplayPolicy::Never {
                    replay.decisions.push(ReplayDecision {
                        history_index: index,
                        disposition: ReplayDisposition::ReconstructedNormalized,
                    });
                } else if let Some(artifact) = &turn.finish.native_replay {
                    match replay::decode(artifact, &turn.message.content, protocol) {
                        Ok(decoded) => {
                            replay.decisions.push(ReplayDecision {
                                history_index: index,
                                disposition: ReplayDisposition::Replayed,
                            });
                            content = Some(
                                replay::eligible_blocks(
                                    artifact,
                                    &descriptor.identity.model_id,
                                    descriptor.capabilities.replay.reasoning,
                                    decoded,
                                    &mut warnings,
                                )?
                                .into_iter()
                                .map(|value| {
                                    Block::Native(NativeBlock {
                                        value,
                                        cache: None,
                                        text: None,
                                    })
                                })
                                .collect::<Vec<_>>(),
                            );
                        }
                        Err(reason) => replay.decisions.push(ReplayDecision {
                            history_index: index,
                            disposition: ReplayDisposition::DiscardedInvalidPayload {
                                reason: reason.into(),
                            },
                        }),
                    }
                } else {
                    replay.decisions.push(ReplayDecision {
                        history_index: index,
                        disposition: ReplayDisposition::NoArtifact,
                    });
                }
                if content.is_none() {
                    if descriptor.capabilities.replay.capability
                        == oven_sdk::ReplayCapability::Required
                        && !request.history[index + 1..]
                            .iter()
                            .any(|turn| matches!(turn, HistoryTurn::User(_)))
                        && turn
                            .message
                            .content
                            .iter()
                            .any(|part| matches!(part, AssistantPart::ToolCall(_)))
                        && turn
                            .message
                            .content
                            .iter()
                            .any(|part| matches!(part, AssistantPart::Reasoning(_)))
                    {
                        return Err(ModelError::replay(format!(
                            "{} reasoning continuation requires a valid native replay artifact at history index {index}",
                            protocol.display_name()
                        )));
                    }
                    if policy != ReplayPolicy::Never {
                        replay.decisions.push(ReplayDecision {
                            history_index: index,
                            disposition: ReplayDisposition::ReconstructedNormalized,
                        });
                    }
                    content = Some(
                        turn.message
                            .content
                            .iter()
                            .filter_map(assistant_block)
                            .collect::<Vec<_>>(),
                    );
                    if turn
                        .message
                        .content
                        .iter()
                        .any(|part| matches!(part, AssistantPart::Reasoning(_)))
                    {
                        warnings.push(format!(
                            "{} reasoning omitted from normalized reconstruction at history index {index}",
                            protocol.display_name()
                        ));
                    }
                }
                let mut content = content.unwrap_or_default();
                attach_assistant_cache_markers(
                    &mut content,
                    &turn.message.content,
                    &cache_plan.history[index],
                );
                if index == request.history.len() - 1 {
                    trim_final_assistant_prefill(&mut content);
                }
                if !content.is_empty() {
                    messages.push(Message {
                        role: "assistant",
                        content,
                    });
                }
            }
        }
    }
    let requested_max = request
        .inference
        .max_output_tokens
        .or(descriptor.capabilities.limits.output)
        .unwrap_or(4096);
    let wire_max = match &opts.thinking {
        Some(AnthropicThinking::Enabled { budget_tokens, .. })
            if request.inference.max_output_tokens.is_some() =>
        {
            requested_max.checked_add(*budget_tokens).ok_or_else(|| {
                ModelError::invalid_request("Anthropic max_tokens arithmetic overflowed")
            })?
        }
        _ => requested_max,
    };
    let tools = if !request.tools.is_empty() && !matches!(request.tool_choice, ToolChoice::None) {
        request
            .tools
            .iter()
            .enumerate()
            .map(|(index, tool)| Tool {
                name: &tool.name,
                description: &tool.description,
                input_schema: tool.input_schema.as_value(),
                strict: parsed.strict_tools[index],
                cache_control: wire_cache(cache_plan.tools[index].clone()),
            })
            .collect()
    } else {
        Vec::new()
    };
    let tool_choice = match &request.tool_choice {
        ToolChoice::Auto if !request.tools.is_empty() => Some(WireToolChoice::Auto),
        ToolChoice::Required if protocol.is_first_party() => Some(WireToolChoice::Any),
        ToolChoice::Tool(name) if protocol.is_first_party() => Some(WireToolChoice::Tool { name }),
        ToolChoice::None if !request.tools.is_empty() => {
            warnings.push(format!(
                "{} tool_choice none omitted declared tools",
                protocol.display_name()
            ));
            None
        }
        _ => None,
    };
    let mut thinking = None;
    if protocol.is_first_party()
        && let Some(value) = &opts.thinking
    {
        thinking = Some(match value {
            AnthropicThinking::Disabled => Thinking {
                mode: "disabled",
                budget_tokens: None,
                display: None,
            },
            AnthropicThinking::Enabled {
                budget_tokens,
                display,
            } => Thinking {
                mode: "enabled",
                budget_tokens: Some(*budget_tokens),
                display: display.as_deref(),
            },
            AnthropicThinking::Adaptive { display } => Thinking {
                mode: "adaptive",
                budget_tokens: None,
                display: display.as_deref(),
            },
        });
    }
    let mut service_tier = None;
    let mut metadata = None;
    if protocol == Protocol::MiniMax {
        if let Some(value) = &minimax_opts.thinking {
            thinking = Some(Thinking {
                mode: match value {
                    MiniMaxThinking::Adaptive => "adaptive",
                    MiniMaxThinking::Disabled => "disabled",
                },
                budget_tokens: None,
                display: None,
            });
        }
        service_tier = minimax_opts.service_tier.as_deref();
        metadata = minimax_opts
            .user_id
            .as_deref()
            .map(|user_id| Metadata { user_id });
    }
    let effort = opts.effort.as_deref().filter(|_| protocol.is_first_party());
    let format = match &request.response_format {
        ResponseFormat::Json {
            schema: Some(schema),
        } if protocol.is_first_party() => Some(OutputFormat {
            kind: "json_schema",
            schema: schema.as_value(),
        }),
        _ => None,
    };
    let output_config =
        (effort.is_some() || format.is_some()).then_some(OutputConfig { effort, format });
    if protocol.is_first_party()
        && let Some(user_id) = &opts.user_id
    {
        metadata = Some(Metadata { user_id });
    }
    let body = MessagesBody {
        model: descriptor.identity.model_id.as_str(),
        messages,
        stream: true,
        max_tokens: wire_max,
        system,
        tools,
        tool_choice,
        thinking,
        temperature: request.inference.temperature,
        top_p: request.inference.top_p,
        service_tier,
        metadata,
        output_config,
        cache_control: if protocol.is_first_party() {
            cache_plan.request.as_ref().map(CacheControl::from)
        } else {
            None
        },
        inference_geo: aws_opts
            .inference_geo
            .as_deref()
            .filter(|_| protocol == Protocol::AnthropicAws),
    }
    .to_bytes()
    .map_err(|_| {
        ModelError::invalid_request("could not serialize Messages request")
            .with_stage(ErrorStage::RequestEncoding)
    })?;
    let betas = if protocol.is_first_party() {
        opts.betas
            .iter()
            .filter(|beta| {
                beta.as_str() != "prompt-caching-2024-07-31"
                    && beta.as_str() != "structured-outputs-2025-11-13"
            })
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    } else {
        Vec::new()
    };
    Ok(Encoded {
        body: Bytes::from(body),
        replay,
        warnings,
        betas,
    })
}
fn input_block<'a>(
    part: &'a InputPart,
    cache: Option<CacheControl>,
    protocol: Protocol,
    minimax_options: Option<&'a MiniMaxMediaOptions>,
    compatible: bool,
) -> Result<Option<Block<'a>>, ModelError> {
    Ok(match part {
        InputPart::Text(t) => (!t.text.is_empty()).then(|| {
            Block::Typed(TypedBlock::Text {
                text: TextValue::Str(&t.text),
                cache_control: cache,
            })
        }),
        InputPart::File(f) => {
            let mut block = file_block(f, protocol, minimax_options, compatible)?;
            if let Some(cache) = cache {
                block.set_cache(cache);
            }
            Some(block)
        }
        InputPart::Custom(_) => None,
    })
}
fn file_block<'a>(
    file: &'a oven_sdk::FilePart,
    protocol: Protocol,
    minimax_options: Option<&'a MiniMaxMediaOptions>,
    compatible: bool,
) -> Result<Block<'a>, ModelError> {
    if protocol == Protocol::MiniMax || (compatible && file.media_type.starts_with("video/")) {
        return minimax_file_block(file, minimax_options);
    }
    let image = file.media_type.starts_with("image/");
    let source = match &file.source {
        FileSource::Text(text) if file.media_type == "text/plain" => {
            return Ok(Block::Typed(TypedBlock::Document {
                source: Source::text("text/plain", text),
                cache_control: None,
            }));
        }
        FileSource::Bytes(bytes) => Source::base64(&file.media_type, bytes),
        FileSource::Url(url) => Source::url(Cow::Borrowed(url.as_str())),
        _ => {
            return Err(ModelError::unsupported("unsupported Anthropic file source")
                .with_stage(ErrorStage::RequestEncoding));
        }
    };
    Ok(Block::Typed(if image {
        TypedBlock::Image {
            source,
            cache_control: None,
        }
    } else {
        TypedBlock::Document {
            source,
            cache_control: None,
        }
    }))
}

fn minimax_file_block<'a>(
    file: &'a oven_sdk::FilePart,
    options: Option<&'a MiniMaxMediaOptions>,
) -> Result<Block<'a>, ModelError> {
    let mut source = match &file.source {
        FileSource::Bytes(bytes) => Source::base64(&file.media_type, bytes),
        FileSource::Url(url) => Source::url(Cow::Borrowed(url.as_str())),
        FileSource::ProviderReference { provider, id } if provider.as_str() == "minimax" => {
            Source::url(Cow::Owned(format!("mm_file://{id}")))
        }
        _ => {
            return Err(ModelError::unsupported("unsupported MiniMax file source")
                .with_stage(ErrorStage::RequestEncoding));
        }
    };
    if let Some(options) = options {
        source.detail = options.detail.as_deref();
        source.fps = options.fps;
        source.max_long_side_pixel = options.max_long_side_pixel;
    }
    Ok(Block::Typed(if file.media_type.starts_with("image/") {
        TypedBlock::Image {
            source,
            cache_control: None,
        }
    } else {
        TypedBlock::Video {
            source,
            cache_control: None,
        }
    }))
}

fn minimax_media_options(
    file: &oven_sdk::FilePart,
) -> Result<Option<MiniMaxMediaOptions>, ModelError> {
    file.metadata
        .as_ref()
        .and_then(|metadata| metadata.get("minimax"))
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| ModelError::invalid_request("invalid MiniMax media options"))
        })
        .transpose()
}
fn tool_result_block(
    result: &ToolResultPart,
    cache: Option<CacheControl>,
    protocol: Protocol,
) -> Result<Block<'_>, ModelError> {
    let content = match &result.content {
        ToolContent::Text(v) => ToolResultContent::Text(TextValue::Str(v)),
        ToolContent::Json(v) => ToolResultContent::Text(TextValue::Json(v)),
        ToolContent::Mixed(v)
            if protocol.is_first_party()
                && v.iter().any(|value| matches!(value, ContentValue::File(_))) =>
        {
            ToolResultContent::Blocks(
                v.iter()
                    .map(|value| match value {
                        ContentValue::Text(value) => Ok(Block::Typed(TypedBlock::Text {
                            text: TextValue::Str(value),
                            cache_control: None,
                        })),
                        ContentValue::Json(value) => Ok(Block::Typed(TypedBlock::Text {
                            text: TextValue::Json(value),
                            cache_control: None,
                        })),
                        ContentValue::File(file) => file_block(file, protocol, None, false),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        ToolContent::Mixed(v) => ToolResultContent::Text(TextValue::Owned(
            serde_json::to_string(v).unwrap_or_default(),
        )),
        ToolContent::Denied { reason } => {
            ToolResultContent::Text(TextValue::Str(reason.as_deref().unwrap_or_default()))
        }
    };
    Ok(Block::Typed(TypedBlock::ToolResult {
        tool_use_id: &result.tool_call_id,
        content,
        is_error: result.is_error,
        cache_control: cache,
    }))
}
pub(crate) fn assistant_block(part: &AssistantPart) -> Option<Block<'_>> {
    match part {
        AssistantPart::Text(t) => (!t.text.is_empty()).then(|| {
            Block::Typed(TypedBlock::Text {
                text: TextValue::Str(&t.text),
                cache_control: None,
            })
        }),
        AssistantPart::Reasoning(_) => None,
        AssistantPart::ToolCall(t) => Some(Block::Typed(TypedBlock::ToolUse {
            id: &t.id,
            name: &t.name,
            input: if t.input.is_object() {
                ToolInput::Object(&t.input)
            } else {
                ToolInput::Raw { raw: &t.input }
            },
            cache_control: None,
        })),
        _ => None,
    }
}

/// Wire block type `assistant_block` produces for a part, without building the block.
fn assistant_block_type(part: &AssistantPart) -> Option<&'static str> {
    match part {
        AssistantPart::Text(t) if !t.text.is_empty() => Some("text"),
        AssistantPart::ToolCall(_) => Some("tool_use"),
        _ => None,
    }
}

fn trim_final_assistant_prefill(content: &mut [Block<'_>]) {
    if let Some(text) = content
        .iter_mut()
        .rev()
        .find(|block| block.block_type() == Some("text"))
    {
        text.trim_text_end();
    }
}

pub(crate) fn validate_serialized_request_size(
    size: usize,
    protocol: Protocol,
) -> Result<(), ModelError> {
    let maximum = if protocol == Protocol::MiniMax {
        MINIMAX_REQUEST_MAX_BYTES
    } else {
        ANTHROPIC_REQUEST_MAX_BYTES
    };
    if size > maximum {
        return Err(ModelError::invalid_request(format!(
            "{} serialized request exceeds {} MiB",
            protocol.display_name(),
            maximum / MIB
        ))
        .with_stage(ErrorStage::RequestEncoding));
    }
    Ok(())
}

fn attach_assistant_cache_markers(
    content: &mut [Block<'_>],
    normalized: &[AssistantPart],
    plan: &TurnCachePlan,
) {
    let mut wire_index = 0;
    for (part_index, part) in normalized.iter().enumerate() {
        let Some(normalized_type) = assistant_block_type(part) else {
            continue;
        };
        while content.get(wire_index).and_then(Block::block_type) != Some(normalized_type) {
            wire_index += 1;
            if wire_index >= content.len() {
                return;
            }
        }
        if let Some(marker) = plan.marker_for(part_index) {
            content[wire_index].set_cache(CacheControl::from(&marker));
        }
        wire_index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn normalized_reasoning_is_never_reconstructed_as_provider_thinking() {
        let part = AssistantPart::Reasoning(oven_sdk::ReasoningPart {
            text: String::new(),
            metadata: Some(BTreeMap::from([(
                "anthropic.redacted".into(),
                serde_json::json!("opaque"),
            )])),
        });
        assert!(assistant_block(&part).is_none());
    }

    #[test]
    fn serialized_request_size_boundaries_are_inclusive() {
        assert!(
            validate_serialized_request_size(ANTHROPIC_REQUEST_MAX_BYTES, Protocol::Anthropic)
                .is_ok()
        );
        assert!(
            validate_serialized_request_size(ANTHROPIC_REQUEST_MAX_BYTES + 1, Protocol::Anthropic)
                .is_err()
        );
        assert!(
            validate_serialized_request_size(MINIMAX_REQUEST_MAX_BYTES, Protocol::MiniMax).is_ok()
        );
        assert!(
            validate_serialized_request_size(MINIMAX_REQUEST_MAX_BYTES + 1, Protocol::MiniMax)
                .is_err()
        );
    }
}
