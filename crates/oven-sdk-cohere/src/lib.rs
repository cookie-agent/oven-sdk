#![warn(missing_docs)]
// `ModelError` is the public structured error contract and is intentionally returned by value.
#![allow(clippy::result_large_err)]
//! Registry-free Cohere native v2 Chat adapter.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
    time::Duration,
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use oven_sdk::provider_support::{SseEvent, SseParser, parse_retry_after};
use oven_sdk::{
    AbortSignal, AdapterId, ApiEndpoint, AssistantPart, BoxFuture, BoxStream,
    CancellationCapability, Capability, CompactionCapability, ContentValue, CustomPart, ErrorStage,
    FilePart, FileSource, Finish, FinishReason, HeaderConfig, HistoryTurn, InputPart, JsonSchema,
    JsonValue, LanguageModel, LanguageModelDescriptor, MediaSourceSupport, ModelCapabilities,
    ModelConfig, ModelError, ModelErrorKind, ModelId, ModelIdentity, NativeContextScope,
    NativeReplayArtifact, ProviderConfig, ProviderMetadata, ReplayCapability, ReplayDecision,
    ReplayDisposition, ReplayOutcome, ReplayPayloadError, ReplayPolicy, Request, RequestMetadata,
    ResourceId, ResponseFormat, ResponseHead, SanitizedBody, SecretString, SourcePart, StreamItem,
    StreamPart, StreamResponse, ToolChoice, ToolContent, ToolResultPart, Usage,
};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod body;

/// Conventional serving-provider identity for Cohere Platform.
pub const COHERE_PROVIDER_ID: &str = "cohere";
/// Stable adapter identity for Cohere native v2 Chat.
pub const COHERE_V2_CHAT_ADAPTER_ID: &str = "cohere.v2.chat";

const REPLAY_SCOPE_VERSION: &str = "cohere.v2.chat.replay_scope.v2";
const REPLAY_FORMAT: &str = "cohere.v2.chat.message.v2";
const REPLAY_FINGERPRINT_KIND: &str = "cohere.v2.chat.replay_fingerprint";
const MAX_IMAGES: usize = 20;
const MAX_INLINE_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// Explicit Cohere bearer authentication.
#[derive(Clone)]
pub struct CohereAuth {
    token: SecretString,
}

impl CohereAuth {
    /// Creates bearer authentication from an already resolved token.
    #[must_use]
    pub fn bearer(token: SecretString) -> Self {
        Self { token }
    }
}

impl std::fmt::Debug for CohereAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CohereAuth(<redacted>)")
    }
}

/// Per-phase HTTP timeouts. There is intentionally no total stream timeout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CohereTimeouts {
    /// Connection timeout.
    pub connect: Duration,
    /// Maximum wait for response headers.
    pub headers: Duration,
    /// Maximum inactivity while reading response bytes.
    pub stream_idle: Duration,
}

impl Default for CohereTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(60),
            headers: Duration::from_secs(300),
            stream_idle: Duration::from_secs(300),
        }
    }
}

/// Explicit Cohere thinking configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohereThinking {
    /// Whether Cohere thinking is enabled.
    pub enabled: bool,
    /// Optional positive thinking-token budget.
    pub token_budget: Option<u64>,
}

/// Complete Cohere v2 wire settings for one configured model.
#[derive(Clone, Debug, Default)]
pub struct CohereSettings {
    /// Transport phase timeouts.
    pub timeouts: CohereTimeouts,
    /// Enforce Cohere strict tool schemas.
    pub strict_tools: bool,
    /// Optional provider-defined safety-mode label.
    pub safety_mode: Option<String>,
    /// Optional default thinking configuration.
    pub thinking: Option<CohereThinking>,
    /// Explicit provider-neutral reasoning-effort mappings.
    pub reasoning_effort: BTreeMap<String, CohereThinking>,
    /// Optional top-k sampling value.
    pub top_k: Option<u32>,
    /// Optional deterministic-sampling seed.
    pub seed: Option<u64>,
    /// Optional frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Optional presence penalty.
    pub presence_penalty: Option<f64>,
    /// Provider stop sequences, limited to five.
    pub stop_sequences: Vec<String>,
    /// Optional request priority.
    pub priority: Option<i64>,
    /// Optional injected HTTP client shared across models. When set, the
    /// connect timeout is whatever the caller configured on that client.
    pub client: Option<reqwest::Client>,
}

/// A caller-provided document available for Cohere grounding and citations.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohereDocument {
    /// Optional stable document identifier.
    pub id: Option<String>,
    /// Arbitrary JSON object passed as Cohere document data.
    pub data: serde_json::Map<String, JsonValue>,
}

/// Request-scoped Cohere v2 options. Provider-defined labels remain strings.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohereRequestOptions {
    /// Grounding documents.
    pub documents: Vec<CohereDocument>,
    /// Optional provider-defined citation mode.
    pub citation_mode: Option<String>,
    /// Optional per-request image detail label.
    pub image_detail: Option<String>,
}

/// Extension methods for typed Cohere request options.
pub trait CohereRequestExt {
    /// Stores typed options under the `cohere` provider namespace.
    fn with_cohere_options(self, options: CohereRequestOptions) -> Self;
}

impl CohereRequestExt for Request {
    fn with_cohere_options(mut self, options: CohereRequestOptions) -> Self {
        self.provider_options.insert(
            "cohere".into(),
            serde_json::to_value(options).expect("Cohere request options are serializable"),
        );
        self
    }
}

/// Complete registry-free Cohere configuration.
pub type CohereConfig = ModelConfig<CohereAuth, CohereSettings>;

/// One configured Cohere native v2 Chat model.
#[derive(Clone)]
pub struct CohereModel {
    config: Arc<Config>,
    descriptor: LanguageModelDescriptor,
}

impl CohereModel {
    /// Constructs one registry-free Cohere model.
    pub fn new(config: CohereConfig) -> Result<Self, ModelError> {
        let (config, descriptor) = Config::build(config)?;
        Ok(Self { config, descriptor })
    }

    /// Returns the exact configured model ID.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        &self.descriptor.identity.model_id
    }
}

#[derive(Clone)]
struct Config {
    auth: CohereAuth,
    client: reqwest::Client,
    endpoint: reqwest::Url,
    headers: HeaderConfig,
    base_headers: HeaderMap,
    settings: CohereSettings,
    capabilities: ModelCapabilities,
    identity: ModelIdentity,
    replay_seed: Sha256,
}

impl Config {
    fn build(value: CohereConfig) -> Result<(Arc<Self>, LanguageModelDescriptor), ModelError> {
        let ModelConfig {
            provider,
            model,
            settings,
        } = value;
        validate_endpoint(&provider.api)?;
        validate_settings(&settings, &model.capabilities)?;
        validate_capabilities(&model.capabilities)?;
        if provider.auth.token.is_empty() {
            return Err(ModelError::invalid_request(
                "Cohere bearer token must not be empty",
            ));
        }
        reject_protected_headers(provider.headers.static_headers.as_map())?;
        model.validate()?;
        let identity = ModelIdentity::new(provider.id.clone(), model.id.clone())?;
        let descriptor = LanguageModelDescriptor::new(
            identity.clone(),
            AdapterId::new(COHERE_V2_CHAT_ADAPTER_ID),
            model.capabilities.clone(),
        )?;
        let replay_seed = replay_seed(&provider, &model, &settings)?;
        let client = match &settings.client {
            Some(client) => client.clone(),
            None => reqwest::Client::builder()
                .connect_timeout(settings.timeouts.connect)
                .build()
                .map_err(|_| ModelError::transport("could not construct Cohere HTTP client"))?,
        };
        let mut base_headers = provider.headers.static_headers.as_map().clone();
        base_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok((
            Arc::new(Self {
                auth: provider.auth,
                client,
                endpoint: provider.api.as_url().clone(),
                headers: provider.headers,
                base_headers,
                settings,
                capabilities: descriptor.capabilities.clone(),
                identity,
                replay_seed,
            }),
            descriptor,
        ))
    }

    fn request_headers(&self, context: &oven_sdk::HeaderContext) -> Result<HeaderMap, ModelError> {
        let mut headers = self.base_headers.clone();
        if let Some(dynamic) = &self.headers.dynamic_headers {
            let dynamic = dynamic.headers(context)?;
            reject_protected_headers(dynamic.as_map())?;
            headers.extend(dynamic.as_map().clone());
        }
        if !oven_sdk::contains_auth_owned_header(&headers) {
            let authorization =
                HeaderValue::from_str(&format!("Bearer {}", self.auth.token.expose_secret()))
                    .map_err(|_| {
                        ModelError::invalid_request(
                            "Cohere bearer token is not a valid header value",
                        )
                    })?;
            headers.insert(reqwest::header::AUTHORIZATION, authorization);
        }
        Ok(headers)
    }

    fn replay_context(
        &self,
        headers: &HeaderMap,
    ) -> Result<(JsonValue, NativeContextScope), ModelError> {
        let mut hasher = self.replay_seed.clone();
        hash_headers(&mut hasher, headers);
        let digest = URL_SAFE_NO_PAD.encode(hasher.finalize());
        let binding = serde_json::json!({"version":REPLAY_SCOPE_VERSION,"sha256":digest});
        let scope = NativeContextScope::new(
            self.identity.provider_id.clone(),
            self.identity.model_id.clone(),
            ResourceId::new(format!("{REPLAY_SCOPE_VERSION}.sha256.{digest}"))?,
        )?;
        Ok((binding, scope))
    }
}

impl LanguageModel for CohereModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.descriptor
    }

    fn validate_request(&self, request: &Request) -> Result<(), ModelError> {
        request.validate_for(&self.config.capabilities)?;
        let options = request_options(request)?;
        validate_request(request, &options, &self.config.settings)
    }

    fn supports_request(&self, request: &Request) -> bool {
        self.validate_request(request).is_ok()
    }

    fn stream<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        Box::pin(async move {
            request.validate_for(&self.config.capabilities)?;
            let options = request_options(&request)?;
            validate_request(&request, &options, &self.config.settings)?;
            if abort.is_aborted() {
                return Err(ModelError::abort("request was aborted before dispatch")
                    .with_stage(ErrorStage::Connect));
            }
            let headers = self.config.request_headers(&request.header_context)?;
            let (binding, scope) = self.config.replay_context(&headers)?;
            let encoded = encode_request(
                &request,
                &options,
                &self.descriptor,
                &self.config.settings,
                &binding,
                &scope,
            )?;
            let send = self
                .config
                .client
                .post(self.config.endpoint.clone())
                .headers(headers)
                .body(encoded.body)
                .send();
            let response = tokio::select! {
                value = tokio::time::timeout(self.config.settings.timeouts.headers, send) => value
                    .map_err(|_| ModelError::timeout("Cohere response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
                    .map_err(|_| ModelError::transport("Cohere request failed").with_stage(ErrorStage::Connect))?,
                _ = abort.aborted() => return Err(ModelError::abort("request was aborted before response headers").with_stage(ErrorStage::Connect)),
            };
            let head = response_head(&response);
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let response_headers = response.headers().clone();
                let request_id = head.request_id.clone();
                let (body, bytes) =
                    read_error_body(response, &abort, self.config.settings.timeouts.stream_idle)
                        .await?;
                return Err(classify_error(
                    status,
                    &body,
                    request_id,
                    ErrorStage::ResponseBody,
                    bytes,
                    &response_headers,
                ));
            }
            if !response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/event-stream"))
            {
                return Err(ModelError::invalid_response("Cohere response is not SSE")
                    .with_stage(ErrorStage::ResponseHeaders));
            }
            let response_headers = response.headers().clone();
            let live = Live {
                bytes: Box::pin(response.bytes_stream()),
                parser: SseParser::new("Cohere SSE contains invalid UTF-8")
                    .clear_name_on_empty_event(),
                state: CohereState::new(
                    self.descriptor.adapter_id.clone(),
                    self.config.capabilities.replay.policy,
                    self.config.capabilities.replay.capability,
                    self.config.capabilities.replay.reasoning,
                    binding,
                    scope,
                ),
                queue: VecDeque::from([Ok(StreamPart::StreamStart {
                    warnings: encoded.warnings,
                })]),
                events: VecDeque::new(),
                pending_error: None,
                deadline: oven_sdk::provider_support::StreamReadDeadline::new(
                    tokio::time::sleep(self.config.settings.timeouts.stream_idle),
                    &abort,
                ),
                idle: self.config.settings.timeouts.stream_idle,
                count: 0,
                eof: false,
                request_id: head.request_id.clone(),
                response_headers,
                saw_event: false,
                include_raw: request.stream_options.include_raw,
            };
            let stream = futures_util::stream::unfold(live, |mut live| async move {
                loop {
                    if let Some(item) = live.queue.pop_front() {
                        return Some((item, live));
                    }
                    if let Some(error) = live.pending_error.take() {
                        live.eof = true;
                        return Some((Err(error), live));
                    }
                    if live.eof {
                        return None;
                    }
                    if let Err(error) = read_live(&mut live).await {
                        live.pending_error = Some(error);
                    }
                }
            });
            Ok(StreamResponse::new(Box::pin(stream))
                .with_request(RequestMetadata {
                    replay: encoded.replay,
                    provider_metadata: ProviderMetadata::new(),
                })
                .with_response(head))
        })
    }
}

struct Encoded {
    /// Serialized request body, sent verbatim.
    body: bytes::Bytes,
    replay: ReplayOutcome,
    warnings: Vec<String>,
}

fn request_options(request: &Request) -> Result<CohereRequestOptions, ModelError> {
    request
        .provider_options
        .get("cohere")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| ModelError::invalid_request("invalid typed Cohere request options"))
        .map(Option::unwrap_or_default)
}

fn validate_request(
    request: &Request,
    options: &CohereRequestOptions,
    settings: &CohereSettings,
) -> Result<(), ModelError> {
    validate_open_label(options.citation_mode.as_deref(), "citation mode")?;
    validate_open_label(options.image_detail.as_deref(), "image detail")?;
    if let Some(value) = request.inference.top_p
        && (!value.is_finite() || !(0.01..=0.99).contains(&value))
    {
        return Err(ModelError::invalid_request(
            "Cohere top_p must be finite and between 0.01 and 0.99",
        ));
    }
    if let Some(effort) = request.inference.reasoning_effort.as_deref() {
        validate_open_label(Some(effort), "reasoning effort")?;
        if !settings.reasoning_effort.contains_key(effort) {
            return Err(ModelError::unsupported(
                "Cohere reasoning_effort has no explicit configured mapping",
            ));
        }
    }
    if let ResponseFormat::Json {
        schema: Some(schema),
    } = &request.response_format
    {
        validate_cohere_schema(schema.as_value(), true)?;
    }
    if matches!(request.response_format, ResponseFormat::Json { .. })
        && (!request.tools.is_empty() || !options.documents.is_empty())
    {
        return Err(ModelError::unsupported(
            "Cohere response_format cannot be combined with tools or documents",
        ));
    }
    if settings.safety_mode.is_some()
        && (!request.tools.is_empty() || !options.documents.is_empty())
    {
        return Err(ModelError::unsupported(
            "Cohere safety_mode cannot be combined with tools or documents",
        ));
    }
    if settings.strict_tools && !request.tools.is_empty() {
        let fields = request
            .tools
            .iter()
            .map(|tool| validate_cohere_schema(tool.input_schema.as_value(), true))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<usize>();
        if fields > 200 {
            return Err(ModelError::invalid_request(
                "Cohere strict tools permit at most 200 schema fields",
            ));
        }
    }
    let mut image_count = 0_usize;
    let mut inline_bytes = 0_usize;
    for turn in &request.history {
        if let HistoryTurn::Assistant(turn) = turn {
            for part in &turn.message.content {
                match part {
                    AssistantPart::Text(_)
                    | AssistantPart::Reasoning(_)
                    | AssistantPart::ToolCall(_) => {}
                    AssistantPart::Source(source) if source.metadata.is_some() => {}
                    AssistantPart::Source(_) => {
                        return Err(ModelError::invalid_request(
                            "Cohere assistant sources require provider citation metadata",
                        ));
                    }
                    AssistantPart::ToolResult(result)
                        if !matches!(
                            &result.content,
                            ToolContent::Mixed(values)
                                if values.iter().any(|value| matches!(value, ContentValue::File(_)))
                        ) => {}
                    AssistantPart::File(_) | AssistantPart::ToolResult(_) => {
                        return Err(ModelError::unsupported(
                            "Cohere does not support files in assistant or tool-result history",
                        ));
                    }
                    AssistantPart::ToolApproval(_) => {
                        return Err(ModelError::unsupported(
                            "Cohere cannot encode tool-approval parts in assistant history",
                        ));
                    }
                    AssistantPart::Custom(part) if part.kind == REPLAY_FINGERPRINT_KIND => {}
                    AssistantPart::Custom(_) => {
                        return Err(ModelError::unsupported(
                            "Cohere cannot encode custom assistant parts",
                        ));
                    }
                }
            }
        }
        if let HistoryTurn::Tool(message) = turn
            && message.results.iter().any(|result| {
                matches!(
                    &result.content,
                    ToolContent::Mixed(values)
                        if values.iter().any(|value| matches!(value, ContentValue::File(_)))
                )
            })
        {
            return Err(ModelError::unsupported(
                "Cohere does not support files in assistant or tool-result history",
            ));
        }
        if let HistoryTurn::User(message) = turn {
            for part in &message.content {
                if let InputPart::File(file) = part {
                    image_count += 1;
                    if let FileSource::Bytes(bytes) = &file.source {
                        inline_bytes = inline_bytes.saturating_add(bytes.len());
                    }
                }
            }
        }
    }
    if image_count > MAX_IMAGES || inline_bytes > MAX_INLINE_IMAGE_BYTES {
        return Err(ModelError::invalid_request(
            "Cohere accepts at most 20 images and 20 MiB of inline image bytes per request",
        ));
    }
    Ok(())
}

fn encode_request(
    request: &Request,
    options: &CohereRequestOptions,
    descriptor: &LanguageModelDescriptor,
    settings: &CohereSettings,
    binding: &JsonValue,
    _scope: &NativeContextScope,
) -> Result<Encoded, ModelError> {
    let mut messages = Vec::with_capacity(request.history.len());
    let mut replay = ReplayOutcome::default();
    let mut warnings = Vec::new();
    warnings.extend(oven_sdk::DeferredTools::of(request).upfront_warning("cohere"));
    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
                messages.extend(body::Message::system(&message.content));
            }
            HistoryTurn::User(message) => {
                let mut content = Vec::new();
                let mut has_file = false;
                for part in &message.content {
                    match part {
                        InputPart::Text(text) if !text.text.is_empty() => {
                            content.push(body::UserPart::Text { text: &text.text })
                        }
                        InputPart::File(file) => {
                            has_file = true;
                            content.push(image_content(file, options.image_detail.as_deref())?)
                        }
                        _ => {}
                    }
                }
                if !content.is_empty() {
                    if has_file {
                        messages.push(body::Message::user_parts(content));
                    } else {
                        messages.push(body::Message::user_text(&message.content));
                    }
                }
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    messages.push(tool_result(result)?);
                }
            }
            HistoryTurn::Assistant(turn) => {
                if settings_replay_never(descriptor) {
                    replay.decisions.push(ReplayDecision {
                        history_index,
                        disposition: ReplayDisposition::ReconstructedNormalized,
                    });
                    normalized_assistant(&turn.message.content, true, &mut messages)?;
                    continue;
                }
                let replayed = match &turn.finish.native_replay {
                    None => {
                        replay.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::NoArtifact,
                        });
                        None
                    }
                    Some(artifact) => match decode_replay(
                        artifact,
                        &turn.message.content,
                        binding,
                        descriptor.capabilities.replay.reasoning,
                    ) {
                        Some(message) => {
                            replay.decisions.push(ReplayDecision {
                                history_index,
                                disposition: ReplayDisposition::Replayed,
                            });
                            Some(message)
                        }
                        None => {
                            replay.decisions.push(ReplayDecision {
                                history_index,
                                disposition: ReplayDisposition::DiscardedInvalidPayload {
                                    reason:
                                        "Cohere replay payload did not match normalized content"
                                            .into(),
                                },
                            });
                            None
                        }
                    },
                };
                if let Some(message) = replayed {
                    messages.push(message);
                } else {
                    replay.decisions.push(ReplayDecision {
                        history_index,
                        disposition: ReplayDisposition::ReconstructedNormalized,
                    });
                    warnings.push(
                        "Cohere normalized replay fallback may omit provider-only citation details"
                            .into(),
                    );
                    normalized_assistant(&turn.message.content, true, &mut messages)?;
                }
            }
        }
    }
    let thinking = resolved_thinking(request, settings)?.map(|config| body::Thinking {
        kind: if config.enabled {
            "enabled"
        } else {
            "disabled"
        },
        token_budget: config.token_budget,
    });
    let (tools, strict_tools, tool_choice) = if !request.tools.is_empty()
        && !matches!(request.tool_choice, ToolChoice::None)
    {
        let tools = request
            .tools
            .iter()
            .filter(|tool| match &request.tool_choice {
                ToolChoice::Tool(name) => &tool.name == name,
                _ => true,
            })
            .map(|tool| {
                body::Tool::function(&tool.name, &tool.description, tool.input_schema.as_value())
            })
            .collect();
        let tool_choice = match &request.tool_choice {
            ToolChoice::Required | ToolChoice::Tool(_) => Some("REQUIRED"),
            _ => None,
        };
        (Some(tools), settings.strict_tools, tool_choice)
    } else {
        (None, false, None)
    };
    let response_format = match &request.response_format {
        ResponseFormat::Text => None,
        ResponseFormat::Json { schema } => Some(body::ResponseFormat::json_object(
            schema.as_ref().map(JsonSchema::as_value),
        )),
    };
    let body = body::ChatBody {
        model: descriptor.identity.model_id.as_str(),
        messages,
        stream: true,
        max_tokens: request.inference.max_output_tokens,
        temperature: request.inference.temperature,
        p: request.inference.top_p,
        k: settings.top_k,
        seed: settings.seed,
        frequency_penalty: settings.frequency_penalty,
        presence_penalty: settings.presence_penalty,
        stop_sequences: &settings.stop_sequences,
        priority: settings.priority,
        safety_mode: settings.safety_mode.as_deref(),
        thinking,
        documents: &options.documents,
        citation_options: options
            .citation_mode
            .as_deref()
            .map(|mode| body::CitationOptions { mode }),
        tools,
        strict_tools,
        tool_choice,
        response_format,
    }
    .to_bytes()
    .map_err(|_| ModelError::invalid_request("could not encode Cohere request"))?;
    Ok(Encoded {
        body: bytes::Bytes::from(body),
        replay,
        warnings,
    })
}

fn settings_replay_never(descriptor: &LanguageModelDescriptor) -> bool {
    descriptor.capabilities.replay.policy == ReplayPolicy::Never
}

fn resolved_thinking<'a>(
    request: &Request,
    settings: &'a CohereSettings,
) -> Result<Option<&'a CohereThinking>, ModelError> {
    match request.inference.reasoning_effort.as_deref() {
        Some(effort) => settings
            .reasoning_effort
            .get(effort)
            .map(Some)
            .ok_or_else(|| {
                ModelError::unsupported(
                    "Cohere reasoning_effort has no explicit configured mapping",
                )
            }),
        None => Ok(settings.thinking.as_ref()),
    }
}

fn image_content<'a>(
    file: &'a FilePart,
    default_detail: Option<&'a str>,
) -> Result<body::UserPart<'a>, ModelError> {
    let url = match &file.source {
        FileSource::Bytes(bytes) => body::ImageSource::Data {
            media_type: &file.media_type,
            bytes,
        },
        FileSource::Url(url) if matches!(url.scheme(), "http" | "https") => {
            body::ImageSource::Url(url.as_str())
        }
        FileSource::Url(_) => {
            return Err(ModelError::unsupported(
                "Cohere image URLs must use HTTP(S)",
            ));
        }
        FileSource::Text(_) => {
            return Err(ModelError::unsupported(
                "Cohere images do not accept inline text sources",
            ));
        }
        FileSource::ProviderReference { .. } => {
            return Err(ModelError::unsupported(
                "Cohere image provider references are unsupported",
            ));
        }
    };
    let detail = file
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("cohere.image_detail"))
        .and_then(JsonValue::as_str)
        .or(default_detail);
    Ok(body::UserPart::ImageUrl {
        image_url: body::ImageUrl { url, detail },
    })
}

fn tool_result(result: &ToolResultPart) -> Result<body::Message<'_>, ModelError> {
    let content = match &result.content {
        ToolContent::Text(value) => body::ToolResultContent::Text(body::TextValue::Str(value)),
        ToolContent::LoadTools { text: value, .. } => {
            body::ToolResultContent::Text(body::TextValue::Str(value))
        }
        ToolContent::Json(value) => body::ToolResultContent::Text(body::TextValue::Json(value)),
        ToolContent::Mixed(values) => body::ToolResultContent::Parts(
            values
                .iter()
                .map(|value| {
                    body::ToolResultPart::from_value(value).ok_or_else(|| {
                        ModelError::unsupported(
                            "Cohere does not support files in tool-result history",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ToolContent::Denied { reason } => body::ToolResultContent::Text(body::TextValue::Str(
            reason.as_deref().unwrap_or("Tool call execution denied."),
        )),
    };
    Ok(body::Message::Tool(body::ToolMessage::new(
        &result.tool_call_id,
        content,
    )))
}

/// Encodes normalized assistant parts: one assistant message (when it has content) followed by
/// any inline tool results.
fn normalized_assistant<'a>(
    parts: &'a [AssistantPart],
    include_thinking: bool,
    messages: &mut Vec<body::Message<'a>>,
) -> Result<(), ModelError> {
    let mut message = body::AssistantMessage::new();
    let mut inline_results = Vec::new();
    for part in parts {
        match part {
            AssistantPart::Text(text) => message
                .content
                .push(body::AssistantContent::Text { text: &text.text }),
            AssistantPart::Reasoning(reasoning) => {
                if body::is_tool_plan(reasoning) {
                    message.tool_plan.0.push(&reasoning.text);
                } else if include_thinking {
                    message.content.push(body::AssistantContent::Thinking {
                        thinking: &reasoning.text,
                    });
                }
            }
            AssistantPart::ToolCall(call) => {
                message.tool_calls.push(body::ToolCall::from_part(call))
            }
            AssistantPart::Source(source) => {
                let metadata = source.metadata.as_ref().ok_or_else(|| {
                    ModelError::invalid_request(
                        "Cohere assistant sources require provider citation metadata",
                    )
                })?;
                message.citations.push(metadata);
            }
            AssistantPart::ToolResult(result) => inline_results.push(tool_result(result)?),
            AssistantPart::Custom(part) if part.kind == REPLAY_FINGERPRINT_KIND => {}
            AssistantPart::File(_) => {
                return Err(ModelError::unsupported(
                    "Cohere cannot encode assistant file parts",
                ));
            }
            AssistantPart::ToolApproval(_) => {
                return Err(ModelError::unsupported(
                    "Cohere cannot encode tool-approval parts in assistant history",
                ));
            }
            AssistantPart::Custom(_) => {
                return Err(ModelError::unsupported(
                    "Cohere cannot encode custom assistant parts",
                ));
            }
        }
    }
    if message.has_content() {
        messages.push(body::Message::Assistant(message));
    }
    messages.extend(inline_results);
    Ok(())
}

struct Live {
    bytes: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    parser: SseParser,
    state: CohereState,
    queue: VecDeque<StreamItem>,
    events: VecDeque<SseEvent>,
    pending_error: Option<ModelError>,
    deadline: oven_sdk::provider_support::StreamReadDeadline<tokio::time::Sleep>,
    idle: Duration,
    count: u64,
    eof: bool,
    request_id: Option<String>,
    response_headers: HeaderMap,
    saw_event: bool,
    include_raw: bool,
}

async fn read_live(live: &mut Live) -> Result<(), ModelError> {
    if live.events.is_empty() {
        let next = match live
            .deadline
            .next(live.bytes.as_mut(), |timer| {
                timer.reset(tokio::time::Instant::now() + live.idle);
            })
            .await
        {
            oven_sdk::provider_support::StreamRead::Aborted => {
                return Err(ModelError::abort("Cohere stream was aborted")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count));
            }
            oven_sdk::provider_support::StreamRead::TimedOut => {
                return Err(ModelError::timeout("Cohere stream idle timeout")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count));
            }
            oven_sdk::provider_support::StreamRead::Item(value) => value,
        };
        match next {
            Some(Ok(chunk)) => {
                live.count = live.count.saturating_add(chunk.len() as u64);
                live.parser.feed_into(&chunk, &mut live.events)?;
            }
            Some(Err(_)) => {
                return Err(ModelError::transport("Cohere stream read failed")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count));
            }
            None => {
                live.eof = true;
                live.events.extend(live.parser.finish()?);
            }
        }
    }
    while let Some(event) = live.events.pop_front() {
        if event.data.is_empty() {
            continue;
        }
        let value: JsonValue = serde_json::from_str(&event.data).map_err(|_| {
            ModelError::invalid_response("Cohere SSE event is invalid JSON")
                .with_stage(ErrorStage::StreamDecode)
                .with_bytes_received(live.count)
        })?;
        let kind = value.get("type").and_then(JsonValue::as_str).unwrap_or("");
        if live.include_raw {
            live.queue.push_back(Ok(StreamPart::Raw {
                value: value.clone(),
            }));
        }
        if kind == "message-end"
            && value
                .pointer("/delta/finish_reason")
                .and_then(JsonValue::as_str)
                == Some("ERROR")
        {
            let error = classify_error(
                500,
                event.data.as_bytes(),
                live.request_id.clone(),
                ErrorStage::StreamEvent,
                live.count,
                &live.response_headers,
            );
            live.state.in_band_error(error, &mut live.queue)?;
            live.eof = true;
            live.events.clear();
            return Ok(());
        }
        live.saw_event = true;
        live.state.apply(value, &mut live.queue, live.count)?;
        if live.state.done {
            live.eof = true;
            live.events.clear();
            return Ok(());
        }
        if !live.queue.is_empty() {
            return Ok(());
        }
    }
    if live.eof && !live.state.done {
        return Err(ModelError::unexpected_eof(if live.saw_event {
            "Cohere stream ended before message-end"
        } else {
            "Cohere stream ended before any event"
        })
        .with_bytes_received(live.count));
    }
    Ok(())
}

#[derive(Default)]
struct ContentState {
    kind: String,
    text: String,
    open: bool,
}
#[derive(Default)]
struct ToolState {
    id: String,
    name: Option<String>,
    arguments: String,
    open: bool,
}

struct CohereState {
    adapter: AdapterId,
    policy: ReplayPolicy,
    replay_capability: ReplayCapability,
    replay_reasoning: bool,
    binding: JsonValue,
    scope: NativeContextScope,
    started: bool,
    done: bool,
    contents: BTreeMap<usize, ContentState>,
    tools: BTreeMap<usize, ToolState>,
    tool_ids: BTreeSet<String>,
    citations: BTreeMap<usize, JsonValue>,
    open_citations: BTreeSet<usize>,
    tool_plan: String,
    tool_plan_open: bool,
    generation_id: Option<String>,
}

impl CohereState {
    fn new(
        adapter: AdapterId,
        policy: ReplayPolicy,
        replay_capability: ReplayCapability,
        replay_reasoning: bool,
        binding: JsonValue,
        scope: NativeContextScope,
    ) -> Self {
        Self {
            adapter,
            policy,
            replay_capability,
            replay_reasoning,
            binding,
            scope,
            started: false,
            done: false,
            contents: BTreeMap::new(),
            tools: BTreeMap::new(),
            tool_ids: BTreeSet::new(),
            citations: BTreeMap::new(),
            open_citations: BTreeSet::new(),
            tool_plan: String::new(),
            tool_plan_open: false,
            generation_id: None,
        }
    }

    fn apply(
        &mut self,
        value: JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self.done {
            return Err(event_error("Cohere event after message-end", bytes));
        }
        let kind = value.get("type").and_then(JsonValue::as_str).unwrap_or("");
        if !self.started && kind != "message-start" {
            return Err(event_error(
                "Cohere stream must start with message-start",
                bytes,
            ));
        }
        match kind {
            "message-start" => {
                if self.started {
                    return Err(event_error("duplicate Cohere message-start", bytes));
                }
                self.started = true;
                self.generation_id = value
                    .get("id")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
            }
            "content-start" => {
                let index = index(&value, bytes)?;
                if self.contents.contains_key(&index) {
                    return Err(event_error("duplicate Cohere content-start index", bytes));
                }
                let kind = value
                    .pointer("/delta/message/content/type")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| event_error("Cohere content-start is missing type", bytes))?
                    .to_owned();
                let id = format!("content-{index}");
                let metadata = part_kind_metadata(&kind);
                queue.push_back(Ok(if kind == "thinking" {
                    StreamPart::ReasoningStart { id, metadata }
                } else if kind == "text" {
                    StreamPart::TextStart { id, metadata }
                } else {
                    return Err(event_error("unsupported Cohere content type", bytes));
                }));
                self.contents.insert(
                    index,
                    ContentState {
                        kind,
                        text: String::new(),
                        open: true,
                    },
                );
            }
            "content-delta" => {
                let index = index(&value, bytes)?;
                let state = self
                    .contents
                    .get_mut(&index)
                    .filter(|state| state.open)
                    .ok_or_else(|| event_error("Cohere content-delta has no open block", bytes))?;
                let delta = if state.kind == "thinking" {
                    value.pointer("/delta/message/content/thinking")
                } else {
                    value.pointer("/delta/message/content/text")
                }
                .and_then(JsonValue::as_str)
                .ok_or_else(|| event_error("Cohere content-delta is missing text", bytes))?;
                state.text.push_str(delta);
                let id = format!("content-{index}");
                queue.push_back(Ok(if state.kind == "thinking" {
                    StreamPart::ReasoningDelta {
                        id,
                        delta: delta.into(),
                        metadata: part_kind_metadata("thinking"),
                    }
                } else {
                    StreamPart::TextDelta {
                        id,
                        delta: delta.into(),
                        metadata: None,
                    }
                }));
            }
            "content-end" => {
                let index = index(&value, bytes)?;
                let state = self
                    .contents
                    .get_mut(&index)
                    .filter(|state| state.open)
                    .ok_or_else(|| event_error("Cohere content-end has no open block", bytes))?;
                state.open = false;
                let id = format!("content-{index}");
                queue.push_back(Ok(if state.kind == "thinking" {
                    StreamPart::ReasoningEnd {
                        id,
                        metadata: part_kind_metadata("thinking"),
                    }
                } else {
                    StreamPart::TextEnd { id, metadata: None }
                }));
            }
            "tool-plan-delta" => {
                if !self.tool_plan_open {
                    self.tool_plan_open = true;
                    queue.push_back(Ok(StreamPart::ReasoningStart {
                        id: "tool-plan".into(),
                        metadata: part_kind_metadata("tool_plan"),
                    }));
                }
                let delta = value
                    .pointer("/delta/message/tool_plan")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| event_error("Cohere tool-plan-delta is missing text", bytes))?;
                self.tool_plan.push_str(delta);
                queue.push_back(Ok(StreamPart::ReasoningDelta {
                    id: "tool-plan".into(),
                    delta: delta.into(),
                    metadata: part_kind_metadata("tool_plan"),
                }));
            }
            "tool-call-start" => {
                self.close_tool_plan(queue);
                let index = index(&value, bytes)?;
                if self.tools.contains_key(&index) {
                    return Err(event_error("duplicate Cohere tool-call-start index", bytes));
                }
                let provider_id = value
                    .pointer("/delta/message/tool_calls/id")
                    .and_then(JsonValue::as_str)
                    .filter(|value| !value.is_empty());
                let id = reserve_tool_id(&mut self.tool_ids, provider_id, index as u64);
                let name = value
                    .pointer("/delta/message/tool_calls/function/name")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                let arguments = value
                    .pointer("/delta/message/tool_calls/function/arguments")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("")
                    .to_owned();
                queue.push_back(Ok(StreamPart::ToolCallStart {
                    id: id.clone(),
                    name: name.clone().unwrap_or_default(),
                    metadata: None,
                }));
                if !arguments.is_empty() {
                    queue.push_back(Ok(StreamPart::ToolCallDelta {
                        id: id.clone(),
                        delta: arguments.clone(),
                        metadata: None,
                    }));
                }
                self.tools.insert(
                    index,
                    ToolState {
                        id,
                        name,
                        arguments,
                        open: true,
                    },
                );
            }
            "tool-call-delta" => {
                let index = index(&value, bytes)?;
                let state = self
                    .tools
                    .get_mut(&index)
                    .filter(|state| state.open)
                    .ok_or_else(|| event_error("Cohere tool-call-delta has no open call", bytes))?;
                let delta = value
                    .pointer("/delta/message/tool_calls/function/arguments")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| {
                        event_error("Cohere tool-call-delta is missing arguments", bytes)
                    })?;
                state.arguments.push_str(delta);
                queue.push_back(Ok(StreamPart::ToolCallDelta {
                    id: state.id.clone(),
                    delta: delta.into(),
                    metadata: None,
                }));
            }
            "tool-call-end" => {
                let index = index(&value, bytes)?;
                let state = self
                    .tools
                    .get_mut(&index)
                    .filter(|state| state.open)
                    .ok_or_else(|| event_error("Cohere tool-call-end has no open call", bytes))?;
                state.open = false;
                let parsed =
                    if state.arguments.trim() == "null" || state.arguments.trim().is_empty() {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str(&state.arguments).map_err(|_| {
                            ModelError::new(
                                ModelErrorKind::InvalidToolInput,
                                "Cohere tool arguments are invalid JSON",
                            )
                            .with_stage(ErrorStage::StreamFinalize)
                            .with_bytes_received(bytes)
                        })?
                    };
                if !parsed.is_object() {
                    return Err(ModelError::new(
                        ModelErrorKind::InvalidToolInput,
                        "Cohere tool arguments must be a JSON object",
                    )
                    .with_stage(ErrorStage::StreamFinalize)
                    .with_bytes_received(bytes));
                }
                queue.push_back(Ok(StreamPart::ToolCallEnd {
                    id: state.id.clone(),
                    metadata: None,
                }));
                let mut call = oven_sdk::ToolCallPart::new(
                    &state.id,
                    state.name.clone().unwrap_or_default(),
                    parsed,
                );
                call.raw_input = Some(state.arguments.clone());
                if state.name.is_none() {
                    call.metadata = Some(BTreeMap::from([(
                        "cohere.name_missing".into(),
                        JsonValue::Bool(true),
                    )]));
                }
                queue.push_back(Ok(StreamPart::ToolCall { tool_call: call }));
            }
            "citation-start" => {
                let index = index(&value, bytes)?;
                if self.citations.contains_key(&index) {
                    return Err(event_error("duplicate Cohere citation index", bytes));
                }
                let citation = value
                    .pointer("/delta/message/citations")
                    .cloned()
                    .ok_or_else(|| {
                        event_error("Cohere citation-start is missing citation", bytes)
                    })?;
                let source = source_from_citation(&citation)?;
                queue.push_back(Ok(StreamPart::Source { source }));
                self.citations.insert(index, citation);
                if !self.open_citations.insert(index) {
                    return Err(event_error(
                        "Cohere citation-start repeated an open citation",
                        bytes,
                    ));
                }
            }
            "citation-end" => {
                let index = index(&value, bytes)?;
                if !self.open_citations.remove(&index) {
                    return Err(event_error(
                        "Cohere citation-end has no citation-start",
                        bytes,
                    ));
                }
            }
            "message-end" => self.finish(value, queue, bytes)?,
            _ => queue.push_back(Ok(StreamPart::ProviderEvent {
                name: format!("cohere.{kind}"),
                data: value,
            })),
        }
        Ok(())
    }

    fn close_tool_plan(&mut self, queue: &mut VecDeque<StreamItem>) {
        if self.tool_plan_open {
            self.tool_plan_open = false;
            queue.push_back(Ok(StreamPart::ReasoningEnd {
                id: "tool-plan".into(),
                metadata: part_kind_metadata("tool_plan"),
            }));
        }
    }

    fn finish(
        &mut self,
        value: JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        self.close_tool_plan(queue);
        if self.contents.values().any(|state| state.open)
            || self.tools.values().any(|state| state.open)
            || !self.open_citations.is_empty()
        {
            return Err(event_error(
                "Cohere message-end arrived with open blocks",
                bytes,
            ));
        }
        let raw_reason = value
            .pointer("/delta/finish_reason")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| event_error("Cohere message-end is missing finish reason", bytes))?;
        let usage_value = value
            .pointer("/delta/usage")
            .cloned()
            .unwrap_or(JsonValue::Null);
        let usage = usage_from(&usage_value);
        let finish_reason = match raw_reason {
            "COMPLETE" | "STOP_SEQUENCE" => FinishReason::Stop,
            "MAX_TOKENS" => FinishReason::Length,
            "TOOL_CALL" => FinishReason::ToolCalls,
            "ERROR" => FinishReason::Error,
            "TIMEOUT" => FinishReason::Timeout,
            other => FinishReason::other(other),
        };
        let message = self.message(self.replay_reasoning);
        let mut finish = Finish::new(usage, finish_reason);
        if let Some(id) = &self.generation_id {
            finish
                .response_metadata
                .insert("cohere.generation_id".into(), id.clone().into());
        }
        finish
            .provider_metadata
            .insert("cohere.finish_reason".into(), raw_reason.into());
        if self.policy != ReplayPolicy::Never {
            let fingerprint = replay_fingerprint(&message)
                .ok_or_else(|| event_error("could not fingerprint Cohere replay message", bytes))?;
            let payload = serde_json::json!({"format":REPLAY_FORMAT,"binding":self.binding,"message":message,"fingerprint":fingerprint});
            match NativeReplayArtifact::capture(self.adapter.clone(), self.scope.clone(), payload) {
                Ok(artifact) => {
                    queue.push_back(Ok(StreamPart::Custom {
                        part: CustomPart::new(REPLAY_FINGERPRINT_KIND, fingerprint.into()),
                    }));
                    finish.native_replay = Some(artifact);
                }
                Err(ReplayPayloadError::TooLarge { .. })
                    if self.replay_capability == ReplayCapability::Optional =>
                {
                    finish.provider_metadata.insert(
                        "cohere.replay_capture".into(),
                        serde_json::json!({
                            "status":"discarded",
                            "reason":"payload_too_large"
                        }),
                    );
                    queue.push_back(Ok(StreamPart::ProviderEvent {
                        name: "cohere.replay.warning".into(),
                        data: serde_json::json!({
                            "message":"Cohere replay artifact was omitted because it exceeded the size limit"
                        }),
                    }));
                }
                Err(_) => {
                    return Err(ModelError::replay(
                        "Cohere replay artifact could not be captured within its size contract",
                    )
                    .with_stage(ErrorStage::ReplayEncode));
                }
            }
        }
        queue.push_back(Ok(StreamPart::Finish { finish }));
        self.done = true;
        Ok(())
    }

    fn message(&self, include_thinking: bool) -> JsonValue {
        let content = self
            .contents
            .values()
            .filter(|state| include_thinking || state.kind != "thinking")
            .map(|state| {
                if state.kind == "thinking" {
                    serde_json::json!({"type":"thinking","thinking":state.text})
                } else {
                    serde_json::json!({"type":"text","text":state.text})
                }
            })
            .collect::<Vec<_>>();
        let tools = self
            .tools
            .values()
            .map(|state| {
                let mut call = serde_json::json!({"id":state.id,"type":"function","function":{"arguments":state.arguments}});
                if let Some(name) = &state.name {
                    call["function"]["name"] = name.clone().into();
                }
                call
            })
            .collect::<Vec<_>>();
        let mut message = serde_json::json!({"role":"assistant"});
        if !content.is_empty() {
            message["content"] = JsonValue::Array(content);
        }
        if !self.tool_plan.is_empty() {
            message["tool_plan"] = self.tool_plan.clone().into();
        }
        if !tools.is_empty() {
            message["tool_calls"] = JsonValue::Array(tools);
        }
        if !self.citations.is_empty() {
            message["citations"] = JsonValue::Array(self.citations.values().cloned().collect());
        }
        message
    }

    fn in_band_error(
        &mut self,
        error: ModelError,
        queue: &mut VecDeque<StreamItem>,
    ) -> Result<(), ModelError> {
        if self.tools.values().any(|state| state.open) {
            return Err(
                ModelError::invalid_response("Cohere error interrupted an open tool call")
                    .with_stage(ErrorStage::StreamEvent),
            );
        }
        self.close_tool_plan(queue);
        for (index, state) in &mut self.contents {
            if state.open {
                state.open = false;
                let id = format!("content-{index}");
                queue.push_back(Ok(if state.kind == "thinking" {
                    StreamPart::ReasoningEnd {
                        id,
                        metadata: part_kind_metadata("thinking"),
                    }
                } else {
                    StreamPart::TextEnd { id, metadata: None }
                }));
            }
        }
        queue.push_back(Ok(StreamPart::Error { error }));
        queue.push_back(Ok(StreamPart::Finish {
            finish: Finish::new(Usage::default(), FinishReason::Error),
        }));
        self.done = true;
        Ok(())
    }
}

fn source_from_citation(citation: &JsonValue) -> Result<SourcePart, ModelError> {
    let mut source = SourcePart::new();
    source.excerpt = citation
        .get("text")
        .and_then(JsonValue::as_str)
        .map(str::to_owned);
    let first = citation
        .get("sources")
        .and_then(JsonValue::as_array)
        .and_then(|values| values.first());
    source.id = first
        .and_then(|value| value.get("id"))
        .and_then(JsonValue::as_str)
        .map(str::to_owned);
    source.title = first
        .and_then(|value| value.pointer("/document/title"))
        .and_then(JsonValue::as_str)
        .map(str::to_owned);
    source.url = first
        .and_then(|value| value.pointer("/document/url"))
        .and_then(JsonValue::as_str)
        .and_then(|value| url::Url::parse(value).ok());
    source.media_type = Some("text/plain".into());
    source.metadata = citation
        .as_object()
        .map(|object| object.clone().into_iter().collect());
    Ok(source)
}

fn part_kind_metadata(kind: &str) -> oven_sdk::PartMetadata {
    Some(BTreeMap::from([("cohere.kind".into(), kind.into())]))
}

fn index(value: &JsonValue, bytes: u64) -> Result<usize, ModelError> {
    value
        .get("index")
        .and_then(JsonValue::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value < 1024)
        .ok_or_else(|| event_error("Cohere event has an invalid index", bytes))
}

fn event_error(message: &str, bytes: u64) -> ModelError {
    ModelError::invalid_response(message)
        .with_stage(ErrorStage::StreamEvent)
        .with_bytes_received(bytes)
}

fn usage_from(value: &JsonValue) -> Usage {
    let input = value
        .pointer("/tokens/input_tokens")
        .and_then(JsonValue::as_u64);
    let output = value
        .pointer("/tokens/output_tokens")
        .and_then(JsonValue::as_u64);
    let cached = value.get("cached_tokens").and_then(JsonValue::as_u64);
    Usage {
        input_tokens: input,
        input_tokens_no_cache: input
            .zip(cached)
            .map(|(input, cached)| input.saturating_sub(cached)),
        input_tokens_cache_read: cached,
        output_tokens: output,
        output_tokens_text: None,
        output_tokens_reasoning: None,
        raw: (!value.is_null()).then(|| value.clone()),
        ..Usage::default()
    }
}

fn replay_fingerprint(message: &JsonValue) -> Option<String> {
    serde_json::to_vec(message)
        .ok()
        .map(Sha256::digest)
        .map(|digest| URL_SAFE_NO_PAD.encode(digest))
}

fn reserve_tool_id(
    used: &mut BTreeSet<String>,
    provider_id: Option<&str>,
    fallback: u64,
) -> String {
    let base = provider_id
        .map(str::to_owned)
        .unwrap_or_else(|| format!("google-call-{fallback}"));
    if used.insert(base.clone()) {
        return base;
    }
    for suffix in 1_u64.. {
        let candidate = format!("{base}-{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("unbounded tool ID suffix space")
}

/// Validates a native replay artifact in place and returns the message to send: the stored
/// message itself, or its normalized form without thinking when reasoning replay is off.
fn decode_replay<'a>(
    artifact: &'a NativeReplayArtifact,
    normalized: &'a [AssistantPart],
    _binding: &JsonValue,
    replay_reasoning: bool,
) -> Option<body::Message<'a>> {
    let object = artifact.payload().as_object()?;
    if object.len() != 4
        || !["binding", "fingerprint", "format", "message"]
            .iter()
            .all(|key| object.contains_key(*key))
        || object.get("format")?.as_str()? != REPLAY_FORMAT
    {
        return None;
    }
    let binding = object.get("binding")?;
    binding.get("version")?.as_str()?;
    binding.get("sha256")?.as_str()?;
    let message = object.get("message")?;
    let fingerprint = object.get("fingerprint")?.as_str()?;
    let mut normalized_fingerprints = normalized.iter().filter_map(|part| match part {
        AssistantPart::Custom(part) if part.kind == REPLAY_FINGERPRINT_KIND => part.data.as_str(),
        _ => None,
    });
    if normalized_fingerprints.next() != Some(fingerprint)
        || normalized_fingerprints.next().is_some()
        || !fingerprint_matches(message, fingerprint)
    {
        return None;
    }
    // Parts normalized reconstruction cannot encode make the artifact unusable.
    if normalized.iter().any(|part| match part {
        AssistantPart::Source(source) => source.metadata.is_none(),
        AssistantPart::File(_) | AssistantPart::ToolApproval(_) => true,
        AssistantPart::Custom(part) => part.kind != REPLAY_FINGERPRINT_KIND,
        AssistantPart::ToolResult(result) => matches!(
            &result.content,
            ToolContent::Mixed(values)
                if values.iter().any(|value| matches!(value, ContentValue::File(_)))
        ),
        _ => false,
    }) {
        return None;
    }
    let full = body::assistant_message_matches(message, normalized, true);
    if !full && !body::assistant_message_matches(message, normalized, false) {
        return None;
    }
    if replay_reasoning {
        Some(body::Message::Native(message))
    } else {
        let mut messages = Vec::with_capacity(1);
        normalized_assistant(normalized, false, &mut messages).ok()?;
        messages.into_iter().next()
    }
}

/// Whether `fingerprint` is the replay fingerprint of `message`, hashing without buffering.
fn fingerprint_matches(message: &JsonValue, fingerprint: &str) -> bool {
    let mut hasher = body::HashWriter(Sha256::new());
    if serde_json::to_writer(&mut hasher, message).is_err() {
        return false;
    }
    let mut encoded = [0_u8; 43];
    URL_SAFE_NO_PAD
        .encode_slice(hasher.0.finalize(), &mut encoded)
        .is_ok_and(|written| written == encoded.len())
        && encoded.as_slice() == fingerprint.as_bytes()
}

fn validate_endpoint(endpoint: &ApiEndpoint) -> Result<(), ModelError> {
    let url = endpoint.as_url();
    let loopback = url.scheme() == "http"
        && url
            .host_str()
            .is_some_and(|host| matches!(host, "localhost" | "127.0.0.1" | "::1"));
    if (url.scheme() != "https" && !loopback)
        || url.query().is_some()
        || !url.path().ends_with("/v2/chat")
    {
        return Err(ModelError::invalid_request(
            "Cohere endpoint must be an HTTPS full /v2/chat URL without query (HTTP loopback is allowed for tests)",
        ));
    }
    Ok(())
}

fn validate_capabilities(capabilities: &ModelCapabilities) -> Result<(), ModelError> {
    capabilities.validate()?;
    if capabilities.cancellation != CancellationCapability::LocalOnly {
        return Err(ModelError::invalid_request(
            "Cohere declares local-only cancellation",
        ));
    }
    if capabilities.compaction != CompactionCapability::Unsupported {
        return Err(ModelError::invalid_request(
            "Cohere does not implement provider-native context compaction",
        ));
    }
    if capabilities.deferred_tools != oven_sdk::DeferredToolsCapability::Unsupported {
        return Err(ModelError::invalid_request(
            "Cohere does not implement deferred tool loading",
        ));
    }
    let text = oven_sdk::Modality::text();
    if !capabilities.modalities.input.contains(&text)
        || capabilities.modalities.output != [text.clone()].into_iter().collect()
    {
        return Err(ModelError::invalid_request(
            "Cohere v2 Chat requires text input and text-only output",
        ));
    }
    for modality in &capabilities.modalities.input {
        if !matches!(modality.as_str(), "text" | "image") {
            return Err(ModelError::invalid_request(
                "Cohere v2 Chat implements only text and image input modalities",
            ));
        }
    }
    if capabilities.features.contains(Capability::PROVIDER_TOOLS) {
        return Err(ModelError::invalid_request(
            "Cohere adapter does not implement provider-hosted tools",
        ));
    }
    for (modality, support) in &capabilities.media.input {
        if modality.as_str() != "image"
            || !support.media_types.iter().all(|value| {
                matches!(
                    value.as_str(),
                    "image/png" | "image/jpeg" | "image/webp" | "image/gif"
                )
            })
            || support.sources.intersects(
                MediaSourceSupport::INLINE_TEXT | MediaSourceSupport::PROVIDER_REFERENCE,
            )
        {
            return Err(ModelError::invalid_request(
                "Cohere image declarations exceed PNG/JPEG/WEBP/non-animated-GIF byte-or-HTTP(S)-URL support",
            ));
        }
    }
    if capabilities.replay.reasoning && !capabilities.features.contains(Capability::REASONING) {
        return Err(ModelError::invalid_request(
            "Cohere reasoning replay requires reasoning capability",
        ));
    }
    Ok(())
}

fn validate_settings(
    settings: &CohereSettings,
    capabilities: &ModelCapabilities,
) -> Result<(), ModelError> {
    validate_open_label(settings.safety_mode.as_deref(), "safety mode")?;
    if settings.stop_sequences.len() > 5 {
        return Err(ModelError::invalid_request(
            "Cohere accepts at most five stop sequences",
        ));
    }
    if settings.top_k.is_some_and(|value| value > 500) {
        return Err(ModelError::invalid_request(
            "Cohere top_k must not exceed 500",
        ));
    }
    for (name, value) in [
        ("frequency penalty", settings.frequency_penalty),
        ("presence penalty", settings.presence_penalty),
    ] {
        if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
            return Err(ModelError::invalid_request(format!(
                "Cohere {name} must be finite and between 0 and 1"
            )));
        }
    }
    for (effort, thinking) in settings
        .reasoning_effort
        .iter()
        .map(|(effort, thinking)| (Some(effort.as_str()), thinking))
        .chain(settings.thinking.iter().map(|thinking| (None, thinking)))
    {
        if let Some(effort) = effort {
            validate_open_label(Some(effort), "reasoning effort mapping")?;
        }
        if thinking.token_budget == Some(0) {
            return Err(ModelError::invalid_request(
                "Cohere thinking token budget must be positive",
            ));
        }
        if !thinking.enabled && thinking.token_budget.is_some() {
            return Err(ModelError::invalid_request(
                "Cohere disabled thinking cannot specify a token budget",
            ));
        }
        if !capabilities.features.contains(Capability::REASONING) {
            return Err(ModelError::invalid_request(
                "Cohere thinking settings require reasoning capability",
            ));
        }
    }
    Ok(())
}

fn validate_open_label(value: Option<&str>, name: &str) -> Result<(), ModelError> {
    if value.is_some_and(|value| value.trim().is_empty() || value.chars().any(char::is_control)) {
        Err(ModelError::invalid_request(format!(
            "Cohere {name} must be a non-empty open string without control characters"
        )))
    } else {
        Ok(())
    }
}

fn validate_cohere_schema(
    schema: &JsonValue,
    require_object_root: bool,
) -> Result<usize, ModelError> {
    validate_cohere_schema_node(schema, require_object_root, 0)
}

fn validate_cohere_schema_node(
    schema: &JsonValue,
    require_object: bool,
    depth: usize,
) -> Result<usize, ModelError> {
    if depth > 128 {
        return Err(ModelError::invalid_request(
            "Cohere JSON Schema nesting exceeds 128 levels",
        ));
    }
    let object = schema.as_object().ok_or_else(|| {
        ModelError::invalid_request("Cohere JSON Schema nodes must be JSON objects")
    })?;
    const ALLOWED: &[&str] = &[
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "const",
        "description",
        "enum",
        "format",
        "items",
        "pattern",
        "properties",
        "required",
        "type",
    ];
    if object.keys().any(|key| !ALLOWED.contains(&key.as_str())) {
        return Err(ModelError::unsupported(
            "Cohere JSON Schema contains an unsupported keyword",
        ));
    }
    let schema_type = object
        .get("type")
        .map(|value| {
            value.as_str().ok_or_else(|| {
                ModelError::unsupported("Cohere JSON Schema type must be one supported string")
            })
        })
        .transpose()?;
    if schema_type.is_some_and(|value| {
        !matches!(
            value,
            "object" | "array" | "string" | "integer" | "number" | "boolean"
        )
    }) {
        return Err(ModelError::unsupported(
            "Cohere JSON Schema contains an unsupported type",
        ));
    }
    if require_object && schema_type != Some("object") {
        return Err(ModelError::invalid_request(
            "Cohere JSON Schema roots must have type object",
        ));
    }
    if let Some(description) = object.get("description")
        && !description.is_string()
    {
        return Err(ModelError::invalid_request(
            "Cohere JSON Schema descriptions must be strings",
        ));
    }
    if let Some(reference) = object.get("$ref") {
        let reference = reference
            .as_str()
            .filter(|value| value.starts_with("#/$defs/"));
        if reference.is_none() {
            return Err(ModelError::unsupported(
                "Cohere JSON Schema supports only local $defs references",
            ));
        }
    }
    if let Some(values) = object.get("enum")
        && values.as_array().is_none_or(Vec::is_empty)
    {
        return Err(ModelError::invalid_request(
            "Cohere JSON Schema enum must be a non-empty array",
        ));
    }
    if let Some(pattern) = object.get("pattern") {
        if schema_type != Some("string") {
            return Err(ModelError::invalid_request(
                "Cohere JSON Schema pattern requires type string",
            ));
        }
        let pattern = pattern.as_str().ok_or_else(|| {
            ModelError::invalid_request("Cohere JSON Schema pattern must be a string")
        })?;
        if ["^", "$", "?=", "?!"]
            .iter()
            .any(|unsupported| pattern.contains(unsupported))
        {
            return Err(ModelError::unsupported(
                "Cohere JSON Schema pattern uses an unsupported expression",
            ));
        }
    }
    if let Some(format) = object.get("format") {
        if schema_type != Some("string") {
            return Err(ModelError::invalid_request(
                "Cohere JSON Schema format requires type string",
            ));
        }
        if !matches!(
            format.as_str(),
            Some("date-time" | "uuid" | "date" | "time")
        ) {
            return Err(ModelError::unsupported(
                "Cohere JSON Schema format is unsupported",
            ));
        }
    }

    let mut fields = 0_usize;
    match schema_type {
        Some("object") => {
            let properties = object
                .get("properties")
                .and_then(JsonValue::as_object)
                .ok_or_else(|| {
                    ModelError::invalid_request("Cohere JSON Schema objects require properties")
                })?;
            let required = object
                .get("required")
                .and_then(JsonValue::as_array)
                .filter(|values| !values.is_empty())
                .ok_or_else(|| {
                    ModelError::invalid_request(
                        "Cohere JSON Schema objects require at least one required property",
                    )
                })?;
            let mut names = BTreeSet::new();
            for value in required {
                let name = value.as_str().ok_or_else(|| {
                    ModelError::invalid_request(
                        "Cohere JSON Schema required entries must be strings",
                    )
                })?;
                if !properties.contains_key(name) || !names.insert(name) {
                    return Err(ModelError::invalid_request(
                        "Cohere JSON Schema required entries must uniquely name declared properties",
                    ));
                }
            }
            fields = fields.saturating_add(properties.len());
            for property in properties.values() {
                fields =
                    fields.saturating_add(validate_cohere_schema_node(property, false, depth + 1)?);
            }
        }
        Some("array") => {
            if let Some(items) = object.get("items") {
                fields =
                    fields.saturating_add(validate_cohere_schema_node(items, false, depth + 1)?);
            }
        }
        _ => {
            if object.contains_key("properties")
                || object.contains_key("required")
                || object.contains_key("items")
                || object.contains_key("additionalProperties")
            {
                return Err(ModelError::invalid_request(
                    "Cohere JSON Schema structural keywords do not match the declared type",
                ));
            }
        }
    }
    if let Some(definitions) = object.get("$defs") {
        let definitions = definitions.as_object().ok_or_else(|| {
            ModelError::invalid_request("Cohere JSON Schema $defs must be an object")
        })?;
        for definition in definitions.values() {
            fields =
                fields.saturating_add(validate_cohere_schema_node(definition, false, depth + 1)?);
        }
    }
    if let Some(branches) = object.get("anyOf") {
        let branches = branches
            .as_array()
            .filter(|branches| !branches.is_empty())
            .ok_or_else(|| {
                ModelError::invalid_request("Cohere JSON Schema anyOf must be a non-empty array")
            })?;
        for branch in branches {
            fields = fields.saturating_add(validate_cohere_schema_node(branch, false, depth + 1)?);
        }
    }
    if let Some(additional) = object.get("additionalProperties") {
        match additional {
            JsonValue::Bool(_) => {}
            JsonValue::Object(_) => {
                fields = fields.saturating_add(validate_cohere_schema_node(
                    additional,
                    false,
                    depth + 1,
                )?);
            }
            _ => {
                return Err(ModelError::invalid_request(
                    "Cohere JSON Schema additionalProperties must be boolean or a schema",
                ));
            }
        }
    }
    Ok(fields)
}

fn reject_protected_headers(headers: &HeaderMap) -> Result<(), ModelError> {
    if ["host", "content-type", "content-length"]
        .iter()
        .any(|name| headers.contains_key(*name))
    {
        Err(ModelError::invalid_request(
            "Cohere authentication and transport headers are protected",
        ))
    } else {
        Ok(())
    }
}

fn replay_seed(
    provider: &ProviderConfig<CohereAuth>,
    model: &oven_sdk::ModelDeclaration,
    settings: &CohereSettings,
) -> Result<Sha256, ModelError> {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, "version", REPLAY_SCOPE_VERSION.as_bytes());
    hash_field(&mut hasher, "provider", provider.id.as_str().as_bytes());
    hash_field(&mut hasher, "model", model.id.as_str().as_bytes());
    hash_field(
        &mut hasher,
        "endpoint",
        provider.api.as_url().as_str().as_bytes(),
    );
    hash_json(&mut hasher, "capabilities", &model.capabilities)?;
    hash_field(&mut hasher, "strict_tools", &[settings.strict_tools as u8]);
    hash_duration(&mut hasher, "connect_timeout", settings.timeouts.connect);
    hash_duration(&mut hasher, "headers_timeout", settings.timeouts.headers);
    hash_duration(
        &mut hasher,
        "stream_idle_timeout",
        settings.timeouts.stream_idle,
    );
    hash_optional(&mut hasher, "safety_mode", settings.safety_mode.as_deref());
    hash_json(&mut hasher, "thinking", &settings.thinking)?;
    hash_json(&mut hasher, "reasoning_effort", &settings.reasoning_effort)?;
    hash_json(&mut hasher, "top_k", &settings.top_k)?;
    hash_json(&mut hasher, "seed", &settings.seed)?;
    hash_json(
        &mut hasher,
        "frequency_penalty",
        &settings.frequency_penalty,
    )?;
    hash_json(&mut hasher, "presence_penalty", &settings.presence_penalty)?;
    hash_json(&mut hasher, "stop_sequences", &settings.stop_sequences)?;
    hash_json(&mut hasher, "priority", &settings.priority)?;
    Ok(hasher)
}

fn hash_json(hasher: &mut Sha256, tag: &str, value: &impl Serialize) -> Result<(), ModelError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| ModelError::invalid_request("could not encode Cohere replay scope inputs"))?;
    hash_field(hasher, tag, &bytes);
    Ok(())
}
fn hash_optional(hasher: &mut Sha256, tag: &str, value: Option<&str>) {
    hash_field(hasher, tag, value.unwrap_or("").as_bytes());
}
fn hash_duration(hasher: &mut Sha256, tag: &str, value: Duration) {
    hash_field(hasher, tag, &value.as_nanos().to_be_bytes());
}
fn hash_headers(hasher: &mut Sha256, headers: &HeaderMap) {
    let mut names = headers.keys().collect::<Vec<_>>();
    names.sort_unstable_by_key(|name| name.as_str());
    for name in names {
        hash_field(hasher, "header_name", name.as_str().as_bytes());
        for value in headers.get_all(name) {
            hash_field(hasher, "header_value", value.as_bytes());
        }
    }
}
fn hash_field(hasher: &mut Sha256, tag: &str, value: &[u8]) {
    hasher.update((tag.len() as u64).to_be_bytes());
    hasher.update(tag.as_bytes());
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn response_head(response: &reqwest::Response) -> ResponseHead {
    ResponseHead {
        http_status: Some(response.status().as_u16()),
        request_id: ["x-request-id", "request-id"].into_iter().find_map(|name| {
            response
                .headers()
                .get(name)?
                .to_str()
                .ok()
                .map(str::to_owned)
        }),
        response_metadata: Default::default(),
    }
}

async fn read_error_body(
    response: reqwest::Response,
    abort: &AbortSignal,
    idle: Duration,
) -> Result<(Vec<u8>, u64), ModelError> {
    oven_sdk::provider_support::read_bounded_body(
        response.bytes_stream(),
        abort,
        oven_sdk::provider_support::BodyReadConfig {
            cap: SanitizedBody::MAX_BYTES + 1,
            limit: oven_sdk::provider_support::BodyLimit::Truncate,
            stage: ErrorStage::ResponseBody,
            timeout_message: "Cohere error response body idle timeout",
            abort_message: "Cohere error response body read was aborted",
            read_message: "Cohere error response body read failed",
            overflow_message: "Cohere error response body byte count overflowed",
        },
        tokio::time::sleep(idle),
        move |timer| timer.reset(tokio::time::Instant::now() + idle),
    )
    .await
}

fn classify_error(
    status: u16,
    body: &[u8],
    request_id: Option<String>,
    stage: ErrorStage,
    bytes: u64,
    headers: &HeaderMap,
) -> ModelError {
    let value: JsonValue = serde_json::from_slice(body).unwrap_or(JsonValue::Null);
    let message = value
        .get("message")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    let lower = message.to_ascii_lowercase();
    let mut error = match status {
        401 => ModelError::new(ModelErrorKind::Auth, "Cohere request failed"),
        403 => ModelError::new(ModelErrorKind::PermissionDenied, "Cohere request failed"),
        404 => ModelError::new(ModelErrorKind::ModelNotFound, "Cohere request failed"),
        408 | 504 => ModelError::timeout("Cohere request failed"),
        429 => ModelError::rate_limited("Cohere request failed"),
        498 => ModelError::new(ModelErrorKind::ContentFilter, "Cohere request failed"),
        499 => ModelError::abort("Cohere request was cancelled"),
        501 => ModelError::unsupported("Cohere feature is not implemented"),
        503 => {
            ModelError::new(ModelErrorKind::Overload, "Cohere request failed").with_retryable(true)
        }
        500..=599 => ModelError::provider("Cohere request failed").with_retryable(true),
        _ if lower.contains("context") || lower.contains("too many tokens") => {
            ModelError::new(ModelErrorKind::ContextLength, "Cohere request failed")
        }
        _ => ModelError::invalid_request("Cohere request failed"),
    }
    .with_http_status(status)
    .with_stage(stage)
    .with_bytes_received(bytes);
    if let Some(body) = oven_sdk::provider_support::sanitize_error_body(body, bytes, stage) {
        error = error.with_sanitized_body(body);
    }
    if let Some(id) = request_id {
        error = error.with_request_id(id);
    }
    if let Some(delay) = parse_retry_after(headers, &[]) {
        error = error.with_retry_after(delay);
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use oven_sdk::{
        CompactionCapability, MediaCapabilities, MediaInputSupport, Modalities, Modality,
        ModelDeclaration, ModelLimits, ProviderId, ReplayCapability, ReplayDeclaration,
    };

    #[test]
    fn provider_body_diagnostics() {
        oven_sdk_conformance::assert_error_body_diagnostics(|status, body, stage, bytes| {
            classify_error(
                status,
                body,
                Some("req-1".into()),
                stage,
                bytes,
                &HeaderMap::new(),
            )
        });
    }
    fn capabilities() -> ModelCapabilities {
        let mut media = MediaCapabilities::default();
        media.input.insert(
            Modality::image(),
            MediaInputSupport::new(
                ["image/png".into()],
                MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
            )
            .unwrap(),
        );
        ModelCapabilities {
            features: Capability::TOOL_CALLING
                | Capability::PARALLEL_TOOLS
                | Capability::TOOL_INPUT_DELTAS
                | Capability::REASONING
                | Capability::STRUCTURED_OUTPUT
                | Capability::TEMPERATURE
                | Capability::TOP_P
                | Capability::MAX_OUTPUT_TOKENS
                | Capability::USAGE
                | Capability::SOURCES,
            limits: ModelLimits::new(Some(128_000), None, Some(32_000)),
            modalities: Modalities::new([Modality::text(), Modality::image()], [Modality::text()]),
            media,
            cancellation: CancellationCapability::LocalOnly,
            compaction: CompactionCapability::Unsupported,
            deferred_tools: oven_sdk::DeferredToolsCapability::Unsupported,
            replay: ReplayDeclaration {
                policy: ReplayPolicy::IfValid,
                capability: ReplayCapability::Optional,
                reasoning: true,
            },
        }
    }

    #[test]
    fn constructor_rejects_non_chat_endpoint_and_protected_headers() {
        let provider = ProviderConfig::new(
            ProviderId::new("cohere"),
            ApiEndpoint::parse("https://example.com/v2/generate").unwrap(),
            CohereAuth::bearer(SecretString::new("x")),
            HeaderConfig::empty(),
        )
        .unwrap();
        let model = ModelDeclaration::new(ModelId::new("opaque"), capabilities()).unwrap();
        assert!(
            CohereModel::new(ModelConfig::new(provider, model, CohereSettings::default())).is_err()
        );
    }

    #[test]
    fn constructor_rejects_native_compaction_declarations() {
        let provider = ProviderConfig::new(
            ProviderId::new("cohere"),
            ApiEndpoint::parse("https://api.cohere.com/v2/chat").unwrap(),
            CohereAuth::bearer(SecretString::new("x")),
            HeaderConfig::empty(),
        )
        .unwrap();
        let mut capabilities = capabilities();
        capabilities.compaction = CompactionCapability::Native;
        let model = ModelDeclaration::new(ModelId::new("opaque"), capabilities).unwrap();
        assert!(
            CohereModel::new(ModelConfig::new(provider, model, CohereSettings::default())).is_err()
        );
    }

    #[test]
    fn sse_parser_handles_arbitrary_utf8_boundaries() {
        let mut parser =
            SseParser::new("Cohere SSE contains invalid UTF-8").clear_name_on_empty_event();
        let input = "event: x\ndata: {\"type\":\"x\",\"text\":\"hé\"}\n\n".as_bytes();
        let mut events = Vec::new();
        for byte in input {
            events.extend(parser.feed(&[*byte]).unwrap());
        }
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "x");
    }

    // Pre-refactor `serde_json::Value` encoders, kept as the reference for parity tests.
    fn reference_tool_result(result: &ToolResultPart) -> Result<JsonValue, ModelError> {
        let content = match &result.content {
            ToolContent::Text(value) => JsonValue::String(value.clone()),
            ToolContent::LoadTools { text: value, .. } => JsonValue::String(value.clone()),
            ToolContent::Json(value) => JsonValue::String(value.to_string()),
            ToolContent::Mixed(values) => JsonValue::Array(
                values
                    .iter()
                    .map(|value| match value {
                        ContentValue::Text(value) => {
                            Ok(serde_json::json!({"type":"text","text":value}))
                        }
                        ContentValue::Json(value) => {
                            Ok(serde_json::json!({"type":"text","text":value.to_string()}))
                        }
                        ContentValue::File(_) => Err(ModelError::unsupported(
                            "Cohere does not support files in tool-result history",
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ToolContent::Denied { reason } => reason
                .clone()
                .unwrap_or_else(|| "Tool call execution denied.".into())
                .into(),
        };
        Ok(serde_json::json!({"role":"tool","tool_call_id":result.tool_call_id,"content":content}))
    }

    fn reference_normalized_assistant(
        parts: &[AssistantPart],
        include_thinking: bool,
    ) -> Result<Vec<JsonValue>, ModelError> {
        let mut content = Vec::new();
        let mut tool_plan = String::new();
        let mut tool_calls = Vec::new();
        let mut citations = Vec::new();
        let mut inline_results = Vec::new();
        for part in parts {
            match part {
                AssistantPart::Text(text) => {
                    content.push(serde_json::json!({"type":"text","text":text.text}))
                }
                AssistantPart::Reasoning(reasoning) => {
                    if reasoning
                        .metadata
                        .as_ref()
                        .and_then(|value| value.get("cohere.kind"))
                        .and_then(JsonValue::as_str)
                        == Some("tool_plan")
                    {
                        tool_plan.push_str(&reasoning.text);
                    } else if include_thinking {
                        content
                            .push(serde_json::json!({"type":"thinking","thinking":reasoning.text}));
                    }
                }
                AssistantPart::ToolCall(call) => {
                    let mut value = serde_json::json!({
                        "id":call.id,"type":"function","function":{"arguments":call.raw_input.clone().unwrap_or_else(|| call.input.to_string())}
                    });
                    if call
                        .metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get("cohere.name_missing"))
                        != Some(&JsonValue::Bool(true))
                    {
                        value["function"]["name"] = call.name.clone().into();
                    }
                    tool_calls.push(value);
                }
                AssistantPart::Source(source) => {
                    let metadata = source.metadata.as_ref().ok_or_else(|| {
                        ModelError::invalid_request(
                            "Cohere assistant sources require provider citation metadata",
                        )
                    })?;
                    citations.push(JsonValue::Object(metadata.clone().into_iter().collect()));
                }
                AssistantPart::ToolResult(result) => {
                    inline_results.push(reference_tool_result(result)?)
                }
                AssistantPart::Custom(part) if part.kind == REPLAY_FINGERPRINT_KIND => {}
                AssistantPart::File(_) => {
                    return Err(ModelError::unsupported(
                        "Cohere cannot encode assistant file parts",
                    ));
                }
                AssistantPart::ToolApproval(_) => {
                    return Err(ModelError::unsupported(
                        "Cohere cannot encode tool-approval parts in assistant history",
                    ));
                }
                AssistantPart::Custom(_) => {
                    return Err(ModelError::unsupported(
                        "Cohere cannot encode custom assistant parts",
                    ));
                }
            }
        }
        let mut message = serde_json::json!({"role":"assistant"});
        if !content.is_empty() {
            message["content"] = JsonValue::Array(content);
        }
        if !tool_plan.is_empty() {
            message["tool_plan"] = tool_plan.into();
        }
        if !tool_calls.is_empty() {
            message["tool_calls"] = JsonValue::Array(tool_calls);
        }
        if !citations.is_empty() {
            message["citations"] = JsonValue::Array(citations);
        }
        let has_assistant_content = message.as_object().is_some_and(|object| object.len() > 1);
        let mut messages = Vec::new();
        if has_assistant_content {
            messages.push(message);
        }
        messages.extend(inline_results);
        Ok(messages)
    }

    fn reference_matches(
        message: &JsonValue,
        parts: &[AssistantPart],
        include_thinking: bool,
    ) -> bool {
        reference_normalized_assistant(parts, include_thinking)
            .is_ok_and(|messages| messages.as_slice() == std::slice::from_ref(message))
    }

    fn reasoning(text: &str, plan: bool) -> AssistantPart {
        let mut part = oven_sdk::ReasoningPart::new(text);
        if plan {
            part.metadata = Some(BTreeMap::from([(
                "cohere.kind".into(),
                JsonValue::from("tool_plan"),
            )]));
        }
        AssistantPart::Reasoning(part)
    }

    fn part_sets() -> Vec<Vec<AssistantPart>> {
        let mut raw = oven_sdk::ToolCallPart::new("c2", "lookup", serde_json::json!({"a":1}));
        raw.raw_input = Some("{\"a\": 1}".into());
        let mut nameless = oven_sdk::ToolCallPart::new("c3", "lookup", serde_json::json!({}));
        nameless.metadata = Some(BTreeMap::from([(
            "cohere.name_missing".into(),
            JsonValue::Bool(true),
        )]));
        let mut source = SourcePart::new();
        source.metadata = Some(BTreeMap::from([
            ("start".into(), serde_json::json!(0)),
            ("text".into(), serde_json::json!("ans")),
        ]));
        let fingerprint = AssistantPart::Custom(CustomPart::new(
            REPLAY_FINGERPRINT_KIND,
            serde_json::json!("fp"),
        ));
        let call = |id: &str, input: JsonValue| {
            AssistantPart::ToolCall(oven_sdk::ToolCallPart::new(id, "lookup", input))
        };
        vec![
            vec![],
            vec![AssistantPart::Text(oven_sdk::TextPart::new(""))],
            vec![reasoning("think", false)],
            vec![
                reasoning("think", false),
                AssistantPart::Text(oven_sdk::TextPart::new("answer")),
                AssistantPart::Source(source),
                fingerprint.clone(),
            ],
            vec![
                reasoning("plan ", true),
                reasoning("", true),
                reasoning("more", true),
                call("c1", serde_json::json!({"q":[1,2.5,null]})),
                AssistantPart::ToolCall(raw),
                AssistantPart::ToolCall(nameless),
                call("c4", serde_json::json!("not an object")),
            ],
            vec![reasoning("", true), call("c1", serde_json::json!({}))],
            vec![
                call("c1", serde_json::json!({})),
                AssistantPart::ToolResult(ToolResultPart::new(
                    "c1",
                    ToolContent::Mixed(vec![
                        ContentValue::Text("t".into()),
                        ContentValue::Json(serde_json::json!({"k":"v"})),
                    ]),
                )),
                AssistantPart::ToolResult(ToolResultPart::new(
                    "c1",
                    ToolContent::Denied { reason: None },
                )),
            ],
            vec![fingerprint],
        ]
    }

    #[test]
    fn normalized_assistant_matches_reference_encoding() {
        for parts in part_sets() {
            for include_thinking in [true, false] {
                let mut messages = Vec::new();
                normalized_assistant(&parts, include_thinking, &mut messages).unwrap();
                let encoded = messages
                    .iter()
                    .map(|message| serde_json::to_value(message).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    encoded,
                    reference_normalized_assistant(&parts, include_thinking).unwrap()
                );
            }
        }
    }

    #[test]
    fn in_place_comparison_matches_reference_equality() {
        let sets = part_sets();
        let mut candidates = Vec::new();
        for parts in &sets {
            for include_thinking in [true, false] {
                candidates.extend(reference_normalized_assistant(parts, include_thinking).unwrap());
            }
        }
        let mutations: [fn(&mut JsonValue); 9] = [
            |message| message["role"] = "user".into(),
            |message| message["extra"] = true.into(),
            |message| message["content"] = serde_json::json!([]),
            |message| message["tool_plan"] = "".into(),
            |message| message["tool_calls"] = serde_json::json!([]),
            |message| message["citations"] = serde_json::json!([{"start":0}]),
            |message| {
                if let Some(text) = message.pointer_mut("/content/0/text") {
                    *text = "changed".into();
                }
            },
            |message| {
                if let Some(arguments) = message.pointer_mut("/tool_calls/0/function/arguments") {
                    *arguments = "{\"q\":[1,2.5,null] }".into();
                }
            },
            |message| {
                if let Some(function) = message.pointer_mut("/tool_calls/0/function") {
                    function["name"] = "other".into();
                }
            },
        ];
        for candidate in candidates.clone() {
            for mutate in mutations {
                let mut changed = candidate.clone();
                mutate(&mut changed);
                candidates.push(changed);
            }
        }
        candidates.push(serde_json::json!("not an object"));
        for parts in &sets {
            for include_thinking in [true, false] {
                for message in &candidates {
                    assert_eq!(
                        body::assistant_message_matches(message, parts, include_thinking),
                        reference_matches(message, parts, include_thinking),
                        "parts={parts:?} include_thinking={include_thinking} message={message}"
                    );
                }
            }
        }
    }

    #[test]
    fn json_text_and_fingerprint_comparisons_are_exact() {
        let value = serde_json::json!({"b":[1,2.5,null,"x\n"],"a":{"c":true}});
        assert!(body::json_text_equals(&value, &value.to_string()));
        assert!(!body::json_text_equals(&value, &format!("{value} ")));
        assert!(!body::json_text_equals(&value, &value.to_string()[1..]));
        assert!(!body::json_text_equals(&value, "{}"));
        let fingerprint = replay_fingerprint(&value).unwrap();
        assert!(fingerprint_matches(&value, &fingerprint));
        assert!(!fingerprint_matches(&value, &fingerprint[1..]));
        assert!(!fingerprint_matches(&serde_json::json!({}), &fingerprint));
    }
}
