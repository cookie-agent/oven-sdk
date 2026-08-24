#![warn(missing_docs)]
// `ModelError` is the public structured error contract and is intentionally returned by value.
#![allow(clippy::result_large_err)]
//! Registry-free standardized Open Responses HTTP/SSE adapter.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
    time::Duration,
};

use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
};
use oven_sdk::provider_support::{SseEvent, SseParser, parse_retry_after};
use oven_sdk::{
    AbortSignal, AdapterId, ApiEndpoint, AssistantPart, BoxFuture, BoxStream,
    CancellationCapability, Capability, CompactionCapability, CompactionRequest, ContentValue,
    CustomPart, ErrorStage, FilePart, FileSource, Finish, FinishReason, HeaderConfig, HistoryTurn,
    InputPart, JsonValue, LanguageModel, LanguageModelDescriptor, MediaSourceSupport,
    ModelCapabilities, ModelConfig, ModelError, ModelErrorKind, ModelId, ModelIdentity,
    NativeContextScope, NativeReplayArtifact, ProviderConfig, ProviderMetadata, ReplayCapability,
    ReplayDecision, ReplayDisposition, ReplayOutcome, ReplayPolicy, Request, RequestMetadata,
    ResourceId, ResponseFormat, ResponseHead, SanitizedBody, SecretString, SourcePart, StreamItem,
    StreamPart, StreamResponse, ToolChoice, ToolContent, ToolResultPart, Usage,
};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Stable adapter identity for the standardized protocol.
pub const OPEN_RESPONSES_ADAPTER_ID: &str = "open.responses.http_sse";
/// Conventional provider identity for the Hugging Face router profile.
pub const HUGGING_FACE_PROVIDER_ID: &str = "huggingface";

const NATIVE_CONTEXT_SCOPE_VERSION: &str = "open.responses.native_context_scope.v2";
const REPLAY_FORMAT: &str = "open.responses.items.v2";
const REPLAY_FINGERPRINT_KIND: &str = "open.responses.replay_fingerprint";
const MAX_ITEMS: usize = 128;
const MAX_CONTENT_PARTS: usize = 128;
const MAX_INLINE_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_INLINE_FILE_BYTES: usize = 32 * 1024 * 1024;
const IMAGE_MEDIA_TYPES: &[&str] = &["image/png", "image/jpeg", "image/webp", "image/gif"];
const MAX_FUNCTION_NAME_BYTES: usize = 64;
const MAX_REQUEST_IDENTIFIER_BYTES: usize = 64;
const PHASE_METADATA_KEY: &str = "open_responses.phase";
const REFUSAL_KIND: &str = "open_responses.refusal";

/// Explicit generic bearer authentication.
#[derive(Clone)]
pub struct OpenResponsesAuth {
    token: SecretString,
}

impl OpenResponsesAuth {
    /// Creates authentication from an already resolved bearer token.
    #[must_use]
    pub fn bearer(token: SecretString) -> Self {
        Self { token }
    }
}

impl std::fmt::Debug for OpenResponsesAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("OpenResponsesAuth(<redacted>)")
    }
}

/// Explicit Hugging Face transport identity.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceTransport {
    /// Caller-declared routing identity already represented by the exact model ID.
    pub routing: String,
}

/// Transport profile. It never selects a model or rewrites its ID.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenResponsesTransport {
    /// Generic standards-compliant endpoint with an open profile label.
    Generic {
        /// Caller-defined implementation/profile label.
        profile: String,
    },
    /// Hugging Face router profile. This explicit selection also gates the
    /// router's documented legacy reasoning event and content variants.
    HuggingFace(HuggingFaceTransport),
}

/// Per-phase HTTP timeouts. There is no total stream timeout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenResponsesTimeouts {
    /// Connection timeout.
    pub connect: Duration,
    /// Maximum wait for response headers.
    pub headers: Duration,
    /// Maximum inactivity while reading response bytes.
    pub stream_idle: Duration,
}

impl Default for OpenResponsesTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(10),
            headers: Duration::from_secs(30),
            stream_idle: Duration::from_secs(60),
        }
    }
}

/// Complete structural settings for one Open Responses model.
#[derive(Clone, Debug)]
pub struct OpenResponsesSettings {
    /// Explicit transport/profile declaration.
    pub transport: OpenResponsesTransport,
    /// Transport phase timeouts.
    pub timeouts: OpenResponsesTimeouts,
    /// Strict JSON Schema flag for response formats.
    pub strict_json_schema: bool,
    /// Strict function-tool schema flag.
    pub strict_tools: bool,
    /// Whether parallel function calls are requested when tools are present.
    pub parallel_tool_calls: bool,
    /// Whether the server may persist responses.
    pub store: bool,
    /// Standard include values, kept as open strings.
    pub include: Vec<String>,
    /// Optional provider-defined reasoning-summary label.
    pub reasoning_summary: Option<String>,
}

/// Request-scoped standard options with open string labels.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenResponsesRequestOptions {
    /// Additional instructions.
    pub instructions: Option<String>,
    /// Up to sixteen metadata key/value pairs.
    pub metadata: BTreeMap<String, String>,
    /// Optional service-tier label.
    pub service_tier: Option<String>,
    /// Optional truncation label.
    pub truncation: Option<String>,
    /// Optional text-verbosity label.
    pub text_verbosity: Option<String>,
    /// Optional maximum number of tool calls.
    pub max_tool_calls: Option<u64>,
    /// Optional safety identifier.
    pub safety_identifier: Option<String>,
    /// Optional prompt-cache key.
    pub prompt_cache_key: Option<String>,
}

/// Extension methods for typed request options.
pub trait OpenResponsesRequestExt {
    /// Stores options under the `open_responses` provider namespace.
    fn with_open_responses_options(self, options: OpenResponsesRequestOptions) -> Self;
}

impl OpenResponsesRequestExt for Request {
    fn with_open_responses_options(mut self, options: OpenResponsesRequestOptions) -> Self {
        self.provider_options.insert(
            "open_responses".into(),
            serde_json::to_value(options).expect("Open Responses options serialize"),
        );
        self
    }
}

/// Complete registry-free configuration.
pub type OpenResponsesConfig = ModelConfig<OpenResponsesAuth, OpenResponsesSettings>;

/// One configured standardized Open Responses model.
#[derive(Clone)]
pub struct OpenResponsesModel {
    config: Arc<Config>,
    descriptor: LanguageModelDescriptor,
}

impl OpenResponsesModel {
    /// Constructs one registry-free Open Responses model.
    pub fn new(config: OpenResponsesConfig) -> Result<Self, ModelError> {
        let (config, descriptor) = Config::build(config)?;
        Ok(Self { config, descriptor })
    }

    /// Returns the exact model ID sent on the wire.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        &self.descriptor.identity.model_id
    }
}

#[derive(Clone)]
struct Config {
    auth: OpenResponsesAuth,
    client: reqwest::Client,
    endpoint: reqwest::Url,
    headers: HeaderConfig,
    settings: OpenResponsesSettings,
    capabilities: ModelCapabilities,
    identity: ModelIdentity,
    replay_seed: Sha256,
}

impl Config {
    fn build(
        value: OpenResponsesConfig,
    ) -> Result<(Arc<Self>, LanguageModelDescriptor), ModelError> {
        let ModelConfig {
            provider,
            model,
            settings,
        } = value;
        validate_endpoint(&provider.api)?;
        validate_settings(&settings, &model.capabilities)?;
        validate_capabilities(&settings.transport, &model.capabilities)?;
        if provider.auth.token.is_empty() {
            return Err(ModelError::invalid_request(
                "Open Responses bearer token must not be empty",
            ));
        }
        reject_protected_headers(provider.headers.static_headers.as_map())?;
        model.validate()?;
        let identity = ModelIdentity::new(provider.id.clone(), model.id.clone())?;
        let mut metadata = ProviderMetadata::new();
        match &settings.transport {
            OpenResponsesTransport::Generic { profile } => {
                metadata.insert("open_responses.profile".into(), profile.clone().into());
            }
            OpenResponsesTransport::HuggingFace(profile) => {
                metadata.insert("open_responses.profile".into(), "huggingface".into());
                metadata.insert("huggingface.routing".into(), profile.routing.clone().into());
            }
        }
        let descriptor = LanguageModelDescriptor::new(
            identity.clone(),
            AdapterId::new(OPEN_RESPONSES_ADAPTER_ID),
            model.capabilities.clone(),
        )?
        .with_provider_metadata(metadata);
        let replay_seed = replay_seed(&provider, &model, &settings)?;
        let client = reqwest::Client::builder()
            .connect_timeout(settings.timeouts.connect)
            .build()
            .map_err(|_| ModelError::transport("could not construct Open Responses HTTP client"))?;
        Ok((
            Arc::new(Self {
                auth: provider.auth,
                client,
                endpoint: provider.api.as_url().clone(),
                headers: provider.headers,
                settings,
                capabilities: descriptor.capabilities.clone(),
                identity,
                replay_seed,
            }),
            descriptor,
        ))
    }

    fn request_headers(&self) -> Result<HeaderMap, ModelError> {
        let mut headers = self.headers.static_headers.as_map().clone();
        if let Some(dynamic) = &self.headers.dynamic_headers {
            headers.extend(dynamic.headers()?.as_map().clone());
        }
        reject_protected_headers(&headers)?;
        headers.insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.auth.token.expose_secret())).map_err(
                |_| ModelError::invalid_request("bearer token is not a valid header value"),
            )?,
        );
        Ok(headers)
    }

    fn replay_context(
        &self,
        headers: &HeaderMap,
    ) -> Result<(JsonValue, NativeContextScope), ModelError> {
        let mut hasher = self.replay_seed.clone();
        hash_headers(&mut hasher, headers);
        let digest = URL_SAFE_NO_PAD.encode(hasher.finalize());
        let binding = serde_json::json!({"version":NATIVE_CONTEXT_SCOPE_VERSION,"sha256":digest});
        let scope = NativeContextScope::new(
            self.identity.provider_id.clone(),
            self.identity.model_id.clone(),
            ResourceId::new(format!("{NATIVE_CONTEXT_SCOPE_VERSION}.sha256.{digest}"))?,
        )?;
        Ok((binding, scope))
    }
}

impl LanguageModel for OpenResponsesModel {
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
    fn validate_compaction(&self, _request: &CompactionRequest) -> Result<(), ModelError> {
        Err(ModelError::unsupported(
            "standardized Open Responses and Hugging Face transports do not support OpenAI native compaction",
        ))
    }
    fn stream<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        Box::pin(async move {
            self.validate_request(&request)?;
            if abort.is_aborted() {
                return Err(ModelError::abort("request was aborted before dispatch")
                    .with_stage(ErrorStage::Connect));
            }
            let headers = self.config.request_headers()?;
            let (binding, scope) = self.config.replay_context(&headers)?;
            let encoded = encode_request(
                &request,
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
                .json(&encoded.body)
                .send();
            let response = tokio::select! {value=tokio::time::timeout(self.config.settings.timeouts.headers,send)=>value.map_err(|_|ModelError::timeout("Open Responses headers timed out").with_stage(ErrorStage::ResponseHeaders))?.map_err(|_|ModelError::transport("Open Responses request failed").with_stage(ErrorStage::Connect))?,_=abort.aborted()=>return Err(ModelError::abort("request was aborted before response headers").with_stage(ErrorStage::Connect))};
            let head = response_head(&response);
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let response_headers = response.headers().clone();
                let request_id = head.request_id.clone();
                let (body, bytes) =
                    read_error_body(response, &abort, self.config.settings.timeouts.stream_idle)
                        .await?;
                return Err(classify_error(
                    Some(status),
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
                return Err(
                    ModelError::invalid_response("Open Responses response is not SSE")
                        .with_stage(ErrorStage::ResponseHeaders),
                );
            }
            let response_headers = response.headers().clone();
            let live = Live {
                bytes: Box::pin(response.bytes_stream()),
                parser: SseParser::new("Open Responses SSE contains invalid UTF-8")
                    .clear_name_on_empty_event(),
                state: State::new(
                    self.descriptor.adapter_id.clone(),
                    self.config.capabilities.replay.policy,
                    self.config.capabilities.replay.capability,
                    self.config.capabilities.replay.reasoning,
                    matches!(
                        &self.config.settings.transport,
                        OpenResponsesTransport::HuggingFace(_)
                    ),
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
                        if live.state.done {
                            return None;
                        }
                        live.state.done = true;
                        return Some((Err(unexpected_eof(live.count)), live));
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
    body: JsonValue,
    replay: ReplayOutcome,
    warnings: Vec<String>,
}

fn request_options(request: &Request) -> Result<OpenResponsesRequestOptions, ModelError> {
    request
        .provider_options
        .get("open_responses")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| ModelError::invalid_request("invalid typed Open Responses request options"))
        .map(Option::unwrap_or_default)
}

fn validate_request(
    request: &Request,
    options: &OpenResponsesRequestOptions,
    settings: &OpenResponsesSettings,
) -> Result<(), ModelError> {
    for (name, value) in [
        ("service tier", options.service_tier.as_deref()),
        ("truncation", options.truncation.as_deref()),
        ("text verbosity", options.text_verbosity.as_deref()),
        ("reasoning summary", settings.reasoning_summary.as_deref()),
    ] {
        validate_label(value, name)?;
    }
    if options.metadata.len() > 16
        || options
            .metadata
            .iter()
            .any(|(key, value)| key.len() > 64 || value.len() > 512)
    {
        return Err(ModelError::invalid_request(
            "Open Responses metadata exceeds 16 entries or key/value limits",
        ));
    }
    if options.max_tool_calls == Some(0) {
        return Err(ModelError::invalid_request(
            "max_tool_calls must be positive",
        ));
    }
    if request
        .inference
        .max_output_tokens
        .is_some_and(|value| value < 16)
    {
        return Err(ModelError::invalid_request(
            "Open Responses max_output_tokens must be at least 16",
        ));
    }
    for (name, value) in [
        ("safety_identifier", options.safety_identifier.as_deref()),
        ("prompt_cache_key", options.prompt_cache_key.as_deref()),
    ] {
        if value.is_some_and(|value| value.chars().count() > MAX_REQUEST_IDENTIFIER_BYTES) {
            return Err(ModelError::invalid_request(format!(
                "Open Responses {name} must not exceed 64 bytes"
            )));
        }
        validate_label(value, name)?;
    }
    for tool in &request.tools {
        validate_function_name(&tool.name)?;
        validate_function_schema(tool.input_schema.as_value())?;
    }
    if let ToolChoice::Tool(name) = &request.tool_choice {
        validate_function_name(name)?;
    }
    if let ResponseFormat::Json {
        schema: Some(schema),
    } = &request.response_format
        && !schema.as_value().is_object()
    {
        return Err(ModelError::invalid_request(
            "Open Responses response schemas must be JSON objects",
        ));
    }
    for turn in &request.history {
        match turn {
            HistoryTurn::User(message) => {
                for part in &message.content {
                    if let InputPart::File(file) = part {
                        validate_media_file(file)?;
                    }
                }
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    validate_tool_result(result)?;
                }
            }
            HistoryTurn::Assistant(turn) => {
                if !turn.message.provider_options.is_empty() {
                    return Err(ModelError::unsupported(
                        "Open Responses assistant history does not support message provider options",
                    ));
                }
                normalized_assistant(&turn.message.content, &settings.transport)?;
            }
            HistoryTurn::System(_) => {}
        }
    }
    Ok(())
}

fn validate_function_name(name: &str) -> Result<(), ModelError> {
    if !valid_function_name(name) {
        return Err(ModelError::invalid_request(
            "Open Responses function names must match ^[a-zA-Z0-9_-]+$ and be at most 64 bytes",
        ));
    }
    Ok(())
}

fn valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_FUNCTION_NAME_BYTES
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn validate_call_id(call_id: &str) -> Result<(), ModelError> {
    if !valid_call_id(call_id) {
        return Err(ModelError::invalid_request(
            "Open Responses function call IDs must be 1 to 64 bytes",
        ));
    }
    Ok(())
}

fn valid_call_id(call_id: &str) -> bool {
    !call_id.is_empty() && call_id.len() <= MAX_REQUEST_IDENTIFIER_BYTES
}

fn validate_tool_result(result: &ToolResultPart) -> Result<(), ModelError> {
    validate_call_id(&result.tool_call_id)?;
    if result.is_error {
        return Err(ModelError::unsupported(
            "Open Responses function outputs cannot represent the normalized is_error flag",
        ));
    }
    if let ToolContent::Mixed(values) = &result.content {
        for value in values {
            if let ContentValue::File(file) = value {
                validate_media_file(file)?;
            }
        }
    }
    Ok(())
}

fn validate_media_file(file: &FilePart) -> Result<(), ModelError> {
    let image = file.media_type.starts_with("image/");
    match &file.source {
        FileSource::Bytes(bytes) => {
            let encoded_len = "data:;base64,".len()
                + file.media_type.len()
                + bytes.len().div_ceil(3).saturating_mul(4);
            let limit = if image {
                MAX_INLINE_IMAGE_BYTES
            } else {
                MAX_INLINE_FILE_BYTES
            };
            if encoded_len > limit {
                return Err(ModelError::invalid_request(
                    "Open Responses encoded inline media exceeds its protocol limit",
                ));
            }
        }
        FileSource::Url(url) => {
            if !matches!(url.scheme(), "http" | "https") {
                return Err(ModelError::unsupported(
                    "Open Responses media URLs must use HTTP(S)",
                ));
            }
            if image && url.as_str().len() > MAX_INLINE_IMAGE_BYTES {
                return Err(ModelError::invalid_request(
                    "Open Responses image URL exceeds its protocol limit",
                ));
            }
        }
        FileSource::Text(_) => {
            return Err(ModelError::unsupported(
                "Open Responses media does not accept inline text sources",
            ));
        }
        FileSource::ProviderReference { .. } => {
            return Err(ModelError::unsupported(
                "Open Responses provider file references are unsupported",
            ));
        }
    }
    Ok(())
}

fn encode_request(
    request: &Request,
    descriptor: &LanguageModelDescriptor,
    settings: &OpenResponsesSettings,
    binding: &JsonValue,
    scope: &NativeContextScope,
) -> Result<Encoded, ModelError> {
    let options = request_options(request)?;
    let mut input = Vec::new();
    let mut replay = ReplayOutcome::default();
    let warnings = Vec::new();
    for (history_index, turn) in request.history.iter().enumerate() {
        match turn {
            HistoryTurn::System(message) => {
                input.extend(encode_system_message(message)?);
            }
            HistoryTurn::User(message) => {
                input.extend(encode_user_message(message)?);
            }
            HistoryTurn::Tool(message) => {
                for result in &message.results {
                    input.push(function_output(result)?);
                }
            }
            HistoryTurn::Assistant(turn) => {
                if descriptor.capabilities.replay.policy == ReplayPolicy::Never {
                    replay.decisions.push(ReplayDecision {
                        history_index,
                        disposition: ReplayDisposition::ReconstructedNormalized,
                    });
                    input.extend(normalized_assistant(
                        &turn.message.content,
                        &settings.transport,
                    )?);
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
                    Some(artifact) if artifact.adapter_id() != &descriptor.adapter_id => {
                        replay.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedForeignAdapter {
                                found: artifact.adapter_id().clone(),
                                expected: descriptor.adapter_id.clone(),
                            },
                        });
                        None
                    }
                    Some(artifact) if artifact.scope() != scope => {
                        replay.decisions.push(ReplayDecision {
                            history_index,
                            disposition: ReplayDisposition::DiscardedForeignScope {
                                found: artifact.scope().clone(),
                                expected: scope.clone(),
                            },
                        });
                        None
                    }
                    Some(artifact) => match decode_replay(artifact, &turn.message.content, binding)
                    {
                        Some(items) => {
                            replay.decisions.push(ReplayDecision {
                                history_index,
                                disposition: ReplayDisposition::Replayed,
                            });
                            Some(items)
                        }
                        None => {
                            replay.decisions.push(ReplayDecision{history_index,disposition:ReplayDisposition::DiscardedInvalidPayload{reason:"Open Responses replay payload did not match normalized content".into()}});
                            None
                        }
                    },
                };
                if let Some(items) = replayed {
                    input.extend(items)
                } else {
                    replay.decisions.push(ReplayDecision {
                        history_index,
                        disposition: ReplayDisposition::ReconstructedNormalized,
                    });
                    input.extend(normalized_assistant(
                        &turn.message.content,
                        &settings.transport,
                    )?);
                }
            }
        }
    }
    let mut include = settings.include.clone();
    if descriptor.capabilities.replay.reasoning
        && !include
            .iter()
            .any(|value| value == "reasoning.encrypted_content")
    {
        include.push("reasoning.encrypted_content".into());
    }
    let mut body = serde_json::json!({"model":descriptor.identity.model_id.as_str(),"input":input,"stream":true,"store":settings.store,"include":include});
    if let Some(value) = request.inference.max_output_tokens {
        body["max_output_tokens"] = value.into();
    }
    if let Some(value) = request.inference.temperature {
        body["temperature"] = value.into();
    }
    if let Some(value) = request.inference.top_p {
        body["top_p"] = value.into();
    }
    if let Some(effort) = &request.inference.reasoning_effort {
        let mut reasoning = serde_json::json!({"effort":effort});
        if let Some(summary) = &settings.reasoning_summary {
            reasoning["summary"] = summary.clone().into();
        }
        body["reasoning"] = reasoning;
    } else if let Some(summary) = &settings.reasoning_summary {
        body["reasoning"] = serde_json::json!({"summary":summary});
    }
    if let Some(value) = options.instructions {
        body["instructions"] = value.into();
    }
    if !options.metadata.is_empty() {
        body["metadata"] = serde_json::to_value(options.metadata).expect("metadata serializes");
    }
    if let Some(value) = options.service_tier {
        body["service_tier"] = value.into();
    }
    if let Some(value) = options.truncation {
        body["truncation"] = value.into();
    }
    if let Some(value) = options.max_tool_calls {
        body["max_tool_calls"] = value.into();
    }
    if let Some(value) = options.safety_identifier {
        body["safety_identifier"] = value.into();
    }
    if let Some(value) = options.prompt_cache_key {
        body["prompt_cache_key"] = value.into();
    }
    if !request.tools.is_empty() {
        body["tools"]=JsonValue::Array(request.tools.iter().map(|tool|serde_json::json!({"type":"function","name":tool.name,"description":tool.description,"parameters":tool.input_schema.as_value(),"strict":settings.strict_tools})).collect());
        body["parallel_tool_calls"] = settings.parallel_tool_calls.into();
    }
    body["tool_choice"] = match &request.tool_choice {
        ToolChoice::Auto => "auto".into(),
        ToolChoice::Required => "required".into(),
        ToolChoice::None => "none".into(),
        ToolChoice::Tool(name) => serde_json::json!({"type":"function","name":name}),
    };
    match &request.response_format {
        ResponseFormat::Text => {
            if let Some(verbosity) = options.text_verbosity {
                body["text"] = serde_json::json!({"verbosity":verbosity});
            }
        }
        ResponseFormat::Json { schema: None } => {
            body["text"] = serde_json::json!({"format":{"type":"json_object"}});
            if let Some(verbosity) = options.text_verbosity {
                body["text"]["verbosity"] = verbosity.into();
            }
        }
        ResponseFormat::Json {
            schema: Some(schema),
        } => {
            body["text"] = serde_json::json!({"format":{"type":"json_schema","name":"response","strict":settings.strict_json_schema,"schema":schema.as_value()}});
            if let Some(verbosity) = options.text_verbosity {
                body["text"]["verbosity"] = verbosity.into();
            }
        }
    }
    Ok(Encoded {
        body,
        replay,
        warnings,
    })
}

fn input_file(file: &FilePart) -> Result<JsonValue, ModelError> {
    validate_media_file(file)?;
    let image = file.media_type.starts_with("image/");
    match &file.source {
        FileSource::Bytes(bytes) => {
            let data = format!("data:{};base64,{}", file.media_type, STANDARD.encode(bytes));
            if image {
                Ok(serde_json::json!({"type":"input_image","image_url":data,"detail":"auto"}))
            } else {
                Ok(
                    serde_json::json!({"type":"input_file","filename":file.filename.clone().unwrap_or_else(||"file".into()),"file_data":data}),
                )
            }
        }
        FileSource::Url(url) if matches!(url.scheme(), "http" | "https") => {
            if image {
                Ok(serde_json::json!({"type":"input_image","image_url":url,"detail":"auto"}))
            } else {
                Ok(serde_json::json!({"type":"input_file","file_url":url}))
            }
        }
        FileSource::Url(_) => Err(ModelError::unsupported(
            "Open Responses media URLs must use HTTP(S)",
        )),
        FileSource::Text(_) => Err(ModelError::unsupported(
            "Open Responses media does not accept inline text sources",
        )),
        FileSource::ProviderReference { .. } => Err(ModelError::unsupported(
            "Open Responses provider file references are unsupported",
        )),
    }
}

fn function_output(result: &ToolResultPart) -> Result<JsonValue, ModelError> {
    validate_tool_result(result)?;
    let output = match &result.content {
        ToolContent::Text(value) => JsonValue::String(value.clone()),
        ToolContent::Json(value) => JsonValue::String(value.to_string()),
        ToolContent::Mixed(values) => JsonValue::Array(
            values
                .iter()
                .map(|value| match value {
                    ContentValue::Text(value) => {
                        Ok(serde_json::json!({"type":"input_text","text":value}))
                    }
                    ContentValue::Json(value) => {
                        Ok(serde_json::json!({"type":"input_text","text":value.to_string()}))
                    }
                    ContentValue::File(file) => input_file(file),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ToolContent::Denied { reason } => reason
            .clone()
            .unwrap_or_else(|| "Tool call execution denied.".into())
            .into(),
    };
    Ok(
        serde_json::json!({"type":"function_call_output","call_id":result.tool_call_id,"output":output,"status":"completed"}),
    )
}

fn encode_system_message(message: &oven_sdk::SystemMessage) -> Result<Vec<JsonValue>, ModelError> {
    let mut output = Vec::new();
    let mut content = Vec::new();
    for part in &message.content {
        match part {
            oven_sdk::SystemPart::Text(text) => {
                content.push(serde_json::json!({"type":"input_text","text":text.text}));
            }
            oven_sdk::SystemPart::Custom(custom) => {
                flush_message(&mut output, "system", &mut content);
                output.push(custom_item(custom)?);
            }
        }
    }
    flush_message(&mut output, "system", &mut content);
    Ok(output)
}

fn encode_user_message(message: &oven_sdk::UserMessage) -> Result<Vec<JsonValue>, ModelError> {
    let mut output = Vec::new();
    let mut content = Vec::new();
    for part in &message.content {
        match part {
            InputPart::Text(text) => {
                content.push(serde_json::json!({"type":"input_text","text":text.text}));
            }
            InputPart::File(file) => content.push(input_file(file)?),
            InputPart::Custom(custom) => {
                flush_message(&mut output, "user", &mut content);
                output.push(custom_item(custom)?);
            }
        }
    }
    flush_message(&mut output, "user", &mut content);
    Ok(output)
}

fn flush_message(output: &mut Vec<JsonValue>, role: &str, content: &mut Vec<JsonValue>) {
    if !content.is_empty() {
        output.push(
            serde_json::json!({"type":"message","role":role,"content":std::mem::take(content)}),
        );
    }
}

fn custom_item(part: &CustomPart) -> Result<JsonValue, ModelError> {
    if !part.kind.contains(':') {
        return Err(ModelError::unsupported(
            "Open Responses custom history requires a provider-prefixed item type",
        ));
    }
    let item = part.data.as_object().ok_or_else(|| {
        ModelError::unsupported("Open Responses custom history item data must be an object")
    })?;
    if item.get("type").and_then(JsonValue::as_str) != Some(part.kind.as_str())
        || item
            .get("id")
            .and_then(JsonValue::as_str)
            .is_none_or(str::is_empty)
        || !matches!(
            item.get("status").and_then(JsonValue::as_str),
            Some("completed" | "incomplete")
        )
    {
        return Err(ModelError::unsupported(
            "Open Responses custom history items require matching type, non-empty id, and terminal status",
        ));
    }
    Ok(JsonValue::Object(item.clone()))
}

fn normalized_assistant(
    parts: &[AssistantPart],
    transport: &OpenResponsesTransport,
) -> Result<Vec<JsonValue>, ModelError> {
    let mut output = Vec::new();
    let mut index = 0;
    while index < parts.len() {
        match &parts[index] {
            AssistantPart::Text(text) => {
                let mut annotations = Vec::new();
                index += 1;
                while let Some(AssistantPart::Source(source)) = parts.get(index) {
                    annotations.push(source_annotation(source)?);
                    index += 1;
                }
                let mut item = serde_json::json!({
                    "type":"message",
                    "role":"assistant",
                    "content":[{"type":"output_text","text":text.text,"annotations":annotations}],
                    "status":"completed"
                });
                apply_phase(&mut item, text.metadata.as_ref())?;
                output.push(item);
                continue;
            }
            AssistantPart::ToolCall(call) => {
                let mut item = serde_json::json!({"type":"function_call","call_id":call.id,"name":call.name,"arguments":call.raw_input.clone().unwrap_or_else(||call.input.to_string()),"status":"completed"});
                if let Some(id) = &call.provider_item_id {
                    item["id"] = id.clone().into();
                }
                output.push(item);
            }
            AssistantPart::ToolResult(result) => output.push(function_output(result)?),
            AssistantPart::Reasoning(reasoning) => {
                let kind = reasoning
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("open_responses.kind"))
                    .and_then(JsonValue::as_str)
                    .unwrap_or("summary_text");
                match kind {
                    "summary_text" => output.push(serde_json::json!({
                        "type":"reasoning",
                        "summary":[{"type":"summary_text","text":reasoning.text}],
                        "content":null,
                        "encrypted_content":null,
                        "status":"completed"
                    })),
                    "reasoning_text"
                        if matches!(transport, OpenResponsesTransport::HuggingFace(_)) =>
                    {
                        output.push(serde_json::json!({
                            "type":"reasoning",
                            "summary":[],
                            "content":[{"type":"reasoning_text","text":reasoning.text}],
                            "status":"completed"
                        }));
                    }
                    "reasoning_summary"
                        if matches!(transport, OpenResponsesTransport::HuggingFace(_)) =>
                    {
                        output.push(serde_json::json!({
                            "type":"reasoning",
                            "summary":[{"type":"reasoning_summary","text":reasoning.text}],
                            "content":[],
                            "status":"completed"
                        }));
                    }
                    _ => {
                        return Err(ModelError::unsupported(
                            "Open Responses normalized reasoning kind is not representable by this transport",
                        ));
                    }
                }
            }
            AssistantPart::Source(_) => {
                return Err(ModelError::unsupported(
                    "Open Responses sources must immediately follow the output text they annotate",
                ));
            }
            AssistantPart::File(_) => {
                return Err(ModelError::unsupported(
                    "Open Responses has no standard assistant output-file history item",
                ));
            }
            AssistantPart::ToolApproval(_) => {
                return Err(ModelError::unsupported(
                    "Open Responses has no standard tool-approval history item",
                ));
            }
            AssistantPart::Custom(custom) if custom.kind == REPLAY_FINGERPRINT_KIND => {
                if custom.data.as_str().is_none_or(str::is_empty) {
                    return Err(ModelError::unsupported(
                        "Open Responses replay fingerprint history must contain a non-empty string",
                    ));
                }
            }
            AssistantPart::Custom(custom) if custom.kind == REFUSAL_KIND => {
                let refusal = custom.data.as_str().ok_or_else(|| {
                    ModelError::unsupported("Open Responses refusal history must contain a string")
                })?;
                let mut item = serde_json::json!({
                    "type":"message",
                    "role":"assistant",
                    "content":[{"type":"refusal","refusal":refusal}],
                    "status":"completed"
                });
                apply_phase(&mut item, custom.metadata.as_ref())?;
                output.push(item);
            }
            AssistantPart::Custom(custom) => output.push(custom_item(custom)?),
        }
        index += 1;
    }
    Ok(output)
}

fn apply_phase(
    item: &mut JsonValue,
    metadata: Option<&BTreeMap<String, JsonValue>>,
) -> Result<(), ModelError> {
    let Some(phase) = metadata
        .and_then(|metadata| metadata.get(PHASE_METADATA_KEY))
        .and_then(JsonValue::as_str)
    else {
        return Ok(());
    };
    if !matches!(phase, "commentary" | "final_answer") {
        return Err(ModelError::invalid_request(
            "Open Responses assistant phase must be commentary or final_answer",
        ));
    }
    item["phase"] = phase.into();
    Ok(())
}

fn source_annotation(source: &SourcePart) -> Result<JsonValue, ModelError> {
    let annotation = source
        .metadata
        .as_ref()
        .map(|metadata| JsonValue::Object(metadata.clone().into_iter().collect()))
        .ok_or_else(|| {
            ModelError::unsupported(
                "Open Responses source history requires its exact URL-citation metadata",
            )
        })?;
    if annotation.get("type").and_then(JsonValue::as_str) != Some("url_citation")
        || annotation.get("url").and_then(JsonValue::as_str).is_none()
        || annotation
            .get("title")
            .and_then(JsonValue::as_str)
            .is_none()
        || annotation
            .get("start_index")
            .and_then(JsonValue::as_u64)
            .is_none()
        || annotation
            .get("end_index")
            .and_then(JsonValue::as_u64)
            .is_none()
    {
        return Err(ModelError::unsupported(
            "Open Responses supports only complete URL-citation sources in assistant history",
        ));
    }
    Ok(annotation)
}

struct Live {
    bytes: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    parser: SseParser,
    state: State,
    queue: VecDeque<StreamItem>,
    events: VecDeque<SseEvent>,
    pending_error: Option<ModelError>,
    deadline: oven_sdk::provider_support::StreamReadDeadline<tokio::time::Sleep>,
    idle: Duration,
    count: u64,
    eof: bool,
    request_id: Option<String>,
    response_headers: HeaderMap,
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
                return Err(ModelError::abort("Open Responses stream was aborted")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count));
            }
            oven_sdk::provider_support::StreamRead::TimedOut => {
                return Err(ModelError::timeout("Open Responses stream idle timeout")
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
                return Err(ModelError::transport("Open Responses stream read failed")
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
        if event.data == "[DONE]" {
            live.state.done(&mut live.queue, live.count)?;
            live.eof = true;
            live.events.clear();
            return Ok(());
        }
        let value: JsonValue = serde_json::from_str(&event.data).map_err(|_| {
            ModelError::invalid_response("Open Responses SSE event is invalid JSON")
                .with_stage(ErrorStage::StreamDecode)
                .with_bytes_received(live.count)
        })?;
        let kind = value.get("type").and_then(JsonValue::as_str).unwrap_or("");
        if event.name != kind {
            return Err(event_error(
                "Open Responses event field must match payload type",
                live.count,
            ));
        }
        if live.include_raw {
            live.queue.push_back(Ok(StreamPart::Raw {
                value: value.clone(),
            }));
        }
        if kind == "error" {
            live.state.stream_error(
                classify_error(
                    value
                        .get("status")
                        .and_then(JsonValue::as_u64)
                        .and_then(|value| u16::try_from(value).ok()),
                    event.data.as_bytes(),
                    live.request_id.clone(),
                    ErrorStage::StreamEvent,
                    live.count,
                    &live.response_headers,
                ),
                &value,
                live.count,
            )?;
        } else {
            live.state.apply(value, &mut live.queue, live.count)?;
        }
        if !live.queue.is_empty() {
            return Ok(());
        }
    }
    if live.eof && !live.state.done {
        return Err(unexpected_eof(live.count));
    }
    Ok(())
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Phase {
    Initial,
    Created,
    Queued,
    InProgress,
    Terminal,
}
struct ContentState {
    kind: String,
    text: String,
    annotations: Vec<JsonValue>,
    deferred: bool,
    open: bool,
    done: bool,
}
struct ItemState {
    item: JsonValue,
    id: String,
    kind: String,
    contents: BTreeMap<usize, ContentState>,
    summaries: BTreeMap<usize, ContentState>,
    arguments: String,
    arguments_done: bool,
    done: bool,
}
struct State {
    adapter: AdapterId,
    policy: ReplayPolicy,
    replay_capability: ReplayCapability,
    replay_reasoning: bool,
    hugging_face: bool,
    binding: JsonValue,
    scope: NativeContextScope,
    phase: Phase,
    sequence: Option<u64>,
    items: BTreeMap<usize, ItemState>,
    incomplete_item: Option<usize>,
    terminal: Option<Finish>,
    stream_error: Option<ModelError>,
    done: bool,
    response_metadata: BTreeMap<String, JsonValue>,
}

impl State {
    fn new(
        adapter: AdapterId,
        policy: ReplayPolicy,
        replay_capability: ReplayCapability,
        replay_reasoning: bool,
        hugging_face: bool,
        binding: JsonValue,
        scope: NativeContextScope,
    ) -> Self {
        Self {
            adapter,
            policy,
            replay_capability,
            replay_reasoning,
            hugging_face,
            binding,
            scope,
            phase: Phase::Initial,
            sequence: None,
            items: BTreeMap::new(),
            incomplete_item: None,
            terminal: None,
            stream_error: None,
            done: false,
            response_metadata: BTreeMap::new(),
        }
    }
    fn sequence(&mut self, value: &JsonValue, bytes: u64) -> Result<(), ModelError> {
        let current = value
            .get("sequence_number")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| event_error("Open Responses events require sequence_number", bytes))?;
        if let Some(previous) = self.sequence
            && current != previous.saturating_add(1)
        {
            return Err(event_error(
                "Open Responses sequence numbers must be contiguous",
                bytes,
            ));
        }
        self.sequence = Some(current);
        Ok(())
    }
    fn apply(
        &mut self,
        value: JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self.done || self.phase == Phase::Terminal {
            return Err(event_error(
                "Open Responses event after terminal response",
                bytes,
            ));
        }
        self.sequence(&value, bytes)?;
        let kind = value.get("type").and_then(JsonValue::as_str).unwrap_or("");
        if self.stream_error.is_some() && kind != "response.failed" {
            return Err(event_error(
                "Open Responses error event must be followed by response.failed",
                bytes,
            ));
        }
        if self.phase == Phase::Initial && kind != "response.created" {
            return Err(event_error(
                "Open Responses stream must start with response.created",
                bytes,
            ));
        }
        match kind {
            "response.created" => {
                if self.phase != Phase::Initial {
                    return Err(event_error("duplicate response.created", bytes));
                }
                let response = event_response(&value, "response.created", "in_progress", bytes)?;
                self.phase = Phase::Created;
                self.capture_response(response);
            }
            "response.queued" => {
                if self.phase != Phase::Created {
                    return Err(event_error("response.queued is out of order", bytes));
                }
                let response = event_response(&value, "response.queued", "queued", bytes)?;
                self.phase = Phase::Queued;
                self.capture_response(response);
            }
            "response.in_progress" => {
                if !matches!(self.phase, Phase::Created | Phase::Queued) {
                    return Err(event_error("response.in_progress is out of order", bytes));
                }
                let response =
                    event_response(&value, "response.in_progress", "in_progress", bytes)?;
                self.phase = Phase::InProgress;
                self.capture_response(response);
            }
            "response.output_item.added" => self.item_added(&value, queue, bytes)?,
            "response.reasoning_summary_part.added" => self.summary_added(&value, queue, bytes)?,
            "response.reasoning_summary_part.done" => {
                self.summary_part_done(&value, queue, bytes)?
            }
            "response.content_part.added" => self.content_added(&value, queue, bytes)?,
            "response.output_text.delta"
            | "response.reasoning.delta"
            | "response.refusal.delta" => self.content_delta(&value, queue, bytes)?,
            "response.reasoning_summary_text.delta" => self.summary_delta(&value, queue, bytes)?,
            "response.reasoning_text.delta" if self.hugging_face => {
                self.content_delta(&value, queue, bytes)?
            }
            "response.output_text.done" | "response.reasoning.done" | "response.refusal.done" => {
                self.content_done(&value, bytes)?
            }
            "response.reasoning_summary_text.done" => self.summary_done(&value, bytes)?,
            "response.reasoning_text.done" if self.hugging_face => {
                self.content_done(&value, bytes)?
            }
            "response.output_text.annotation.added" => {
                self.annotation_added(&value, queue, bytes)?
            }
            "response.content_part.done" => self.content_part_done(&value, queue, bytes)?,
            "response.function_call_arguments.delta" => {
                self.arguments_delta(&value, queue, bytes)?
            }
            "response.function_call_arguments.done" => self.arguments_done(&value, bytes)?,
            "response.output_item.done" => self.item_done(&value, queue, bytes)?,
            "response.completed" => self.terminal_response(value, "completed", bytes)?,
            "response.incomplete" => self.terminal_response(value, "incomplete", bytes)?,
            "response.failed" => self.failed_response(value, bytes)?,
            _ if kind.contains(':') => queue.push_back(Ok(StreamPart::ProviderEvent {
                name: kind.into(),
                data: value,
            })),
            _ => {
                return Err(event_error(
                    "unknown unprefixed Open Responses event",
                    bytes,
                ));
            }
        }
        Ok(())
    }
    fn ensure_progress(&mut self, bytes: u64) -> Result<(), ModelError> {
        if self.phase != Phase::InProgress {
            Err(event_error(
                "Open Responses output event before response progress",
                bytes,
            ))
        } else {
            Ok(())
        }
    }
    fn item_added(
        &mut self,
        value: &JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        self.ensure_progress(bytes)?;
        if self.incomplete_item.is_some() {
            return Err(event_error(
                "Open Responses cannot add an item after an incomplete item",
                bytes,
            ));
        }
        let index = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        if self.items.contains_key(&index) || index != self.items.len() {
            return Err(event_error(
                "Open Responses output items must be added once in contiguous order",
                bytes,
            ));
        }
        let item = value
            .get("item")
            .cloned()
            .ok_or_else(|| event_error("output_item.added is missing item", bytes))?;
        let id = required_string(&item, "id", bytes)?.to_owned();
        let kind = required_string(&item, "type", bytes)?.to_owned();
        if item.get("status").and_then(JsonValue::as_str) != Some("in_progress") {
            return Err(event_error(
                "added Open Responses item must be in_progress",
                bytes,
            ));
        }
        if kind == "message" {
            validate_message_phase(&item, bytes)?;
        } else if !matches!(
            kind.as_str(),
            "function_call" | "function_call_output" | "reasoning"
        ) && !kind.contains(':')
        {
            return Err(event_error("unsupported standard output item type", bytes));
        }
        if kind == "function_call" {
            let call_id = required_string(&item, "call_id", bytes)?;
            let name = required_string(&item, "name", bytes)?;
            if !valid_call_id(call_id) || !valid_function_name(name) {
                return Err(event_error(
                    "function item call ID or name violates Open Responses limits",
                    bytes,
                ));
            }
            if item.get("arguments").and_then(JsonValue::as_str) != Some("") {
                return Err(event_error(
                    "added function item must start with empty arguments",
                    bytes,
                ));
            }
            queue.push_back(Ok(StreamPart::ToolCallStart {
                id: call_id.into(),
                name: name.into(),
                metadata: None,
            }));
        }
        self.items.insert(
            index,
            ItemState {
                item,
                id,
                kind,
                contents: BTreeMap::new(),
                summaries: BTreeMap::new(),
                arguments: String::new(),
                arguments_done: false,
                done: false,
            },
        );
        Ok(())
    }
    fn content_added(
        &mut self,
        value: &JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let content = bounded_index(value, "content_index", MAX_CONTENT_PARTS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        if item.done {
            return Err(event_error("content part has no open item", bytes));
        }
        if content != item.contents.len() {
            return Err(event_error(
                "content parts must be added in contiguous order",
                bytes,
            ));
        }
        let part = value
            .get("part")
            .ok_or_else(|| event_error("content_part.added is missing part", bytes))?;
        let kind = required_string(part, "type", bytes)?.to_owned();
        if !matches!(
            (item.kind.as_str(), kind.as_str()),
            ("message", "output_text" | "refusal") | ("reasoning", "reasoning_text")
        ) {
            return Err(event_error(
                "content part type does not belong to its output item",
                bytes,
            ));
        }
        let initial = part
            .get(if kind == "refusal" { "refusal" } else { "text" })
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        if !initial.is_empty() {
            return Err(event_error(
                "added streamable content part must start empty",
                bytes,
            ));
        }
        let id = format!("{}:{content}", item.id);
        let deferred = kind == "output_text"
            && item
                .contents
                .values()
                .any(|state| state.kind == "refusal" && state.open);
        match kind.as_str() {
            "output_text" if !deferred => queue.push_back(Ok(StreamPart::TextStart {
                id,
                metadata: phase_metadata(&item.item),
            })),
            "output_text" => {}
            "reasoning_text" => queue.push_back(Ok(StreamPart::ReasoningStart {
                id,
                metadata: reasoning_metadata(&kind),
            })),
            "refusal" => {}
            _ => return Err(event_error("unsupported standard content part", bytes)),
        }
        item.contents.insert(
            content,
            ContentState {
                kind,
                text: String::new(),
                annotations: Vec::new(),
                deferred,
                open: true,
                done: false,
            },
        );
        Ok(())
    }
    fn summary_added(
        &mut self,
        value: &JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let summary = bounded_index(value, "summary_index", MAX_CONTENT_PARTS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        if item.done || item.kind != "reasoning" || summary != item.summaries.len() {
            return Err(event_error(
                "reasoning summary parts must be added once in contiguous order",
                bytes,
            ));
        }
        let part = value
            .get("part")
            .ok_or_else(|| event_error("reasoning_summary_part.added is missing part", bytes))?;
        if required_string(part, "type", bytes)? != "summary_text"
            || part.get("text").and_then(JsonValue::as_str) != Some("")
        {
            return Err(event_error(
                "added reasoning summary part must be empty summary_text",
                bytes,
            ));
        }
        let id = format!("{}:summary:{summary}", item.id);
        queue.push_back(Ok(StreamPart::ReasoningStart {
            id,
            metadata: reasoning_metadata("summary_text"),
        }));
        item.summaries.insert(
            summary,
            ContentState {
                kind: "summary_text".into(),
                text: String::new(),
                annotations: Vec::new(),
                deferred: false,
                open: true,
                done: false,
            },
        );
        Ok(())
    }
    fn summary_delta(
        &mut self,
        value: &JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let summary = bounded_index(value, "summary_index", MAX_CONTENT_PARTS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        let state = item
            .summaries
            .get_mut(&summary)
            .filter(|state| state.open && !state.done)
            .ok_or_else(|| event_error("reasoning summary delta has no open part", bytes))?;
        let delta = value
            .get("delta")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| event_error("reasoning summary delta is missing delta", bytes))?;
        state.text.push_str(delta);
        queue.push_back(Ok(StreamPart::ReasoningDelta {
            id: format!("{}:summary:{summary}", item.id),
            delta: delta.into(),
            metadata: reasoning_metadata("summary_text"),
        }));
        Ok(())
    }
    fn summary_done(&mut self, value: &JsonValue, bytes: u64) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let summary = bounded_index(value, "summary_index", MAX_CONTENT_PARTS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        let state = item
            .summaries
            .get_mut(&summary)
            .filter(|state| state.open && !state.done)
            .ok_or_else(|| event_error("reasoning summary done has no open part", bytes))?;
        let text = value
            .get("text")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| event_error("reasoning summary done is missing text", bytes))?;
        if state.text != text {
            return Err(event_error(
                "streamed and authoritative reasoning summary differ",
                bytes,
            ));
        }
        state.done = true;
        Ok(())
    }
    fn summary_part_done(
        &mut self,
        value: &JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let summary = bounded_index(value, "summary_index", MAX_CONTENT_PARTS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        let state = item
            .summaries
            .get_mut(&summary)
            .filter(|state| state.open && state.done)
            .ok_or_else(|| {
                event_error(
                    "reasoning summary part done requires matching text done",
                    bytes,
                )
            })?;
        let part = value
            .get("part")
            .ok_or_else(|| event_error("reasoning_summary_part.done is missing part", bytes))?;
        if required_string(part, "type", bytes)? != "summary_text"
            || part.get("text").and_then(JsonValue::as_str) != Some(state.text.as_str())
        {
            return Err(event_error(
                "reasoning summary part done contradicts its deltas",
                bytes,
            ));
        }
        state.open = false;
        queue.push_back(Ok(StreamPart::ReasoningEnd {
            id: format!("{}:summary:{summary}", item.id),
            metadata: reasoning_metadata("summary_text"),
        }));
        Ok(())
    }
    fn content_delta(
        &mut self,
        value: &JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let content = bounded_index(value, "content_index", MAX_CONTENT_PARTS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        let event = value.get("type").and_then(JsonValue::as_str).unwrap_or("");
        if self.hugging_face && !item.contents.contains_key(&content) {
            if content != item.contents.len() {
                return Err(event_error(
                    "Hugging Face content deltas must start in contiguous order",
                    bytes,
                ));
            }
            let kind = match (item.kind.as_str(), event) {
                ("message", "response.output_text.delta") => "output_text",
                ("reasoning", "response.reasoning_text.delta") => "reasoning_text",
                _ => {
                    return Err(event_error(
                        "Hugging Face delta has no matching content item",
                        bytes,
                    ));
                }
            };
            let id = format!("{}:{content}", item.id);
            if kind == "output_text" {
                queue.push_back(Ok(StreamPart::TextStart {
                    id,
                    metadata: phase_metadata(&item.item),
                }));
            } else {
                queue.push_back(Ok(StreamPart::ReasoningStart {
                    id,
                    metadata: reasoning_metadata(kind),
                }));
            }
            item.contents.insert(
                content,
                ContentState {
                    kind: kind.into(),
                    text: String::new(),
                    annotations: Vec::new(),
                    deferred: false,
                    open: true,
                    done: false,
                },
            );
        }
        let state = item
            .contents
            .get_mut(&content)
            .filter(|state| state.open && !state.done)
            .ok_or_else(|| event_error("content delta has no open content part", bytes))?;
        let matches_kind = match state.kind.as_str() {
            "output_text" => event == "response.output_text.delta",
            "reasoning_text" => {
                event == "response.reasoning.delta"
                    || (self.hugging_face && event == "response.reasoning_text.delta")
            }
            "refusal" => event == "response.refusal.delta",
            _ => false,
        };
        if !matches_kind {
            return Err(event_error(
                "content delta type does not match content part",
                bytes,
            ));
        }
        let delta = value
            .get("delta")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| event_error("content delta is missing delta", bytes))?;
        state.text.push_str(delta);
        let id = format!("{}:{content}", item.id);
        if state.kind == "output_text" {
            if !state.deferred {
                queue.push_back(Ok(StreamPart::TextDelta {
                    id,
                    delta: delta.into(),
                    metadata: phase_metadata(&item.item),
                }));
            }
        } else if state.kind != "refusal" {
            queue.push_back(Ok(StreamPart::ReasoningDelta {
                id,
                delta: delta.into(),
                metadata: reasoning_metadata(&state.kind),
            }));
        }
        Ok(())
    }
    fn annotation_added(
        &mut self,
        value: &JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let content = bounded_index(value, "content_index", MAX_CONTENT_PARTS, bytes)?;
        let annotation_index = bounded_index(value, "annotation_index", MAX_CONTENT_PARTS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        let state = item
            .contents
            .get_mut(&content)
            .filter(|state| state.kind == "output_text" && state.open)
            .ok_or_else(|| event_error("annotation has no open output text part", bytes))?;
        if annotation_index != state.annotations.len() {
            return Err(event_error(
                "annotations must be added in contiguous order",
                bytes,
            ));
        }
        let annotation = value
            .get("annotation")
            .cloned()
            .ok_or_else(|| event_error("annotation event is missing annotation", bytes))?;
        if !state.deferred
            && annotation.get("type").and_then(JsonValue::as_str) == Some("url_citation")
        {
            queue.push_back(Ok(StreamPart::Source {
                source: source_from_annotation(&annotation)?,
            }));
        }
        state.annotations.push(annotation);
        Ok(())
    }
    fn content_done(&mut self, value: &JsonValue, bytes: u64) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let content = bounded_index(value, "content_index", MAX_CONTENT_PARTS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        let state = item
            .contents
            .get_mut(&content)
            .filter(|state| state.open && !state.done)
            .ok_or_else(|| event_error("content done has no open content part", bytes))?;
        let event = value.get("type").and_then(JsonValue::as_str).unwrap_or("");
        let matches_kind = match state.kind.as_str() {
            "output_text" => event == "response.output_text.done",
            "reasoning_text" => {
                event == "response.reasoning.done"
                    || (self.hugging_face && event == "response.reasoning_text.done")
            }
            "refusal" => event == "response.refusal.done",
            _ => false,
        };
        if !matches_kind {
            return Err(event_error(
                "content done type does not match content part",
                bytes,
            ));
        }
        let final_text = value
            .get(if state.kind == "refusal" {
                "refusal"
            } else {
                "text"
            })
            .and_then(JsonValue::as_str)
            .ok_or_else(|| event_error("content done is missing authoritative text", bytes))?;
        if state.text != final_text {
            return Err(event_error(
                "streamed and authoritative content text differ",
                bytes,
            ));
        }
        state.done = true;
        Ok(())
    }
    fn content_part_done(
        &mut self,
        value: &JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let content = bounded_index(value, "content_index", MAX_CONTENT_PARTS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        let state = item
            .contents
            .get_mut(&content)
            .filter(|state| state.open && state.done)
            .ok_or_else(|| {
                event_error("content part done requires matching content done", bytes)
            })?;
        let part = value
            .get("part")
            .ok_or_else(|| event_error("content_part.done is missing part", bytes))?;
        if required_string(part, "type", bytes)? != state.kind {
            return Err(event_error("content_part.done changed content type", bytes));
        }
        let final_text = part
            .get(if state.kind == "refusal" {
                "refusal"
            } else {
                "text"
            })
            .and_then(JsonValue::as_str)
            .ok_or_else(|| event_error("content_part.done is missing text", bytes))?;
        if final_text != state.text {
            return Err(event_error("content_part.done contradicts deltas", bytes));
        }
        state.open = false;
        let id = format!("{}:{content}", item.id);
        if state.kind == "output_text" && !state.deferred {
            queue.push_back(Ok(StreamPart::TextEnd {
                id,
                metadata: phase_metadata(&item.item),
            }));
        } else if state.kind == "reasoning_text" {
            queue.push_back(Ok(StreamPart::ReasoningEnd {
                id,
                metadata: reasoning_metadata(&state.kind),
            }));
        } else if state.kind == "refusal" {
            queue.push_back(Ok(StreamPart::Custom {
                part: CustomPart {
                    kind: REFUSAL_KIND.into(),
                    data: state.text.clone().into(),
                    metadata: phase_metadata(&item.item),
                },
            }));
            flush_deferred_message_text(item, queue)?;
        }
        Ok(())
    }
    fn arguments_delta(
        &mut self,
        value: &JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        if item.kind != "function_call" || item.done || item.arguments_done {
            return Err(event_error(
                "function arguments delta has no open function item",
                bytes,
            ));
        }
        let delta = value
            .get("delta")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| event_error("function arguments delta is missing delta", bytes))?;
        item.arguments.push_str(delta);
        let id = required_string(&item.item, "call_id", bytes)?;
        queue.push_back(Ok(StreamPart::ToolCallDelta {
            id: id.into(),
            delta: delta.into(),
            metadata: None,
        }));
        Ok(())
    }
    fn arguments_done(&mut self, value: &JsonValue, bytes: u64) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let item = event_item_mut(&mut self.items, value, output, bytes)?;
        if item.kind != "function_call" || item.done || item.arguments_done {
            return Err(event_error(
                "function arguments done has no function item",
                bytes,
            ));
        }
        let arguments = value
            .get("arguments")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| event_error("function arguments done is missing arguments", bytes))?;
        if item.arguments != arguments {
            return Err(event_error(
                "streamed and authoritative function arguments differ",
                bytes,
            ));
        }
        item.arguments_done = true;
        Ok(())
    }
    fn item_done(
        &mut self,
        value: &JsonValue,
        queue: &mut VecDeque<StreamItem>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let output = bounded_index(value, "output_index", MAX_ITEMS, bytes)?;
        let item = self
            .items
            .get_mut(&output)
            .filter(|item| !item.done)
            .ok_or_else(|| event_error("output item done has no open item", bytes))?;
        if !self.hugging_face
            && (item.contents.values().any(|state| state.open)
                || item.summaries.values().any(|state| state.open))
        {
            return Err(event_error(
                "output item done arrived with open content parts",
                bytes,
            ));
        }
        let authoritative = value
            .get("item")
            .cloned()
            .ok_or_else(|| event_error("output_item.done is missing item", bytes))?;
        if required_string(&authoritative, "id", bytes)? != item.id
            || required_string(&authoritative, "type", bytes)? != item.kind
        {
            return Err(event_error("output_item.done identity changed", bytes));
        }
        if item.kind == "message" {
            validate_message_phase(&authoritative, bytes)?;
            if authoritative.get("phase") != item.item.get("phase") {
                return Err(event_error("output message phase changed", bytes));
            }
        }
        let terminal_status = authoritative
            .get("status")
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        if !matches!(terminal_status, "completed" | "incomplete") {
            return Err(event_error("done item requires terminal status", bytes));
        }
        if item.kind == "function_call" {
            if !item.arguments_done {
                return Err(event_error(
                    "function item completion requires function_call_arguments.done",
                    bytes,
                ));
            }
            let arguments = required_string(&authoritative, "arguments", bytes)?;
            if item.arguments != arguments {
                return Err(event_error(
                    "function item arguments contradict streamed arguments",
                    bytes,
                ));
            }
            let parsed: JsonValue = serde_json::from_str(arguments).map_err(|_| {
                ModelError::new(
                    ModelErrorKind::InvalidToolInput,
                    "Open Responses tool arguments are invalid JSON",
                )
                .with_stage(ErrorStage::StreamFinalize)
                .with_bytes_received(bytes)
            })?;
            if !parsed.is_object() {
                return Err(ModelError::new(
                    ModelErrorKind::InvalidToolInput,
                    "Open Responses tool arguments must be a JSON object",
                )
                .with_stage(ErrorStage::StreamFinalize)
                .with_bytes_received(bytes));
            }
            let call_id = required_string(&authoritative, "call_id", bytes)?;
            let name = required_string(&authoritative, "name", bytes)?;
            if item.item.get("call_id") != authoritative.get("call_id")
                || item.item.get("name") != authoritative.get("name")
            {
                return Err(event_error("function item identity changed", bytes));
            }
            queue.push_back(Ok(StreamPart::ToolCallEnd {
                id: call_id.into(),
                metadata: None,
            }));
            let mut call = oven_sdk::ToolCallPart::new(call_id, name, parsed);
            call.provider_item_id = Some(item.id.clone());
            call.raw_input = Some(arguments.into());
            queue.push_back(Ok(StreamPart::ToolCall { tool_call: call }));
        } else if item.kind == "function_call_output" {
            queue.push_back(Ok(StreamPart::ToolResult {
                tool_result: tool_result_from_item(&authoritative, bytes)?,
            }));
        } else if item.kind.contains(':') {
            queue.push_back(Ok(StreamPart::Custom {
                part: CustomPart::new(item.kind.clone(), authoritative.clone()),
            }));
        } else {
            validate_item_content(item, &authoritative, queue, bytes, self.hugging_face)?;
        }
        if self.hugging_face {
            close_hugging_face_item(item, queue);
        }
        if terminal_status == "incomplete" {
            self.incomplete_item = Some(output);
        }
        item.item = authoritative;
        item.done = true;
        Ok(())
    }
    fn terminal_response(
        &mut self,
        value: JsonValue,
        status: &str,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self.phase != Phase::InProgress {
            return Err(event_error(
                "terminal response requires response.in_progress",
                bytes,
            ));
        }
        if self.items.values().any(|item| !item.done) {
            return Err(event_error(
                "terminal response arrived before all items completed",
                bytes,
            ));
        }
        let response = value
            .get("response")
            .ok_or_else(|| event_error("terminal response event is missing response", bytes))?;
        if response.get("status").and_then(JsonValue::as_str) != Some(status) {
            return Err(event_error(
                "terminal response status contradicts event",
                bytes,
            ));
        }
        let output = response
            .get("output")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| event_error("terminal response is missing output array", bytes))?;
        let items = self
            .items
            .values()
            .map(|item| item.item.clone())
            .collect::<Vec<_>>();
        if output != &items {
            return Err(event_error(
                "terminal response output differs from completed items",
                bytes,
            ));
        }
        let incomplete_reason = response
            .pointer("/incomplete_details/reason")
            .and_then(JsonValue::as_str)
            .filter(|reason| !reason.is_empty());
        if status == "incomplete" {
            if incomplete_reason.is_none() {
                return Err(event_error(
                    "incomplete response requires incomplete_details.reason",
                    bytes,
                ));
            }
            if let Some(index) = self.incomplete_item
                && index + 1 != self.items.len()
            {
                return Err(event_error(
                    "an incomplete item must be the last response item",
                    bytes,
                ));
            }
        } else if self.incomplete_item.is_some() {
            return Err(event_error(
                "an incomplete item requires an incomplete response",
                bytes,
            ));
        } else if incomplete_reason.is_some() {
            return Err(event_error(
                "completed response must not include an incomplete reason",
                bytes,
            ));
        }
        if self.replay_reasoning
            && items
                .iter()
                .filter(|item| item.get("type").and_then(JsonValue::as_str) == Some("reasoning"))
                .any(|item| {
                    item.get("encrypted_content")
                        .and_then(JsonValue::as_str)
                        .is_none_or(str::is_empty)
                })
        {
            return Err(event_error(
                "provider-authoritative reasoning replay requires encrypted_content",
                bytes,
            ));
        }
        self.capture_response(response);
        let mut finish = Finish::new(
            usage_from(response.get("usage").unwrap_or(&JsonValue::Null)),
            if status == "incomplete" {
                match incomplete_reason {
                    Some("max_output_tokens") => FinishReason::Length,
                    Some("content_filter") => FinishReason::ContentFilter,
                    Some(value) => FinishReason::other(value),
                    None => FinishReason::Unknown,
                }
            } else if items
                .iter()
                .any(|item| item.get("type").and_then(JsonValue::as_str) == Some("function_call"))
            {
                FinishReason::ToolCalls
            } else {
                FinishReason::Stop
            },
        );
        finish.response_metadata = self.response_metadata.clone();
        if self.policy != ReplayPolicy::Never {
            let fingerprint = replay_fingerprint(&items)
                .ok_or_else(|| event_error("could not fingerprint replay items", bytes))?;
            let payload = serde_json::json!({"format":REPLAY_FORMAT,"binding":self.binding,"items":items,"fingerprint":fingerprint});
            match NativeReplayArtifact::new(self.adapter.clone(), self.scope.clone(), payload) {
                Ok(artifact) => {
                    finish.native_replay = Some(artifact);
                    finish.provider_metadata.insert(
                        "open_responses.replay_fingerprint".into(),
                        fingerprint.into(),
                    );
                }
                Err(_) if self.replay_capability == ReplayCapability::Optional => {}
                Err(_) => {
                    return Err(ModelError::replay(
                        "Open Responses required replay artifact exceeds its size limit",
                    )
                    .with_stage(ErrorStage::ReplayEncode));
                }
            }
        }
        self.terminal = Some(finish);
        self.phase = Phase::Terminal;
        Ok(())
    }
    fn failed_response(&mut self, value: JsonValue, bytes: u64) -> Result<(), ModelError> {
        if self.incomplete_item.is_some() {
            return Err(event_error(
                "an incomplete item requires response.incomplete",
                bytes,
            ));
        }
        if self
            .items
            .values()
            .any(|item| item.kind == "function_call" && !item.done)
        {
            return Err(event_error(
                "response.failed interrupted an open function call",
                bytes,
            ));
        }
        let response = value
            .get("response")
            .ok_or_else(|| event_error("response.failed is missing response", bytes))?;
        if response.get("status").and_then(JsonValue::as_str) != Some("failed") {
            return Err(event_error("response.failed requires failed status", bytes));
        }
        if self.stream_error.is_none() {
            let error_value = response.get("error").cloned().unwrap_or(JsonValue::Null);
            self.stream_error = Some(classify_error(
                None,
                error_value.to_string().as_bytes(),
                None,
                ErrorStage::StreamEvent,
                bytes,
                &HeaderMap::new(),
            ));
        }
        self.capture_response(response);
        let mut finish = Finish::new(
            usage_from(response.get("usage").unwrap_or(&JsonValue::Null)),
            FinishReason::Error,
        );
        finish.response_metadata = self.response_metadata.clone();
        self.terminal = Some(finish);
        self.phase = Phase::Terminal;
        Ok(())
    }
    fn stream_error(
        &mut self,
        error: ModelError,
        value: &JsonValue,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self.phase == Phase::Initial || self.stream_error.is_some() {
            return Err(event_error(
                "Open Responses error event is out of order",
                bytes,
            ));
        }
        self.sequence(value, bytes)?;
        self.stream_error = Some(error);
        Ok(())
    }
    fn done(&mut self, queue: &mut VecDeque<StreamItem>, bytes: u64) -> Result<(), ModelError> {
        if self.done || self.phase != Phase::Terminal {
            return Err(event_error(
                "[DONE] requires exactly one preceding terminal response event",
                bytes,
            ));
        }
        if let Some(error) = self.stream_error.take() {
            for (item_index, item) in &mut self.items {
                let phase = phase_metadata(&item.item);
                for (content_index, state) in &mut item.contents {
                    if state.open {
                        state.open = false;
                        let id = format!("{}:{content_index}", item.id);
                        if state.kind == "output_text" {
                            queue.push_back(Ok(StreamPart::TextEnd {
                                id,
                                metadata: phase.clone(),
                            }));
                        } else if state.kind != "refusal" {
                            queue.push_back(Ok(StreamPart::ReasoningEnd {
                                id,
                                metadata: reasoning_metadata(&state.kind),
                            }));
                        }
                    }
                }
                for (summary_index, state) in &mut item.summaries {
                    if state.open {
                        state.open = false;
                        queue.push_back(Ok(StreamPart::ReasoningEnd {
                            id: format!("{}:summary:{summary_index}", item.id),
                            metadata: reasoning_metadata("summary_text"),
                        }));
                    }
                }
                let _ = item_index;
            }
            queue.push_back(Ok(StreamPart::Error { error }));
        }
        let finish = self
            .terminal
            .take()
            .ok_or_else(|| event_error("[DONE] has no terminal finish", bytes))?;
        if let Some(fingerprint) = finish
            .provider_metadata
            .get("open_responses.replay_fingerprint")
            .and_then(JsonValue::as_str)
        {
            queue.push_back(Ok(StreamPart::Custom {
                part: CustomPart::new(REPLAY_FINGERPRINT_KIND, fingerprint.into()),
            }));
        }
        queue.push_back(Ok(StreamPart::Finish { finish }));
        self.done = true;
        Ok(())
    }
    fn capture_response(&mut self, response: &JsonValue) {
        for (source, key) in [
            ("id", "open_responses.response_id"),
            ("model", "open_responses.model"),
            ("created_at", "open_responses.created_at"),
            ("service_tier", "open_responses.service_tier"),
        ] {
            if let Some(value) = response
                .get(source)
                .cloned()
                .filter(|value| !value.is_null())
            {
                self.response_metadata.insert(key.into(), value);
            }
        }
    }
}

fn event_item_mut<'a>(
    items: &'a mut BTreeMap<usize, ItemState>,
    value: &JsonValue,
    output: usize,
    bytes: u64,
) -> Result<&'a mut ItemState, ModelError> {
    let item = items
        .get_mut(&output)
        .ok_or_else(|| event_error("Open Responses event has no output item", bytes))?;
    if required_string(value, "item_id", bytes)? != item.id {
        return Err(event_error(
            "Open Responses event item_id does not match its output item",
            bytes,
        ));
    }
    Ok(item)
}

fn event_response<'a>(
    value: &'a JsonValue,
    event: &str,
    status: &str,
    bytes: u64,
) -> Result<&'a JsonValue, ModelError> {
    let response = value
        .get("response")
        .filter(|response| response.is_object())
        .ok_or_else(|| event_error(&format!("{event} is missing response object"), bytes))?;
    if response.get("status").and_then(JsonValue::as_str) != Some(status) {
        return Err(event_error(
            &format!("{event} response status must be {status}"),
            bytes,
        ));
    }
    Ok(response)
}

fn validate_message_phase(item: &JsonValue, bytes: u64) -> Result<(), ModelError> {
    if item
        .get("phase")
        .and_then(JsonValue::as_str)
        .is_some_and(|phase| !matches!(phase, "commentary" | "final_answer"))
        || item.get("phase").is_some_and(|phase| !phase.is_string())
    {
        return Err(event_error(
            "assistant message phase must be commentary or final_answer",
            bytes,
        ));
    }
    Ok(())
}

fn phase_metadata(item: &JsonValue) -> Option<BTreeMap<String, JsonValue>> {
    item.get("phase")
        .and_then(JsonValue::as_str)
        .map(|phase| BTreeMap::from([(PHASE_METADATA_KEY.into(), phase.into())]))
}

fn reasoning_metadata(kind: &str) -> Option<BTreeMap<String, JsonValue>> {
    Some(BTreeMap::from([(
        "open_responses.kind".into(),
        kind.into(),
    )]))
}

fn close_hugging_face_item(item: &mut ItemState, queue: &mut VecDeque<StreamItem>) {
    let phase = phase_metadata(&item.item);
    for (content_index, state) in &mut item.contents {
        if !state.open {
            continue;
        }
        state.open = false;
        state.done = true;
        let id = format!("{}:{content_index}", item.id);
        if state.kind == "output_text" {
            queue.push_back(Ok(StreamPart::TextEnd {
                id,
                metadata: phase.clone(),
            }));
        } else if state.kind == "reasoning_text" {
            queue.push_back(Ok(StreamPart::ReasoningEnd {
                id,
                metadata: reasoning_metadata(&state.kind),
            }));
        }
    }
}

fn flush_deferred_message_text(
    item: &mut ItemState,
    queue: &mut VecDeque<StreamItem>,
) -> Result<(), ModelError> {
    let next_refusal = item
        .contents
        .iter()
        .find(|(_, state)| state.kind == "refusal" && state.open)
        .map(|(index, _)| *index)
        .unwrap_or(usize::MAX);
    let metadata = phase_metadata(&item.item);
    for (content_index, state) in &mut item.contents {
        if *content_index >= next_refusal || state.kind != "output_text" || !state.deferred {
            continue;
        }
        state.deferred = false;
        let id = format!("{}:{content_index}", item.id);
        queue.push_back(Ok(StreamPart::TextStart {
            id: id.clone(),
            metadata: metadata.clone(),
        }));
        if !state.text.is_empty() {
            queue.push_back(Ok(StreamPart::TextDelta {
                id: id.clone(),
                delta: state.text.clone(),
                metadata: metadata.clone(),
            }));
        }
        for annotation in &state.annotations {
            if annotation.get("type").and_then(JsonValue::as_str) == Some("url_citation") {
                queue.push_back(Ok(StreamPart::Source {
                    source: source_from_annotation(annotation)?,
                }));
            }
        }
        if !state.open {
            queue.push_back(Ok(StreamPart::TextEnd {
                id,
                metadata: metadata.clone(),
            }));
        }
    }
    Ok(())
}

fn tool_result_from_item(item: &JsonValue, bytes: u64) -> Result<ToolResultPart, ModelError> {
    let call_id = required_string(item, "call_id", bytes)?;
    let status = required_string(item, "status", bytes)?;
    if !matches!(status, "completed" | "incomplete") {
        return Err(event_error(
            "function output item requires terminal status",
            bytes,
        ));
    }
    let output = item
        .get("output")
        .ok_or_else(|| event_error("function output item is missing output", bytes))?;
    let content = if let Some(text) = output.as_str() {
        ToolContent::Text(text.into())
    } else if let Some(values) = output.as_array() {
        ToolContent::Mixed(
            values
                .iter()
                .map(|value| content_value_from_output(value, bytes))
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        return Err(event_error(
            "function output must be a string or content array",
            bytes,
        ));
    };
    Ok(ToolResultPart::new(call_id, content))
}

fn content_value_from_output(value: &JsonValue, bytes: u64) -> Result<ContentValue, ModelError> {
    match required_string(value, "type", bytes)? {
        "input_text" => Ok(ContentValue::Text(
            required_string(value, "text", bytes)?.into(),
        )),
        "input_image" => Ok(ContentValue::File(file_from_url_field(
            value,
            "image_url",
            true,
            bytes,
        )?)),
        "input_file" => {
            if value.get("file_data").and_then(JsonValue::as_str).is_some() {
                file_from_data_url(
                    required_string(value, "file_data", bytes)?,
                    value.get("filename").and_then(JsonValue::as_str),
                    false,
                    bytes,
                )
                .map(ContentValue::File)
            } else {
                file_from_url_field(value, "file_url", false, bytes).map(ContentValue::File)
            }
        }
        _ => Err(event_error(
            "unsupported function output content type",
            bytes,
        )),
    }
}

fn file_from_url_field(
    value: &JsonValue,
    field: &str,
    image: bool,
    bytes: u64,
) -> Result<FilePart, ModelError> {
    let raw = required_string(value, field, bytes)?;
    if raw.starts_with("data:") {
        return file_from_data_url(
            raw,
            value.get("filename").and_then(JsonValue::as_str),
            image,
            bytes,
        );
    }
    let limit = response_media_limit(image);
    if raw.len() > limit {
        return Err(event_error(
            "function output media URL exceeds its protocol limit",
            bytes,
        ));
    }
    let url = url::Url::parse(raw)
        .map_err(|_| event_error("function output media URL is invalid", bytes))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(event_error(
            "function output media URL must use HTTP(S)",
            bytes,
        ));
    }
    let mut file = FilePart::new(
        if image {
            "image/*"
        } else {
            "application/octet-stream"
        },
        FileSource::Url(url),
    );
    file.filename = value
        .get("filename")
        .and_then(JsonValue::as_str)
        .map(str::to_owned);
    Ok(file)
}

fn file_from_data_url(
    value: &str,
    filename: Option<&str>,
    image: bool,
    bytes: u64,
) -> Result<FilePart, ModelError> {
    let limit = response_media_limit(image);
    let (media_type, data) = decode_function_output_data_url(value, image, limit, limit, bytes)?;
    let mut file = FilePart::new(media_type, FileSource::Bytes(data.into()));
    file.filename = filename.map(str::to_owned);
    Ok(file)
}

fn decode_function_output_data_url(
    value: &str,
    image: bool,
    encoded_limit: usize,
    decoded_limit: usize,
    bytes: u64,
) -> Result<(&str, Vec<u8>), ModelError> {
    if value.len() > encoded_limit {
        return Err(event_error(
            "function output encoded data URL exceeds its protocol limit",
            bytes,
        ));
    }
    let encoded = value
        .strip_prefix("data:")
        .and_then(|value| value.split_once(";base64,"))
        .ok_or_else(|| event_error("function output data URL is invalid", bytes))?;
    let (media_type, payload) = encoded;
    if !response_media_type_allowed(media_type, image) {
        return Err(event_error(
            "function output data URL MIME type is unsupported",
            bytes,
        ));
    }
    let decoded_len = base64_decoded_len(payload)
        .filter(|length| *length <= decoded_limit)
        .ok_or_else(|| {
            event_error(
                "function output decoded data URL exceeds its protocol limit",
                bytes,
            )
        })?;
    let mut data = Vec::with_capacity(decoded_len);
    STANDARD
        .decode_vec(payload, &mut data)
        .map_err(|_| event_error("function output data URL base64 is invalid", bytes))?;
    if data.len() != decoded_len {
        return Err(event_error(
            "function output data URL decoded length is invalid",
            bytes,
        ));
    }
    Ok((media_type, data))
}

fn response_media_limit(image: bool) -> usize {
    if image {
        MAX_INLINE_IMAGE_BYTES
    } else {
        MAX_INLINE_FILE_BYTES
    }
}

fn response_media_type_allowed(media_type: &str, image: bool) -> bool {
    if image {
        IMAGE_MEDIA_TYPES.contains(&media_type)
    } else {
        media_type == "application/pdf"
    }
}

fn base64_decoded_len(value: &str) -> Option<usize> {
    if value.is_empty() || value.len() & 3 != 0 {
        return None;
    }
    let padding = value
        .as_bytes()
        .iter()
        .rev()
        .take_while(|byte| **byte == b'=')
        .count();
    if padding > 2 {
        return None;
    }
    value
        .len()
        .checked_div(4)?
        .checked_mul(3)?
        .checked_sub(padding)
}

fn validate_item_content(
    item: &ItemState,
    authoritative: &JsonValue,
    queue: &mut VecDeque<StreamItem>,
    bytes: u64,
    hugging_face: bool,
) -> Result<(), ModelError> {
    if item.kind == "message" {
        if authoritative.get("role").and_then(JsonValue::as_str) != Some("assistant") {
            return Err(event_error("output message requires assistant role", bytes));
        }
        let content = authoritative
            .get("content")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| event_error("output message requires content", bytes))?;
        if !hugging_face && content.len() != item.contents.len() {
            return Err(event_error(
                "output message content count differs from streamed parts",
                bytes,
            ));
        }
        for (index, part) in content.iter().enumerate() {
            let kind = required_string(part, "type", bytes)?;
            let text = part
                .get(if kind == "refusal" { "refusal" } else { "text" })
                .and_then(JsonValue::as_str)
                .ok_or_else(|| event_error("message content is missing text", bytes))?;
            if let Some(state) = item.contents.get(&index) {
                if kind != state.kind || text != state.text {
                    return Err(event_error(
                        "message content differs from streamed content",
                        bytes,
                    ));
                }
            } else if !hugging_face {
                return Err(event_error("missing streamed message content part", bytes));
            } else if kind == "output_text" {
                let id = format!("{}:{index}", item.id);
                let metadata = phase_metadata(authoritative);
                queue.push_back(Ok(StreamPart::TextStart {
                    id: id.clone(),
                    metadata: metadata.clone(),
                }));
                queue.push_back(Ok(StreamPart::TextDelta {
                    id: id.clone(),
                    delta: text.into(),
                    metadata: metadata.clone(),
                }));
                queue.push_back(Ok(StreamPart::TextEnd { id, metadata }));
            }
            if kind == "output_text" {
                let annotations = part
                    .get("annotations")
                    .and_then(JsonValue::as_array)
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                let streamed_annotations = item
                    .contents
                    .get(&index)
                    .map(|state| state.annotations.as_slice())
                    .unwrap_or_default();
                if !streamed_annotations.is_empty()
                    && streamed_annotations.iter().collect::<Vec<_>>() != annotations
                {
                    return Err(event_error(
                        "message annotations differ from streamed annotations",
                        bytes,
                    ));
                }
                if streamed_annotations.is_empty() {
                    for annotation in annotations {
                        if annotation.get("type").and_then(JsonValue::as_str)
                            == Some("url_citation")
                        {
                            queue.push_back(Ok(StreamPart::Source {
                                source: source_from_annotation(annotation)?,
                            }));
                        }
                    }
                }
            } else if kind == "refusal" {
                if !item.contents.contains_key(&index) {
                    queue.push_back(Ok(StreamPart::Custom {
                        part: CustomPart {
                            kind: REFUSAL_KIND.into(),
                            data: text.into(),
                            metadata: phase_metadata(authoritative),
                        },
                    }));
                }
            } else {
                return Err(event_error("unsupported message content type", bytes));
            }
        }
    } else if item.kind == "reasoning" {
        validate_reasoning_content(item, authoritative, queue, bytes, hugging_face)?;
    } else {
        return Err(event_error("unsupported standard output item type", bytes));
    }
    Ok(())
}

fn validate_reasoning_content(
    item: &ItemState,
    authoritative: &JsonValue,
    queue: &mut VecDeque<StreamItem>,
    bytes: u64,
    hugging_face: bool,
) -> Result<(), ModelError> {
    let summaries = authoritative
        .get("summary")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| event_error("reasoning item requires summary", bytes))?;
    let contents = authoritative
        .get("content")
        .and_then(JsonValue::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if !hugging_face
        && (summaries.len() != item.summaries.len() || contents.len() != item.contents.len())
    {
        return Err(event_error(
            "reasoning item part counts differ from streamed lifecycle",
            bytes,
        ));
    }
    for (index, part) in summaries.iter().enumerate() {
        let kind = required_string(part, "type", bytes)?;
        if kind != "summary_text" && !(hugging_face && kind == "reasoning_summary") {
            return Err(event_error("unsupported reasoning summary content", bytes));
        }
        validate_or_emit_reasoning(
            item.summaries.get(&index),
            part,
            kind,
            format!("{}:summary:{index}", item.id),
            queue,
            bytes,
            hugging_face,
        )?;
    }
    for (index, part) in contents.iter().enumerate() {
        let kind = required_string(part, "type", bytes)?;
        if kind != "reasoning_text" {
            return Err(event_error("unsupported reasoning content", bytes));
        }
        validate_or_emit_reasoning(
            item.contents.get(&index),
            part,
            kind,
            format!("{}:{index}", item.id),
            queue,
            bytes,
            hugging_face,
        )?;
    }
    Ok(())
}

fn validate_or_emit_reasoning(
    streamed: Option<&ContentState>,
    part: &JsonValue,
    kind: &str,
    id: String,
    queue: &mut VecDeque<StreamItem>,
    bytes: u64,
    hugging_face: bool,
) -> Result<(), ModelError> {
    let text = required_string(part, "text", bytes)?;
    let expected_kind = if kind == "reasoning_text" {
        "reasoning_text"
    } else {
        "summary_text"
    };
    if let Some(streamed) = streamed {
        if streamed.text != text || streamed.kind != expected_kind {
            return Err(event_error(
                "reasoning item differs from streamed content",
                bytes,
            ));
        }
        return Ok(());
    }
    if !hugging_face {
        return Err(event_error("missing streamed reasoning part", bytes));
    }
    let metadata = reasoning_metadata(kind);
    queue.push_back(Ok(StreamPart::ReasoningStart {
        id: id.clone(),
        metadata: metadata.clone(),
    }));
    queue.push_back(Ok(StreamPart::ReasoningDelta {
        id: id.clone(),
        delta: text.into(),
        metadata: metadata.clone(),
    }));
    queue.push_back(Ok(StreamPart::ReasoningEnd { id, metadata }));
    Ok(())
}

fn source_from_annotation(annotation: &JsonValue) -> Result<SourcePart, ModelError> {
    let mut source = SourcePart::new();
    source.url = annotation
        .get("url")
        .and_then(JsonValue::as_str)
        .map(url::Url::parse)
        .transpose()
        .map_err(|_| ModelError::invalid_response("Open Responses citation URL is invalid"))?;
    source.title = annotation
        .get("title")
        .and_then(JsonValue::as_str)
        .map(str::to_owned);
    source.excerpt = None;
    source.metadata = annotation
        .as_object()
        .map(|object| object.clone().into_iter().collect());
    Ok(source)
}
fn bounded_index(
    value: &JsonValue,
    field: &str,
    limit: usize,
    bytes: u64,
) -> Result<usize, ModelError> {
    value
        .get(field)
        .and_then(JsonValue::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value < limit)
        .ok_or_else(|| event_error(&format!("Open Responses event has invalid {field}"), bytes))
}
fn required_string<'a>(
    value: &'a JsonValue,
    field: &str,
    bytes: u64,
) -> Result<&'a str, ModelError> {
    value
        .get(field)
        .and_then(JsonValue::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| event_error(&format!("Open Responses object is missing {field}"), bytes))
}
fn event_error(message: &str, bytes: u64) -> ModelError {
    ModelError::invalid_response(message)
        .with_stage(ErrorStage::StreamEvent)
        .with_bytes_received(bytes)
}

fn unexpected_eof(bytes: u64) -> ModelError {
    ModelError::unexpected_eof("Open Responses stream ended before terminal [DONE]")
        .with_stage(ErrorStage::StreamRead)
        .with_bytes_received(bytes)
}

fn usage_from(value: &JsonValue) -> Usage {
    let input = value.get("input_tokens").and_then(JsonValue::as_u64);
    let cached = value
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(JsonValue::as_u64);
    let output = value.get("output_tokens").and_then(JsonValue::as_u64);
    let reasoning = value
        .pointer("/output_tokens_details/reasoning_tokens")
        .and_then(JsonValue::as_u64);
    Usage {
        input_tokens: input,
        input_tokens_no_cache: input
            .zip(cached)
            .map(|(input, cached)| input.saturating_sub(cached)),
        input_tokens_cache_read: cached,
        input_tokens_cache_write: None,
        output_tokens: output,
        output_tokens_text: output
            .zip(reasoning)
            .map(|(output, reasoning)| output.saturating_sub(reasoning)),
        output_tokens_reasoning: reasoning,
        raw: (!value.is_null()).then(|| value.clone()),
    }
}
fn replay_fingerprint(items: &[JsonValue]) -> Option<String> {
    serde_json::to_vec(items)
        .ok()
        .map(Sha256::digest)
        .map(|digest| URL_SAFE_NO_PAD.encode(digest))
}
fn replay_semantics(items: &[JsonValue]) -> Option<JsonValue> {
    let mut semantics = Vec::new();
    for item in items {
        match item.get("type")?.as_str()? {
            "message" => {
                for part in item.get("content")?.as_array()? {
                    match part.get("type")?.as_str()? {
                        "output_text" => {
                            semantics.push(serde_json::json!({
                                "type":"text",
                                "text":part.get("text")?,
                                "phase":item.get("phase")
                            }));
                            for annotation in part
                                .get("annotations")
                                .and_then(JsonValue::as_array)
                                .into_iter()
                                .flatten()
                            {
                                semantics.push(serde_json::json!({
                                    "type":"source",
                                    "annotation":annotation
                                }));
                            }
                        }
                        "refusal" => semantics.push(serde_json::json!({
                            "type":"refusal",
                            "text":part.get("refusal")?,
                            "phase":item.get("phase")
                        })),
                        _ => return None,
                    }
                }
            }
            "reasoning" => {
                for part in item
                    .get("summary")
                    .and_then(JsonValue::as_array)
                    .into_iter()
                    .flatten()
                {
                    semantics.push(serde_json::json!({
                        "type":"reasoning",
                        "kind":part.get("type")?.as_str()?,
                        "text": part.get("text")?.as_str()?,
                    }));
                }
                for part in item
                    .get("content")
                    .and_then(JsonValue::as_array)
                    .into_iter()
                    .flatten()
                {
                    semantics.push(serde_json::json!({
                        "type":"reasoning",
                        "kind":part.get("type")?.as_str()?,
                        "text": part.get("text")?.as_str()?,
                    }));
                }
            }
            "function_call" => semantics.push(serde_json::json!({
                "type":"tool_call",
                "item_id": item.get("id")?.as_str()?,
                "call_id": item.get("call_id")?.as_str()?,
                "name": item.get("name")?.as_str()?,
                "arguments": item.get("arguments")?.as_str()?,
            })),
            "function_call_output" => semantics.push(serde_json::json!({
                "type":"tool_result",
                "call_id":item.get("call_id")?,
                "output":item.get("output")?
            })),
            kind if kind.contains(':') => semantics.push(serde_json::json!({
                "type":"custom",
                "kind":kind,
                "data":item
            })),
            _ => return None,
        }
    }
    Some(JsonValue::Array(semantics))
}
fn normalized_semantics(parts: &[AssistantPart]) -> Option<JsonValue> {
    let mut semantics = Vec::new();
    for part in parts {
        match part {
            AssistantPart::Text(part) => semantics.push(serde_json::json!({
                "type":"text",
                "text":part.text,
                "phase":part.metadata.as_ref().and_then(|metadata|metadata.get(PHASE_METADATA_KEY))
            })),
            AssistantPart::Reasoning(part) => semantics.push(serde_json::json!({
                "type":"reasoning",
                "kind":part
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("open_responses.kind"))
                    .and_then(JsonValue::as_str)
                    .unwrap_or("summary_text"),
                "text": part.text,
            })),
            AssistantPart::ToolCall(part) => semantics.push(serde_json::json!({
                "type":"tool_call",
                "item_id": part.provider_item_id,
                "call_id": part.id,
                "name": part.name,
                "arguments": part.raw_input.clone().unwrap_or_else(|| part.input.to_string()),
            })),
            AssistantPart::Source(part) => {
                semantics.push(serde_json::json!({
                    "type":"source",
                    "annotation":source_annotation(part).ok()?
                }));
            }
            AssistantPart::ToolResult(part) => {
                let item = function_output(part).ok()?;
                semantics.push(serde_json::json!({
                    "type":"tool_result",
                    "call_id":item.get("call_id")?,
                    "output":item.get("output")?
                }));
            }
            AssistantPart::Custom(part) if part.kind == REPLAY_FINGERPRINT_KIND => {}
            AssistantPart::Custom(part) if part.kind == REFUSAL_KIND => {
                semantics.push(serde_json::json!({
                    "type":"refusal",
                    "text":part.data.as_str()?,
                    "phase":part.metadata.as_ref().and_then(|metadata|metadata.get(PHASE_METADATA_KEY))
                }));
            }
            AssistantPart::Custom(part) => semantics.push(serde_json::json!({
                "type":"custom",
                "kind":part.kind,
                "data":custom_item(part).ok()?
            })),
            AssistantPart::File(_) | AssistantPart::ToolApproval(_) => return None,
        }
    }
    Some(JsonValue::Array(semantics))
}
fn decode_replay(
    artifact: &NativeReplayArtifact,
    normalized: &[AssistantPart],
    binding: &JsonValue,
) -> Option<Vec<JsonValue>> {
    let object = artifact.payload().as_object()?;
    let keys = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if keys
        != ["binding", "fingerprint", "format", "items"]
            .into_iter()
            .collect()
        || object.get("format")?.as_str()? != REPLAY_FORMAT
        || object.get("binding")? != binding
    {
        return None;
    }
    let items = object.get("items")?.as_array()?.clone();
    let fingerprint = object.get("fingerprint")?.as_str()?;
    if replay_fingerprint(&items)?.as_str() != fingerprint {
        return None;
    }
    let normalized_fingerprints = normalized
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Custom(part) if part.kind == REPLAY_FINGERPRINT_KIND => {
                part.data.as_str()
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if normalized_fingerprints.as_slice() != [fingerprint]
        || replay_semantics(&items)? != normalized_semantics(normalized)?
    {
        return None;
    }
    Some(items)
}

fn validate_endpoint(endpoint: &ApiEndpoint) -> Result<(), ModelError> {
    let url = endpoint.as_url();
    let loopback = url.scheme() == "http"
        && url
            .host_str()
            .is_some_and(|host| matches!(host, "localhost" | "127.0.0.1" | "::1"));
    if (url.scheme() != "https" && !loopback)
        || url.query().is_some()
        || !url.path().ends_with("/responses")
    {
        Err(ModelError::invalid_request(
            "Open Responses endpoint must be an HTTPS full /responses URL without query (HTTP loopback is allowed for tests)",
        ))
    } else {
        Ok(())
    }
}
fn validate_settings(
    settings: &OpenResponsesSettings,
    capabilities: &ModelCapabilities,
) -> Result<(), ModelError> {
    match &settings.transport {
        OpenResponsesTransport::Generic { profile } => {
            validate_label(Some(profile), "generic profile")?
        }
        OpenResponsesTransport::HuggingFace(profile) => {
            validate_label(Some(&profile.routing), "Hugging Face routing")?
        }
    }
    validate_label(settings.reasoning_summary.as_deref(), "reasoning summary")?;
    for value in &settings.include {
        validate_label(Some(value), "include value")?;
    }
    if settings.parallel_tool_calls && !capabilities.features.contains(Capability::PARALLEL_TOOLS) {
        return Err(ModelError::invalid_request(
            "parallel_tool_calls setting requires parallel-tools capability",
        ));
    }
    if settings.reasoning_summary.is_some()
        && !capabilities.features.contains(Capability::REASONING)
    {
        return Err(ModelError::invalid_request(
            "reasoning summary setting requires reasoning capability",
        ));
    }
    Ok(())
}
fn validate_capabilities(
    transport: &OpenResponsesTransport,
    capabilities: &ModelCapabilities,
) -> Result<(), ModelError> {
    capabilities.validate()?;
    if capabilities.cancellation != CancellationCapability::LocalOnly {
        return Err(ModelError::invalid_request(
            "Open Responses declares local-only cancellation",
        ));
    }
    if capabilities.compaction != CompactionCapability::Unsupported {
        return Err(ModelError::invalid_request(
            "standardized Open Responses and Hugging Face transports do not inherit OpenAI native compaction",
        ));
    }
    let text = oven_sdk::Modality::text();
    if !capabilities.modalities.input.contains(&text)
        || capabilities.modalities.output != [text.clone()].into_iter().collect()
    {
        return Err(ModelError::invalid_request(
            "Open Responses requires text input and text-only output",
        ));
    }
    for modality in &capabilities.modalities.input {
        if !matches!(modality.as_str(), "text" | "image" | "pdf") {
            return Err(ModelError::invalid_request(
                "Open Responses implements only text, image, and PDF input modalities",
            ));
        }
        if matches!(transport, OpenResponsesTransport::HuggingFace(_)) && modality.as_str() == "pdf"
        {
            return Err(ModelError::invalid_request(
                "Hugging Face Open Responses profile implements image but not PDF input",
            ));
        }
    }
    if capabilities.features.contains(Capability::PROVIDER_TOOLS) {
        return Err(ModelError::invalid_request(
            "standardized Open Responses crate does not implement provider-hosted tools",
        ));
    }
    if capabilities
        .features
        .contains(Capability::MAX_OUTPUT_TOKENS)
        && capabilities.limits.output.is_some_and(|limit| limit < 16)
    {
        return Err(ModelError::invalid_request(
            "Open Responses declared output-token limits must be at least 16",
        ));
    }
    for (modality, support) in &capabilities.media.input {
        let valid = match modality.as_str() {
            "image" => support
                .media_types
                .iter()
                .all(|value| value.starts_with("image/")),
            "pdf" => support
                .media_types
                .iter()
                .all(|value| value == "application/pdf"),
            _ => false,
        };
        if !valid
            || support.sources.intersects(
                MediaSourceSupport::INLINE_TEXT | MediaSourceSupport::PROVIDER_REFERENCE,
            )
        {
            return Err(ModelError::invalid_request(
                "Open Responses media declarations exceed implemented byte-or-HTTP(S)-URL support",
            ));
        }
    }
    Ok(())
}
fn validate_function_schema(schema: &JsonValue) -> Result<(), ModelError> {
    if !schema.is_object() {
        return Err(ModelError::invalid_request(
            "Open Responses function schemas must be JSON objects",
        ));
    }
    Ok(())
}
fn validate_label(value: Option<&str>, name: &str) -> Result<(), ModelError> {
    if value.is_some_and(|value| value.trim().is_empty() || value.chars().any(char::is_control)) {
        Err(ModelError::invalid_request(format!(
            "Open Responses {name} must be a non-empty open string without control characters"
        )))
    } else {
        Ok(())
    }
}
fn reject_protected_headers(headers: &HeaderMap) -> Result<(), ModelError> {
    if ["authorization", "host", "content-type", "content-length"]
        .iter()
        .any(|name| headers.contains_key(*name))
    {
        Err(ModelError::invalid_request(
            "Open Responses authentication and transport headers are protected",
        ))
    } else {
        Ok(())
    }
}

fn replay_seed(
    provider: &ProviderConfig<OpenResponsesAuth>,
    model: &oven_sdk::ModelDeclaration,
    settings: &OpenResponsesSettings,
) -> Result<Sha256, ModelError> {
    let mut hasher = Sha256::new();
    hash_field(
        &mut hasher,
        "version",
        NATIVE_CONTEXT_SCOPE_VERSION.as_bytes(),
    );
    hash_field(&mut hasher, "provider", provider.id.as_str().as_bytes());
    hash_field(&mut hasher, "model", model.id.as_str().as_bytes());
    hash_field(
        &mut hasher,
        "endpoint",
        provider.api.as_url().as_str().as_bytes(),
    );
    hash_json(&mut hasher, "capabilities", &model.capabilities)?;
    hash_json(&mut hasher, "transport", &settings.transport)?;
    hash_field(
        &mut hasher,
        "strict_json_schema",
        &[settings.strict_json_schema as u8],
    );
    hash_field(&mut hasher, "strict_tools", &[settings.strict_tools as u8]);
    hash_field(
        &mut hasher,
        "parallel_tool_calls",
        &[settings.parallel_tool_calls as u8],
    );
    hash_field(&mut hasher, "store", &[settings.store as u8]);
    hash_json(&mut hasher, "include", &settings.include)?;
    hash_json(
        &mut hasher,
        "reasoning_summary",
        &settings.reasoning_summary,
    )?;
    Ok(hasher)
}
fn hash_json(hasher: &mut Sha256, tag: &str, value: &impl Serialize) -> Result<(), ModelError> {
    let bytes = serde_json::to_vec(value).map_err(|_| {
        ModelError::invalid_request("could not encode Open Responses replay scope inputs")
    })?;
    hash_field(hasher, tag, &bytes);
    Ok(())
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
            timeout_message: "Open Responses error body idle timeout",
            abort_message: "Open Responses error body read was aborted",
            read_message: "Open Responses error body read failed",
            overflow_message: "Open Responses error body byte count overflowed",
        },
        tokio::time::sleep(idle),
        move |timer| timer.reset(tokio::time::Instant::now() + idle),
    )
    .await
}
fn classify_error(
    status: Option<u16>,
    body: &[u8],
    request_id: Option<String>,
    stage: ErrorStage,
    bytes: u64,
    headers: &HeaderMap,
) -> ModelError {
    let value: JsonValue = serde_json::from_slice(body).unwrap_or(JsonValue::Null);
    let code = value
        .pointer("/error/code")
        .or_else(|| value.get("code"))
        .and_then(JsonValue::as_str);
    let kind = value
        .pointer("/error/type")
        .or_else(|| value.get("type"))
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    let lower = format!("{} {}", code.unwrap_or(""), kind).to_ascii_lowercase();
    let mut error = if status == Some(401)
        || lower.contains("authentication")
        || lower.contains("invalid_api_key")
        || lower.contains("unauthorized")
    {
        ModelError::new(ModelErrorKind::Auth, "Open Responses request failed")
    } else if status == Some(403) || lower.contains("permission") {
        ModelError::new(
            ModelErrorKind::PermissionDenied,
            "Open Responses request failed",
        )
    } else if status == Some(404) || lower.contains("not_found") {
        ModelError::new(
            ModelErrorKind::ModelNotFound,
            "Open Responses request failed",
        )
    } else if lower.contains("context") {
        ModelError::new(
            ModelErrorKind::ContextLength,
            "Open Responses request failed",
        )
    } else if lower.contains("quota") {
        ModelError::new(ModelErrorKind::Quota, "Open Responses request failed")
    } else if status == Some(429)
        || lower.contains("too_many_requests")
        || lower.contains("rate_limit")
    {
        ModelError::rate_limited("Open Responses request failed")
    } else if matches!(status, Some(408 | 504)) || lower.contains("timeout") {
        ModelError::timeout("Open Responses request failed")
    } else if status == Some(503) || lower.contains("overload") {
        ModelError::new(ModelErrorKind::Overload, "Open Responses request failed")
            .with_retryable(true)
    } else if lower.contains("content_filter") {
        ModelError::new(
            ModelErrorKind::ContentFilter,
            "Open Responses request failed",
        )
    } else if status.is_some_and(|status| status >= 500) || lower.contains("server_error") {
        ModelError::provider("Open Responses request failed").with_retryable(true)
    } else if lower.contains("model_error") {
        ModelError::provider("Open Responses request failed")
    } else if status.is_some_and(|status| (400..500).contains(&status))
        || lower.contains("invalid_request")
    {
        ModelError::invalid_request("Open Responses request failed")
    } else {
        ModelError::provider("Open Responses request failed")
    };
    error = error.with_stage(stage).with_bytes_received(bytes);
    if let Some(status) = status {
        error = error.with_http_status(status);
    }
    if let Some(code) = code.filter(|code| {
        !code.is_empty()
            && code.len() <= 128
            && code
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    }) {
        error = error.with_vendor_code(code);
    }
    if let Some(id) = request_id {
        error = error.with_request_id(id);
    }
    if let Some(delay) = parse_retry_after(headers, &[]) {
        error = error.with_retry_after(delay);
    }
    if let Some(body) = sanitized_error_body(&value, body) {
        error = error.with_sanitized_body(body);
    }
    error
}

fn sanitized_error_body(value: &JsonValue, raw: &[u8]) -> Option<SanitizedBody> {
    let error = value.get("error").unwrap_or(value);
    if let Some(object) = error.as_object() {
        let mut safe = serde_json::Map::new();
        for key in ["type", "code", "param", "message"] {
            if let Some(text) = object.get(key).and_then(JsonValue::as_str) {
                safe.insert(key.into(), sanitize_provider_text(text).into());
            }
        }
        if !safe.is_empty() {
            return serde_json::to_string(&JsonValue::Object(safe))
                .ok()
                .map(SanitizedBody::new);
        }
    }
    (!raw.is_empty()).then(|| SanitizedBody::new("[non-JSON provider error body omitted]"))
}

fn sanitize_provider_text(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_control() && !matches!(character, '\n' | '\r' | '\t') {
                '\u{fffd}'
            } else {
                character
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parser_requires_event_and_preserves_utf8() {
        let mut parser =
            SseParser::new("Open Responses SSE contains invalid UTF-8").clear_name_on_empty_event();
        let input="event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"hé\"}}\n\n".as_bytes();
        let mut events = Vec::new();
        for byte in input {
            events.extend(parser.feed(&[*byte]).unwrap());
        }
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "response.created");
    }
    #[test]
    fn open_labels_are_not_enums() {
        assert!(validate_label(Some("future-provider-label"), "label").is_ok());
    }
    #[test]
    fn response_data_urls_are_bounded_before_decode_allocation() {
        assert!(
            decode_function_output_data_url("data:image/png;base64,AAAA", true, 8, 8, 0,).is_err()
        );
        assert!(
            decode_function_output_data_url("data:image/png;base64,AAAA", true, 64, 2, 0,).is_err()
        );
        assert!(
            decode_function_output_data_url("data:text/plain;base64,AAAA", true, 64, 64, 0,)
                .is_err()
        );
        assert_eq!(
            decode_function_output_data_url("data:image/png;base64,AAAA", true, 64, 64, 0,)
                .unwrap()
                .1,
            [0, 0, 0]
        );
    }
}
