//! Explicit registry-free Google language-model construction.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, BoxFuture, CompactionCapability,
    CompleteResult, CompletedTurn, ErrorStage, Finish, JsonValue, LanguageModel,
    LanguageModelDescriptor, ModelConfig, ModelError, ModelId, ModelIdentity, NativeContextScope,
    ProviderConfig, ProviderMetadata, ReasoningPart, Request, RequestMetadata, ResourceId,
    SecretString, StreamPart, StreamResponse, TextPart,
};
use reqwest::{
    Client,
    header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use sha2::{Digest as _, Sha256};
use url::Url;

use crate::{GOOGLE_GENERATE_CONTENT_ADAPTER_ID, GOOGLE_PROVIDER_ID, transport::GoogleTimeouts};

/// Explicit Google AI Studio API-key authentication.
#[derive(Clone, Debug)]
pub struct GoogleApiKeyAuth {
    api_key: SecretString,
}

impl GoogleApiKeyAuth {
    /// Wraps a caller-resolved API key. Empty keys are rejected by [`GoogleModel::new`].
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::new(api_key),
        }
    }
}

/// Explicit request-level thinking controls accepted by a configured endpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GoogleThinkingSettings {
    /// Reject all Google thinking options.
    Unsupported,
    /// Accept `thinkingBudget` and reject `thinkingLevel`.
    Budget {
        /// Exact normalized effort label to Google thinking-budget mapping.
        effort_budgets: BTreeMap<String, i64>,
    },
    /// Accept `thinkingLevel` and reject `thinkingBudget`.
    Level {
        /// Exact normalized effort label to Google thinking-level mapping.
        effort_levels: BTreeMap<String, String>,
    },
}

/// Explicit wire behavior for client and provider tools.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GoogleToolSettings {
    /// Whether strict client functions may use validated mode.
    pub strict_functions: bool,
    /// Whether client functions may be mixed with provider-executed tools.
    pub mixed_client_and_provider_tools: bool,
    /// Whether normalized current-turn client calls may use Google's documented sentinel.
    pub current_turn_signature_sentinel: bool,
}

/// Structural settings for one configured Google `generateContent` endpoint.
#[derive(Clone, Debug)]
pub struct GoogleGenerateContentSettings {
    /// Exact provider model resource placed before the method suffix, for example
    /// `models/gemini-2.5-flash`.
    pub model_resource: String,
    /// Adapter transport timeouts.
    pub timeouts: GoogleTimeouts,
    /// Explicit thinking behavior; never inferred from the model ID.
    pub thinking: GoogleThinkingSettings,
    /// Explicit tool behavior; never inferred from the model ID.
    pub tools: GoogleToolSettings,
}

#[derive(Clone)]
struct Config {
    provider: ProviderConfig<GoogleApiKeyAuth>,
    settings: GoogleGenerateContentSettings,
    descriptor: LanguageModelDescriptor,
    client: Client,
    native_context_scope: NativeContextScope,
    base_headers: HeaderMap,
}

/// A configured Google Gemini `generateContent` model.
#[derive(Clone)]
pub struct GoogleModel {
    config: Arc<Config>,
}

impl GoogleModel {
    /// Creates one model solely from explicit 0.4 configuration.
    pub fn new(
        config: ModelConfig<GoogleApiKeyAuth, GoogleGenerateContentSettings>,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        if config.provider.id.as_str() != GOOGLE_PROVIDER_ID {
            return Err(ModelError::invalid_request(
                "Google provider configuration must use provider ID `google`",
            ));
        }
        validate_auth(&config.provider.auth)?;
        validate_endpoint(config.provider.api.as_url())?;
        validate_model_resource(&config.settings.model_resource)?;
        validate_settings(&config)?;
        if config.model.capabilities.compaction == CompactionCapability::Native {
            return Err(ModelError::unsupported(
                "Google AI Studio does not support provider-native context compaction",
            ));
        }
        let native_context_scope = derived_native_context_scope(
            &config.provider,
            &config.model.id,
            &config.settings.model_resource,
        )?;
        let client = Client::builder()
            .connect_timeout(config.settings.timeouts.connect)
            .build()
            .map_err(|_| ModelError::transport("could not construct Google HTTP client"))?;
        let identity = ModelIdentity::new(config.provider.id.clone(), config.model.id.clone())?;
        let descriptor = LanguageModelDescriptor::new(
            identity,
            AdapterId::new(GOOGLE_GENERATE_CONTENT_ADAPTER_ID),
            config.model.capabilities.clone(),
        )?;
        let mut base_headers = config.provider.headers.static_headers.as_map().clone();
        base_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(Self {
            config: Arc::new(Config {
                provider: config.provider,
                settings: config.settings,
                descriptor,
                client,
                native_context_scope,
                base_headers,
            }),
        })
    }

    /// Returns the explicitly configured model identity.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        &self.config.descriptor.identity.model_id
    }

    /// Returns the internally derived native-context scope used by replay.
    #[must_use]
    pub fn native_context_scope(&self) -> &NativeContextScope {
        &self.config.native_context_scope
    }

    /// Calls the non-streaming `generateContent` method and returns a normalized completed turn.
    pub fn generate_content<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<CompleteResult, ModelError>> {
        Box::pin(async move {
            let response = self.execute(request, abort, false).await?;
            collect_direct(response).await
        })
    }

    fn endpoint(&self, streaming: bool) -> Result<Url, ModelError> {
        let method = if streaming {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        Url::parse(&format!(
            "{}/{}:{method}",
            self.config
                .provider
                .api
                .as_url()
                .as_str()
                .trim_end_matches('/'),
            self.config.settings.model_resource
        ))
        .map_err(|_| {
            ModelError::invalid_request("Google Gemini endpoint URL is invalid")
                .with_stage(ErrorStage::RequestEncoding)
        })
    }

    fn execute<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
        streaming: bool,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        Box::pin(async move {
            let options = crate::request::options(&request)?;
            crate::request::validate_request(
                &request,
                &options,
                &self.config.descriptor.capabilities,
                &self.config.provider.id,
                &self.config.settings,
            )?;
            if abort.is_aborted() {
                return Err(ModelError::abort("request was aborted before dispatch")
                    .with_stage(ErrorStage::Connect));
            }
            let encoded = crate::request::encode_request(
                &request,
                &options,
                &self.config.descriptor,
                &self.config.settings,
                &self.config.native_context_scope,
            )?;
            let body = serde_json::to_vec(&encoded.body).map_err(|_| {
                ModelError::invalid_request("could not serialize Google Gemini request")
                    .with_stage(ErrorStage::RequestEncoding)
            })?;
            crate::request::validate_request_body_size(body.len())?;
            let mut headers = self.config.base_headers.clone();
            if let Some(provider) = &self.config.provider.headers.dynamic_headers {
                let dynamic = provider.headers(&request.header_context)?;
                headers.extend(dynamic.as_map().clone());
            }
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            if !headers.contains_key("x-goog-api-key") {
                headers.insert(
                    HeaderName::from_static("x-goog-api-key"),
                    HeaderValue::from_str(self.config.provider.auth.api_key.expose_secret())
                        .map_err(|_| {
                            ModelError::invalid_request(
                                "Google Gemini API key is not a valid header value",
                            )
                        })?,
                );
            }
            let send = self
                .config
                .client
                .post(self.endpoint(streaming)?)
                .headers(headers)
                .body(body)
                .send();
            let response = tokio::select! {
                value = tokio::time::timeout(self.config.settings.timeouts.headers, send) => value
                    .map_err(|_| ModelError::timeout("Google response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
                    .map_err(|_| ModelError::transport("Google Gemini request failed").with_stage(ErrorStage::Connect))?,
                _ = abort.aborted() => return Err(ModelError::abort("request was aborted before response headers").with_stage(ErrorStage::ResponseHeaders)),
            };
            let mut head = crate::transport::response_head(&response);
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let request_id = head.request_id.clone();
                let error_headers = response.headers().clone();
                let (body, count) = crate::transport::read_body(
                    response,
                    &abort,
                    self.config.settings.timeouts.stream_idle,
                    oven_sdk::SanitizedBody::MAX_BYTES,
                )
                .await?;
                return Err(crate::error::classify_error(
                    status,
                    &body,
                    request_id,
                    ErrorStage::ResponseBody,
                    count,
                    &error_headers,
                ));
            }
            let request_metadata = RequestMetadata {
                replay: encoded.replay,
                provider_metadata: ProviderMetadata::new(),
            };
            if !streaming {
                const MAX_SUCCESS_BODY: usize = 32 * 1024 * 1024;
                let request_id = head.request_id.clone();
                let (body, count) = crate::transport::read_body(
                    response,
                    &abort,
                    self.config.settings.timeouts.stream_idle,
                    MAX_SUCCESS_BODY,
                )
                .await?;
                if count > MAX_SUCCESS_BODY as u64 {
                    return Err(ModelError::invalid_response(
                        "Google generateContent response exceeds 32 MiB",
                    )
                    .with_stage(ErrorStage::ResponseBody)
                    .with_bytes_received(count));
                }
                let value: JsonValue = serde_json::from_slice(&body).map_err(|_| {
                    ModelError::invalid_response("Google generateContent response is invalid JSON")
                        .with_stage(ErrorStage::ResponseBody)
                        .with_bytes_received(count)
                })?;
                let (mut parts, metadata) = crate::stream::normalize_single(
                    value,
                    self.config.descriptor.capabilities.replay.policy,
                    self.config.native_context_scope.clone(),
                    request_id,
                    request.stream_options.include_raw,
                    count,
                )?;
                if let Some(StreamPart::StreamStart { warnings }) = parts.first_mut() {
                    *warnings = encoded.warnings;
                }
                head.response_metadata.extend(metadata);
                return Ok(StreamResponse::new(Box::pin(futures_util::stream::iter(
                    parts.into_iter().map(Ok),
                )))
                .with_request(request_metadata)
                .with_response(head));
            }
            let mut live = crate::stream::LiveState {
                bytes: Box::pin(response.bytes_stream()),
                parser: crate::sse::Parser::default(),
                state: crate::stream::State::new(
                    self.config.descriptor.capabilities.replay.policy,
                    self.config.native_context_scope.clone(),
                ),
                queue: VecDeque::from([Ok(StreamPart::StreamStart {
                    warnings: encoded.warnings,
                })]),
                pending_events: VecDeque::new(),
                pending_error: None,
                deadline: oven_sdk::provider_support::StreamReadDeadline::new(
                    tokio::time::sleep(self.config.settings.timeouts.stream_idle),
                    &abort,
                ),
                idle: self.config.settings.timeouts.stream_idle,
                count: 0,
                eof: false,
                include_raw: request.stream_options.include_raw,
            };
            live.state.set_request_id(head.request_id.clone());
            crate::stream::early_peek(&mut live).await?;
            head.response_metadata
                .extend(live.state.response_metadata().clone());
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
                    if let Err(error) = crate::stream::read_live(&mut live, false).await {
                        live.pending_error = Some(error);
                    }
                }
            });
            Ok(StreamResponse::new(Box::pin(stream))
                .with_request(request_metadata)
                .with_response(head))
        })
    }
}

impl LanguageModel for GoogleModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.config.descriptor
    }

    fn validate_request(&self, request: &Request) -> Result<(), ModelError> {
        let options = crate::request::options(request)?;
        crate::request::validate_request(
            request,
            &options,
            &self.config.descriptor.capabilities,
            &self.config.provider.id,
            &self.config.settings,
        )
    }

    fn supports_request(&self, request: &Request) -> bool {
        self.validate_request(request).is_ok()
    }

    fn stream<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        self.execute(request, abort, true)
    }
}

async fn collect_direct(mut response: StreamResponse) -> Result<CompleteResult, ModelError> {
    use futures_util::StreamExt as _;

    let mut content = Vec::new();
    let mut warnings = Vec::new();
    let mut finish: Option<Finish> = None;
    while let Some(item) = response.stream.next().await {
        match item? {
            StreamPart::StreamStart { warnings: value } => warnings = value,
            StreamPart::TextDelta { delta, .. } => {
                content.push(AssistantPart::Text(TextPart::new(delta)));
            }
            StreamPart::ReasoningDelta { delta, .. } => {
                content.push(AssistantPart::Reasoning(ReasoningPart::new(delta)));
            }
            StreamPart::ToolCall { tool_call } => {
                content.push(AssistantPart::ToolCall(tool_call));
            }
            StreamPart::File { file } => content.push(AssistantPart::File(file)),
            StreamPart::Source { source } => content.push(AssistantPart::Source(source)),
            StreamPart::Custom { part } => content.push(AssistantPart::Custom(part)),
            StreamPart::Error { error } => return Err(error),
            StreamPart::Finish { finish: value } => finish = Some(value),
            _ => {}
        }
    }
    let finish = finish.ok_or_else(|| {
        ModelError::unexpected_eof("Google generateContent ended without Finish")
            .with_stage(ErrorStage::StreamFinalize)
    })?;
    Ok(CompleteResult {
        turn: CompletedTurn {
            message: AssistantMessage::new(content),
            finish,
            warnings,
        },
        request: response.request,
        response: response.response,
    })
}

fn validate_auth(auth: &GoogleApiKeyAuth) -> Result<(), ModelError> {
    if auth.api_key.expose_secret().trim().is_empty() {
        return Err(ModelError::invalid_request(
            "Google Gemini requires an explicit non-empty API key",
        ));
    }
    Ok(())
}

fn validate_endpoint(endpoint: &Url) -> Result<(), ModelError> {
    if endpoint.query().is_some() {
        return Err(ModelError::invalid_request(
            "Google API endpoint must not include a query",
        ));
    }
    Ok(())
}

fn validate_model_resource(resource: &str) -> Result<(), ModelError> {
    let Some(id) = resource.strip_prefix("models/") else {
        return Err(ModelError::invalid_request(
            "Google model_resource must use `models/{id}`",
        ));
    };
    if id.is_empty()
        || id.contains('/')
        || id.contains(':')
        || id.contains('?')
        || id.contains('#')
        || id.chars().any(char::is_whitespace)
    {
        return Err(ModelError::invalid_request(
            "Google model_resource must use `models/{id}`",
        ));
    }
    Ok(())
}

fn validate_settings(
    config: &ModelConfig<GoogleApiKeyAuth, GoogleGenerateContentSettings>,
) -> Result<(), ModelError> {
    let features = config.model.capabilities.features;
    if config.settings.thinking != GoogleThinkingSettings::Unsupported
        && !features.contains(oven_sdk::Capability::REASONING)
    {
        return Err(ModelError::invalid_request(
            "Google thinking settings require declared reasoning support",
        ));
    }
    if (config.settings.tools.strict_functions
        || config.settings.tools.mixed_client_and_provider_tools
        || config.settings.tools.current_turn_signature_sentinel)
        && !features.contains(oven_sdk::Capability::TOOL_CALLING)
    {
        return Err(ModelError::invalid_request(
            "Google tool settings require declared tool-calling support",
        ));
    }
    if config.settings.tools.mixed_client_and_provider_tools
        && !features.contains(oven_sdk::Capability::PROVIDER_TOOLS)
    {
        return Err(ModelError::invalid_request(
            "mixed Google tools require declared provider-tool support",
        ));
    }
    Ok(())
}

fn derived_native_context_scope(
    provider: &ProviderConfig<GoogleApiKeyAuth>,
    model_id: &ModelId,
    model_resource: &str,
) -> Result<NativeContextScope, ModelError> {
    const SCOPE_VERSION: &str = "oven-sdk-google-generate-content-replay-v1";
    let endpoint = canonical_endpoint(provider.api.as_url());
    let mut hasher = Sha256::new();
    for component in [
        SCOPE_VERSION.as_bytes(),
        endpoint.as_bytes(),
        b"generateContent",
        model_resource.as_bytes(),
    ] {
        hasher.update((component.len() as u64).to_be_bytes());
        hasher.update(component);
    }
    let resource = ResourceId::new(format!(
        "google-generate-content-v1-sha256:{}",
        hex::encode(hasher.finalize())
    ))?;
    NativeContextScope::new(provider.id.clone(), model_id.clone(), resource)
}

fn canonical_endpoint(endpoint: &Url) -> String {
    let mut canonical = endpoint.clone();
    let path = canonical.path().trim_end_matches('/').to_owned();
    canonical.set_path(if path.is_empty() { "/" } else { &path });
    canonical.set_query(None);
    canonical.to_string()
}
