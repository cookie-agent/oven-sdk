//! Explicit registry-free Vertex configuration and language-model implementation.

use std::{
    collections::VecDeque,
    future::Future,
    sync::{Arc, Mutex},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use oven_sdk::{
    AbortSignal, AdapterId, ApiEndpoint, BoxFuture, CompactionCapability, CompleteResult,
    ErrorStage, LanguageModel, LanguageModelDescriptor, ModelCapabilities, ModelConfig, ModelError,
    ModelErrorKind, ModelId, ModelIdentity, NativeContextScope, ProviderConfig, ProviderId,
    ProviderMetadata, Request, RequestMetadata, ResourceId, SecretString, StreamPart,
    StreamResponse,
};
use reqwest::{
    Client,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use url::Url;

use crate::{GOOGLE_VERTEX_GENERATE_CONTENT_ADAPTER_ID, transport::GoogleVertexTimeouts};

/// Caller-managed asynchronous OAuth token provider.
pub trait VertexTokenProvider: Send + Sync {
    /// Resolves a current bearer token. Implementations own caching and refresh.
    fn token(&self) -> BoxFuture<'static, Result<SecretString, ModelError>>;
}

impl<F, Fut> VertexTokenProvider for F
where
    F: Fn() -> Fut + Send + Sync,
    Fut: Future<Output = Result<SecretString, ModelError>> + Send + 'static,
{
    fn token(&self) -> BoxFuture<'static, Result<SecretString, ModelError>> {
        Box::pin(self())
    }
}

/// Explicit Vertex authentication selection.
#[derive(Clone)]
pub enum VertexAuth {
    /// A static bearer token, mainly useful for short-lived jobs and tests.
    AccessToken(SecretString),
    /// A caller-managed refreshing token provider.
    TokenProvider(Arc<dyn VertexTokenProvider>),
}

impl std::fmt::Debug for VertexAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AccessToken(_) => formatter.write_str("AccessToken(<redacted>)"),
            Self::TokenProvider(_) => formatter.write_str("TokenProvider(<opaque>)"),
        }
    }
}

/// Typed Vertex resource path; model identity never determines this value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GoogleVertexResource {
    /// A publisher model resource.
    PublisherModel {
        /// Publisher resource segment.
        publisher: String,
        /// Publisher model resource segment.
        model: String,
    },
    /// A deployed endpoint resource.
    Endpoint {
        /// Endpoint resource segment.
        endpoint: String,
    },
}

impl GoogleVertexResource {
    fn path(&self) -> String {
        match self {
            Self::PublisherModel { publisher, model } => {
                format!("publishers/{publisher}/models/{model}")
            }
            Self::Endpoint { endpoint } => format!("endpoints/{endpoint}"),
        }
    }

    fn validate(&self) -> Result<(), ModelError> {
        match self {
            Self::PublisherModel { publisher, model } => {
                validate_resource_segment(publisher, "publisher")?;
                validate_resource_segment(model, "publisher model")
            }
            Self::Endpoint { endpoint } => validate_resource_segment(endpoint, "endpoint"),
        }
    }
}

const NATIVE_CONTEXT_SCOPE_VERSION: &str = "google.vertex.generate-content.native-context-scope.v1";

/// Builds the exact native-context scope for an explicit Vertex API endpoint and resource.
///
/// The resource ID is a stable SHA-256 fingerprint. It binds replay to the canonical API
/// endpoint, project, location, and typed resource without serializing any of those values.
pub fn google_vertex_native_context_scope(
    provider_id: ProviderId,
    model_id: ModelId,
    api: &ApiEndpoint,
    project: &str,
    location: &str,
    resource: &GoogleVertexResource,
) -> Result<NativeContextScope, ModelError> {
    validate_resource_segment(project, "project")?;
    validate_resource_segment(location, "location")?;
    resource.validate()?;
    if api.as_url().query().is_some() {
        return Err(ModelError::invalid_request(
            "Vertex API endpoint must not include a query",
        ));
    }

    let mut endpoint = api.as_url().clone();
    let path = endpoint.path().trim_end_matches('/').to_owned();
    endpoint.set_path(if path.is_empty() { "/" } else { &path });

    let mut binding = Vec::new();
    hash_field(
        &mut binding,
        "version",
        NATIVE_CONTEXT_SCOPE_VERSION.as_bytes(),
    );
    hash_field(&mut binding, "api", endpoint.as_str().as_bytes());
    hash_field(&mut binding, "project", project.as_bytes());
    hash_field(&mut binding, "location", location.as_bytes());
    match resource {
        GoogleVertexResource::PublisherModel { publisher, model } => {
            hash_field(&mut binding, "resource_type", b"publisher_model");
            hash_field(&mut binding, "publisher", publisher.as_bytes());
            hash_field(&mut binding, "model", model.as_bytes());
        }
        GoogleVertexResource::Endpoint { endpoint } => {
            hash_field(&mut binding, "resource_type", b"endpoint");
            hash_field(&mut binding, "endpoint", endpoint.as_bytes());
        }
    }
    let fingerprint = URL_SAFE_NO_PAD.encode(sha256(&binding));
    NativeContextScope::new(
        provider_id,
        model_id,
        ResourceId::new(format!(
            "{NATIVE_CONTEXT_SCOPE_VERSION}.sha256.{fingerprint}"
        ))?,
    )
}

/// Explicit Vertex thinking-control wire family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GoogleVertexThinkingMode {
    /// Thinking controls are rejected.
    Unsupported,
    /// `thinkingBudget` is accepted and `thinkingLevel` is rejected.
    Budget,
    /// `thinkingLevel` is accepted and `thinkingBudget` is rejected.
    Level,
}

/// Explicit Vertex tool behavior independent of model identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GoogleVertexToolSettings {
    /// Whether provider-executed tools are enabled.
    pub provider_tools: bool,
    /// Whether client functions and provider tools may be mixed.
    pub mixed_client_and_provider_tools: bool,
    /// Whether strict validated client functions are enabled.
    pub strict_functions: bool,
}

/// Explicit Vertex media count, source, and inline-size limits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GoogleVertexMediaSettings {
    /// Maximum image inputs.
    pub max_images: usize,
    /// Maximum HTTPS image inputs.
    pub max_https_images: usize,
    /// Maximum document inputs.
    pub max_documents: usize,
    /// Maximum audio inputs.
    pub max_audio: usize,
    /// Maximum video inputs.
    pub max_videos: usize,
    /// Maximum HTTPS video inputs.
    pub max_https_videos: usize,
    /// Maximum inline image bytes.
    pub max_inline_image_bytes: usize,
    /// Maximum inline PDF bytes.
    pub max_inline_pdf_bytes: usize,
    /// Maximum inline text bytes.
    pub max_inline_text_bytes: usize,
    /// Explicit URL schemes accepted for media, such as `https` and `gs`.
    pub url_schemes: Vec<String>,
}

/// Vertex-specific structural settings supplied explicitly by the caller.
#[derive(Clone, Debug)]
pub struct GoogleVertexSettings {
    /// Google Cloud project resource segment.
    pub project: String,
    /// Google Cloud location resource segment.
    pub location: String,
    /// Typed publisher-model or endpoint resource.
    pub resource: GoogleVertexResource,
    /// Explicit thinking-control family.
    pub thinking: GoogleVertexThinkingMode,
    /// Explicit tool behavior.
    pub tools: GoogleVertexToolSettings,
    /// Whether Vertex streamed `partialArgs` are requested and normalized.
    pub stream_function_call_arguments: bool,
    /// Explicit media limits and URL schemes.
    pub media: GoogleVertexMediaSettings,
    /// Exact provider/model/resource native-context scope used by replay.
    pub native_context_scope: NativeContextScope,
    /// Optional caller-built HTTP client.
    pub client: Option<Client>,
    /// Adapter-controlled phase timeouts.
    pub timeouts: GoogleVertexTimeouts,
}

#[derive(Clone)]
struct Config {
    provider: ProviderConfig<VertexAuth>,
    settings: GoogleVertexSettings,
    descriptor: LanguageModelDescriptor,
    client: Client,
    base_headers: HeaderMap,
}

/// A configured Vertex Gemini publisher model or deployed endpoint.
#[derive(Clone)]
pub struct GoogleVertexModel {
    config: Arc<Config>,
}

impl GoogleVertexModel {
    /// Constructs one Vertex model from explicit provider, declaration, and structural settings.
    pub fn new(config: ModelConfig<VertexAuth, GoogleVertexSettings>) -> Result<Self, ModelError> {
        config.validate()?;
        validate_resource_segment(&config.settings.project, "project")?;
        validate_resource_segment(&config.settings.location, "location")?;
        config.settings.resource.validate()?;
        validate_headers(config.provider.headers.static_headers.as_map())?;
        if config.provider.api.as_url().query().is_some() {
            return Err(ModelError::invalid_request(
                "Vertex API endpoint must not include a query",
            ));
        }
        if matches!(&config.provider.auth, VertexAuth::AccessToken(value) if value.expose_secret().trim().is_empty())
        {
            return Err(ModelError::invalid_request(
                "Vertex access token must not be empty",
            ));
        }
        if config.model.capabilities.compaction == CompactionCapability::Native {
            return Err(ModelError::unsupported(
                "Vertex does not support provider-native context compaction",
            ));
        }
        let expected_scope = google_vertex_native_context_scope(
            config.provider.id.clone(),
            config.model.id.clone(),
            &config.provider.api,
            &config.settings.project,
            &config.settings.location,
            &config.settings.resource,
        )?;
        if config.settings.native_context_scope != expected_scope {
            return Err(ModelError::invalid_request(
                "Vertex replay scope must match the configured provider, model, API endpoint, project, location, and resource",
            ));
        }
        let features = config.model.capabilities.features;
        if config.settings.stream_function_call_arguments
            != features.contains(oven_sdk::Capability::TOOL_INPUT_DELTAS)
        {
            return Err(ModelError::invalid_request(
                "Vertex partial-argument settings must match TOOL_INPUT_DELTAS",
            ));
        }
        if config.settings.tools.provider_tools
            != features.contains(oven_sdk::Capability::PROVIDER_TOOLS)
        {
            return Err(ModelError::invalid_request(
                "Vertex provider-tool settings must match PROVIDER_TOOLS",
            ));
        }
        if (config.settings.tools.strict_functions
            || config.settings.tools.mixed_client_and_provider_tools)
            && !features.contains(oven_sdk::Capability::TOOL_CALLING)
        {
            return Err(ModelError::invalid_request(
                "Vertex strict or mixed tool settings require TOOL_CALLING",
            ));
        }
        if config.settings.tools.mixed_client_and_provider_tools
            && !config.settings.tools.provider_tools
        {
            return Err(ModelError::invalid_request(
                "Vertex mixed tool settings require provider tools",
            ));
        }
        if config.settings.thinking != GoogleVertexThinkingMode::Unsupported
            && !features.contains(oven_sdk::Capability::REASONING)
        {
            return Err(ModelError::invalid_request(
                "Vertex thinking controls require REASONING",
            ));
        }
        if config
            .settings
            .media
            .url_schemes
            .iter()
            .any(|scheme| scheme.trim().is_empty())
        {
            return Err(ModelError::invalid_request(
                "Vertex media URL schemes must not be empty",
            ));
        }
        let identity = ModelIdentity::new(config.provider.id.clone(), config.model.id.clone())?;
        let descriptor = LanguageModelDescriptor::new(
            identity,
            AdapterId::new(GOOGLE_VERTEX_GENERATE_CONTENT_ADAPTER_ID),
            config.model.capabilities.clone(),
        )?;
        let client = match &config.settings.client {
            Some(client) => client.clone(),
            None => Client::builder()
                .connect_timeout(config.settings.timeouts.connect)
                .build()
                .map_err(|_| ModelError::transport("could not construct Vertex HTTP client"))?,
        };
        let mut base_headers = config.provider.headers.static_headers.as_map().clone();
        base_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(Self {
            config: Arc::new(Config {
                provider: config.provider,
                settings: config.settings,
                descriptor,
                client,
                base_headers,
            }),
        })
    }

    /// Returns the explicitly declared model identity.
    #[must_use]
    pub fn model_id(&self) -> &oven_sdk::ModelId {
        &self.config.descriptor.identity.model_id
    }

    /// Returns the exact configured native-context scope used by replay.
    #[must_use]
    pub fn native_context_scope(&self) -> &NativeContextScope {
        &self.config.settings.native_context_scope
    }

    /// Calls the non-streaming Vertex `generateContent` method.
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
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let mut endpoint = self.config.provider.api.as_url().clone();
        let prefix = endpoint.path().trim_end_matches('/');
        endpoint.set_path(&format!(
            "{prefix}/projects/{}/locations/{}/{}:{method}",
            self.config.settings.project,
            self.config.settings.location,
            self.config.settings.resource.path(),
        ));
        endpoint.set_query(streaming.then_some("alt=sse"));
        Ok(endpoint)
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
                &self.descriptor().capabilities,
                &self.config.settings,
            )?;
            if abort.is_aborted() {
                return Err(ModelError::abort("request was aborted before dispatch")
                    .with_stage(ErrorStage::Connect));
            }
            let descriptor = self.descriptor();
            let stream_function_call_arguments =
                streaming && self.config.settings.stream_function_call_arguments;
            let encoded = crate::request::encode_request(
                &request,
                &options,
                descriptor,
                descriptor.capabilities.replay.policy,
                stream_function_call_arguments,
                &self.config.settings.native_context_scope,
            )?;
            let body = serde_json::to_vec(&encoded.body).map_err(|_| {
                ModelError::invalid_request("could not serialize Vertex Gemini request")
                    .with_stage(ErrorStage::RequestEncoding)
            })?;
            let token = self.resolve_token(&abort).await?;
            let mut headers = self.config.base_headers.clone();
            if let Some(provider) = &self.config.provider.headers.dynamic_headers {
                let dynamic = provider.headers()?;
                validate_headers(dynamic.as_map())?;
                headers.extend(dynamic.as_map().clone());
            }
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| {
                    ModelError::new(
                        ModelErrorKind::Auth,
                        "Vertex OAuth token is not a valid header value",
                    )
                })?,
            );
            if let Some(value) = encoded.shared_request_type {
                headers.insert(
                    HeaderName::from_static("x-vertex-ai-llm-shared-request-type"),
                    header_value(&value, "invalid Vertex shared request type header")?,
                );
            }
            if let Some(value) = encoded.request_type {
                headers.insert(
                    HeaderName::from_static("x-vertex-ai-llm-request-type"),
                    header_value(&value, "invalid Vertex request type header")?,
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
                    .map_err(|_| ModelError::timeout("Vertex response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
                    .map_err(|_| ModelError::transport("Vertex Gemini request failed").with_stage(ErrorStage::Connect))?,
                _ = abort.aborted() => return Err(ModelError::abort("request was aborted before Vertex response headers").with_stage(ErrorStage::ResponseHeaders)),
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
                        "Vertex generateContent response exceeds 32 MiB",
                    )
                    .with_stage(ErrorStage::ResponseBody)
                    .with_bytes_received(count));
                }
                let value: serde_json::Value = serde_json::from_slice(&body).map_err(|_| {
                    ModelError::invalid_response("Vertex generateContent response is invalid JSON")
                        .with_stage(ErrorStage::ResponseBody)
                        .with_bytes_received(count)
                })?;
                let (mut parts, metadata) = crate::stream::normalize_single(
                    value,
                    descriptor.capabilities.replay.policy,
                    self.config.settings.native_context_scope.clone(),
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
                    descriptor.capabilities.replay.policy,
                    self.config.settings.native_context_scope.clone(),
                    stream_function_call_arguments,
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

    async fn resolve_token(&self, abort: &AbortSignal) -> Result<String, ModelError> {
        let credentials = async {
            match &self.config.provider.auth {
                VertexAuth::AccessToken(value) => Ok(value.expose_secret().to_owned()),
                VertexAuth::TokenProvider(provider) => provider
                    .token()
                    .await
                    .map(|token| token.expose_secret().to_owned()),
            }
        };
        tokio::select! {
            value = tokio::time::timeout(self.config.settings.timeouts.credentials, credentials) => value
                .map_err(|_| ModelError::timeout("Vertex credential resolution timed out").with_stage(ErrorStage::RequestEncoding))?,
            _ = abort.aborted() => Err(ModelError::abort("request was aborted while resolving Vertex credentials").with_stage(ErrorStage::RequestEncoding)),
        }
    }
}

fn hash_field(output: &mut Vec<u8>, tag: &str, value: &[u8]) {
    output.extend_from_slice(&(tag.len() as u64).to_be_bytes());
    output.extend_from_slice(tag.as_bytes());
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value);
}

fn sha256(input: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_len = (input.len() as u64).wrapping_mul(8);
    let padded_len = (input.len() + 9).div_ceil(64) * 64;
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend_from_slice(input);
    padded.push(0x80);
    padded.resize(padded_len - 8, 0);
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut schedule = [0_u32; 64];
        for (word, bytes) in schedule.iter_mut().zip(chunk.chunks_exact(4)) {
            *word = u32::from_be_bytes(bytes.try_into().expect("SHA-256 word has four bytes"));
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ (!e & g);
            let first = h
                .wrapping_add(sum1)
                .wrapping_add(choice)
                .wrapping_add(ROUND[index])
                .wrapping_add(schedule[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let second = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(first);
            d = c;
            c = b;
            b = a;
            a = first.wrapping_add(second);
        }
        for (value, addition) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *value = value.wrapping_add(addition);
        }
    }

    let mut digest = [0_u8; 32];
    for (bytes, value) in digest.chunks_exact_mut(4).zip(state) {
        bytes.copy_from_slice(&value.to_be_bytes());
    }
    digest
}

impl LanguageModel for GoogleVertexModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.config.descriptor
    }

    fn validate_request(&self, request: &Request) -> Result<(), ModelError> {
        let options = crate::request::options(request)?;
        crate::request::validate_request(
            request,
            &options,
            &self.descriptor().capabilities,
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

struct DirectResponseCollector {
    response: Mutex<Option<StreamResponse>>,
    descriptor: LanguageModelDescriptor,
}

impl LanguageModel for DirectResponseCollector {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.descriptor
    }

    fn validate_request(&self, _request: &Request) -> Result<(), ModelError> {
        Ok(())
    }

    fn supports_request(&self, _request: &Request) -> bool {
        true
    }

    fn stream<'a>(
        &'a self,
        _request: Request,
        _abort: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        let response = self
            .response
            .lock()
            .map_err(|_| ModelError::invalid_response("Vertex direct response collector poisoned"))
            .and_then(|mut response| {
                response.take().ok_or_else(|| {
                    ModelError::invalid_response("Vertex direct response was already collected")
                })
            });
        Box::pin(std::future::ready(response))
    }
}

async fn collect_direct(response: StreamResponse) -> Result<CompleteResult, ModelError> {
    let collector = DirectResponseCollector {
        response: Mutex::new(Some(response)),
        descriptor: LanguageModelDescriptor::new(
            ModelIdentity::new(
                oven_sdk::ProviderId::new("google.vertex.direct-collector"),
                oven_sdk::ModelId::new("direct-response-collector"),
            )
            .expect("constant collector identity is valid"),
            AdapterId::new(GOOGLE_VERTEX_GENERATE_CONTENT_ADAPTER_ID),
            ModelCapabilities::conservative(),
        )
        .expect("constant collector descriptor is valid"),
    };
    collector
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
}

fn validate_resource_segment(value: &str, name: &str) -> Result<(), ModelError> {
    if value.trim().is_empty()
        || value.chars().any(|character| {
            character.is_whitespace() || matches!(character, '/' | ':' | '?' | '#')
        })
    {
        Err(ModelError::invalid_request(format!(
            "Vertex {name} contains invalid resource characters"
        )))
    } else {
        Ok(())
    }
}

fn validate_headers(headers: &HeaderMap) -> Result<(), ModelError> {
    if headers.contains_key(AUTHORIZATION) || headers.contains_key("x-goog-api-key") {
        return Err(ModelError::invalid_request(
            "Vertex caller headers cannot set authentication headers",
        ));
    }
    Ok(())
}

fn header_value(value: &str, message: &str) -> Result<HeaderValue, ModelError> {
    HeaderValue::from_str(value).map_err(|_| ModelError::invalid_request(message))
}

#[cfg(test)]
mod tests {
    #[test]
    fn sha256_matches_standard_vector() {
        assert_eq!(
            super::sha256(b"abc"),
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad,
            ]
        );
    }
}
