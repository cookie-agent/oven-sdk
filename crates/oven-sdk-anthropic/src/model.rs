//! Anthropic-compatible Messages language-model implementations.

use std::{collections::VecDeque, sync::Arc};

use oven_sdk::{
    AbortSignal, AdapterId, BoxFuture, CancellationCapability, Capability, CompactionCapability,
    ErrorStage, HeaderConfig, LanguageModel, LanguageModelDescriptor, MediaSourceSupport, Modality,
    ModelCapabilities, ModelConfig, ModelError, ModelIdentity, NativeContextScope,
    ProviderMetadata, Request, RequestMetadata, ResourceId, StreamPart,
};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use sha2::{Digest, Sha256};
use url::Url;

use crate::{
    config::{
        AnthropicAuth, AnthropicAwsAuth, AnthropicAwsCredentials, AnthropicAwsSettings,
        AnthropicCompatibleAuth, AnthropicCompatibleSettings, AnthropicProtocolSettings,
        AnthropicSettings, AnthropicThinkingSupport, MiniMaxAuth, MiniMaxProtocolSettings,
        MiniMaxSettings,
    },
    error::classify_error_for,
    wire::{Protocol, VERSION},
};

#[derive(Clone)]
enum Auth {
    Anthropic(AnthropicAuth),
    Compatible(AnthropicCompatibleAuth),
    MiniMax(MiniMaxAuth),
    AnthropicAws(AnthropicAwsAuth),
}

#[derive(Clone)]
pub(crate) enum ProtocolSettings {
    Anthropic(AnthropicProtocolSettings),
    MiniMax(MiniMaxProtocolSettings),
}

#[derive(Clone)]
struct Config {
    protocol: Protocol,
    compatible: bool,
    auth: Auth,
    endpoint: oven_sdk::ApiEndpoint,
    headers: HeaderConfig,
    base_headers: HeaderMap,
    client: reqwest::Client,
    timeouts: crate::transport::AnthropicTimeouts,
    protocol_settings: ProtocolSettings,
    native_context_scope: NativeContextScope,
    aws_region: Option<String>,
    workspace_id: Option<String>,
}

#[derive(Clone)]
struct InnerModel {
    config: Arc<Config>,
    descriptor: LanguageModelDescriptor,
}

impl InnerModel {
    fn validate_request(&self, request: &Request) -> Result<(), ModelError> {
        let options =
            crate::request::parse_options(request, self.config.protocol, self.config.compatible)?;
        crate::request::validate_request(
            request,
            &options,
            &self.descriptor.capabilities,
            self.config.protocol,
            &self.config.protocol_settings,
            self.config.compatible,
        )
    }

    fn stream<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<oven_sdk::StreamResponse, ModelError>> {
        Box::pin(async move {
            let options = crate::request::parse_options(
                &request,
                self.config.protocol,
                self.config.compatible,
            )?;
            crate::request::validate_request(
                &request,
                &options,
                &self.descriptor.capabilities,
                self.config.protocol,
                &self.config.protocol_settings,
                self.config.compatible,
            )?;
            if abort.is_aborted() {
                return Err(ModelError::abort("request was aborted before dispatch")
                    .with_stage(ErrorStage::Connect));
            }
            let replay_policy = if self.descriptor.capabilities.replay.capability
                == oven_sdk::ReplayCapability::Unsupported
            {
                oven_sdk::ReplayPolicy::Never
            } else {
                self.descriptor.capabilities.replay.policy
            };
            let encoded = crate::request::encode_request(
                &request,
                &options,
                &self.descriptor,
                &self.config.native_context_scope,
                replay_policy,
                self.config.protocol,
                self.config.compatible,
            )?;
            let body = serde_json::to_vec(&encoded.body).map_err(|_| {
                ModelError::invalid_request("could not serialize Messages request")
                    .with_stage(ErrorStage::RequestEncoding)
            })?;
            crate::request::validate_serialized_request_size(body.len(), self.config.protocol)?;
            let mut headers = self.config.base_headers.clone();
            if let Some(provider) = &self.config.headers.dynamic_headers {
                headers.extend(provider.headers()?.as_map().clone());
            }
            if self.config.protocol == Protocol::AnthropicAws {
                validate_aws_caller_headers(&headers)?;
            }
            apply_transport_headers(&mut headers, self.config.protocol);
            if !encoded.betas.is_empty() && self.config.protocol.is_first_party() {
                headers.insert(
                    HeaderName::from_static("anthropic-beta"),
                    HeaderValue::from_str(&encoded.betas.join(",")).map_err(|_| {
                        ModelError::invalid_request("invalid Anthropic beta header")
                    })?,
                );
            }
            let url = messages_url(&self.config.endpoint)?;
            self.authenticate(&url, &body, &mut headers, &abort).await?;
            let send = self
                .config
                .client
                .post(url)
                .headers(headers)
                .body(body)
                .send();
            let response = tokio::select! {
                value = tokio::time::timeout(self.config.timeouts.headers, send) => value
                    .map_err(|_| ModelError::timeout("response headers timed out").with_stage(ErrorStage::ResponseHeaders))?
                    .map_err(|_| ModelError::transport(format!("{} request failed", self.config.protocol.display_name())).with_stage(ErrorStage::Connect))?,
                _ = abort.aborted() => return Err(ModelError::abort("request was aborted before response headers").with_stage(ErrorStage::ResponseHeaders)),
            };
            let mut head = crate::transport::response_head(&response);
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let request_id = head.request_id.clone();
                let error_headers = response.headers().clone();
                let (body, bytes_received) = crate::transport::read_error_body(
                    response,
                    &abort,
                    self.config.timeouts.stream_idle,
                )
                .await
                .map_err(|error| {
                    let error = error.with_http_status(status);
                    if let Some(request_id) = request_id.clone() {
                        error.with_request_id(request_id)
                    } else {
                        error
                    }
                })?;
                return Err(classify_error_for(
                    self.config.protocol,
                    status,
                    &body,
                    request_id,
                    ErrorStage::ResponseBody,
                    bytes_received,
                    &error_headers,
                ));
            }
            let mut live = crate::stream::LiveState {
                bytes: Box::pin(response.bytes_stream()),
                parser: crate::sse::Parser::default(),
                state: crate::stream::state::State::new(
                    replay_policy,
                    self.config.protocol,
                    self.descriptor.adapter_id.clone(),
                    self.config.native_context_scope.clone(),
                ),
                queue: VecDeque::from([Ok(StreamPart::StreamStart {
                    warnings: encoded.warnings,
                })]),
                pending_events: VecDeque::new(),
                pending_error: None,
                deadline: oven_sdk::provider_support::StreamReadDeadline::new(
                    tokio::time::sleep(self.config.timeouts.stream_idle),
                    &abort,
                ),
                idle: self.config.timeouts.stream_idle,
                count: 0,
                eof: false,
                request_id: head.request_id.clone(),
                protocol: self.config.protocol,
            };
            live.state.set_request_id(head.request_id.clone());
            crate::stream::early_peek(&mut live).await?;
            head.response_metadata
                .extend(live.state.response_metadata().clone());
            Ok(
                oven_sdk::StreamResponse::new(Box::pin(futures_util::stream::unfold(
                    live,
                    |mut live| async move {
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
                    },
                )))
                .with_request(RequestMetadata {
                    replay: encoded.replay,
                    provider_metadata: ProviderMetadata::new(),
                })
                .with_response(head),
            )
        })
    }

    async fn authenticate(
        &self,
        url: &Url,
        body: &[u8],
        headers: &mut HeaderMap,
        abort: &AbortSignal,
    ) -> Result<(), ModelError> {
        match &self.config.auth {
            Auth::Anthropic(auth) => {
                if !headers.contains_key("x-api-key")
                    && !headers.contains_key(AUTHORIZATION)
                    && let AnthropicAuth::ApiKey(key) = auth
                {
                    headers.insert(
                        HeaderName::from_static("x-api-key"),
                        header_value(key.expose_secret(), "invalid Anthropic API key header")?,
                    );
                }
            }
            Auth::Compatible(auth) => {
                if !headers.contains_key("x-api-key") && !headers.contains_key(AUTHORIZATION) {
                    match auth {
                        AnthropicCompatibleAuth::ApiKey(key) => {
                            headers.insert(
                                HeaderName::from_static("x-api-key"),
                                header_value(
                                    key.expose_secret(),
                                    "invalid Anthropic-compatible API key header",
                                )?,
                            );
                        }
                        AnthropicCompatibleAuth::Bearer(token) => {
                            headers.insert(
                                AUTHORIZATION,
                                header_value(
                                    &format!("Bearer {}", token.expose_secret()),
                                    "invalid Anthropic-compatible bearer header",
                                )?,
                            );
                        }
                        AnthropicCompatibleAuth::None => {}
                    }
                }
            }
            Auth::MiniMax(auth) => {
                if !headers.contains_key(AUTHORIZATION)
                    && let MiniMaxAuth::Bearer(key) = auth
                {
                    headers.insert(
                        AUTHORIZATION,
                        header_value(
                            &format!("Bearer {}", key.expose_secret()),
                            "invalid MiniMax bearer header",
                        )?,
                    );
                }
            }
            Auth::AnthropicAws(AnthropicAwsAuth::BearerKey(key)) => {
                headers.insert(
                    HeaderName::from_static("x-api-key"),
                    header_value(
                        key.expose_secret(),
                        "invalid Anthropic AWS bearer key header",
                    )?,
                );
                self.insert_workspace_header(headers)?;
            }
            Auth::AnthropicAws(AnthropicAwsAuth::StaticCredentials(credentials)) => {
                self.insert_workspace_header(headers)?;
                self.sign(url, body, headers, credentials)?;
            }
            Auth::AnthropicAws(AnthropicAwsAuth::CredentialProvider(provider)) => {
                self.insert_workspace_header(headers)?;
                let credentials = tokio::select! {
                    value = tokio::time::timeout(self.config.timeouts.credentials, provider()) => value
                        .map_err(|_| ModelError::timeout("Anthropic AWS credential provider timed out").with_stage(ErrorStage::RequestEncoding))?
                        .map_err(|_| {
                            ModelError::new(
                                oven_sdk::ModelErrorKind::Auth,
                                "Anthropic AWS credential provider failed",
                            )
                            .with_stage(ErrorStage::RequestEncoding)
                        })?,
                    _ = abort.aborted() => return Err(
                        ModelError::abort("request was aborted while awaiting AWS credentials")
                            .with_stage(ErrorStage::RequestEncoding)
                    ),
                };
                self.sign(url, body, headers, &credentials)?;
            }
        }
        Ok(())
    }

    fn sign(
        &self,
        url: &Url,
        body: &[u8],
        headers: &mut HeaderMap,
        credentials: &AnthropicAwsCredentials,
    ) -> Result<(), ModelError> {
        crate::signing::sign(
            "POST",
            url,
            body,
            headers,
            self.config
                .aws_region
                .as_deref()
                .expect("AWS configuration always has a region"),
            credentials,
        )
    }

    fn insert_workspace_header(&self, headers: &mut HeaderMap) -> Result<(), ModelError> {
        headers.insert(
            HeaderName::from_static("anthropic-workspace-id"),
            header_value(
                self.config
                    .workspace_id
                    .as_deref()
                    .expect("AWS configuration always has a workspace ID"),
                "invalid Anthropic AWS workspace ID header",
            )?,
        );
        Ok(())
    }
}

fn messages_url(endpoint: &oven_sdk::ApiEndpoint) -> Result<Url, ModelError> {
    validate_endpoint(endpoint)?;
    let mut url = endpoint.as_url().clone();
    url.path_segments_mut()
        .map_err(|_| ModelError::invalid_request("Messages API endpoint cannot be a base URL"))?
        .pop_if_empty()
        .push("messages");
    Ok(url)
}

fn header_value(value: &str, message: &str) -> Result<HeaderValue, ModelError> {
    HeaderValue::from_str(value).map_err(|_| ModelError::invalid_request(message))
}

fn validate_endpoint(endpoint: &oven_sdk::ApiEndpoint) -> Result<(), ModelError> {
    if endpoint.as_url().query().is_some() {
        return Err(ModelError::invalid_request(
            "Messages API base endpoint must not contain a query",
        ));
    }
    Ok(())
}

fn canonical_endpoint(endpoint: &oven_sdk::ApiEndpoint) -> Result<String, ModelError> {
    validate_endpoint(endpoint)?;
    Ok(endpoint.as_url().as_str().trim_end_matches('/').to_owned())
}

fn derive_native_context_resource(
    adapter_id: &AdapterId,
    endpoint: &oven_sdk::ApiEndpoint,
    aws: Option<(&str, &str)>,
    discriminator: Option<&ResourceId>,
) -> Result<ResourceId, ModelError> {
    const DOMAIN: &str = "oven-sdk-anthropic/native-context-resource/v1";
    fn field(hasher: &mut Sha256, value: &str) {
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value.as_bytes());
    }

    let mut hasher = Sha256::new();
    field(&mut hasher, DOMAIN);
    field(&mut hasher, adapter_id.as_str());
    field(&mut hasher, &canonical_endpoint(endpoint)?);
    if let Some((region, workspace_id)) = aws {
        field(&mut hasher, region);
        field(&mut hasher, workspace_id);
    }
    if let Some(discriminator) = discriminator {
        field(&mut hasher, discriminator.as_str());
    }
    ResourceId::new(format!(
        "anthropic-context-v1-{}",
        hex::encode(hasher.finalize())
    ))
}

fn validate_capability_ceiling(
    protocol: Protocol,
    capabilities: &ModelCapabilities,
    compatible: bool,
) -> Result<(), ModelError> {
    let allowed_features = match protocol {
        Protocol::Anthropic | Protocol::AnthropicAws => {
            Capability::TOOL_CALLING
                | Capability::PARALLEL_TOOLS
                | Capability::TOOL_INPUT_DELTAS
                | Capability::REASONING
                | Capability::STRUCTURED_OUTPUT
                | Capability::TEMPERATURE
                | Capability::TOP_P
                | Capability::MAX_OUTPUT_TOKENS
                | Capability::PROMPT_CACHING
                | Capability::USAGE
        }
        Protocol::MiniMax => {
            Capability::TOOL_CALLING
                | Capability::TOOL_INPUT_DELTAS
                | Capability::REASONING
                | Capability::TEMPERATURE
                | Capability::TOP_P
                | Capability::MAX_OUTPUT_TOKENS
                | Capability::USAGE
        }
    };
    if !(capabilities.features & !allowed_features).is_empty() {
        return Err(ModelError::invalid_request(format!(
            "{} declaration exceeds the protocol feature ceiling",
            protocol.display_name()
        )));
    }
    if capabilities.cancellation == CancellationCapability::RemoteBestEffort {
        return Err(ModelError::invalid_request(format!(
            "{} does not support remote cancellation",
            protocol.display_name()
        )));
    }
    if capabilities.compaction != CompactionCapability::Unsupported {
        return Err(ModelError::invalid_request(format!(
            "{} does not implement provider-native compaction",
            protocol.display_name()
        )));
    }
    let allowed_input = match protocol {
        Protocol::Anthropic if compatible => [
            Modality::text(),
            Modality::image(),
            Modality::pdf(),
            Modality::video(),
        ]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>(),
        Protocol::Anthropic | Protocol::AnthropicAws => {
            [Modality::text(), Modality::image(), Modality::pdf()]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
        }
        Protocol::MiniMax => [Modality::text(), Modality::image(), Modality::video()]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>(),
    };
    if !capabilities.modalities.input.is_subset(&allowed_input)
        || capabilities.modalities.output != [Modality::text()].into_iter().collect()
    {
        return Err(ModelError::invalid_request(format!(
            "{} declaration exceeds the protocol modality ceiling",
            protocol.display_name()
        )));
    }
    for (modality, support) in &capabilities.media.input {
        let (media_types, sources): (&[&str], MediaSourceSupport) = match protocol {
            Protocol::Anthropic | Protocol::AnthropicAws if modality == &Modality::image() => (
                &["image/jpeg", "image/png", "image/gif", "image/webp"],
                MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
            ),
            Protocol::Anthropic | Protocol::AnthropicAws if modality == &Modality::pdf() => (
                &["application/pdf"],
                MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
            ),
            Protocol::Anthropic | Protocol::AnthropicAws if modality == &Modality::text() => {
                (&["text/plain"], MediaSourceSupport::INLINE_TEXT)
            }
            Protocol::Anthropic if compatible && modality == &Modality::video() => (
                &[
                    "video/mp4",
                    "video/avi",
                    "video/x-msvideo",
                    "video/quicktime",
                    "video/mov",
                    "video/x-matroska",
                ],
                MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
            ),
            Protocol::MiniMax if modality == &Modality::image() => (
                &["image/jpeg", "image/png", "image/gif", "image/webp"],
                MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
            ),
            Protocol::MiniMax if modality == &Modality::video() => (
                &[
                    "video/mp4",
                    "video/avi",
                    "video/x-msvideo",
                    "video/quicktime",
                    "video/mov",
                    "video/x-matroska",
                ],
                MediaSourceSupport::INLINE_BYTES
                    | MediaSourceSupport::URL
                    | MediaSourceSupport::PROVIDER_REFERENCE,
            ),
            _ => {
                return Err(ModelError::invalid_request(format!(
                    "{} declaration exceeds the protocol media ceiling",
                    protocol.display_name()
                )));
            }
        };
        if support
            .media_types
            .iter()
            .any(|media_type| !media_types.contains(&media_type.as_str()))
            || !(support.sources & !sources).is_empty()
        {
            return Err(ModelError::invalid_request(format!(
                "{} declaration exceeds the protocol media ceiling",
                protocol.display_name()
            )));
        }
    }
    Ok(())
}

fn validate_anthropic_protocol(
    settings: &AnthropicProtocolSettings,
    features: Capability,
) -> Result<(), ModelError> {
    let thinking = settings.thinking != AnthropicThinkingSupport::None;
    if thinking && !features.contains(Capability::REASONING) {
        return Err(ModelError::invalid_request(
            "Anthropic thinking settings require the reasoning capability",
        ));
    }
    if !thinking
        && (settings.thinking_default_active
            || settings.thinking_disable_allowed
            || !settings.thinking_disable_forbidden_efforts.is_empty())
    {
        return Err(ModelError::invalid_request(
            "Anthropic thinking flags require explicit thinking support",
        ));
    }
    if settings
        .thinking_disable_forbidden_efforts
        .iter()
        .any(|effort| effort.trim().is_empty())
    {
        return Err(ModelError::invalid_request(
            "disabled-thinking effort exclusions must not be empty",
        ));
    }
    Ok(())
}

fn validate_minimax_protocol(
    settings: &MiniMaxProtocolSettings,
    features: Capability,
) -> Result<(), ModelError> {
    if settings.thinking && !features.contains(Capability::REASONING) {
        return Err(ModelError::invalid_request(
            "MiniMax thinking settings require the reasoning capability",
        ));
    }
    if settings.thinking_disable_allowed && !settings.thinking {
        return Err(ModelError::invalid_request(
            "MiniMax disabled-thinking support requires thinking support",
        ));
    }
    Ok(())
}

fn validate_aws_caller_headers(headers: &HeaderMap) -> Result<(), ModelError> {
    const PROTECTED: &[&str] = &[
        "authorization",
        "x-api-key",
        "x-amz-date",
        "x-amz-security-token",
        "x-amz-content-sha256",
        "host",
        "anthropic-workspace-id",
    ];
    if let Some(name) = PROTECTED.iter().find(|name| headers.contains_key(**name)) {
        return Err(ModelError::invalid_request(format!(
            "Claude Platform on AWS caller headers cannot set protected header {name}"
        )));
    }
    Ok(())
}

macro_rules! impl_language_model {
    ($name:ident) => {
        impl LanguageModel for $name {
            fn descriptor(&self) -> &LanguageModelDescriptor {
                &self.0.descriptor
            }

            fn validate_request(&self, request: &Request) -> Result<(), ModelError> {
                self.0.validate_request(request)
            }

            fn supports_request(&self, request: &Request) -> bool {
                self.validate_request(request).is_ok()
            }

            fn stream<'a>(
                &'a self,
                request: Request,
                abort: AbortSignal,
            ) -> BoxFuture<'a, Result<oven_sdk::StreamResponse, ModelError>> {
                self.0.stream(request, abort)
            }
        }
    };
}

fn base_headers(configured: &HeaderConfig, protocol: Protocol) -> HeaderMap {
    let mut headers = configured.static_headers.as_map().clone();
    apply_transport_headers(&mut headers, protocol);
    headers
}

fn apply_transport_headers(headers: &mut HeaderMap, protocol: Protocol) {
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if protocol != Protocol::MiniMax {
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(VERSION),
        );
    }
}

/// A caller-configured direct Anthropic Messages model.
#[derive(Clone)]
pub struct AnthropicModel(InnerModel);

impl AnthropicModel {
    /// Constructs one model from a complete explicit registry-free configuration.
    pub fn new(config: ModelConfig<AnthropicAuth, AnthropicSettings>) -> Result<Self, ModelError> {
        config.validate()?;
        validate_endpoint(&config.provider.api)?;
        validate_capability_ceiling(Protocol::Anthropic, &config.model.capabilities, false)?;
        validate_anthropic_protocol(
            &config.settings.protocol,
            config.model.capabilities.features,
        )?;
        if config.provider.id.as_str() != crate::wire::ANTHROPIC_PROVIDER_ID {
            return Err(ModelError::invalid_request(
                "direct Anthropic requires provider ID `anthropic`",
            ));
        }
        let identity = ModelIdentity::new(config.provider.id.clone(), config.model.id.clone())?;
        let descriptor = LanguageModelDescriptor::new(
            identity,
            Protocol::Anthropic.adapter_id(),
            config.model.capabilities.clone(),
        )?;
        let native_context_resource = derive_native_context_resource(
            &descriptor.adapter_id,
            &config.provider.api,
            None,
            config.settings.native_context_discriminator.as_ref(),
        )?;
        let native_context_scope = NativeContextScope::new(
            config.provider.id.clone(),
            config.model.id.clone(),
            native_context_resource,
        )?;
        let base_headers = base_headers(&config.provider.headers, Protocol::Anthropic);
        Ok(Self(InnerModel {
            descriptor,
            config: Arc::new(Config {
                protocol: Protocol::Anthropic,
                compatible: false,
                auth: Auth::Anthropic(config.provider.auth),
                endpoint: config.provider.api,
                headers: config.provider.headers,
                base_headers,
                client: config.settings.client,
                timeouts: config.settings.timeouts,
                protocol_settings: ProtocolSettings::Anthropic(config.settings.protocol),
                native_context_scope,
                aws_region: None,
                workspace_id: None,
            }),
        }))
    }
}

impl_language_model!(AnthropicModel);

/// A caller-configured Anthropic Messages-compatible model.
#[derive(Clone)]
pub struct AnthropicCompatibleModel(InnerModel);

impl AnthropicCompatibleModel {
    /// Constructs one compatible model from a complete explicit registry-free configuration.
    pub fn new(
        config: ModelConfig<AnthropicCompatibleAuth, AnthropicCompatibleSettings>,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        validate_endpoint(&config.provider.api)?;
        config.settings.adapter_id.validate()?;
        if matches!(
            config.settings.adapter_id.as_str(),
            crate::wire::ANTHROPIC_MESSAGES_ADAPTER_ID | crate::wire::MINIMAX_MESSAGES_ADAPTER_ID
        ) {
            return Err(ModelError::invalid_request(
                "official Anthropic and MiniMax adapter IDs are reserved",
            ));
        }
        validate_capability_ceiling(Protocol::Anthropic, &config.model.capabilities, true)?;
        validate_anthropic_protocol(
            &config.settings.protocol,
            config.model.capabilities.features,
        )?;
        let identity = ModelIdentity::new(config.provider.id.clone(), config.model.id.clone())?;
        let descriptor = LanguageModelDescriptor::new(
            identity,
            config.settings.adapter_id,
            config.model.capabilities.clone(),
        )?;
        let native_context_resource = derive_native_context_resource(
            &descriptor.adapter_id,
            &config.provider.api,
            None,
            config.settings.native_context_discriminator.as_ref(),
        )?;
        let native_context_scope = NativeContextScope::new(
            config.provider.id.clone(),
            config.model.id.clone(),
            native_context_resource,
        )?;
        let base_headers = base_headers(&config.provider.headers, Protocol::Anthropic);
        Ok(Self(InnerModel {
            descriptor,
            config: Arc::new(Config {
                protocol: Protocol::Anthropic,
                compatible: true,
                auth: Auth::Compatible(config.provider.auth),
                endpoint: config.provider.api,
                headers: config.provider.headers,
                base_headers,
                client: config.settings.client,
                timeouts: config.settings.timeouts,
                protocol_settings: ProtocolSettings::Anthropic(config.settings.protocol),
                native_context_scope,
                aws_region: None,
                workspace_id: None,
            }),
        }))
    }
}

impl_language_model!(AnthropicCompatibleModel);

/// A caller-configured MiniMax Messages-compatible model.
#[derive(Clone)]
pub struct MiniMaxModel(InnerModel);

impl MiniMaxModel {
    /// Constructs one model from a complete explicit registry-free configuration.
    pub fn new(config: ModelConfig<MiniMaxAuth, MiniMaxSettings>) -> Result<Self, ModelError> {
        config.validate()?;
        validate_endpoint(&config.provider.api)?;
        validate_capability_ceiling(Protocol::MiniMax, &config.model.capabilities, false)?;
        validate_minimax_protocol(
            &config.settings.protocol,
            config.model.capabilities.features,
        )?;
        if config.provider.id.as_str() != crate::wire::MINIMAX_PROVIDER_ID {
            return Err(ModelError::invalid_request(
                "MiniMax requires provider ID `minimax`",
            ));
        }
        let identity = ModelIdentity::new(config.provider.id.clone(), config.model.id.clone())?;
        let descriptor = LanguageModelDescriptor::new(
            identity,
            Protocol::MiniMax.adapter_id(),
            config.model.capabilities.clone(),
        )?;
        let native_context_resource = derive_native_context_resource(
            &descriptor.adapter_id,
            &config.provider.api,
            None,
            config.settings.native_context_discriminator.as_ref(),
        )?;
        let native_context_scope = NativeContextScope::new(
            config.provider.id.clone(),
            config.model.id.clone(),
            native_context_resource,
        )?;
        let base_headers = base_headers(&config.provider.headers, Protocol::MiniMax);
        Ok(Self(InnerModel {
            descriptor,
            config: Arc::new(Config {
                protocol: Protocol::MiniMax,
                compatible: false,
                auth: Auth::MiniMax(config.provider.auth),
                endpoint: config.provider.api,
                headers: config.provider.headers,
                base_headers,
                client: config.settings.client,
                timeouts: config.settings.timeouts,
                protocol_settings: ProtocolSettings::MiniMax(config.settings.protocol),
                native_context_scope,
                aws_region: None,
                workspace_id: None,
            }),
        }))
    }
}

impl_language_model!(MiniMaxModel);

/// A caller-configured Claude Platform on AWS Messages model.
#[derive(Clone)]
pub struct AnthropicAwsModel(InnerModel);

impl AnthropicAwsModel {
    /// Constructs one model from a complete explicit registry-free configuration.
    pub fn new(
        config: ModelConfig<AnthropicAwsAuth, AnthropicAwsSettings>,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        validate_endpoint(&config.provider.api)?;
        validate_capability_ceiling(Protocol::AnthropicAws, &config.model.capabilities, false)?;
        validate_anthropic_protocol(
            &config.settings.protocol,
            config.model.capabilities.features,
        )?;
        if config.provider.id.as_str() != crate::wire::ANTHROPIC_AWS_PROVIDER_ID {
            return Err(ModelError::invalid_request(
                "Claude Platform on AWS requires provider ID `anthropic-aws`",
            ));
        }
        if config.settings.region.trim().is_empty()
            || config.settings.workspace_id.trim().is_empty()
        {
            return Err(ModelError::invalid_request(
                "Claude Platform on AWS requires non-empty region and workspace ID",
            ));
        }
        validate_aws_caller_headers(config.provider.headers.static_headers.as_map())?;
        let identity = ModelIdentity::new(config.provider.id.clone(), config.model.id.clone())?;
        let descriptor = LanguageModelDescriptor::new(
            identity,
            Protocol::AnthropicAws.adapter_id(),
            config.model.capabilities.clone(),
        )?;
        let native_context_resource = derive_native_context_resource(
            &descriptor.adapter_id,
            &config.provider.api,
            Some((&config.settings.region, &config.settings.workspace_id)),
            config.settings.native_context_discriminator.as_ref(),
        )?;
        let native_context_scope = NativeContextScope::new(
            config.provider.id.clone(),
            config.model.id.clone(),
            native_context_resource,
        )?;
        let base_headers = base_headers(&config.provider.headers, Protocol::AnthropicAws);
        Ok(Self(InnerModel {
            descriptor,
            config: Arc::new(Config {
                protocol: Protocol::AnthropicAws,
                compatible: false,
                auth: Auth::AnthropicAws(config.provider.auth),
                endpoint: config.provider.api,
                headers: config.provider.headers,
                base_headers,
                client: config.settings.client,
                timeouts: config.settings.timeouts,
                protocol_settings: ProtocolSettings::Anthropic(config.settings.protocol),
                native_context_scope,
                aws_region: Some(config.settings.region),
                workspace_id: Some(config.settings.workspace_id),
            }),
        }))
    }
}

impl_language_model!(AnthropicAwsModel);
