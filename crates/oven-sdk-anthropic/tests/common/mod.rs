#![allow(dead_code)]
// Test constructors intentionally preserve the public `ModelError` result type used by models.
#![allow(clippy::result_large_err)]

use std::{collections::BTreeMap, future::Future, sync::Arc};

use oven_sdk::{
    AdapterId, ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    HeaderOverrides, HeaderProvider, MediaCapabilities, MediaInputSupport, MediaSourceSupport,
    Modalities, Modality, ModelCapabilities, ModelConfig, ModelDeclaration, ModelError, ModelId,
    ModelLimits, ProviderConfig, ProviderId, ReplayCapability, ReplayDeclaration, ReplayPolicy,
    ResourceId, SecretString,
};
use oven_sdk_anthropic::{
    AnthropicAuth, AnthropicAwsAuth, AnthropicAwsCredentialProvider, AnthropicAwsCredentials,
    AnthropicAwsModel, AnthropicAwsSettings, AnthropicCompatibleAuth, AnthropicCompatibleModel,
    AnthropicCompatibleSettings, AnthropicModel, AnthropicProtocolSettings, AnthropicSettings,
    AnthropicThinkingSupport, AnthropicTimeouts, MiniMaxAuth, MiniMaxModel,
    MiniMaxProtocolSettings, MiniMaxSettings,
};
use reqwest::{Client, header::HeaderMap};
use sha2::{Digest, Sha256};

pub fn anthropic_protocol() -> AnthropicProtocolSettings {
    AnthropicProtocolSettings {
        thinking: AnthropicThinkingSupport::Both,
        thinking_default_active: false,
        thinking_disable_allowed: true,
        thinking_disable_forbidden_efforts: Default::default(),
        effort: true,
        assistant_prefill: true,
        reject_non_default_sampling: false,
    }
}

pub fn anthropic_capabilities(policy: ReplayPolicy) -> ModelCapabilities {
    let mut media = MediaCapabilities::default();
    media.input.insert(
        Modality::image(),
        MediaInputSupport::new(
            [
                "image/jpeg".into(),
                "image/png".into(),
                "image/gif".into(),
                "image/webp".into(),
            ],
            MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
        )
        .unwrap(),
    );
    media.input.insert(
        Modality::pdf(),
        MediaInputSupport::new(
            ["application/pdf".into()],
            MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
        )
        .unwrap(),
    );
    media.input.insert(
        Modality::text(),
        MediaInputSupport::new(["text/plain".into()], MediaSourceSupport::INLINE_TEXT).unwrap(),
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
            | Capability::PROMPT_CACHING
            | Capability::USAGE,
        limits: ModelLimits::new(Some(200_000), None, Some(64_000)),
        modalities: Modalities::new(
            [Modality::text(), Modality::image(), Modality::pdf()],
            [Modality::text()],
        ),
        media,
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        replay: replay(policy, true),
    }
}

pub fn minimax_capabilities(policy: ReplayPolicy, media_enabled: bool) -> ModelCapabilities {
    let mut media = MediaCapabilities::default();
    let mut input = vec![Modality::text()];
    if media_enabled {
        input.extend([Modality::image(), Modality::video()]);
        media.input.insert(
            Modality::image(),
            MediaInputSupport::new(
                [
                    "image/jpeg".into(),
                    "image/png".into(),
                    "image/gif".into(),
                    "image/webp".into(),
                ],
                MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
            )
            .unwrap(),
        );
        media.input.insert(
            Modality::video(),
            MediaInputSupport::new(
                [
                    "video/mp4".into(),
                    "video/avi".into(),
                    "video/x-msvideo".into(),
                    "video/quicktime".into(),
                    "video/mov".into(),
                    "video/x-matroska".into(),
                ],
                MediaSourceSupport::INLINE_BYTES
                    | MediaSourceSupport::URL
                    | MediaSourceSupport::PROVIDER_REFERENCE,
            )
            .unwrap(),
        );
    }
    ModelCapabilities {
        features: Capability::TOOL_CALLING
            | Capability::TOOL_INPUT_DELTAS
            | Capability::REASONING
            | Capability::TEMPERATURE
            | Capability::TOP_P
            | Capability::MAX_OUTPUT_TOKENS
            | Capability::USAGE,
        limits: ModelLimits::new(Some(1_000_000), None, Some(524_288)),
        modalities: Modalities::new(input, [Modality::text()]),
        media,
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        replay: replay(policy, true),
    }
}

pub fn conservative_capabilities() -> ModelCapabilities {
    let mut capabilities = ModelCapabilities::conservative();
    capabilities.features = Capability::MAX_OUTPUT_TOKENS;
    capabilities.cancellation = CancellationCapability::LocalOnly;
    capabilities
}

fn replay(policy: ReplayPolicy, reasoning: bool) -> ReplayDeclaration {
    if policy == ReplayPolicy::Never {
        ReplayDeclaration {
            policy,
            capability: ReplayCapability::Unsupported,
            reasoning: false,
        }
    } else {
        ReplayDeclaration {
            policy,
            capability: if reasoning {
                ReplayCapability::Required
            } else {
                ReplayCapability::Optional
            },
            reasoning,
        }
    }
}

pub fn expected_native_context_scope(
    provider_id: &str,
    adapter_id: &str,
    endpoint: &str,
    model_id: &str,
    aws: Option<(&str, &str)>,
    discriminator: Option<&str>,
) -> oven_sdk::NativeContextScope {
    fn field(hasher: &mut Sha256, value: &str) {
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value.as_bytes());
    }
    let endpoint = url::Url::parse(endpoint).unwrap();
    let mut hasher = Sha256::new();
    field(&mut hasher, "oven-sdk-anthropic/native-context-resource/v1");
    field(&mut hasher, adapter_id);
    field(&mut hasher, endpoint.as_str().trim_end_matches('/'));
    if let Some((region, workspace_id)) = aws {
        field(&mut hasher, region);
        field(&mut hasher, workspace_id);
    }
    if let Some(discriminator) = discriminator {
        field(&mut hasher, discriminator);
    }
    oven_sdk::NativeContextScope::new(
        ProviderId::new(provider_id),
        ModelId::new(model_id),
        ResourceId::new(format!(
            "anthropic-context-v1-{}",
            hex::encode(hasher.finalize())
        ))
        .unwrap(),
    )
    .unwrap()
}

#[derive(Clone)]
struct DynamicHeaders(Arc<dyn Fn() -> HeaderMap + Send + Sync>);

impl HeaderProvider for DynamicHeaders {
    fn headers(&self) -> Result<HeaderOverrides, ModelError> {
        Ok(HeaderOverrides::new((self.0)()))
    }
}

fn headers(static_headers: HeaderMap, dynamic: Option<DynamicHeaders>) -> HeaderConfig {
    HeaderConfig {
        static_headers: HeaderOverrides::new(static_headers),
        dynamic_headers: dynamic.map(|provider| Arc::new(provider) as _),
    }
}

fn client(client: Option<Client>) -> Client {
    client.unwrap_or_default()
}

pub fn try_anthropic_model(
    endpoint: &str,
    model_id: &str,
    capabilities: ModelCapabilities,
    protocol: AnthropicProtocolSettings,
    discriminator: Option<&str>,
) -> Result<AnthropicModel, ModelError> {
    AnthropicModel::new(ModelConfig::new(
        ProviderConfig::new(
            ProviderId::new("anthropic"),
            ApiEndpoint::parse(endpoint)?,
            AnthropicAuth::None,
            HeaderConfig::empty(),
        )?,
        ModelDeclaration::new(ModelId::new(model_id), capabilities)?,
        AnthropicSettings {
            client: Client::new(),
            timeouts: AnthropicTimeouts::default(),
            protocol,
            native_context_discriminator: discriminator.map(ResourceId::new).transpose()?,
        },
    ))
}

pub fn try_minimax_model(
    endpoint: &str,
    model_id: &str,
    capabilities: ModelCapabilities,
    protocol: MiniMaxProtocolSettings,
    discriminator: Option<&str>,
) -> Result<MiniMaxModel, ModelError> {
    MiniMaxModel::new(ModelConfig::new(
        ProviderConfig::new(
            ProviderId::new("minimax"),
            ApiEndpoint::parse(endpoint)?,
            MiniMaxAuth::None,
            HeaderConfig::empty(),
        )?,
        ModelDeclaration::new(ModelId::new(model_id), capabilities)?,
        MiniMaxSettings {
            client: Client::new(),
            timeouts: AnthropicTimeouts::default(),
            protocol,
            native_context_discriminator: discriminator.map(ResourceId::new).transpose()?,
        },
    ))
}

pub fn try_compatible_model(
    endpoint: &str,
    provider_id: &str,
    model_id: &str,
    adapter_id: &str,
    auth: AnthropicCompatibleAuth,
    capabilities: ModelCapabilities,
    protocol: AnthropicProtocolSettings,
) -> Result<AnthropicCompatibleModel, ModelError> {
    AnthropicCompatibleModel::new(ModelConfig::new(
        ProviderConfig::new(
            ProviderId::new(provider_id),
            ApiEndpoint::parse(endpoint)?,
            auth,
            HeaderConfig::empty(),
        )?,
        ModelDeclaration::new(ModelId::new(model_id), capabilities)?,
        AnthropicCompatibleSettings {
            adapter_id: AdapterId::new(adapter_id),
            client: Client::new(),
            timeouts: AnthropicTimeouts::default(),
            protocol,
            native_context_discriminator: None,
        },
    ))
}

pub fn try_aws_model(
    endpoint: &str,
    model_id: &str,
    capabilities: ModelCapabilities,
    protocol: AnthropicProtocolSettings,
    region: &str,
    workspace_id: &str,
    discriminator: Option<&str>,
) -> Result<AnthropicAwsModel, ModelError> {
    AnthropicAwsModel::new(ModelConfig::new(
        ProviderConfig::new(
            ProviderId::new("anthropic-aws"),
            ApiEndpoint::parse(endpoint)?,
            AnthropicAwsAuth::BearerKey(SecretString::new("test-key")),
            HeaderConfig::empty(),
        )?,
        ModelDeclaration::new(ModelId::new(model_id), capabilities)?,
        AnthropicAwsSettings {
            client: Client::new(),
            timeouts: AnthropicTimeouts::default(),
            protocol,
            region: region.into(),
            workspace_id: workspace_id.into(),
            native_context_discriminator: discriminator.map(ResourceId::new).transpose()?,
        },
    ))
}

pub struct Anthropic {
    builder: AnthropicBuilder,
}

pub struct AnthropicBuilder {
    api_key: Option<String>,
    endpoint: String,
    client: Option<Client>,
    headers: HeaderMap,
    dynamic_headers: Option<DynamicHeaders>,
    timeouts: AnthropicTimeouts,
    replay_policy: ReplayPolicy,
    native_context_discriminator: Option<String>,
    capabilities: Option<ModelCapabilities>,
    protocol: AnthropicProtocolSettings,
}

impl Anthropic {
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder {
            api_key: None,
            endpoint: "https://api.anthropic.com/v1".into(),
            client: None,
            headers: HeaderMap::new(),
            dynamic_headers: None,
            timeouts: AnthropicTimeouts::default(),
            replay_policy: ReplayPolicy::IfValid,
            native_context_discriminator: None,
            capabilities: None,
            protocol: anthropic_protocol(),
        }
    }

    pub fn model(&self, model_id: impl Into<String>) -> AnthropicModel {
        let capabilities = self
            .builder
            .capabilities
            .clone()
            .unwrap_or_else(|| anthropic_capabilities(self.builder.replay_policy));
        AnthropicModel::new(ModelConfig::new(
            ProviderConfig::new(
                ProviderId::new("anthropic"),
                ApiEndpoint::parse(&self.builder.endpoint).unwrap(),
                self.builder
                    .api_key
                    .as_ref()
                    .map_or(AnthropicAuth::None, |key| {
                        AnthropicAuth::ApiKey(SecretString::new(key.clone()))
                    }),
                headers(
                    self.builder.headers.clone(),
                    self.builder.dynamic_headers.clone(),
                ),
            )
            .unwrap(),
            ModelDeclaration::new(ModelId::new(model_id), capabilities).unwrap(),
            AnthropicSettings {
                client: client(self.builder.client.clone()),
                timeouts: self.builder.timeouts.clone(),
                protocol: self.builder.protocol.clone(),
                native_context_discriminator: self
                    .builder
                    .native_context_discriminator
                    .as_ref()
                    .map(|value| ResourceId::new(value.clone()).unwrap()),
            },
        ))
        .unwrap()
    }
}

impl AnthropicBuilder {
    pub fn api_key(mut self, value: impl Into<String>) -> Self {
        self.api_key = Some(value.into());
        self
    }
    pub fn base_url(mut self, value: impl Into<String>) -> Self {
        self.endpoint = value.into();
        self
    }
    pub fn client(mut self, value: Client) -> Self {
        self.client = Some(value);
        self
    }
    pub fn default_headers(mut self, value: HeaderMap) -> Self {
        self.headers = value;
        self
    }
    pub fn header_provider(mut self, value: Arc<dyn Fn() -> HeaderMap + Send + Sync>) -> Self {
        self.dynamic_headers = Some(DynamicHeaders(value));
        self
    }
    pub fn timeouts(mut self, value: AnthropicTimeouts) -> Self {
        self.timeouts = value;
        self
    }
    pub fn replay_policy(mut self, value: ReplayPolicy) -> Self {
        self.replay_policy = value;
        self
    }
    pub fn native_context_discriminator(mut self, value: impl Into<String>) -> Self {
        self.native_context_discriminator = Some(value.into());
        self
    }
    pub fn capabilities(mut self, value: ModelCapabilities) -> Self {
        self.capabilities = Some(value);
        self
    }
    pub fn protocol(mut self, value: AnthropicProtocolSettings) -> Self {
        self.protocol = value;
        self
    }
    pub fn build(self) -> Result<Anthropic, ModelError> {
        Ok(Anthropic { builder: self })
    }
}

pub struct MiniMax {
    builder: MiniMaxBuilder,
}

pub struct MiniMaxBuilder {
    api_key: Option<String>,
    endpoint: String,
    client: Option<Client>,
    headers: HeaderMap,
    dynamic_headers: Option<DynamicHeaders>,
    timeouts: AnthropicTimeouts,
    replay_policy: ReplayPolicy,
    capabilities: Option<ModelCapabilities>,
    protocol: MiniMaxProtocolSettings,
}

impl MiniMax {
    pub fn builder() -> MiniMaxBuilder {
        MiniMaxBuilder {
            api_key: None,
            endpoint: "https://api.minimax.io/anthropic/v1".into(),
            client: None,
            headers: HeaderMap::new(),
            dynamic_headers: None,
            timeouts: AnthropicTimeouts::default(),
            replay_policy: ReplayPolicy::IfValid,
            capabilities: None,
            protocol: MiniMaxProtocolSettings {
                thinking: true,
                thinking_disable_allowed: true,
            },
        }
    }

    pub fn model(&self, model_id: impl Into<String>) -> MiniMaxModel {
        let capabilities = self
            .builder
            .capabilities
            .clone()
            .unwrap_or_else(|| minimax_capabilities(self.builder.replay_policy, true));
        MiniMaxModel::new(ModelConfig::new(
            ProviderConfig::new(
                ProviderId::new("minimax"),
                ApiEndpoint::parse(&self.builder.endpoint).unwrap(),
                self.builder
                    .api_key
                    .as_ref()
                    .map_or(MiniMaxAuth::None, |key| {
                        MiniMaxAuth::Bearer(SecretString::new(key.clone()))
                    }),
                headers(
                    self.builder.headers.clone(),
                    self.builder.dynamic_headers.clone(),
                ),
            )
            .unwrap(),
            ModelDeclaration::new(ModelId::new(model_id), capabilities).unwrap(),
            MiniMaxSettings {
                client: client(self.builder.client.clone()),
                timeouts: self.builder.timeouts.clone(),
                protocol: self.builder.protocol.clone(),
                native_context_discriminator: None,
            },
        ))
        .unwrap()
    }
}

impl MiniMaxBuilder {
    pub fn api_key(mut self, value: impl Into<String>) -> Self {
        self.api_key = Some(value.into());
        self
    }
    pub fn base_url(mut self, value: impl Into<String>) -> Self {
        self.endpoint = value.into();
        self
    }
    pub fn client(mut self, value: Client) -> Self {
        self.client = Some(value);
        self
    }
    pub fn default_headers(mut self, value: HeaderMap) -> Self {
        self.headers = value;
        self
    }
    pub fn header_provider(mut self, value: Arc<dyn Fn() -> HeaderMap + Send + Sync>) -> Self {
        self.dynamic_headers = Some(DynamicHeaders(value));
        self
    }
    pub fn timeouts(mut self, value: AnthropicTimeouts) -> Self {
        self.timeouts = value;
        self
    }
    pub fn replay_policy(mut self, value: ReplayPolicy) -> Self {
        self.replay_policy = value;
        self
    }
    pub fn capabilities(mut self, value: ModelCapabilities) -> Self {
        self.capabilities = Some(value);
        self
    }
    pub fn protocol(mut self, value: MiniMaxProtocolSettings) -> Self {
        self.protocol = value;
        self
    }
    pub fn build(self) -> Result<MiniMax, ModelError> {
        Ok(MiniMax { builder: self })
    }
}

pub struct AnthropicAws {
    builder: AnthropicAwsBuilder,
}

pub struct AnthropicAwsBuilder {
    region: String,
    workspace_id: String,
    endpoint: Option<String>,
    auth: Option<AnthropicAwsAuth>,
    client: Option<Client>,
    headers: HeaderMap,
    dynamic_headers: Option<DynamicHeaders>,
    timeouts: AnthropicTimeouts,
    replay_policy: ReplayPolicy,
    capabilities: Option<ModelCapabilities>,
    protocol: AnthropicProtocolSettings,
}

impl AnthropicAws {
    pub fn builder(
        region: impl Into<String>,
        workspace_id: impl Into<String>,
    ) -> AnthropicAwsBuilder {
        AnthropicAwsBuilder {
            region: region.into(),
            workspace_id: workspace_id.into(),
            endpoint: None,
            auth: None,
            client: None,
            headers: HeaderMap::new(),
            dynamic_headers: None,
            timeouts: AnthropicTimeouts::default(),
            replay_policy: ReplayPolicy::IfValid,
            capabilities: None,
            protocol: anthropic_protocol(),
        }
    }

    pub fn model(&self, model_id: impl Into<String>) -> AnthropicAwsModel {
        let capabilities = self
            .builder
            .capabilities
            .clone()
            .unwrap_or_else(|| anthropic_capabilities(self.builder.replay_policy));
        let endpoint = self.builder.endpoint.clone().unwrap_or_else(|| {
            format!(
                "https://aws-external-anthropic.{}.api.aws/v1",
                self.builder.region
            )
        });
        AnthropicAwsModel::new(ModelConfig::new(
            ProviderConfig::new(
                ProviderId::new("anthropic-aws"),
                ApiEndpoint::parse(endpoint).unwrap(),
                self.builder.auth.clone().expect("test AWS auth configured"),
                headers(
                    self.builder.headers.clone(),
                    self.builder.dynamic_headers.clone(),
                ),
            )
            .unwrap(),
            ModelDeclaration::new(ModelId::new(model_id), capabilities).unwrap(),
            AnthropicAwsSettings {
                client: client(self.builder.client.clone()),
                timeouts: self.builder.timeouts.clone(),
                protocol: self.builder.protocol.clone(),
                region: self.builder.region.clone(),
                workspace_id: self.builder.workspace_id.clone(),
                native_context_discriminator: None,
            },
        ))
        .unwrap()
    }
}

impl AnthropicAwsBuilder {
    pub fn bearer_key(mut self, value: impl Into<String>) -> Self {
        self.auth = Some(AnthropicAwsAuth::BearerKey(SecretString::new(value)));
        self
    }
    pub fn static_credentials(mut self, value: AnthropicAwsCredentials) -> Self {
        self.auth = Some(AnthropicAwsAuth::StaticCredentials(value));
        self
    }
    pub fn credential_provider<F, Fut>(mut self, provider: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<AnthropicAwsCredentials, ModelError>> + Send + 'static,
    {
        let provider: AnthropicAwsCredentialProvider = Arc::new(move || Box::pin(provider()));
        self.auth = Some(AnthropicAwsAuth::CredentialProvider(provider));
        self
    }
    pub fn base_url(mut self, value: impl Into<String>) -> Self {
        self.endpoint = Some(value.into());
        self
    }
    pub fn client(mut self, value: Client) -> Self {
        self.client = Some(value);
        self
    }
    pub fn default_headers(mut self, value: HeaderMap) -> Self {
        self.headers = value;
        self
    }
    pub fn header_provider(mut self, value: Arc<dyn Fn() -> HeaderMap + Send + Sync>) -> Self {
        self.dynamic_headers = Some(DynamicHeaders(value));
        self
    }
    pub fn timeouts(mut self, value: AnthropicTimeouts) -> Self {
        self.timeouts = value;
        self
    }
    pub fn replay_policy(mut self, value: ReplayPolicy) -> Self {
        self.replay_policy = value;
        self
    }
    pub fn capabilities(mut self, value: ModelCapabilities) -> Self {
        self.capabilities = Some(value);
        self
    }
    pub fn protocol(mut self, value: AnthropicProtocolSettings) -> Self {
        self.protocol = value;
        self
    }
    pub fn build(self) -> Result<AnthropicAws, ModelError> {
        if self.auth.is_none() {
            return Err(ModelError::invalid_request("test AWS auth is required"));
        }
        for name in [
            "authorization",
            "x-api-key",
            "x-amz-date",
            "x-amz-security-token",
            "x-amz-content-sha256",
            "host",
            "anthropic-workspace-id",
        ] {
            if self.headers.contains_key(name) {
                return Err(ModelError::invalid_request(
                    "test AWS headers contain a protected name",
                ));
            }
        }
        Ok(AnthropicAws { builder: self })
    }
}

pub fn metadata(
    values: impl IntoIterator<Item = (String, serde_json::Value)>,
) -> BTreeMap<String, serde_json::Value> {
    values.into_iter().collect()
}
