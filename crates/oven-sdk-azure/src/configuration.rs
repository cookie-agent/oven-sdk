//! Azure-specific authentication, revisions, routes, and wire settings.

use std::{future::Future, pin::Pin, sync::Arc};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use oven_sdk::{
    AdapterId, ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    LanguageModelDescriptor, MediaSourceSupport, ModelCapabilities, ModelConfig, ModelError,
    ModelIdentity, NativeContextScope, ProviderConfig, ReplayCapability, ResourceId, SecretString,
};
use reqwest::{
    Client,
    header::{CONTENT_TYPE, HeaderMap, HeaderValue},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    transport::AzureOpenAiTimeouts,
    wire::{
        AZURE_OPENAI_CHAT_ADAPTER_ID, AZURE_OPENAI_PROVIDER_ID, AZURE_OPENAI_RESPONSES_ADAPTER_ID,
        AzureApiRoute, Endpoint,
    },
};

/// Boxed asynchronous Microsoft Entra token result.
pub type AzureTokenFuture =
    Pin<Box<dyn Future<Output = Result<String, ModelError>> + Send + 'static>>;

/// Caller-managed asynchronous Microsoft Entra token provider.
pub type AzureTokenProvider = dyn Fn() -> AzureTokenFuture + Send + Sync;

/// Explicit Azure authentication configuration.
#[derive(Clone)]
pub enum AzureOpenAiAuth {
    /// Azure OpenAI API key.
    ApiKey(SecretString),
    /// Caller-managed asynchronous Microsoft Entra token provider.
    Entra(Arc<AzureTokenProvider>),
}

impl std::fmt::Debug for AzureOpenAiAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(_) => formatter.write_str("AzureOpenAiAuth::ApiKey(<redacted>)"),
            Self::Entra(_) => formatter.write_str("AzureOpenAiAuth::Entra(<provider>)"),
        }
    }
}

/// Complete caller-known Azure deployment revision used to scope native replay.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AzureOpenAiRevision {
    /// Underlying Azure model name.
    pub model: String,
    /// Underlying Azure model version.
    pub version: String,
    /// Azure deployment/SKU type.
    pub deployment_type: String,
}

impl AzureOpenAiRevision {
    fn validate(&self) -> Result<(), ModelError> {
        for (name, value) in [
            ("model", self.model.as_str()),
            ("version", self.version.as_str()),
            ("deployment type", self.deployment_type.as_str()),
        ] {
            if value.trim().is_empty() || value.chars().any(char::is_control) {
                return Err(ModelError::invalid_request(format!(
                    "Azure revision {name} must be non-empty and contain no control characters"
                )));
            }
        }
        Ok(())
    }
}

/// Chat system-message wire role.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AzureSystemMessageRole {
    /// Send `system`.
    #[default]
    System,
    /// Send `developer`.
    Developer,
    /// Omit system messages with a warning.
    Omit,
}

/// Chat maximum-output-token field.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AzureMaxTokensField {
    /// Send `max_tokens`.
    #[default]
    MaxTokens,
    /// Send `max_completion_tokens`.
    MaxCompletionTokens,
    /// Omit the requested limit with a warning.
    Omit,
}

/// Chat structured-output wire support.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AzureStructuredOutputSupport {
    /// Reject structured output.
    #[default]
    Unsupported,
    /// Downgrade JSON Schema requests to JSON object mode.
    JsonObject,
    /// Send native JSON Schema output configuration.
    JsonSchema,
}

/// Chat reasoning-history wire field.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AzureReasoningField {
    /// Do not send normalized reasoning history.
    #[default]
    None,
    /// Send `reasoning_content`.
    ReasoningContent,
    /// Send `reasoning`.
    Reasoning,
}

/// Caller-declared Chat Completions wire behavior.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AzureOpenAiCompletionsConfig {
    /// System-message role behavior.
    pub system_role: AzureSystemMessageRole,
    /// Maximum-output-token field behavior.
    pub max_tokens_field: AzureMaxTokensField,
    /// Request streamed usage from Azure.
    pub stream_usage: bool,
    /// Structured-output wire behavior.
    pub structured_output: AzureStructuredOutputSupport,
    /// Reasoning-history field behavior.
    pub reasoning_field: AzureReasoningField,
    /// Omit sampling controls that reasoning deployments reject.
    pub omit_reasoning_sampling: bool,
}

/// Azure-specific settings for a concrete Chat Completions model.
#[derive(Clone, Debug)]
pub struct AzureOpenAiChatSettings {
    /// Typed Azure route family.
    pub route: AzureApiRoute,
    /// Complete caller-known revision required whenever native replay is enabled.
    pub revision: Option<AzureOpenAiRevision>,
    /// Transport phase timeouts.
    pub timeouts: AzureOpenAiTimeouts,
    /// Explicit Chat Completions wire behavior.
    pub completions: AzureOpenAiCompletionsConfig,
}

impl Default for AzureOpenAiChatSettings {
    fn default() -> Self {
        Self {
            route: AzureApiRoute::V1,
            revision: None,
            timeouts: AzureOpenAiTimeouts::default(),
            completions: AzureOpenAiCompletionsConfig::default(),
        }
    }
}

/// Azure-specific settings for a concrete Responses model.
#[derive(Clone, Debug)]
pub struct AzureOpenAiResponsesSettings {
    /// Typed Azure route family.
    pub route: AzureApiRoute,
    /// Complete caller-known revision required whenever replay or native compaction is enabled.
    pub revision: Option<AzureOpenAiRevision>,
    /// Transport phase timeouts.
    pub timeouts: AzureOpenAiTimeouts,
    /// Explicit standalone Responses compaction behavior.
    pub compaction: AzureOpenAiResponsesCompaction,
}

impl Default for AzureOpenAiResponsesSettings {
    fn default() -> Self {
        Self {
            route: AzureApiRoute::V1,
            revision: None,
            timeouts: AzureOpenAiTimeouts::default(),
            compaction: AzureOpenAiResponsesCompaction::Unsupported,
        }
    }
}

/// Explicit Azure Responses native-compaction surface.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AzureOpenAiResponsesCompaction {
    /// Provider-native compaction and native context are disabled.
    #[default]
    Unsupported,
    /// Azure Responses v1 standalone `/responses/compact`.
    V1 {
        /// Stable non-secret label distinguishing caller-controlled dynamic routing.
        routing_discriminator: String,
    },
}

/// Complete registry-free configuration for one Azure Chat Completions model.
pub type AzureOpenAiChatConfig = ModelConfig<AzureOpenAiAuth, AzureOpenAiChatSettings>;

/// Complete registry-free configuration for one Azure Responses model.
pub type AzureOpenAiResponsesConfig = ModelConfig<AzureOpenAiAuth, AzureOpenAiResponsesSettings>;

#[derive(Clone)]
pub(crate) struct Config {
    pub(crate) auth: AzureOpenAiAuth,
    pub(crate) client: Client,
    pub(crate) headers: HeaderConfig,
    pub(crate) base_headers: HeaderMap,
    pub(crate) timeouts: AzureOpenAiTimeouts,
    pub(crate) capabilities: ModelCapabilities,
    pub(crate) completions: AzureOpenAiCompletionsConfig,
    pub(crate) base_url: String,
    pub(crate) query: Vec<(String, String)>,
    native_scope_seed: Sha256,
    native_context_scope: Option<NativeContextScope>,
    identity: ModelIdentity,
}

const NATIVE_SCOPE_VERSION: &str = "azure.openai.native_context_scope.v1";

impl Config {
    pub(crate) fn caller_headers(&self) -> Result<HeaderMap, ModelError> {
        let mut headers = self.base_headers.clone();
        if let Some(provider) = &self.headers.dynamic_headers {
            let dynamic = provider.headers()?;
            reject_protected_headers(dynamic.as_map())?;
            headers.extend(dynamic.as_map().clone());
        }
        Ok(headers)
    }

    pub(crate) fn native_context(
        &self,
        headers: &HeaderMap,
    ) -> Result<(serde_json::Value, NativeContextScope), ModelError> {
        let scope = if let Some(scope) = &self.native_context_scope {
            scope.clone()
        } else {
            let mut hasher = self.native_scope_seed.clone();
            hash_headers(&mut hasher, headers);
            scope_from_hasher(&self.identity, hasher)?
        };
        Ok((binding_from_scope(&scope)?, scope))
    }

    pub(crate) fn configured_native_context(
        &self,
    ) -> Result<Option<(serde_json::Value, NativeContextScope)>, ModelError> {
        self.native_context_scope
            .clone()
            .map(|scope| binding_from_scope(&scope).map(|binding| (binding, scope)))
            .transpose()
    }

    pub(crate) fn configured_native_context_scope(&self) -> Option<&NativeContextScope> {
        self.native_context_scope.as_ref()
    }
}

fn binding_from_scope(scope: &NativeContextScope) -> Result<serde_json::Value, ModelError> {
    let fingerprint = scope
        .resource_id
        .as_str()
        .strip_prefix(&format!("{NATIVE_SCOPE_VERSION}.sha256."))
        .ok_or_else(|| ModelError::invalid_request("invalid Azure native-context scope"))?;
    let binding = serde_json::json!({
        "version": NATIVE_SCOPE_VERSION,
        "sha256": fingerprint,
    });
    Ok(binding)
}

pub(crate) fn build_chat(
    value: AzureOpenAiChatConfig,
) -> Result<(Arc<Config>, LanguageModelDescriptor), ModelError> {
    let ModelConfig {
        provider,
        model,
        settings,
    } = value;
    validate_chat(&model.capabilities, &settings)?;
    build(
        provider,
        model,
        settings.route,
        settings.revision,
        settings.timeouts,
        settings.completions,
        AzureOpenAiResponsesCompaction::Unsupported,
        Endpoint::Chat,
        AZURE_OPENAI_CHAT_ADAPTER_ID,
    )
}

pub(crate) fn build_responses(
    value: AzureOpenAiResponsesConfig,
) -> Result<(Arc<Config>, LanguageModelDescriptor), ModelError> {
    let ModelConfig {
        provider,
        model,
        settings,
    } = value;
    validate_responses(&model.capabilities, &settings)?;
    build(
        provider,
        model,
        settings.route,
        settings.revision,
        settings.timeouts,
        AzureOpenAiCompletionsConfig::default(),
        settings.compaction,
        Endpoint::Responses,
        AZURE_OPENAI_RESPONSES_ADAPTER_ID,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the arguments are the explicit core and Azure construction boundary"
)]
fn build(
    provider: ProviderConfig<AzureOpenAiAuth>,
    model: oven_sdk::ModelDeclaration,
    route: AzureApiRoute,
    revision: Option<AzureOpenAiRevision>,
    timeouts: AzureOpenAiTimeouts,
    completions: AzureOpenAiCompletionsConfig,
    compaction: AzureOpenAiResponsesCompaction,
    endpoint: Endpoint,
    adapter_id: &str,
) -> Result<(Arc<Config>, LanguageModelDescriptor), ModelError> {
    if provider.id.as_str() != AZURE_OPENAI_PROVIDER_ID {
        return Err(ModelError::invalid_request(
            "Azure models require provider ID azure.openai",
        ));
    }
    validate_api_endpoint(&provider.api)?;
    validate_auth(&provider.auth)?;
    reject_protected_headers(provider.headers.static_headers.as_map())?;
    model.validate()?;
    validate_replay_and_compaction(
        &model.capabilities,
        revision.as_ref(),
        &compaction,
        &route,
        endpoint,
    )?;

    let identity = ModelIdentity::new(provider.id.clone(), model.id.clone())?;
    let descriptor = LanguageModelDescriptor::new(
        identity.clone(),
        AdapterId::new(adapter_id),
        model.capabilities.clone(),
    )?;
    let (base_url, query) = endpoint_base(&provider.api, &route, &model.id, endpoint)?;
    let native_scope_seed = native_scope_seed(
        &provider,
        &model,
        &route,
        revision.as_ref(),
        &completions,
        &compaction,
        endpoint,
    )?;
    let native_context_scope = match &compaction {
        AzureOpenAiResponsesCompaction::V1 {
            routing_discriminator,
        } => {
            let mut hasher = native_scope_seed.clone();
            hash_headers(&mut hasher, provider.headers.static_headers.as_map());
            hash_field(
                &mut hasher,
                "routing_discriminator",
                routing_discriminator.as_bytes(),
            );
            Some(scope_from_hasher(&identity, hasher)?)
        }
        AzureOpenAiResponsesCompaction::Unsupported => None,
    };
    let client = Client::builder()
        .connect_timeout(timeouts.connect)
        .build()
        .map_err(|_| ModelError::transport("could not construct Azure OpenAI HTTP client"))?;
    let mut base_headers = provider.headers.static_headers.as_map().clone();
    base_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok((
        Arc::new(Config {
            auth: provider.auth,
            client,
            headers: provider.headers,
            base_headers,
            timeouts,
            capabilities: descriptor.capabilities.clone(),
            completions,
            base_url,
            query,
            native_scope_seed,
            native_context_scope,
            identity,
        }),
        descriptor,
    ))
}

fn validate_chat(
    capabilities: &ModelCapabilities,
    settings: &AzureOpenAiChatSettings,
) -> Result<(), ModelError> {
    validate_capabilities(capabilities, Endpoint::Chat)?;
    if capabilities.compaction != CompactionCapability::Unsupported {
        return Err(ModelError::invalid_request(
            "Azure Chat does not support provider-native compaction",
        ));
    }
    let structured = capabilities
        .features
        .contains(Capability::STRUCTURED_OUTPUT);
    if structured
        != (settings.completions.structured_output != AzureStructuredOutputSupport::Unsupported)
    {
        return Err(ModelError::invalid_request(
            "Azure structured-output capability and Chat settings must agree",
        ));
    }
    if capabilities.features.contains(Capability::USAGE) != settings.completions.stream_usage {
        return Err(ModelError::invalid_request(
            "Azure usage capability and Chat stream-usage setting must agree",
        ));
    }
    if settings.completions.omit_reasoning_sampling
        && !capabilities.features.contains(Capability::REASONING)
    {
        return Err(ModelError::invalid_request(
            "Azure reasoning sampling behavior requires reasoning capability",
        ));
    }
    Ok(())
}

fn validate_responses(
    capabilities: &ModelCapabilities,
    settings: &AzureOpenAiResponsesSettings,
) -> Result<(), ModelError> {
    validate_capabilities(capabilities, Endpoint::Responses)?;
    match (&settings.compaction, capabilities.compaction) {
        (AzureOpenAiResponsesCompaction::Unsupported, CompactionCapability::Unsupported) => Ok(()),
        (AzureOpenAiResponsesCompaction::V1 { .. }, CompactionCapability::Native) => Ok(()),
        (AzureOpenAiResponsesCompaction::Unsupported, CompactionCapability::Native) => {
            Err(ModelError::invalid_request(
                "Azure native compaction requires explicit Responses V1 compaction settings",
            ))
        }
        (AzureOpenAiResponsesCompaction::V1 { .. }, CompactionCapability::Unsupported) => {
            Err(ModelError::invalid_request(
                "Azure Responses V1 compaction settings require native compaction capability",
            ))
        }
    }
}

fn validate_capabilities(
    capabilities: &ModelCapabilities,
    endpoint: Endpoint,
) -> Result<(), ModelError> {
    capabilities.validate()?;
    if capabilities.cancellation != CancellationCapability::LocalOnly {
        return Err(ModelError::invalid_request(
            "Azure models declare local-only cancellation",
        ));
    }
    let text = oven_sdk::Modality::text();
    if !capabilities.modalities.input.contains(&text)
        || capabilities.modalities.output.len() != 1
        || !capabilities.modalities.output.contains(&text)
    {
        return Err(ModelError::invalid_request(
            "Azure models require text input and text-only output modalities",
        ));
    }
    for modality in &capabilities.modalities.input {
        if !matches!(modality.as_str(), "text" | "image" | "pdf") {
            return Err(ModelError::invalid_request(
                "Azure adapters implement only text, image, and PDF input modalities",
            ));
        }
    }
    if endpoint == Endpoint::Chat
        && capabilities
            .modalities
            .input
            .contains(&oven_sdk::Modality::pdf())
    {
        return Err(ModelError::invalid_request(
            "Azure Chat does not implement PDF input",
        ));
    }
    validate_media(capabilities, endpoint)
}

fn validate_media(capabilities: &ModelCapabilities, endpoint: Endpoint) -> Result<(), ModelError> {
    const IMAGE_TYPES: &[&str] = &["image/png", "image/jpeg", "image/webp", "image/gif"];
    for (modality, support) in &capabilities.media.input {
        let valid_types = match modality.as_str() {
            "image" => support
                .media_types
                .iter()
                .all(|media_type| IMAGE_TYPES.contains(&media_type.as_str())),
            "pdf" if endpoint == Endpoint::Responses => support
                .media_types
                .iter()
                .all(|media_type| media_type == "application/pdf"),
            _ => false,
        };
        let valid_sources = !support
            .sources
            .contains(MediaSourceSupport::PROVIDER_REFERENCE);
        if !valid_types || !valid_sources {
            return Err(ModelError::invalid_request(
                "Azure media declarations exceed implemented MIME or source support",
            ));
        }
    }
    Ok(())
}

fn validate_replay_and_compaction(
    capabilities: &ModelCapabilities,
    revision: Option<&AzureOpenAiRevision>,
    compaction: &AzureOpenAiResponsesCompaction,
    route: &AzureApiRoute,
    endpoint: Endpoint,
) -> Result<(), ModelError> {
    if capabilities.replay.capability != ReplayCapability::Unsupported
        || capabilities.compaction == CompactionCapability::Native
    {
        revision
            .ok_or_else(|| {
                ModelError::invalid_request(
                    "Azure native replay requires a complete explicit deployment revision",
                )
            })?
            .validate()?;
    }
    match compaction {
        AzureOpenAiResponsesCompaction::Unsupported => {}
        AzureOpenAiResponsesCompaction::V1 {
            routing_discriminator,
        } => {
            if endpoint != Endpoint::Responses || route != &AzureApiRoute::V1 {
                return Err(ModelError::invalid_request(
                    "Azure native compaction is available only on the Responses V1 route",
                ));
            }
            validate_routing_discriminator(routing_discriminator)?;
        }
    }
    if endpoint == Endpoint::Chat && capabilities.replay.reasoning {
        return Err(ModelError::invalid_request(
            "Azure Chat does not declare provider-authoritative reasoning replay",
        ));
    }
    if endpoint == Endpoint::Responses
        && capabilities.features.contains(Capability::REASONING)
        && capabilities.replay.capability != ReplayCapability::Unsupported
        && !capabilities.replay.reasoning
    {
        return Err(ModelError::invalid_request(
            "Azure Responses reasoning replay must be declared explicitly",
        ));
    }
    Ok(())
}

fn validate_auth(auth: &AzureOpenAiAuth) -> Result<(), ModelError> {
    if matches!(auth, AzureOpenAiAuth::ApiKey(key) if key.is_empty()) {
        return Err(ModelError::invalid_request(
            "Azure API key must not be empty",
        ));
    }
    Ok(())
}

fn validate_api_endpoint(api: &ApiEndpoint) -> Result<(), ModelError> {
    let api = api.as_url();
    let loopback_http = api.scheme() == "http"
        && api
            .host_str()
            .is_some_and(|host| host == "localhost" || host == "127.0.0.1" || host == "::1");
    if (api.scheme() != "https" && !loopback_http)
        || api.query().is_some()
        || !matches!(api.path(), "" | "/")
    {
        return Err(ModelError::invalid_request(
            "Azure API endpoint must be an HTTPS base URL without path or query",
        ));
    }
    Ok(())
}

fn endpoint_base(
    api: &ApiEndpoint,
    route: &AzureApiRoute,
    model_id: &oven_sdk::ModelId,
    endpoint: Endpoint,
) -> Result<(String, Vec<(String, String)>), ModelError> {
    let mut url = route.endpoint(api.as_url(), model_id.as_str(), endpoint)?;
    let suffix = match endpoint {
        Endpoint::Chat => "/chat/completions",
        Endpoint::Responses => "/responses",
    };
    let path = url
        .path()
        .strip_suffix(suffix)
        .ok_or_else(|| ModelError::invalid_request("Azure route produced an invalid endpoint"))?
        .to_owned();
    url.set_path(&path);
    let query = url
        .query_pairs()
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    url.set_query(None);
    Ok((url.to_string().trim_end_matches('/').to_owned(), query))
}

pub(crate) fn reject_protected_headers(headers: &HeaderMap) -> Result<(), ModelError> {
    const PROTECTED: &[&str] = &[
        "api-key",
        "authorization",
        "host",
        "content-type",
        "content-length",
    ];
    if PROTECTED.iter().any(|name| headers.contains_key(*name)) {
        Err(ModelError::invalid_request(
            "Azure authentication and transport headers are protected",
        ))
    } else {
        Ok(())
    }
}

fn native_scope_seed(
    provider: &ProviderConfig<AzureOpenAiAuth>,
    model: &oven_sdk::ModelDeclaration,
    route: &AzureApiRoute,
    revision: Option<&AzureOpenAiRevision>,
    completions: &AzureOpenAiCompletionsConfig,
    compaction: &AzureOpenAiResponsesCompaction,
    endpoint: Endpoint,
) -> Result<Sha256, ModelError> {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, "version", NATIVE_SCOPE_VERSION.as_bytes());
    hash_field(&mut hasher, "provider_id", provider.id.as_str().as_bytes());
    hash_field(&mut hasher, "model_id", model.id.as_str().as_bytes());
    hash_field(
        &mut hasher,
        "endpoint_scheme",
        provider.api.as_url().scheme().as_bytes(),
    );
    hash_field(
        &mut hasher,
        "endpoint_host",
        provider
            .api
            .as_url()
            .host_str()
            .expect("validated Azure endpoint has a host")
            .as_bytes(),
    );
    hash_field(
        &mut hasher,
        "endpoint_port",
        &provider
            .api
            .as_url()
            .port_or_known_default()
            .expect("validated HTTP(S) endpoint has a port")
            .to_be_bytes(),
    );
    hash_serialized(&mut hasher, "route", route)?;
    hash_field(&mut hasher, "surface", endpoint.as_str().as_bytes());
    hash_serialized(&mut hasher, "revision", &revision)?;
    hash_serialized(&mut hasher, "capabilities", &model.capabilities)?;
    hash_serialized(&mut hasher, "completions", completions)?;
    hash_serialized(&mut hasher, "compaction", compaction)?;
    Ok(hasher)
}

fn scope_from_hasher(
    identity: &ModelIdentity,
    hasher: Sha256,
) -> Result<NativeContextScope, ModelError> {
    let fingerprint = URL_SAFE_NO_PAD.encode(hasher.finalize());
    NativeContextScope::new(
        identity.provider_id.clone(),
        identity.model_id.clone(),
        ResourceId::new(format!("{NATIVE_SCOPE_VERSION}.sha256.{fingerprint}"))?,
    )
}

fn validate_routing_discriminator(value: &str) -> Result<(), ModelError> {
    if value.trim().is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        return Err(ModelError::invalid_request(
            "Azure compaction routing_discriminator must be a non-empty non-secret label of at most 256 bytes",
        ));
    }
    Ok(())
}

fn hash_serialized(
    hasher: &mut Sha256,
    tag: &str,
    value: &impl Serialize,
) -> Result<(), ModelError> {
    let value = serde_json::to_vec(value)
        .map_err(|_| ModelError::invalid_request("could not encode Azure replay scope inputs"))?;
    hash_field(hasher, tag, &value);
    Ok(())
}

fn hash_headers(hasher: &mut Sha256, headers: &HeaderMap) {
    let mut names = headers.keys().collect::<Vec<_>>();
    names.sort_unstable_by(|left, right| left.as_str().cmp(right.as_str()));
    hash_usize(hasher, names.len());
    for name in names {
        hash_field(hasher, "header_name", name.as_str().as_bytes());
        let values = headers.get_all(name);
        hash_usize(hasher, values.iter().count());
        for value in values {
            hash_field(hasher, "header_value", value.as_bytes());
        }
    }
}

fn hash_field(hasher: &mut Sha256, tag: &str, value: &[u8]) {
    hash_usize(hasher, tag.len());
    hasher.update(tag.as_bytes());
    hash_usize(hasher, value.len());
    hasher.update(value);
}

fn hash_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_be_bytes());
}

#[cfg(test)]
mod tests {
    use reqwest::header::{HeaderMap, HeaderValue};
    use sha2::{Digest, Sha256};

    use super::hash_headers;

    fn header_fingerprint(headers: &HeaderMap) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hash_headers(&mut hasher, headers);
        hasher.finalize().to_vec()
    }

    #[test]
    fn header_fingerprint_is_deterministic_across_name_insertion_order() {
        let mut first = HeaderMap::new();
        first.insert("x-second", HeaderValue::from_static("two"));
        first.insert("x-first", HeaderValue::from_static("one"));

        let mut second = HeaderMap::new();
        second.insert("x-first", HeaderValue::from_static("one"));
        second.insert("x-second", HeaderValue::from_static("two"));

        assert_eq!(header_fingerprint(&first), header_fingerprint(&second));
    }

    #[test]
    fn header_fingerprint_preserves_repeated_value_multiplicity_and_order() {
        let mut one = HeaderMap::new();
        one.append("x-repeat", HeaderValue::from_static("one"));

        let mut repeated = one.clone();
        repeated.append("x-repeat", HeaderValue::from_static("two"));

        let mut reversed = HeaderMap::new();
        reversed.append("x-repeat", HeaderValue::from_static("two"));
        reversed.append("x-repeat", HeaderValue::from_static("one"));

        assert_ne!(header_fingerprint(&one), header_fingerprint(&repeated));
        assert_ne!(header_fingerprint(&repeated), header_fingerprint(&reversed));
    }
}
