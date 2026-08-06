//! Explicit authentication and structural adapter settings.

use std::{fmt, sync::Arc};

use oven_sdk::{
    AdapterId, Capability, DynHeaderProvider, HeaderConfig, MediaSourceSupport, Modality,
    ModelCapabilities, ModelError, ResourceId, SecretString,
};
use reqwest::{
    Client,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::transport::OpenAiTimeouts;

/// Explicit official OpenAI authentication and account headers.
#[derive(Clone, Debug)]
pub struct OpenAiAuth {
    /// Bearer API key resolved by the caller.
    pub api_key: SecretString,
    /// Optional `OpenAI-Organization` value.
    pub organization: Option<String>,
    /// Optional `OpenAI-Project` value.
    pub project: Option<String>,
}

impl OpenAiAuth {
    /// Creates bearer authentication without organization or project headers.
    #[must_use]
    pub fn new(api_key: SecretString) -> Self {
        Self {
            api_key,
            organization: None,
            project: None,
        }
    }
}

/// Explicit authentication for one OpenAI-compatible Chat endpoint.
#[derive(Clone, Default)]
pub struct OpenAiCompatibleAuth {
    /// Optional Bearer token.
    pub bearer: Option<SecretString>,
    /// Optional caller-managed authentication header provider.
    pub header_provider: Option<DynHeaderProvider>,
}

impl OpenAiCompatibleAuth {
    /// Creates an endpoint configuration without authentication.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Creates Bearer authentication.
    #[must_use]
    pub fn bearer(token: SecretString) -> Self {
        Self {
            bearer: Some(token),
            header_provider: None,
        }
    }

    /// Creates caller-managed authentication headers.
    #[must_use]
    pub fn headers(provider: Arc<dyn oven_sdk::HeaderProvider>) -> Self {
        Self {
            bearer: None,
            header_provider: Some(provider),
        }
    }
}

impl fmt::Debug for OpenAiCompatibleAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleAuth")
            .field("bearer", &self.bearer)
            .field(
                "header_provider",
                &self.header_provider.as_ref().map(|_| "<provider>"),
            )
            .finish()
    }
}

/// Compatible reasoning delta field.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningField {
    /// No visible Chat reasoning field.
    #[default]
    None,
    /// `reasoning_content`.
    ReasoningContent,
    /// `reasoning`.
    Reasoning,
}

/// Chat structured-output wire behavior.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredOutputSupport {
    /// Structured output is unsupported.
    #[default]
    Unsupported,
    /// JSON object mode only.
    JsonObject,
    /// Native JSON Schema mode.
    JsonSchema,
}

/// Role used for normalized Chat system messages.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemMessageRole {
    /// `system` role.
    #[default]
    System,
    /// `developer` role.
    Developer,
    /// Omit system messages with a warning.
    Omit,
}

/// Chat output-token request field.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxTokensField {
    /// `max_tokens`.
    #[default]
    MaxTokens,
    /// `max_completion_tokens`.
    MaxCompletionTokens,
    /// Omit the field with a warning.
    Omit,
}

/// Structural settings for official OpenAI Chat Completions.
#[derive(Clone, Debug)]
pub struct OpenAiChatSettings {
    /// Role used for system messages.
    pub system_message_role: SystemMessageRole,
    /// Output-token field.
    pub max_tokens_field: MaxTokensField,
    /// Whether to request streamed usage.
    pub stream_usage: bool,
    /// Structured-output wire behavior.
    pub structured_output: StructuredOutputSupport,
    /// Optional visible reasoning wire field.
    pub reasoning_field: ReasoningField,
    /// Stable non-secret routing identity required when dynamic headers can
    /// select a different backend or account.
    pub routing_discriminator: Option<String>,
    /// Optional injected HTTP client.
    pub client: Option<Client>,
    /// Transport phase timeouts.
    pub timeouts: OpenAiTimeouts,
}

impl OpenAiChatSettings {
    /// Creates explicit Chat wire settings.
    #[must_use]
    pub fn new(
        system_message_role: SystemMessageRole,
        max_tokens_field: MaxTokensField,
        structured_output: StructuredOutputSupport,
    ) -> Self {
        Self {
            system_message_role,
            max_tokens_field,
            stream_usage: true,
            structured_output,
            reasoning_field: ReasoningField::None,
            routing_discriminator: None,
            client: None,
            timeouts: OpenAiTimeouts::default(),
        }
    }
}

/// Structural settings for official OpenAI Responses.
#[derive(Clone, Debug)]
pub struct OpenAiResponsesSettings {
    /// Stable non-secret routing identity required when dynamic headers can
    /// select a different backend or account.
    pub routing_discriminator: Option<String>,
    /// Explicit standalone Responses compaction surface.
    pub compaction: OpenAiResponsesCompaction,
    /// Optional injected HTTP client.
    pub client: Option<Client>,
    /// Transport phase timeouts.
    pub timeouts: OpenAiTimeouts,
}

impl OpenAiResponsesSettings {
    /// Creates Responses settings with adapter transport defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            routing_discriminator: None,
            compaction: OpenAiResponsesCompaction::Unsupported,
            client: None,
            timeouts: OpenAiTimeouts::default(),
        }
    }
}

/// Explicit official Responses native-compaction surface.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiResponsesCompaction {
    /// Provider-native compaction and native context are disabled.
    #[default]
    Unsupported,
    /// Official Responses v1 standalone `/responses/compact`.
    V1,
}

impl Default for OpenAiResponsesSettings {
    fn default() -> Self {
        Self::new()
    }
}

/// Structural settings for one OpenAI-compatible Chat endpoint.
#[derive(Clone, Debug)]
pub struct OpenAiCompatibleChatSettings {
    /// Caller-owned stable adapter identity.
    pub adapter_id: AdapterId,
    /// Role used for system messages.
    pub system_message_role: SystemMessageRole,
    /// Output-token field.
    pub max_tokens_field: MaxTokensField,
    /// Whether to request streamed usage.
    pub stream_usage: bool,
    /// Structured-output wire behavior.
    pub structured_output: StructuredOutputSupport,
    /// Optional visible reasoning wire field.
    pub reasoning_field: ReasoningField,
    /// Query parameters appended to every Chat request.
    pub query: Vec<(String, String)>,
    /// Additional response request-ID header names.
    pub request_id_headers: Vec<String>,
    /// Whether successful responses must use an SSE content type.
    pub strict_sse_content_type: bool,
    /// Stable non-secret routing identity required when dynamic headers or
    /// dynamic authentication can select a different backend or account.
    pub routing_discriminator: Option<String>,
    /// Optional injected HTTP client.
    pub client: Option<Client>,
    /// Transport phase timeouts.
    pub timeouts: OpenAiTimeouts,
}

impl OpenAiCompatibleChatSettings {
    /// Creates explicit compatible Chat wire settings.
    #[must_use]
    pub fn new(
        adapter_id: AdapterId,
        system_message_role: SystemMessageRole,
        max_tokens_field: MaxTokensField,
        structured_output: StructuredOutputSupport,
        reasoning_field: ReasoningField,
    ) -> Self {
        Self {
            adapter_id,
            system_message_role,
            max_tokens_field,
            stream_usage: false,
            structured_output,
            reasoning_field,
            query: Vec::new(),
            request_id_headers: vec!["x-request-id".into()],
            strict_sse_content_type: false,
            routing_discriminator: None,
            client: None,
            timeouts: OpenAiTimeouts::default(),
        }
    }
}

pub(crate) fn build_client(
    client: Option<Client>,
    timeouts: &OpenAiTimeouts,
) -> Result<Client, ModelError> {
    match client {
        Some(client) => Ok(client),
        None => Client::builder()
            .connect_timeout(timeouts.connect)
            .build()
            .map_err(|_| ModelError::transport("could not construct OpenAI HTTP client")),
    }
}

pub(crate) fn official_headers(
    auth: &OpenAiAuth,
    configured: &HeaderConfig,
) -> Result<HeaderMap, ModelError> {
    if auth.api_key.is_empty() {
        return Err(ModelError::invalid_request(
            "OpenAI authentication requires a non-empty API key",
        ));
    }
    let mut headers = HeaderMap::new();
    insert_header(
        &mut headers,
        AUTHORIZATION,
        &format!("Bearer {}", auth.api_key.expose_secret()),
    )?;
    if let Some(organization) = &auth.organization {
        insert_named_header(&mut headers, "openai-organization", organization)?;
    }
    if let Some(project) = &auth.project {
        insert_named_header(&mut headers, "openai-project", project)?;
    }
    extend_configured_headers(&mut headers, configured, true)?;
    Ok(headers)
}

pub(crate) fn compatible_headers(
    auth: &OpenAiCompatibleAuth,
    configured: &HeaderConfig,
) -> Result<HeaderMap, ModelError> {
    let mut headers = HeaderMap::new();
    if let Some(token) = &auth.bearer {
        if token.is_empty() {
            return Err(ModelError::invalid_request(
                "compatible Bearer authentication must not be empty",
            ));
        }
        insert_header(
            &mut headers,
            AUTHORIZATION,
            &format!("Bearer {}", token.expose_secret()),
        )?;
    }
    if let Some(provider) = &auth.header_provider {
        let supplied = provider.headers()?;
        validate_headers(supplied.as_map(), false)?;
        headers.extend(supplied.as_map().clone());
    }
    extend_configured_headers(&mut headers, configured, true)?;
    Ok(headers)
}

fn extend_configured_headers(
    headers: &mut HeaderMap,
    configured: &HeaderConfig,
    protect_auth: bool,
) -> Result<(), ModelError> {
    validate_headers(configured.static_headers.as_map(), protect_auth)?;
    headers.extend(configured.static_headers.as_map().clone());
    if let Some(provider) = &configured.dynamic_headers {
        let supplied = provider.headers()?;
        validate_headers(supplied.as_map(), protect_auth)?;
        headers.extend(supplied.as_map().clone());
    }
    Ok(())
}

fn validate_headers(headers: &HeaderMap, protect_auth: bool) -> Result<(), ModelError> {
    for name in headers.keys() {
        if name == CONTENT_TYPE || (protect_auth && name == AUTHORIZATION) {
            return Err(ModelError::invalid_request(format!(
                "OpenAI header override `{name}` is protected"
            )));
        }
        if protect_auth && matches!(name.as_str(), "openai-organization" | "openai-project") {
            return Err(ModelError::invalid_request(format!(
                "OpenAI header override `{name}` is protected"
            )));
        }
    }
    Ok(())
}

fn insert_named_header(headers: &mut HeaderMap, name: &str, value: &str) -> Result<(), ModelError> {
    let name = HeaderName::from_bytes(name.as_bytes())
        .map_err(|_| ModelError::invalid_request("invalid OpenAI header name"))?;
    insert_header(headers, name, value)
}

fn insert_header(headers: &mut HeaderMap, name: HeaderName, value: &str) -> Result<(), ModelError> {
    let value = HeaderValue::from_str(value)
        .map_err(|_| ModelError::invalid_request("invalid OpenAI header value"))?;
    headers.insert(name, value);
    Ok(())
}

pub(crate) fn validate_chat_declaration(
    capabilities: &ModelCapabilities,
    max_tokens_field: MaxTokensField,
    structured_output: StructuredOutputSupport,
    reasoning_field: ReasoningField,
    stream_usage: bool,
) -> Result<(), ModelError> {
    validate_features(capabilities)?;
    validate_cancellation(capabilities)?;
    if capabilities.compaction == oven_sdk::CompactionCapability::Native {
        return Err(ModelError::invalid_request(
            "Chat adapters do not support provider-native compaction",
        ));
    }
    if stream_usage != capabilities.features.contains(Capability::USAGE) {
        return Err(ModelError::invalid_request(
            "Chat stream-usage settings and usage capability must agree",
        ));
    }
    if capabilities
        .features
        .contains(Capability::MAX_OUTPUT_TOKENS)
        && max_tokens_field == MaxTokensField::Omit
    {
        return Err(ModelError::invalid_request(
            "Chat max-output-token support requires an enabled token field",
        ));
    }
    if capabilities
        .features
        .contains(Capability::STRUCTURED_OUTPUT)
        && structured_output == StructuredOutputSupport::Unsupported
    {
        return Err(ModelError::invalid_request(
            "Chat structured-output capability requires an enabled wire shape",
        ));
    }
    if reasoning_field != ReasoningField::None
        && !capabilities.features.contains(Capability::REASONING)
    {
        return Err(ModelError::invalid_request(
            "Chat reasoning wire fields require reasoning capability",
        ));
    }
    validate_media(capabilities, false)
}

pub(crate) fn validate_responses_declaration(
    capabilities: &ModelCapabilities,
    compaction: OpenAiResponsesCompaction,
) -> Result<(), ModelError> {
    validate_features(capabilities)?;
    validate_cancellation(capabilities)?;
    if capabilities.features.contains(Capability::REASONING)
        && (!capabilities.replay.reasoning
            || capabilities.replay.capability == oven_sdk::ReplayCapability::Unsupported)
    {
        return Err(ModelError::invalid_request(
            "Responses reasoning requires reasoning-aware native replay",
        ));
    }
    match (compaction, capabilities.compaction) {
        (OpenAiResponsesCompaction::Unsupported, oven_sdk::CompactionCapability::Unsupported)
        | (OpenAiResponsesCompaction::V1, oven_sdk::CompactionCapability::Native) => {}
        (OpenAiResponsesCompaction::Unsupported, oven_sdk::CompactionCapability::Native) => {
            return Err(ModelError::invalid_request(
                "native compaction requires explicit Responses V1 compaction settings",
            ));
        }
        (OpenAiResponsesCompaction::V1, oven_sdk::CompactionCapability::Unsupported) => {
            return Err(ModelError::invalid_request(
                "Responses V1 compaction settings require native compaction capability",
            ));
        }
    }
    validate_media(capabilities, true)
}

fn validate_cancellation(capabilities: &ModelCapabilities) -> Result<(), ModelError> {
    if capabilities.cancellation == oven_sdk::CancellationCapability::RemoteBestEffort {
        return Err(ModelError::invalid_request(
            "OpenAI adapters implement local cancellation only",
        ));
    }
    Ok(())
}

fn validate_features(capabilities: &ModelCapabilities) -> Result<(), ModelError> {
    let unsupported = Capability::PROMPT_CACHING | Capability::PROVIDER_TOOLS | Capability::SOURCES;
    if capabilities.features.intersects(unsupported) {
        return Err(ModelError::invalid_request(
            "this OpenAI adapter does not implement declared prompt-caching, provider-tool, or source capabilities",
        ));
    }
    Ok(())
}

fn validate_media(capabilities: &ModelCapabilities, responses: bool) -> Result<(), ModelError> {
    for (modality, support) in &capabilities.media.input {
        let allowed_sources = if modality == &Modality::image() {
            MediaSourceSupport::INLINE_BYTES
                | MediaSourceSupport::INLINE_TEXT
                | MediaSourceSupport::URL
        } else if modality == &Modality::pdf() {
            let inline = MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::INLINE_TEXT;
            if responses {
                inline | MediaSourceSupport::URL
            } else {
                inline
            }
        } else {
            return Err(ModelError::invalid_request(
                "OpenAI adapters implement media wire encoding only for image and PDF input",
            ));
        };
        if !allowed_sources.contains(support.sources) {
            return Err(ModelError::invalid_request(
                "declared OpenAI media source form is not supported by this API surface",
            ));
        }
        for media_type in &support.media_types {
            let valid = if modality == &Modality::image() {
                media_type.starts_with("image/")
            } else {
                media_type == "application/pdf"
            };
            if !valid {
                return Err(ModelError::invalid_request(
                    "declared OpenAI media MIME type is not supported by this API surface",
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn replay_resource_id(parts: &[&str]) -> Result<ResourceId, ModelError> {
    let mut input = Vec::new();
    input.extend_from_slice(b"oven-sdk-openai-resource-scope-v1\0");
    for part in parts {
        input.extend_from_slice(&(part.len() as u64).to_be_bytes());
        input.extend_from_slice(part.as_bytes());
    }
    ResourceId::new(format!("openai-scope-v1-sha256-{}", sha256_hex(&input)))
}

pub(crate) fn canonical_endpoint(endpoint: &Url) -> String {
    let mut endpoint = endpoint.clone();
    let mut path = endpoint.path().trim_end_matches('/').to_owned();
    if path.is_empty() {
        path.push('/');
    }
    endpoint.set_path(&path);
    let mut query = endpoint
        .query_pairs()
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    query.sort();
    endpoint.set_query(None);
    if !query.is_empty() {
        endpoint.query_pairs_mut().extend_pairs(query);
    }
    endpoint.to_string()
}

pub(crate) fn header_scope_component(headers: &HeaderConfig) -> String {
    let mut values = headers
        .static_headers
        .as_map()
        .iter()
        .map(|(name, value)| {
            let encoded = value
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            (name.as_str().to_owned(), encoded)
        })
        .collect::<Vec<_>>();
    values.sort();
    values
        .into_iter()
        .map(|(name, value)| format!("{name}:{value}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn validate_routing_discriminator(
    has_dynamic_headers: bool,
    value: Option<&str>,
) -> Result<(), ModelError> {
    if has_dynamic_headers && value.is_none_or(|value| value.trim().is_empty()) {
        return Err(ModelError::invalid_request(
            "dynamic OpenAI routing headers require a non-empty routing_discriminator",
        ));
    }
    if value.is_some_and(|value| value.trim().is_empty()) {
        return Err(ModelError::invalid_request(
            "OpenAI routing_discriminator must not be empty",
        ));
    }
    Ok(())
}

pub(crate) fn sha256_hex(value: &[u8]) -> String {
    sha256(value)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256(value: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    Sha256::digest(value).into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn sha256_matches_standard_vector() {
        assert_eq!(
            super::sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn endpoint_canonicalization_normalizes_slashes_ports_and_query_order() {
        let first = url::Url::parse("https://EXAMPLE.test:443/v1/?b=2&a=1").unwrap();
        let second = url::Url::parse("https://example.test/v1?a=1&b=2").unwrap();
        assert_eq!(
            super::canonical_endpoint(&first),
            super::canonical_endpoint(&second)
        );
    }
}
