#![warn(missing_docs)]
// ModelError deliberately owns the spec-mandated typed diagnostics so callers
// can inspect them without an allocation or a second error channel.
#![allow(clippy::result_large_err)]
//! Runtime-neutral normalized contracts for language-model providers.
//!
//! The crate owns typed model contracts, stream lifecycle, cancellation,
//! capabilities, and replay artifacts. Transport, logging, persistence, and
//! retry or fallback policy belong to adapters and calling harnesses.

/// Runtime-neutral implementation helpers shared by provider adapters.
pub mod provider_support;

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    future::Future,
    io,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
    time::Duration,
};

use bitflags::bitflags;
use bytes::Bytes;
use futures_core::Stream;
use http::HeaderMap;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{SeqAccess, Visitor},
    ser::SerializeSeq,
};
use thiserror::Error;
use url::Url;

/// A boxed, sendable future used by [`LanguageModel`].
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A boxed, sendable, runtime-neutral stream.
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

/// JSON values used by the normalized contract.
pub type JsonValue = serde_json::Value;

/// Provider-defined options keyed by adapter or provider namespace.
pub type ProviderOptions = BTreeMap<String, JsonValue>;

/// Provider-defined metadata attached to one normalized content part.
pub type PartMetadata = Option<BTreeMap<String, JsonValue>>;

/// Provider-defined response metadata.
pub type ResponseMetadata = BTreeMap<String, JsonValue>;

/// Provider-defined terminal metadata.
pub type ProviderMetadata = BTreeMap<String, JsonValue>;

#[derive(Default)]
struct CountingWriter {
    bytes: usize,
}

impl io::Write for CountingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(buffer.len())
            .ok_or_else(|| io::Error::other("serialized JSON size overflow"))?;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialized_json_size(value: &impl Serialize) -> Result<usize, serde_json::Error> {
    let mut writer = CountingWriter::default();
    serde_json::to_writer(&mut writer, value)?;
    Ok(writer.bytes)
}

/// A stable provider namespace.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ProviderId(String);

impl ProviderId {
    /// Creates a provider identity.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the stable provider string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Validates that the provider identity is non-empty and contains no control characters.
    pub fn validate(&self) -> Result<(), ModelError> {
        validate_identifier("provider ID", &self.0)
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ProviderId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Self(String::deserialize(deserializer)?);
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

/// A configured model identifier.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ModelId(String);

impl ModelId {
    /// Creates a model identity.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the model identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Validates that the model identity is non-empty and contains no control characters.
    pub fn validate(&self) -> Result<(), ModelError> {
        validate_identifier("model ID", &self.0)
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ModelId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Self(String::deserialize(deserializer)?);
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

/// A stable adapter identity used to decide native replay compatibility.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct AdapterId(String);

impl AdapterId {
    /// Creates an adapter identity.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the stable adapter identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Validates that the adapter identity is non-empty and contains no control characters.
    pub fn validate(&self) -> Result<(), ModelError> {
        validate_identifier("adapter ID", &self.0)
    }
}

impl fmt::Display for AdapterId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for AdapterId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Self(String::deserialize(deserializer)?);
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

fn validate_identifier(name: &str, value: &str) -> Result<(), ModelError> {
    if value.trim().is_empty() {
        return Err(ModelError::invalid_request(format!(
            "{name} must not be empty"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(ModelError::invalid_request(format!(
            "{name} must not contain control characters"
        )));
    }
    Ok(())
}

/// Runtime identity of one provider-served model offering.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ModelIdentity {
    /// Serving provider identity.
    pub provider_id: ProviderId,
    /// Exact provider model, deployment, or resource identifier.
    pub model_id: ModelId,
}

impl ModelIdentity {
    /// Creates and validates one runtime identity.
    pub fn new(provider_id: ProviderId, model_id: ModelId) -> Result<Self, ModelError> {
        provider_id.validate()?;
        model_id.validate()?;
        Ok(Self {
            provider_id,
            model_id,
        })
    }

    /// Validates both identity components.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.provider_id.validate()?;
        self.model_id.validate()
    }
}

/// A validated base API endpoint supplied explicitly by the caller.
#[derive(Clone, Eq, PartialEq)]
pub struct ApiEndpoint {
    base_url: Url,
}

impl ApiEndpoint {
    /// Parses an endpoint without credentials, fragments, or environment templates.
    pub fn parse(value: impl AsRef<str>) -> Result<Self, ModelError> {
        let value = value.as_ref();
        if value.contains("${") {
            return Err(ModelError::invalid_request(
                "API endpoint must not contain unresolved environment templates",
            ));
        }
        let base_url = Url::parse(value).map_err(|error| {
            ModelError::invalid_request(format!("invalid API endpoint: {error}"))
        })?;
        if !matches!(base_url.scheme(), "http" | "https") {
            return Err(ModelError::invalid_request(
                "API endpoint scheme must be http or https",
            ));
        }
        if base_url.host_str().is_none() {
            return Err(ModelError::invalid_request(
                "API endpoint must include a host",
            ));
        }
        if !base_url.username().is_empty() || base_url.password().is_some() {
            return Err(ModelError::invalid_request(
                "API endpoint must not include user information",
            ));
        }
        if base_url.fragment().is_some() {
            return Err(ModelError::invalid_request(
                "API endpoint must not include a fragment",
            ));
        }
        if base_url.query().is_some() {
            return Err(ModelError::invalid_request(
                "API endpoint must not include a query",
            ));
        }
        Ok(Self { base_url })
    }

    /// Returns the validated endpoint URL.
    #[must_use]
    pub fn as_url(&self) -> &Url {
        &self.base_url
    }
}

impl fmt::Debug for ApiEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApiEndpoint(<redacted>)")
    }
}

impl Serialize for ApiEndpoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.base_url.as_str())
    }
}

impl<'de> Deserialize<'de> for ApiEndpoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::parse(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// A secret string whose formatting is always redacted and which is never serializable.
#[derive(Clone, Eq, PartialEq)]
pub struct SecretString(String);

impl SecretString {
    /// Wraps a caller-resolved secret.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Exposes the secret only to code that must construct authentication material.
    #[must_use]
    pub fn expose_secret(&self) -> &str {
        &self.0
    }

    /// Returns whether the wrapped secret is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretString(<redacted>)")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted>")
    }
}

/// Validated caller header overrides. Debug output exposes names only.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct HeaderOverrides {
    headers: HeaderMap,
}

impl HeaderOverrides {
    /// Creates header overrides from an already validated HTTP header map.
    #[must_use]
    pub fn new(headers: HeaderMap) -> Self {
        Self { headers }
    }

    /// Returns the configured header map.
    #[must_use]
    pub fn as_map(&self) -> &HeaderMap {
        &self.headers
    }
}

impl fmt::Debug for HeaderOverrides {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HeaderOverrides")
            .field(
                "names",
                &self
                    .headers
                    .keys()
                    .map(http::HeaderName::as_str)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Caller-managed source of dynamic headers.
pub trait HeaderProvider: Send + Sync {
    /// Resolves headers for one provider request without environment lookup in core.
    fn headers(&self) -> Result<HeaderOverrides, ModelError>;
}

/// Shared dynamic header provider.
pub type DynHeaderProvider = Arc<dyn HeaderProvider>;

/// Static and optional caller-resolved dynamic headers.
#[derive(Clone, Default)]
pub struct HeaderConfig {
    /// Static header overrides.
    pub static_headers: HeaderOverrides,
    /// Dynamic header source.
    pub dynamic_headers: Option<DynHeaderProvider>,
}

impl HeaderConfig {
    /// Creates an empty header configuration.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }
}

impl fmt::Debug for HeaderConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HeaderConfig")
            .field("static_headers", &self.static_headers)
            .field(
                "dynamic_headers",
                &self.dynamic_headers.as_ref().map(|_| "<provider>"),
            )
            .finish()
    }
}

/// Provider-level identity, endpoint, authentication, and headers.
#[derive(Clone)]
pub struct ProviderConfig<A> {
    /// Serving provider identity.
    pub id: ProviderId,
    /// Explicit API endpoint.
    pub api: ApiEndpoint,
    /// Provider-specific resolved authentication.
    pub auth: A,
    /// Caller header configuration.
    pub headers: HeaderConfig,
}

impl<A> fmt::Debug for ProviderConfig<A> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderConfig")
            .field("id", &self.id)
            .field("api", &self.api)
            .field("auth", &"<redacted>")
            .field("headers", &self.headers)
            .finish()
    }
}

impl<A> ProviderConfig<A> {
    /// Creates provider configuration after validating its identity.
    pub fn new(
        id: ProviderId,
        api: ApiEndpoint,
        auth: A,
        headers: HeaderConfig,
    ) -> Result<Self, ModelError> {
        id.validate()?;
        Ok(Self {
            id,
            api,
            auth,
            headers,
        })
    }
}

/// Normalized token accounting. Inclusive totals are never computed by adding
/// component fields.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Usage {
    /// Total input tokens.
    pub input_tokens: Option<u64>,
    /// Input tokens excluding cache tokens.
    pub input_tokens_no_cache: Option<u64>,
    /// Input tokens read from a cache.
    pub input_tokens_cache_read: Option<u64>,
    /// Input tokens written to a cache.
    pub input_tokens_cache_write: Option<u64>,
    /// Total output tokens.
    pub output_tokens: Option<u64>,
    /// Text output tokens.
    pub output_tokens_text: Option<u64>,
    /// Reasoning output tokens.
    pub output_tokens_reasoning: Option<u64>,
    /// Provider usage with no normalized representation.
    pub raw: Option<JsonValue>,
}

/// Aggregated normalized usage across completed calls.
///
/// Each field is summed independently across calls. Inclusive totals are never
/// derived by adding component fields within a single call; only like-for-like
/// values are summed across calls.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct UsageTotals {
    usage: Usage,
    calls: u64,
}

impl UsageTotals {
    /// Records one completed call's usage, ignoring provider-specific raw usage.
    pub fn record(&mut self, usage: &Usage) {
        add_usage_total(&mut self.usage.input_tokens, usage.input_tokens);
        add_usage_total(
            &mut self.usage.input_tokens_no_cache,
            usage.input_tokens_no_cache,
        );
        add_usage_total(
            &mut self.usage.input_tokens_cache_read,
            usage.input_tokens_cache_read,
        );
        add_usage_total(
            &mut self.usage.input_tokens_cache_write,
            usage.input_tokens_cache_write,
        );
        add_usage_total(&mut self.usage.output_tokens, usage.output_tokens);
        add_usage_total(&mut self.usage.output_tokens_text, usage.output_tokens_text);
        add_usage_total(
            &mut self.usage.output_tokens_reasoning,
            usage.output_tokens_reasoning,
        );
        self.calls = self.calls.saturating_add(1);
    }

    /// Returns aggregated normalized usage with `raw` always omitted.
    #[must_use]
    pub fn usage(&self) -> Usage {
        Usage {
            raw: None,
            ..self.usage.clone()
        }
    }

    /// Returns the number of completed calls recorded.
    #[must_use]
    pub const fn calls(&self) -> u64 {
        self.calls
    }
}

fn add_usage_total(total: &mut Option<u64>, value: Option<u64>) {
    if let Some(value) = value {
        *total = Some(total.unwrap_or_default().saturating_add(value));
    }
}

/// A normalized semantic completion reason.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model stopped normally.
    Stop,
    /// The model requested one or more tool calls.
    ToolCalls,
    /// An output limit was reached.
    Length,
    /// A content filter stopped output.
    ContentFilter,
    /// The operation was cancelled.
    Cancelled,
    /// The provider reported an in-band error.
    Error,
    /// The provider aborted generation.
    Aborted,
    /// A timeout ended generation.
    Timeout,
    /// The provider refused output.
    Refused,
    /// The provider did not provide a recognized reason.
    Unknown,
    /// An unnormalized provider reason.
    Other(String),
}

impl FinishReason {
    /// Creates a normal-stop reason.
    #[must_use]
    pub const fn stop() -> Self {
        Self::Stop
    }

    /// Creates a tool-calls reason.
    #[must_use]
    pub const fn tool_calls() -> Self {
        Self::ToolCalls
    }

    /// Creates a length-limit reason.
    #[must_use]
    pub const fn length() -> Self {
        Self::Length
    }

    /// Creates a content-filter reason.
    #[must_use]
    pub const fn content_filter() -> Self {
        Self::ContentFilter
    }

    /// Creates a cancellation reason.
    #[must_use]
    pub const fn cancelled() -> Self {
        Self::Cancelled
    }

    /// Creates an in-band-error reason.
    #[must_use]
    pub const fn error() -> Self {
        Self::Error
    }

    /// Creates an abort reason.
    #[must_use]
    pub const fn aborted() -> Self {
        Self::Aborted
    }

    /// Creates a timeout reason.
    #[must_use]
    pub const fn timeout() -> Self {
        Self::Timeout
    }

    /// Creates a refusal reason.
    #[must_use]
    pub const fn refused() -> Self {
        Self::Refused
    }

    /// Creates an unknown normalized reason.
    #[must_use]
    pub const fn unknown() -> Self {
        Self::Unknown
    }

    /// Preserves an unnormalized provider reason.
    #[must_use]
    pub fn other(value: impl Into<String>) -> Self {
        Self::Other(value.into())
    }
}

impl fmt::Display for FinishReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stop => formatter.write_str("stop"),
            Self::ToolCalls => formatter.write_str("tool_calls"),
            Self::Length => formatter.write_str("length"),
            Self::ContentFilter => formatter.write_str("content_filter"),
            Self::Cancelled => formatter.write_str("cancelled"),
            Self::Error => formatter.write_str("error"),
            Self::Aborted => formatter.write_str("aborted"),
            Self::Timeout => formatter.write_str("timeout"),
            Self::Refused => formatter.write_str("refused"),
            Self::Unknown => formatter.write_str("unknown"),
            Self::Other(value) => formatter.write_str(value),
        }
    }
}

/// Mandatory terminal data proving semantic stream completion.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Finish {
    /// Authoritative final token accounting.
    pub usage: Usage,
    /// Semantic completion reason.
    pub finish_reason: FinishReason,
    /// Provider response metadata.
    pub response_metadata: ResponseMetadata,
    /// Provider-specific terminal metadata.
    pub provider_metadata: ProviderMetadata,
    /// Provider-native replay state, when captured.
    pub native_replay: Option<NativeReplayArtifact>,
}

impl Finish {
    /// Creates terminal data with empty metadata and no replay artifact.
    #[must_use]
    pub fn new(usage: Usage, finish_reason: FinishReason) -> Self {
        Self {
            usage,
            finish_reason,
            response_metadata: ResponseMetadata::new(),
            provider_metadata: ProviderMetadata::new(),
            native_replay: None,
        }
    }
}

/// A typed text part.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TextPart {
    /// Text content.
    pub text: String,
    /// Provider-specific part metadata.
    pub metadata: PartMetadata,
}

impl TextPart {
    /// Creates text without provider metadata.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            metadata: None,
        }
    }
}

/// A typed visible reasoning or reasoning-summary part.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReasoningPart {
    /// Reasoning text.
    pub text: String,
    /// Provider-specific part metadata.
    pub metadata: PartMetadata,
}

impl ReasoningPart {
    /// Creates reasoning without provider metadata.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            metadata: None,
        }
    }
}

/// A generic MIME-typed file source.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum FileSource {
    /// Inline binary data.
    Bytes(Bytes),
    /// Inline textual data.
    Text(String),
    /// A caller-owned URL. The SDK never downloads it automatically.
    Url(Url),
    /// A provider-native file reference.
    ProviderReference {
        /// Provider namespace owning the identifier.
        provider: ProviderId,
        /// Provider-native identifier.
        id: String,
    },
}

/// A generic MIME-typed file for images, documents, audio, video, and future media.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FilePart {
    /// MIME type.
    pub media_type: String,
    /// Optional filename.
    pub filename: Option<String>,
    /// Actual file source.
    pub source: FileSource,
    /// Provider-specific part metadata.
    pub metadata: PartMetadata,
}

impl FilePart {
    /// Creates a generic file.
    #[must_use]
    pub fn new(media_type: impl Into<String>, source: FileSource) -> Self {
        Self {
            media_type: media_type.into(),
            filename: None,
            source,
            metadata: None,
        }
    }

    /// Creates an audio file; this remains an ordinary [`FilePart`].
    #[must_use]
    pub fn audio(media_type: impl Into<String>, source: FileSource) -> Self {
        Self::new(media_type, source)
    }

    /// Creates a video file; this remains an ordinary [`FilePart`].
    #[must_use]
    pub fn video(media_type: impl Into<String>, source: FileSource) -> Self {
        Self::new(media_type, source)
    }

    /// Creates an image file; this remains an ordinary [`FilePart`].
    #[must_use]
    pub fn image(media_type: impl Into<String>, source: FileSource) -> Self {
        Self::new(media_type, source)
    }

    /// Creates a document file; this remains an ordinary [`FilePart`].
    #[must_use]
    pub fn document(media_type: impl Into<String>, source: FileSource) -> Self {
        Self::new(media_type, source)
    }
}

/// A finalized model-requested tool call.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolCallPart {
    /// Stable semantic identifier used to pair results.
    pub id: String,
    /// Provider item identifier, distinct from `id`.
    pub provider_item_id: Option<String>,
    /// Requested tool name.
    pub name: String,
    /// Parsed JSON input.
    pub input: JsonValue,
    /// Exact assembled provider argument text when needed.
    pub raw_input: Option<String>,
    /// Provider-specific part metadata.
    pub metadata: PartMetadata,
}

impl ToolCallPart {
    /// Creates a finalized tool call without provider-only fields.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: JsonValue) -> Self {
        Self {
            id: id.into(),
            provider_item_id: None,
            name: name.into(),
            input,
            raw_input: None,
            metadata: None,
        }
    }
}

/// Model-visible tool output.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ToolContent {
    /// Plain text output.
    Text(String),
    /// JSON output.
    Json(JsonValue),
    /// Mixed text, files, and JSON values.
    Mixed(Vec<ContentValue>),
    /// A harness-owned denied execution.
    Denied {
        /// Optional reason visible to the model.
        reason: Option<String>,
    },
}

/// A value accepted inside mixed tool output.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ContentValue {
    /// Text.
    Text(String),
    /// Generic file.
    File(FilePart),
    /// JSON.
    Json(JsonValue),
}

/// A tool execution result.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolResultPart {
    /// The tool call answered by this result.
    pub tool_call_id: String,
    /// Model-visible output.
    pub content: ToolContent,
    /// Whether execution failed.
    pub is_error: bool,
    /// Provider-specific part metadata.
    pub metadata: PartMetadata,
}

impl ToolResultPart {
    /// Creates a successful result without provider metadata.
    #[must_use]
    pub fn new(tool_call_id: impl Into<String>, content: ToolContent) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content,
            is_error: false,
            metadata: None,
        }
    }
}

/// An output citation or provenance record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SourcePart {
    /// Optional provider source identifier.
    pub id: Option<String>,
    /// Optional cited URL.
    pub url: Option<Url>,
    /// Optional title.
    pub title: Option<String>,
    /// Optional MIME type.
    pub media_type: Option<String>,
    /// Optional cited excerpt.
    pub excerpt: Option<String>,
    /// Provider-specific part metadata.
    pub metadata: PartMetadata,
}

impl SourcePart {
    /// Creates an empty source record that callers can populate.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: None,
            url: None,
            title: None,
            media_type: None,
            excerpt: None,
            metadata: None,
        }
    }
}

impl Default for SourcePart {
    fn default() -> Self {
        Self::new()
    }
}

/// A harness-owned request for tool approval.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolApprovalPart {
    /// Tool call requiring approval.
    pub tool_call_id: String,
    /// Optional provider-provided detail.
    pub message: Option<String>,
    /// Provider-specific part metadata.
    pub metadata: PartMetadata,
}

impl ToolApprovalPart {
    /// Creates an approval request without provider metadata.
    #[must_use]
    pub fn new(tool_call_id: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            message: None,
            metadata: None,
        }
    }
}

/// An inspectable namespaced extension, never a replay substitute.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CustomPart {
    /// Namespaced extension kind.
    pub kind: String,
    /// Extension data.
    pub data: JsonValue,
    /// Provider-specific part metadata.
    pub metadata: PartMetadata,
}

impl CustomPart {
    /// Creates an extension without provider metadata.
    #[must_use]
    pub fn new(kind: impl Into<String>, data: JsonValue) -> Self {
        Self {
            kind: kind.into(),
            data,
            metadata: None,
        }
    }
}

/// Content allowed in a system history message.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum SystemPart {
    /// System text.
    Text(TextPart),
    /// A system-scoped extension.
    Custom(CustomPart),
}

/// A role-safe system history message.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SystemMessage {
    /// Ordered system content.
    pub content: Vec<SystemPart>,
    /// Provider options scoped to this message.
    pub provider_options: ProviderOptions,
}

impl SystemMessage {
    /// Creates a system message with no provider options.
    #[must_use]
    pub fn new(content: Vec<SystemPart>) -> Self {
        Self {
            content,
            provider_options: ProviderOptions::new(),
        }
    }
}

/// Content allowed in a user history message.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum InputPart {
    /// User text.
    Text(TextPart),
    /// User-provided MIME-typed media.
    File(FilePart),
    /// A user-scoped extension.
    Custom(CustomPart),
}

/// A role-safe user history message.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UserMessage {
    /// Ordered user content.
    pub content: Vec<InputPart>,
    /// Provider options scoped to this message.
    pub provider_options: ProviderOptions,
}

impl UserMessage {
    /// Creates a user message with no provider options.
    #[must_use]
    pub fn new(content: Vec<InputPart>) -> Self {
        Self {
            content,
            provider_options: ProviderOptions::new(),
        }
    }
}

/// Content allowed in a completed assistant message.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum AssistantPart {
    /// Assistant text.
    Text(TextPart),
    /// Visible assistant reasoning.
    Reasoning(ReasoningPart),
    /// A finalized assistant tool call.
    ToolCall(ToolCallPart),
    /// An assistant-visible tool result.
    ToolResult(ToolResultPart),
    /// An assistant output file.
    File(FilePart),
    /// An assistant source record.
    Source(SourcePart),
    /// A harness-owned tool approval request.
    ToolApproval(ToolApprovalPart),
    /// An assistant-scoped extension.
    Custom(CustomPart),
}

/// A role-safe assistant history message.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AssistantMessage {
    /// Ordered assistant content.
    pub content: Vec<AssistantPart>,
    /// Provider options scoped to this message.
    pub provider_options: ProviderOptions,
}

impl AssistantMessage {
    /// Creates an assistant message with no provider options.
    #[must_use]
    pub fn new(content: Vec<AssistantPart>) -> Self {
        Self {
            content,
            provider_options: ProviderOptions::new(),
        }
    }
}

/// A role-safe tool-result history message.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolMessage {
    /// Ordered results following an assistant turn.
    pub results: Vec<ToolResultPart>,
    /// Provider options scoped to this message.
    pub provider_options: ProviderOptions,
}

impl ToolMessage {
    /// Creates a tool-result message with no provider options.
    #[must_use]
    pub fn new(results: Vec<ToolResultPart>) -> Self {
        Self {
            results,
            provider_options: ProviderOptions::new(),
        }
    }
}

/// A completed assistant turn retaining its mandatory finish and native replay.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompletedTurn {
    /// Role-safe assistant content.
    pub message: AssistantMessage,
    /// Mandatory terminal state.
    pub finish: Finish,
    /// Non-fatal collection warnings, including hosted-tool results without a
    /// normalized in-stream tool call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl CompletedTurn {
    /// Creates a completed assistant turn.
    #[must_use]
    pub fn new(message: AssistantMessage, finish: Finish) -> Self {
        Self {
            message,
            finish,
            warnings: Vec::new(),
        }
    }

    /// Concatenates text content in order.
    #[must_use]
    pub fn text(&self) -> String {
        self.message
            .content
            .iter()
            .filter_map(|part| match part {
                AssistantPart::Text(part) => Some(part.text.as_str()),
                _ => None,
            })
            .collect()
    }
}

/// One role-safe history turn supplied to a model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum HistoryTurn {
    /// A system message.
    System(SystemMessage),
    /// A user message.
    User(UserMessage),
    /// An assistant turn retaining native replay state.
    Assistant(Box<CompletedTurn>),
    /// A tool-result message.
    Tool(ToolMessage),
}

impl HistoryTurn {
    /// Wraps a role-safe system message.
    #[must_use]
    pub fn system(message: SystemMessage) -> Self {
        Self::System(message)
    }

    /// Wraps a role-safe user message.
    #[must_use]
    pub fn user(message: UserMessage) -> Self {
        Self::User(message)
    }

    /// Wraps a completed assistant turn.
    #[must_use]
    pub fn assistant(turn: CompletedTurn) -> Self {
        Self::Assistant(Box::new(turn))
    }

    /// Wraps a role-safe tool-result message.
    #[must_use]
    pub fn tool(message: ToolMessage) -> Self {
        Self::Tool(message)
    }
}

/// Options controlling a streamed model response.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub struct StreamOptions {
    /// Requests safe, opt-in raw provider events.
    pub include_raw: bool,
}

impl StreamOptions {
    /// Creates default stream options.
    #[must_use]
    pub const fn new() -> Self {
        Self { include_raw: false }
    }
}

/// A tool declaration supplied to a model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[non_exhaustive]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Human-readable tool description.
    pub description: String,
    /// Parsed JSON Schema for tool input.
    pub input_schema: JsonSchema,
    /// Provider-specific tool options.
    pub provider_options: ProviderOptions,
}

impl ToolDefinition {
    /// Creates a tool definition with no provider options.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: JsonSchema,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            provider_options: ProviderOptions::new(),
        }
    }
}

/// Caller preference for model tool selection.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide.
    #[default]
    Auto,
    /// Require at least one tool call.
    Required,
    /// Do not allow tool calls.
    None,
    /// Require one named tool.
    Tool(String),
}

/// The requested response representation.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Ordinary text/content output.
    #[default]
    Text,
    /// JSON output with an optional parsed JSON Schema.
    Json {
        /// Optional schema constraining JSON output.
        schema: Option<JsonSchema>,
    },
}

impl ResponseFormat {
    /// Requests schema-constrained JSON output.
    #[must_use]
    pub fn structured(schema: JsonSchema) -> Self {
        Self::Json {
            schema: Some(schema),
        }
    }
}

/// Normalized inference controls.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[non_exhaustive]
pub struct InferenceOptions {
    /// Maximum requested output tokens.
    pub max_output_tokens: Option<u64>,
    /// Requested sampling temperature.
    pub temperature: Option<f64>,
    /// Requested top-p sampling value.
    pub top_p: Option<f64>,
    /// Provider-neutral reasoning effort label.
    pub reasoning_effort: Option<String>,
}

impl InferenceOptions {
    /// Creates empty inference controls.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            reasoning_effort: None,
        }
    }
}

/// One normalized request for a configured model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[non_exhaustive]
pub struct Request {
    /// Ordered history, including completed assistant turns and their artifacts.
    pub history: Vec<HistoryTurn>,
    /// Provider-native compacted context prepended to this request, when present.
    pub native_context: Option<NativeContextWindow>,
    /// Declared callable tools.
    pub tools: Vec<ToolDefinition>,
    /// Tool selection preference.
    pub tool_choice: ToolChoice,
    /// Requested response representation.
    pub response_format: ResponseFormat,
    /// Inference controls.
    pub inference: InferenceOptions,
    /// Stream behavior.
    pub stream_options: StreamOptions,
    /// Provider options scoped to the request.
    pub provider_options: ProviderOptions,
}

impl Request {
    /// Creates a request with normalized defaults.
    #[must_use]
    pub fn new(history: Vec<HistoryTurn>) -> Self {
        Self {
            history,
            native_context: None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            response_format: ResponseFormat::Text,
            inference: InferenceOptions::new(),
            stream_options: StreamOptions::new(),
            provider_options: ProviderOptions::new(),
        }
    }

    /// Replaces declared tools.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Attaches a provider-native compacted context window.
    #[must_use]
    pub fn with_native_context(mut self, native_context: NativeContextWindow) -> Self {
        self.native_context = Some(native_context);
        self
    }

    /// Replaces the tool-selection preference.
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Replaces the response format.
    #[must_use]
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = response_format;
        self
    }

    /// Replaces inference controls.
    #[must_use]
    pub fn with_inference(mut self, inference: InferenceOptions) -> Self {
        self.inference = inference;
        self
    }

    /// Validates structural request invariants and declared model capabilities.
    pub fn validate_for(&self, capabilities: &ModelCapabilities) -> Result<(), ModelError> {
        capabilities.validate()?;
        if self.native_context.is_some()
            && capabilities.compaction == CompactionCapability::Unsupported
        {
            return Err(ModelError::unsupported(
                "provider-native context is not supported by this model",
            ));
        }
        let mut names = BTreeSet::new();
        for tool in &self.tools {
            if tool.name.trim().is_empty() {
                return Err(ModelError::invalid_request("tool names must not be empty"));
            }
            if !names.insert(tool.name.as_str()) {
                return Err(ModelError::invalid_request("tool names must be unique"));
            }
        }
        self.validate_history_tool_pairing()?;
        match &self.tool_choice {
            ToolChoice::Required if self.tools.is_empty() => {
                return Err(ModelError::invalid_request(
                    "required tool choice needs at least one tool",
                ));
            }
            ToolChoice::Tool(name) if !names.contains(name.as_str()) => {
                return Err(ModelError::invalid_request(
                    "named tool choice references an undeclared tool",
                ));
            }
            _ => {}
        }
        if (!self.tools.is_empty()
            || matches!(self.tool_choice, ToolChoice::Required | ToolChoice::Tool(_)))
            && !capabilities.features.contains(Capability::TOOL_CALLING)
        {
            return Err(ModelError::unsupported(
                "tool calling is not supported by this model",
            ));
        }
        if matches!(self.response_format, ResponseFormat::Json { .. })
            && !capabilities
                .features
                .contains(Capability::STRUCTURED_OUTPUT)
        {
            return Err(ModelError::unsupported(
                "structured output is not supported by this model",
            ));
        }
        if let Some(temperature) = self.inference.temperature
            && (!temperature.is_finite() || !(0.0..=2.0).contains(&temperature))
        {
            return Err(ModelError::invalid_request(
                "temperature must be finite and between 0 and 2",
            ));
        }
        if self.inference.temperature.is_some()
            && !capabilities.features.contains(Capability::TEMPERATURE)
        {
            return Err(ModelError::unsupported(
                "temperature is not supported by this model",
            ));
        }
        if let Some(top_p) = self.inference.top_p
            && (!top_p.is_finite() || !(0.0..=1.0).contains(&top_p))
        {
            return Err(ModelError::invalid_request(
                "top_p must be finite and between 0 and 1",
            ));
        }
        if self.inference.top_p.is_some() && !capabilities.features.contains(Capability::TOP_P) {
            return Err(ModelError::unsupported(
                "top_p is not supported by this model",
            ));
        }
        if let Some(max_output_tokens) = self.inference.max_output_tokens {
            if max_output_tokens == 0 {
                return Err(ModelError::invalid_request(
                    "max_output_tokens must be greater than zero",
                ));
            }
            if !capabilities
                .features
                .contains(Capability::MAX_OUTPUT_TOKENS)
            {
                return Err(ModelError::unsupported(
                    "max_output_tokens is not supported by this model",
                ));
            }
            if capabilities
                .limits
                .output
                .is_some_and(|limit| max_output_tokens > limit)
            {
                return Err(ModelError::invalid_request(
                    "max_output_tokens exceeds the declared model output limit",
                ));
            }
        }
        if self.inference.reasoning_effort.is_some()
            && !capabilities.features.contains(Capability::REASONING)
        {
            return Err(ModelError::unsupported(
                "reasoning is not supported by this model",
            ));
        }
        if !capabilities.modalities.output.contains(&Modality::text()) {
            return Err(ModelError::unsupported(
                "the normalized request contract requires declared text output",
            ));
        }
        self.validate_history_features_and_media(capabilities)?;
        Ok(())
    }

    fn validate_history_features_and_media(
        &self,
        capabilities: &ModelCapabilities,
    ) -> Result<(), ModelError> {
        for turn in &self.history {
            match turn {
                HistoryTurn::System(message) => {
                    if message
                        .content
                        .iter()
                        .any(|part| matches!(part, SystemPart::Text(_)))
                    {
                        require_input_modality(capabilities, &Modality::text())?;
                    }
                }
                HistoryTurn::User(message) => {
                    for part in &message.content {
                        match part {
                            InputPart::Text(_) => {
                                require_input_modality(capabilities, &Modality::text())?
                            }
                            InputPart::File(file) => validate_file_for(file, capabilities)?,
                            InputPart::Custom(_) => {}
                        }
                    }
                }
                HistoryTurn::Assistant(turn) => {
                    for part in &turn.message.content {
                        match part {
                            AssistantPart::Text(_) => {
                                require_input_modality(capabilities, &Modality::text())?
                            }
                            AssistantPart::Reasoning(_) => {
                                if !capabilities.features.contains(Capability::REASONING) {
                                    return Err(ModelError::unsupported(
                                        "assistant reasoning history requires reasoning support",
                                    ));
                                }
                            }
                            AssistantPart::ToolCall(_)
                            | AssistantPart::ToolResult(_)
                            | AssistantPart::ToolApproval(_) => {
                                if !capabilities.features.contains(Capability::TOOL_CALLING) {
                                    return Err(ModelError::unsupported(
                                        "assistant tool history requires tool-calling support",
                                    ));
                                }
                            }
                            AssistantPart::File(file) => validate_file_for(file, capabilities)?,
                            AssistantPart::Source(_) | AssistantPart::Custom(_) => {}
                        }
                        if let AssistantPart::ToolResult(result) = part {
                            validate_tool_content_media(&result.content, capabilities)?;
                        }
                    }
                }
                HistoryTurn::Tool(message) => {
                    if !capabilities.features.contains(Capability::TOOL_CALLING) {
                        return Err(ModelError::unsupported(
                            "tool-result history requires tool-calling support",
                        ));
                    }
                    for result in &message.results {
                        validate_tool_content_media(&result.content, capabilities)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_history_tool_pairing(&self) -> Result<(), ModelError> {
        for (index, turn) in self.history.iter().enumerate() {
            let HistoryTurn::Assistant(assistant) = turn else {
                continue;
            };
            let mut calls = BTreeSet::new();
            for part in &assistant.message.content {
                if let AssistantPart::ToolCall(call) = part
                    && !calls.insert(call.id.as_str())
                {
                    return Err(ModelError::invalid_request(
                        "assistant history contains duplicate tool call IDs",
                    ));
                }
            }
            let mut result_ids = BTreeSet::new();
            for part in &assistant.message.content {
                let AssistantPart::ToolResult(result) = part else {
                    continue;
                };
                if !result_ids.insert(result.tool_call_id.as_str()) {
                    return Err(ModelError::invalid_request(
                        "assistant history contains duplicate tool result IDs",
                    ));
                }
                if !calls.contains(result.tool_call_id.as_str()) {
                    return Err(ModelError::invalid_request(
                        "assistant tool result does not pair with its tool call",
                    ));
                }
            }
            let following_tool = match self.history.get(index + 1) {
                Some(HistoryTurn::Tool(tool)) => Some(tool),
                _ => None,
            };
            if let Some(tool) = following_tool {
                if calls.is_empty() {
                    return Err(ModelError::invalid_request(
                        "tool results require tool calls in the preceding assistant turn",
                    ));
                }
                if calls == result_ids {
                    return Err(ModelError::invalid_request(
                        "tool history follows an assistant turn with no unresolved tool calls",
                    ));
                }
                for result in &tool.results {
                    if !result_ids.insert(result.tool_call_id.as_str()) {
                        return Err(ModelError::invalid_request(
                            "tool result ID is duplicated across assistant and tool history",
                        ));
                    }
                    if !calls.contains(result.tool_call_id.as_str()) {
                        return Err(ModelError::invalid_request(
                            "tool result does not pair with the preceding assistant turn",
                        ));
                    }
                }
            }
            if calls != result_ids {
                return Err(ModelError::invalid_request(
                    "assistant tool calls are missing results",
                ));
            }
        }
        for (index, turn) in self.history.iter().enumerate() {
            if matches!(turn, HistoryTurn::Tool(_))
                && !matches!(
                    index
                        .checked_sub(1)
                        .and_then(|previous| self.history.get(previous)),
                    Some(HistoryTurn::Assistant(_))
                )
            {
                return Err(ModelError::invalid_request(
                    "tool results must immediately follow an assistant turn",
                ));
            }
        }
        Ok(())
    }
}

fn validate_tool_content_media(
    content: &ToolContent,
    capabilities: &ModelCapabilities,
) -> Result<(), ModelError> {
    if let ToolContent::Mixed(values) = content {
        for value in values {
            if let ContentValue::File(file) = value {
                validate_file_for(file, capabilities)?;
            }
        }
    }
    Ok(())
}

fn require_input_modality(
    capabilities: &ModelCapabilities,
    modality: &Modality,
) -> Result<(), ModelError> {
    if capabilities.modalities.input.contains(modality) {
        Ok(())
    } else {
        Err(ModelError::unsupported(format!(
            "input modality `{}` is not supported by this model",
            modality.as_str()
        )))
    }
}

fn validate_file_for(file: &FilePart, capabilities: &ModelCapabilities) -> Result<(), ModelError> {
    if file.media_type.trim().is_empty() || !file.media_type.contains('/') {
        return Err(ModelError::invalid_request(
            "file media_type must be a non-empty MIME type",
        ));
    }
    let supported = capabilities.media.input.iter().any(|(modality, support)| {
        capabilities.modalities.input.contains(modality) && support.supports(file)
    });
    if !supported {
        return Err(ModelError::unsupported(format!(
            "media type `{}` or its source form is not supported by any declared input modality",
            file.media_type
        )));
    }
    Ok(())
}

/// The phase in which a model error occurred.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorStage {
    /// The adapter did not identify a more specific stage.
    #[default]
    Unknown,
    /// Core or adapter request validation.
    RequestValidation,
    /// Provider request encoding.
    RequestEncoding,
    /// Connection establishment.
    Connect,
    /// Response header processing.
    ResponseHeaders,
    /// Non-success response body processing.
    ResponseBody,
    /// Stream transport reading.
    StreamRead,
    /// UTF-8, SSE, or JSON stream decoding.
    StreamDecode,
    /// Provider event shape or sequencing.
    StreamEvent,
    /// Normalized stream finalization.
    StreamFinalize,
    /// Native replay encoding.
    ReplayEncode,
    /// Native replay decoding.
    ReplayDecode,
    /// Provider-native context encoding.
    NativeContextEncode,
    /// Provider-native context decoding.
    NativeContextDecode,
    /// Middleware processing.
    Middleware,
}

/// The normalized category of a model error.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum ModelErrorKind {
    /// Transport failure.
    Transport,
    /// Timeout.
    Timeout,
    /// Rate limiting.
    RateLimited,
    /// Authentication failure.
    Auth,
    /// Authorization failure.
    PermissionDenied,
    /// Invalid request.
    InvalidRequest,
    /// Unknown or unavailable model.
    ModelNotFound,
    /// Context-length limit.
    ContextLength,
    /// Quota exhaustion.
    Quota,
    /// Provider overload.
    Overload,
    /// Unsupported feature.
    Unsupported,
    /// EOF before terminal finish.
    UnexpectedEof,
    /// Invalid normalized or provider response.
    InvalidResponse,
    /// Invalid finalized tool input.
    InvalidToolInput,
    /// Content filtering.
    ContentFilter,
    /// Native replay handling.
    Replay,
    /// Provider-native compacted context handling.
    NativeContext,
    /// Provider-defined failure.
    Provider,
    /// Local or provider abort.
    Abort,
    /// Unknown failure.
    Unknown,
}

impl ModelErrorKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Transport => "transport",
            Self::Timeout => "timeout",
            Self::RateLimited => "rate_limited",
            Self::Auth => "auth",
            Self::PermissionDenied => "permission_denied",
            Self::InvalidRequest => "invalid_request",
            Self::ModelNotFound => "model_not_found",
            Self::ContextLength => "context_length",
            Self::Quota => "quota",
            Self::Overload => "overload",
            Self::Unsupported => "unsupported",
            Self::UnexpectedEof => "unexpected_eof",
            Self::InvalidResponse => "invalid_response",
            Self::InvalidToolInput => "invalid_tool_input",
            Self::ContentFilter => "content_filter",
            Self::Replay => "replay",
            Self::NativeContext => "native_context",
            Self::Provider => "provider",
            Self::Abort => "abort",
            Self::Unknown => "unknown",
        }
    }
}

/// A bounded, sanitized provider response body.
#[derive(Clone, Eq, PartialEq, Serialize)]
pub struct SanitizedBody {
    text: String,
    truncated: bool,
}

impl SanitizedBody {
    /// Maximum retained body size in bytes.
    pub const MAX_BYTES: usize = 64 * 1024;

    /// Creates a body, truncating at a valid UTF-8 boundary when necessary.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        let mut text = text.into();
        let truncated = text.len() > Self::MAX_BYTES;
        if truncated {
            let mut end = Self::MAX_BYTES;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
        }
        Self { text, truncated }
    }

    /// Returns the retained sanitized text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns whether truncation occurred.
    #[must_use]
    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    /// Returns the retained length in bytes.
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        self.text.len()
    }
}

impl fmt::Debug for SanitizedBody {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SanitizedBody")
            .field("len_bytes", &self.len_bytes())
            .field("truncated", &self.truncated)
            .finish()
    }
}

impl<'de> Deserialize<'de> for SanitizedBody {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireBody {
            text: String,
            truncated: bool,
        }
        let wire = WireBody::deserialize(deserializer)?;
        let mut body = Self::new(wire.text);
        body.truncated |= wire.truncated;
        Ok(body)
    }
}

mod option_duration_millis {
    use super::*;

    pub fn serialize<S>(value: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value
            .map(|duration| duration.as_millis())
            .map(u64::try_from)
            .transpose()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<u64>::deserialize(deserializer).map(|value| value.map(Duration::from_millis))
    }
}

/// Typed diagnostics accompanying one model error.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ErrorDiagnostics {
    /// HTTP status when the transport uses HTTP.
    pub http_status: Option<u16>,
    /// Failure stage.
    pub stage: ErrorStage,
    /// Bytes received before failure.
    pub bytes_received: u64,
    /// Provider-specific error code.
    pub vendor_code: Option<String>,
    /// Provider request identifier.
    pub request_id: Option<String>,
    /// Provider-requested retry delay.
    #[serde(default, with = "option_duration_millis")]
    pub retry_after: Option<Duration>,
    /// Bounded sanitized response body.
    pub sanitized_body: Option<SanitizedBody>,
}

/// A structured model error. Harnesses own retry and fallback policy.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelError {
    /// Normalized error category.
    pub kind: ModelErrorKind,
    /// Sanitized message.
    pub message: String,
    /// Factual retryability hint.
    pub retryable: bool,
    /// Typed safe diagnostics.
    pub diagnostics: ErrorDiagnostics,
}

impl ModelError {
    /// Creates an error with default diagnostics and no retryability hint.
    #[must_use]
    pub fn new(kind: ModelErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            retryable: false,
            diagnostics: ErrorDiagnostics::default(),
        }
    }

    /// Creates an invalid-request error.
    #[must_use]
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::InvalidRequest, message).with_stage(ErrorStage::RequestValidation)
    }
    /// Creates an unsupported error.
    #[must_use]
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::Unsupported, message).with_stage(ErrorStage::RequestValidation)
    }
    /// Creates an invalid-response error.
    #[must_use]
    pub fn invalid_response(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::InvalidResponse, message).with_stage(ErrorStage::StreamFinalize)
    }
    /// Creates an unexpected-EOF error.
    #[must_use]
    pub fn unexpected_eof(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::UnexpectedEof, message).with_stage(ErrorStage::StreamFinalize)
    }
    /// Creates a retryable transport error.
    #[must_use]
    pub fn transport(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::Transport, message)
            .with_retryable(true)
            .with_stage(ErrorStage::Connect)
    }
    /// Creates a retryable timeout error.
    #[must_use]
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::Timeout, message).with_retryable(true)
    }
    /// Creates a retryable rate-limited error.
    #[must_use]
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::RateLimited, message)
            .with_retryable(true)
            .with_stage(ErrorStage::ResponseHeaders)
    }
    /// Creates a provider error.
    #[must_use]
    pub fn provider(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::Provider, message).with_stage(ErrorStage::ResponseHeaders)
    }
    /// Creates an abort error.
    #[must_use]
    pub fn abort(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::Abort, message)
    }
    /// Creates a replay error.
    #[must_use]
    pub fn replay(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::Replay, message).with_stage(ErrorStage::ReplayDecode)
    }
    /// Creates a provider-native context error.
    #[must_use]
    pub fn native_context(message: impl Into<String>) -> Self {
        Self::new(ModelErrorKind::NativeContext, message)
            .with_stage(ErrorStage::NativeContextDecode)
    }
    /// Returns the normalized error kind.
    #[must_use]
    pub const fn kind(&self) -> ModelErrorKind {
        self.kind
    }
    /// Returns whether this error has the supplied kind.
    #[must_use]
    pub const fn is_kind(&self, kind: ModelErrorKind) -> bool {
        self.kind as u8 == kind as u8
    }
    /// Replaces the retryability hint.
    #[must_use]
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }
    /// Replaces the error stage.
    #[must_use]
    pub fn with_stage(mut self, stage: ErrorStage) -> Self {
        self.diagnostics.stage = stage;
        self
    }
    /// Sets the HTTP status.
    #[must_use]
    pub fn with_http_status(mut self, status: u16) -> Self {
        self.diagnostics.http_status = Some(status);
        self
    }
    /// Sets received byte count.
    #[must_use]
    pub fn with_bytes_received(mut self, bytes: u64) -> Self {
        self.diagnostics.bytes_received = bytes;
        self
    }
    /// Sets provider error code.
    #[must_use]
    pub fn with_vendor_code(mut self, code: impl Into<String>) -> Self {
        self.diagnostics.vendor_code = Some(code.into());
        self
    }
    /// Sets provider request ID.
    #[must_use]
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.diagnostics.request_id = Some(id.into());
        self
    }
    /// Sets retry-after duration.
    #[must_use]
    pub fn with_retry_after(mut self, duration: Duration) -> Self {
        self.diagnostics.retry_after = Some(duration);
        self
    }
    /// Sets the bounded sanitized body.
    #[must_use]
    pub fn with_sanitized_body(mut self, body: SanitizedBody) -> Self {
        self.diagnostics.sanitized_body = Some(body);
        self
    }
    /// Replaces all diagnostics.
    #[must_use]
    pub fn with_diagnostics(mut self, diagnostics: ErrorDiagnostics) -> Self {
        self.diagnostics = diagnostics;
        self
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} error: {}", self.kind.as_str(), self.message)
    }
}

impl std::error::Error for ModelError {}

bitflags! {
    /// Explicit normalized features supported by one configured model.
    ///
    /// Serde serialization and deserialization use a sequence of snake_case
    /// flag names in every format, including non-human-readable formats. Plain
    /// strings are rejected.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct Capability: u128 {
        /// Tool calls.
        const TOOL_CALLING = 1 << 0;
        /// Parallel tool calls.
        const PARALLEL_TOOLS = 1 << 1;
        /// Tool-input deltas.
        const TOOL_INPUT_DELTAS = 1 << 2;
        /// Visible reasoning.
        const REASONING = 1 << 3;
        /// Structured output.
        const STRUCTURED_OUTPUT = 1 << 4;
        /// Temperature sampling control.
        const TEMPERATURE = 1 << 5;
        /// Top-p sampling control.
        const TOP_P = 1 << 6;
        /// Explicit maximum-output-token control.
        const MAX_OUTPUT_TOKENS = 1 << 7;
        /// Prompt caching.
        const PROMPT_CACHING = 1 << 8;
        /// Usage reporting.
        const USAGE = 1 << 9;
        /// Provider-executed tools.
        const PROVIDER_TOOLS = 1 << 10;
        /// Citation output.
        const SOURCES = 1 << 11;
    }
}

const CAPABILITY_NAMES: &[(Capability, &str)] = &[
    (Capability::TOOL_CALLING, "tool_calling"),
    (Capability::PARALLEL_TOOLS, "parallel_tools"),
    (Capability::TOOL_INPUT_DELTAS, "tool_input_deltas"),
    (Capability::REASONING, "reasoning"),
    (Capability::STRUCTURED_OUTPUT, "structured_output"),
    (Capability::TEMPERATURE, "temperature"),
    (Capability::TOP_P, "top_p"),
    (Capability::MAX_OUTPUT_TOKENS, "max_output_tokens"),
    (Capability::PROMPT_CACHING, "prompt_caching"),
    (Capability::USAGE, "usage"),
    (Capability::PROVIDER_TOOLS, "provider_tools"),
    (Capability::SOURCES, "sources"),
];

const CAPABILITY_VALID_NAMES: &str = "tool_calling, parallel_tools, tool_input_deltas, reasoning, structured_output, temperature, top_p, max_output_tokens, prompt_caching, usage, provider_tools, sources";

fn capability_from_name<E>(name: &str) -> Result<Capability, E>
where
    E: serde::de::Error,
{
    CAPABILITY_NAMES
        .iter()
        .find_map(|(capability, valid_name)| (valid_name == &name).then_some(*capability))
        .ok_or_else(|| {
            E::custom(format!(
                "unknown capability `{name}`; valid capabilities: {CAPABILITY_VALID_NAMES}"
            ))
        })
}

/// Token limits declared for one provider-served model offering.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelLimits {
    /// Maximum combined context/window tokens, when known.
    pub context: Option<u64>,
    /// Maximum input tokens, when separately documented.
    pub input: Option<u64>,
    /// Maximum output tokens, when known.
    pub output: Option<u64>,
}

impl<'de> Deserialize<'de> for ModelLimits {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireLimits {
            context: Option<u64>,
            input: Option<u64>,
            output: Option<u64>,
        }
        let wire = WireLimits::deserialize(deserializer)?;
        let limits = Self::new(wire.context, wire.input, wire.output);
        limits.validate().map_err(serde::de::Error::custom)?;
        Ok(limits)
    }
}

impl ModelLimits {
    /// Creates explicit limits, including intentionally unknown values.
    #[must_use]
    pub const fn new(context: Option<u64>, input: Option<u64>, output: Option<u64>) -> Self {
        Self {
            context,
            input,
            output,
        }
    }

    /// Validates relationships that are consistently meaningful across providers.
    pub fn validate(&self) -> Result<(), ModelError> {
        if let Some(context) = self.context.filter(|value| *value > 0) {
            if self.input.is_some_and(|input| input > context) {
                return Err(ModelError::invalid_request(
                    "model input limit must not exceed its context limit",
                ));
            }
            if self.output.is_some_and(|output| output > context) {
                return Err(ModelError::invalid_request(
                    "model output limit must not exceed its context limit",
                ));
            }
        }
        Ok(())
    }
}

/// An open modality label. Standard constructors cover current models.dev values.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Modality(String);

impl Modality {
    /// Creates a non-empty open modality label.
    pub fn new(value: impl Into<String>) -> Result<Self, ModelError> {
        let value = value.into();
        validate_identifier("modality", &value)?;
        Ok(Self(value))
    }

    /// Text modality.
    #[must_use]
    pub fn text() -> Self {
        Self("text".into())
    }

    /// Image modality.
    #[must_use]
    pub fn image() -> Self {
        Self("image".into())
    }

    /// Audio modality.
    #[must_use]
    pub fn audio() -> Self {
        Self("audio".into())
    }

    /// Video modality.
    #[must_use]
    pub fn video() -> Self {
        Self("video".into())
    }

    /// PDF modality.
    #[must_use]
    pub fn pdf() -> Self {
        Self("pdf".into())
    }

    /// Returns the modality label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Modality {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Explicit input and output modalities.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Modalities {
    /// Accepted input modalities.
    pub input: BTreeSet<Modality>,
    /// Produced output modalities.
    pub output: BTreeSet<Modality>,
}

impl Modalities {
    /// Creates modality sets without inferring support from model names.
    #[must_use]
    pub fn new(
        input: impl IntoIterator<Item = Modality>,
        output: impl IntoIterator<Item = Modality>,
    ) -> Self {
        Self {
            input: input.into_iter().collect(),
            output: output.into_iter().collect(),
        }
    }
}

bitflags! {
    /// Source forms accepted for media input.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct MediaSourceSupport: u8 {
        /// Inline binary bytes.
        const INLINE_BYTES = 1 << 0;
        /// Inline text.
        const INLINE_TEXT = 1 << 1;
        /// Caller-owned URL.
        const URL = 1 << 2;
        /// Provider-native reference.
        const PROVIDER_REFERENCE = 1 << 3;
    }
}

const MEDIA_SOURCE_NAMES: &[(MediaSourceSupport, &str)] = &[
    (MediaSourceSupport::INLINE_BYTES, "inline_bytes"),
    (MediaSourceSupport::INLINE_TEXT, "inline_text"),
    (MediaSourceSupport::URL, "url"),
    (MediaSourceSupport::PROVIDER_REFERENCE, "provider_reference"),
];

impl Serialize for MediaSourceSupport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(
            MEDIA_SOURCE_NAMES
                .iter()
                .filter(|(source, _)| self.contains(*source))
                .count(),
        ))?;
        for (source, name) in MEDIA_SOURCE_NAMES {
            if self.contains(*source) {
                sequence.serialize_element(name)?;
            }
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for MediaSourceSupport {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SourcesVisitor;

        impl<'de> Visitor<'de> for SourcesVisitor {
            type Value = MediaSourceSupport;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a sequence of media source names")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut sources = MediaSourceSupport::empty();
                while let Some(name) = sequence.next_element::<String>()? {
                    let source = MEDIA_SOURCE_NAMES
                        .iter()
                        .find_map(|(source, valid)| (name == *valid).then_some(*source))
                        .ok_or_else(|| {
                            serde::de::Error::custom(format!("unknown media source `{name}`"))
                        })?;
                    sources |= source;
                }
                Ok(sources)
            }
        }

        deserializer.deserialize_seq(SourcesVisitor)
    }
}

/// Exact media input support for one modality.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MediaInputSupport {
    /// Exact MIME strings or one trailing `*` prefix pattern such as `image/*`.
    pub media_types: Vec<String>,
    /// Accepted source forms.
    pub sources: MediaSourceSupport,
}

impl<'de> Deserialize<'de> for MediaInputSupport {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireSupport {
            media_types: Vec<String>,
            sources: MediaSourceSupport,
        }
        let wire = WireSupport::deserialize(deserializer)?;
        Self::new(wire.media_types, wire.sources).map_err(serde::de::Error::custom)
    }
}

impl MediaInputSupport {
    /// Creates and validates media support.
    pub fn new(
        media_types: impl IntoIterator<Item = String>,
        sources: MediaSourceSupport,
    ) -> Result<Self, ModelError> {
        let support = Self {
            media_types: media_types.into_iter().collect(),
            sources,
        };
        support.validate()?;
        Ok(support)
    }

    /// Validates MIME patterns and requires at least one source form.
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.media_types.is_empty() {
            return Err(ModelError::invalid_request(
                "media support must declare at least one MIME type",
            ));
        }
        if self.sources.is_empty() {
            return Err(ModelError::invalid_request(
                "media support must declare at least one source form",
            ));
        }
        for media_type in &self.media_types {
            if media_type.trim().is_empty()
                || !media_type.contains('/')
                || media_type[..media_type.len().saturating_sub(1)].contains('*')
            {
                return Err(ModelError::invalid_request(
                    "media MIME declarations must be exact values or trailing-* prefixes",
                ));
            }
        }
        Ok(())
    }

    fn supports(&self, file: &FilePart) -> bool {
        let source = match file.source {
            FileSource::Bytes(_) => MediaSourceSupport::INLINE_BYTES,
            FileSource::Text(_) => MediaSourceSupport::INLINE_TEXT,
            FileSource::Url(_) => MediaSourceSupport::URL,
            FileSource::ProviderReference { .. } => MediaSourceSupport::PROVIDER_REFERENCE,
        };
        self.sources.contains(source)
            && self.media_types.iter().any(|declared| {
                declared.strip_suffix('*').map_or_else(
                    || declared.eq_ignore_ascii_case(&file.media_type),
                    |prefix| {
                        file.media_type
                            .get(..prefix.len())
                            .is_some_and(|value| value.eq_ignore_ascii_case(prefix))
                    },
                )
            })
    }
}

/// Exact per-modality media input support.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MediaCapabilities {
    /// Input support keyed by open modality labels.
    pub input: BTreeMap<Modality, MediaInputSupport>,
}

impl MediaCapabilities {
    /// Validates every declared media rule.
    pub fn validate(&self) -> Result<(), ModelError> {
        for support in self.input.values() {
            support.validate()?;
        }
        Ok(())
    }
}

impl Serialize for Capability {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let count = CAPABILITY_NAMES
            .iter()
            .filter(|(capability, _)| self.contains(*capability))
            .count();
        let mut sequence = serializer.serialize_seq(Some(count))?;
        for (capability, name) in CAPABILITY_NAMES {
            if self.contains(*capability) {
                sequence.serialize_element(name)?;
            }
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for Capability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CapabilityVisitor;

        impl<'de> Visitor<'de> for CapabilityVisitor {
            type Value = Capability;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a sequence of snake_case capability names")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut capabilities = Capability::empty();
                while let Some(name) = sequence.next_element::<String>()? {
                    capabilities |= capability_from_name(&name)?;
                }
                Ok(capabilities)
            }
        }

        deserializer.deserialize_seq(CapabilityVisitor)
    }
}

/// Cancellation semantics offered by a configured model.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationCapability {
    /// Local cancellation drops local work only.
    #[default]
    LocalOnly,
    /// The adapter attempts a provider cancellation operation.
    RemoteBestEffort,
    /// Cancellation is unsupported by the provider profile.
    Unsupported,
}

/// Provider-native context compaction support offered by a configured model.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionCapability {
    /// The adapter exposes provider-native compaction and accepts native context windows.
    Native,
    /// Provider-native compaction and native context windows are unsupported.
    #[default]
    Unsupported,
}

/// Native replay support offered by a configured model.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCapability {
    /// Native replay is required for correct history encoding.
    Required,
    /// Native replay may be used when a compatible artifact exists.
    #[default]
    Optional,
    /// No native replay is supported.
    Unsupported,
}

/// Explicit native replay declaration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReplayDeclaration {
    /// Caller replay policy.
    pub policy: ReplayPolicy,
    /// Native replay support.
    pub capability: ReplayCapability,
    /// Whether replay carries provider-authoritative reasoning state.
    pub reasoning: bool,
}

/// Complete explicit capabilities of one configured model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelCapabilities {
    /// Explicit feature bitset.
    pub features: Capability,
    /// Token limits.
    pub limits: ModelLimits,
    /// Input and output modalities.
    pub modalities: Modalities,
    /// Exact media input rules.
    pub media: MediaCapabilities,
    /// Cancellation semantics.
    pub cancellation: CancellationCapability,
    /// Provider-native context compaction support.
    pub compaction: CompactionCapability,
    /// Replay semantics.
    pub replay: ReplayDeclaration,
}

impl<'de> Deserialize<'de> for ModelCapabilities {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireCapabilities {
            features: Capability,
            limits: ModelLimits,
            modalities: Modalities,
            media: MediaCapabilities,
            cancellation: CancellationCapability,
            compaction: CompactionCapability,
            replay: ReplayDeclaration,
        }
        let wire = WireCapabilities::deserialize(deserializer)?;
        let capabilities = Self {
            features: wire.features,
            limits: wire.limits,
            modalities: wire.modalities,
            media: wire.media,
            cancellation: wire.cancellation,
            compaction: wire.compaction,
            replay: wire.replay,
        };
        capabilities.validate().map_err(serde::de::Error::custom)?;
        Ok(capabilities)
    }
}

impl ModelCapabilities {
    /// Creates a conservative explicit text-only declaration.
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            features: Capability::empty(),
            limits: ModelLimits::new(None, None, None),
            modalities: Modalities::new([Modality::text()], [Modality::text()]),
            media: MediaCapabilities::default(),
            cancellation: CancellationCapability::Unsupported,
            compaction: CompactionCapability::Unsupported,
            replay: ReplayDeclaration {
                policy: ReplayPolicy::Never,
                capability: ReplayCapability::Unsupported,
                reasoning: false,
            },
        }
    }

    /// Validates capability dependencies and declaration consistency.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.limits.validate()?;
        self.media.validate()?;
        if self.modalities.input.is_empty() || self.modalities.output.is_empty() {
            return Err(ModelError::invalid_request(
                "model declarations require at least one input and output modality",
            ));
        }
        if self.features.contains(Capability::PARALLEL_TOOLS)
            && !self.features.contains(Capability::TOOL_CALLING)
        {
            return Err(ModelError::invalid_request(
                "parallel tools require tool calling",
            ));
        }
        if self.features.contains(Capability::TOOL_INPUT_DELTAS)
            && !self.features.contains(Capability::TOOL_CALLING)
        {
            return Err(ModelError::invalid_request(
                "tool input deltas require tool calling",
            ));
        }
        if self.replay.reasoning && !self.features.contains(Capability::REASONING) {
            return Err(ModelError::invalid_request(
                "reasoning replay requires reasoning support",
            ));
        }
        if self.replay.reasoning && self.replay.capability == ReplayCapability::Unsupported {
            return Err(ModelError::invalid_request(
                "reasoning replay requires native replay support",
            ));
        }
        if !matches!(
            (self.replay.policy, self.replay.capability),
            (ReplayPolicy::Never, ReplayCapability::Unsupported)
                | (ReplayPolicy::IfValid, ReplayCapability::Optional)
                | (ReplayPolicy::IfValid, ReplayCapability::Required)
                | (ReplayPolicy::Always, ReplayCapability::Required)
        ) {
            return Err(ModelError::invalid_request(
                "invalid replay policy/capability combination",
            ));
        }
        for modality in self.media.input.keys() {
            if !self.modalities.input.contains(modality) {
                return Err(ModelError::invalid_request(
                    "media rules require the corresponding input modality",
                ));
            }
        }
        Ok(())
    }
}

/// Explicit declaration of one provider-served model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelDeclaration {
    /// Exact provider model, deployment, or resource ID.
    pub id: ModelId,
    /// Explicit capabilities and limits.
    pub capabilities: ModelCapabilities,
}

impl<'de> Deserialize<'de> for ModelDeclaration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireDeclaration {
            id: ModelId,
            capabilities: ModelCapabilities,
        }
        let wire = WireDeclaration::deserialize(deserializer)?;
        Self::new(wire.id, wire.capabilities).map_err(serde::de::Error::custom)
    }
}

impl ModelDeclaration {
    /// Creates a validated declaration.
    pub fn new(id: ModelId, capabilities: ModelCapabilities) -> Result<Self, ModelError> {
        id.validate()?;
        capabilities.validate()?;
        Ok(Self { id, capabilities })
    }

    /// Validates the declaration.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.id.validate()?;
        self.capabilities.validate()
    }
}

/// Complete provider, model, and adapter-specific settings for construction.
#[derive(Clone)]
pub struct ModelConfig<A, S> {
    /// Provider-level configuration.
    pub provider: ProviderConfig<A>,
    /// Explicit model declaration.
    pub model: ModelDeclaration,
    /// Adapter-specific structural settings.
    pub settings: S,
}

impl<A, S> fmt::Debug for ModelConfig<A, S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("settings", &"<redacted>")
            .finish()
    }
}

impl<A, S> ModelConfig<A, S> {
    /// Creates configuration without inferring behavior from the model ID.
    #[must_use]
    pub fn new(provider: ProviderConfig<A>, model: ModelDeclaration, settings: S) -> Self {
        Self {
            provider,
            model,
            settings,
        }
    }

    /// Validates common provider and declaration invariants.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.provider.id.validate()?;
        self.model.validate()
    }
}

/// Identity and contract data for one configured model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LanguageModelDescriptor {
    /// Provider and model identity.
    pub identity: ModelIdentity,
    /// Stable wire adapter identity.
    pub adapter_id: AdapterId,
    /// Explicit model capabilities.
    pub capabilities: ModelCapabilities,
    /// Provider-defined descriptor metadata.
    #[serde(default)]
    pub provider_metadata: ProviderMetadata,
}

impl<'de> Deserialize<'de> for LanguageModelDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireDescriptor {
            identity: ModelIdentity,
            adapter_id: AdapterId,
            capabilities: ModelCapabilities,
            provider_metadata: ProviderMetadata,
        }
        let wire = WireDescriptor::deserialize(deserializer)?;
        Self::new(wire.identity, wire.adapter_id, wire.capabilities)
            .map(|descriptor| descriptor.with_provider_metadata(wire.provider_metadata))
            .map_err(serde::de::Error::custom)
    }
}

impl LanguageModelDescriptor {
    /// Creates a validated descriptor with empty provider metadata.
    pub fn new(
        identity: ModelIdentity,
        adapter_id: AdapterId,
        capabilities: ModelCapabilities,
    ) -> Result<Self, ModelError> {
        identity.validate()?;
        adapter_id.validate()?;
        capabilities.validate()?;
        Ok(Self {
            identity,
            adapter_id,
            capabilities,
            provider_metadata: ProviderMetadata::new(),
        })
    }

    /// Replaces safe provider metadata.
    #[must_use]
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = provider_metadata;
        self
    }
}

/// A parsed JSON Schema document, restricted to object and boolean schemas.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct JsonSchema(JsonValue);

impl JsonSchema {
    /// Creates a schema after validating the JSON Schema root shape.
    pub fn new(value: JsonValue) -> Result<Self, JsonSchemaError> {
        if value.is_object() || value.is_boolean() {
            Ok(Self(value))
        } else {
            Err(JsonSchemaError::InvalidRoot)
        }
    }

    /// Returns the validated JSON Schema value.
    #[must_use]
    pub fn as_value(&self) -> &JsonValue {
        &self.0
    }
}

impl<'de> Deserialize<'de> for JsonSchema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(JsonValue::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Failure while constructing a [`JsonSchema`].
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum JsonSchemaError {
    /// JSON Schema roots must be objects or booleans.
    #[error("JSON Schema root must be an object or boolean")]
    InvalidRoot,
}

/// One item from a model stream.
pub type StreamItem = Result<StreamPart, ModelError>;

/// A normalized stream part. Adapter-internal tool input assembly events are
/// not represented; tool-call events are validated as lifecycle markers and a
/// finalized [`ToolCallPart`] is emitted separately.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamPart {
    /// Starts a stream and reports warnings.
    StreamStart {
        /// Non-fatal adapter warnings.
        warnings: Vec<String>,
    },
    /// Opens a text block.
    TextStart {
        /// Block identifier.
        id: String,
        /// Part metadata.
        metadata: PartMetadata,
    },
    /// Adds a text delta.
    TextDelta {
        /// Block identifier.
        id: String,
        /// Incremental text.
        delta: String,
        /// Part metadata.
        metadata: PartMetadata,
    },
    /// Closes a text block.
    TextEnd {
        /// Block identifier.
        id: String,
        /// Part metadata.
        metadata: PartMetadata,
    },
    /// Opens a reasoning block.
    ReasoningStart {
        /// Block identifier.
        id: String,
        /// Part metadata.
        metadata: PartMetadata,
    },
    /// Adds a reasoning delta.
    ReasoningDelta {
        /// Block identifier.
        id: String,
        /// Incremental reasoning text.
        delta: String,
        /// Part metadata.
        metadata: PartMetadata,
    },
    /// Closes a reasoning block.
    ReasoningEnd {
        /// Block identifier.
        id: String,
        /// Part metadata.
        metadata: PartMetadata,
    },
    /// Opens a tool-call lifecycle block.
    ToolCallStart {
        /// Call identifier.
        id: String,
        /// Tool name.
        name: String,
        /// Part metadata.
        metadata: PartMetadata,
    },
    /// Adds an argument delta to a tool-call block.
    ToolCallDelta {
        /// Call identifier.
        id: String,
        /// Incremental argument text.
        delta: String,
        /// Part metadata.
        metadata: PartMetadata,
    },
    /// Closes a tool-call lifecycle block.
    ToolCallEnd {
        /// Call identifier.
        id: String,
        /// Part metadata.
        metadata: PartMetadata,
    },
    /// Emits a finalized tool call.
    ToolCall {
        /// Finalized call.
        tool_call: ToolCallPart,
    },
    /// Emits a tool result.
    ToolResult {
        /// Tool result.
        tool_result: ToolResultPart,
    },
    /// Emits source provenance.
    Source {
        /// Source record.
        source: SourcePart,
    },
    /// Emits a generic output file.
    File {
        /// Output file.
        file: FilePart,
    },
    /// Requests harness approval.
    ApprovalRequested {
        /// Approval request.
        approval: ToolApprovalPart,
    },
    /// Proves semantic completion and must be final.
    Finish {
        /// Terminal data.
        finish: Finish,
    },
    /// Reports an in-band provider failure; only `Finish(Error)` may follow.
    Error {
        /// In-band error.
        error: ModelError,
    },
    /// Emits opt-in raw provider data, never automatic replay state.
    Raw {
        /// Raw provider value.
        value: JsonValue,
    },
    /// Emits a named provider event.
    ProviderEvent {
        /// Provider event name.
        name: String,
        /// Provider event data.
        data: JsonValue,
    },
    /// Emits a namespaced extension.
    Custom {
        /// Extension part.
        part: CustomPart,
    },
}

/// Stable safe identity of the behavior-affecting provider resource tuple.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ResourceId(String);

impl ResourceId {
    /// Creates a validated resource identity or safe fingerprint.
    pub fn new(value: impl Into<String>) -> Result<Self, ModelError> {
        let value = value.into();
        validate_identifier("resource ID", &value)?;
        Ok(Self(value))
    }

    /// Returns the stable resource identity.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ResourceId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Provider, model, and behavior-affecting resource scope for native context.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct NativeContextScope {
    /// Serving provider identity.
    pub provider_id: ProviderId,
    /// Exact model, deployment, or resource identity.
    pub model_id: ModelId,
    /// Canonical safe resource tuple or fingerprint.
    pub resource_id: ResourceId,
}

impl NativeContextScope {
    /// Creates a validated native-context scope.
    pub fn new(
        provider_id: ProviderId,
        model_id: ModelId,
        resource_id: ResourceId,
    ) -> Result<Self, ModelError> {
        provider_id.validate()?;
        model_id.validate()?;
        Ok(Self {
            provider_id,
            model_id,
            resource_id,
        })
    }
}

/// A bounded opaque provider-native context window returned by compaction.
///
/// The payload is provider correctness state and is intentionally omitted from
/// [`fmt::Debug`] output. Construction and deserialization enforce the size cap.
#[derive(Clone, PartialEq, Serialize)]
pub struct NativeContextWindow {
    adapter_id: AdapterId,
    scope: NativeContextScope,
    payload: JsonValue,
}

impl NativeContextWindow {
    /// Largest permitted serialized payload size, in bytes.
    pub const MAX_PAYLOAD_BYTES: usize = 32 * 1024 * 1024;

    /// Creates a native context window after fail-closed serialized-size validation.
    pub fn new(
        adapter_id: AdapterId,
        scope: NativeContextScope,
        payload: JsonValue,
    ) -> Result<Self, NativeContextPayloadError> {
        adapter_id
            .validate()
            .map_err(|error| NativeContextPayloadError::InvalidIdentity(error.to_string()))?;
        scope
            .provider_id
            .validate()
            .and_then(|()| scope.model_id.validate())
            .map_err(|error| NativeContextPayloadError::InvalidIdentity(error.to_string()))?;
        let size =
            serialized_json_size(&payload).map_err(NativeContextPayloadError::Serialization)?;
        if size > Self::MAX_PAYLOAD_BYTES {
            return Err(NativeContextPayloadError::TooLarge {
                size,
                maximum: Self::MAX_PAYLOAD_BYTES,
            });
        }
        Ok(Self {
            adapter_id,
            scope,
            payload,
        })
    }

    /// Parses a JSON payload and creates a bounded native context window.
    pub fn payload_from_str(
        adapter_id: AdapterId,
        scope: NativeContextScope,
        payload: &str,
    ) -> Result<Self, NativeContextPayloadError> {
        if payload.len() > Self::MAX_PAYLOAD_BYTES {
            return Err(NativeContextPayloadError::TooLarge {
                size: payload.len(),
                maximum: Self::MAX_PAYLOAD_BYTES,
            });
        }
        let parsed =
            serde_json::from_str(payload).map_err(NativeContextPayloadError::InvalidJson)?;
        Self::new(adapter_id, scope, parsed)
    }

    /// Returns the stable adapter identity.
    #[must_use]
    pub fn adapter_id(&self) -> &AdapterId {
        &self.adapter_id
    }

    /// Returns the native-context compatibility scope.
    #[must_use]
    pub fn scope(&self) -> &NativeContextScope {
        &self.scope
    }

    /// Returns the opaque bounded payload.
    #[must_use]
    pub fn payload(&self) -> &JsonValue {
        &self.payload
    }
}

impl fmt::Debug for NativeContextWindow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeContextWindow")
            .field("adapter_id", &self.adapter_id)
            .field("scope", &self.scope)
            .field("payload", &"<redacted>")
            .finish()
    }
}

impl<'de> Deserialize<'de> for NativeContextWindow {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireWindow {
            adapter_id: AdapterId,
            scope: NativeContextScope,
            payload: JsonValue,
        }
        let wire = WireWindow::deserialize(deserializer)?;
        Self::new(wire.adapter_id, wire.scope, wire.payload).map_err(serde::de::Error::custom)
    }
}

/// Failure while parsing, serializing, or bounding a native context payload.
#[derive(Debug, Error)]
pub enum NativeContextPayloadError {
    /// Adapter or scope identity was invalid.
    #[error("invalid native context identity: {0}")]
    InvalidIdentity(String),
    /// Payload exceeds the fail-closed maximum.
    #[error("native context payload is {size} bytes, exceeding the {maximum}-byte limit")]
    TooLarge {
        /// Observed serialized size.
        size: usize,
        /// Maximum permitted serialized size.
        maximum: usize,
    },
    /// JSON input could not be parsed.
    #[error("invalid native context payload JSON: {0}")]
    InvalidJson(serde_json::Error),
    /// An in-memory JSON value could not be serialized for sizing.
    #[error("could not serialize native context payload: {0}")]
    Serialization(serde_json::Error),
}

/// A replay artifact containing stable adapter/scope identity and opaque native state.
///
/// The payload is correctness state and is intentionally omitted from
/// [`fmt::Debug`] output. Construction and deserialization enforce the size cap.
#[derive(Clone, PartialEq, Serialize)]
pub struct NativeReplayArtifact {
    adapter_id: AdapterId,
    scope: NativeContextScope,
    payload: JsonValue,
}

impl NativeReplayArtifact {
    /// Largest permitted serialized payload size, in bytes.
    pub const MAX_PAYLOAD_BYTES: usize = 2 * 1024 * 1024;

    /// Creates an artifact after fail-closed serialized-size validation.
    pub fn new(
        adapter_id: AdapterId,
        scope: NativeContextScope,
        payload: JsonValue,
    ) -> Result<Self, ReplayPayloadError> {
        adapter_id
            .validate()
            .map_err(|error| ReplayPayloadError::InvalidIdentity(error.to_string()))?;
        scope
            .provider_id
            .validate()
            .and_then(|()| scope.model_id.validate())
            .map_err(|error| ReplayPayloadError::InvalidIdentity(error.to_string()))?;
        let size = serialized_json_size(&payload).map_err(ReplayPayloadError::Serialization)?;
        if size > Self::MAX_PAYLOAD_BYTES {
            return Err(ReplayPayloadError::TooLarge {
                size,
                maximum: Self::MAX_PAYLOAD_BYTES,
            });
        }
        Ok(Self {
            adapter_id,
            scope,
            payload,
        })
    }

    /// Parses a JSON payload and creates a bounded artifact.
    pub fn payload_from_str(
        adapter_id: AdapterId,
        scope: NativeContextScope,
        payload: &str,
    ) -> Result<Self, ReplayPayloadError> {
        if payload.len() > Self::MAX_PAYLOAD_BYTES {
            return Err(ReplayPayloadError::TooLarge {
                size: payload.len(),
                maximum: Self::MAX_PAYLOAD_BYTES,
            });
        }
        let parsed = serde_json::from_str(payload).map_err(ReplayPayloadError::InvalidJson)?;
        Self::new(adapter_id, scope, parsed)
    }

    /// Returns the stable adapter identity.
    #[must_use]
    pub fn adapter_id(&self) -> &AdapterId {
        &self.adapter_id
    }

    /// Returns the replay compatibility scope.
    #[must_use]
    pub fn scope(&self) -> &NativeContextScope {
        &self.scope
    }

    /// Returns the opaque bounded payload.
    #[must_use]
    pub fn payload(&self) -> &JsonValue {
        &self.payload
    }
}

impl fmt::Debug for NativeReplayArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeReplayArtifact")
            .field("adapter_id", &self.adapter_id)
            .field("scope", &self.scope)
            .field("payload", &"<redacted>")
            .finish()
    }
}

impl<'de> Deserialize<'de> for NativeReplayArtifact {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireArtifact {
            adapter_id: AdapterId,
            scope: NativeContextScope,
            payload: JsonValue,
        }
        let wire = WireArtifact::deserialize(deserializer)?;
        Self::new(wire.adapter_id, wire.scope, wire.payload).map_err(serde::de::Error::custom)
    }
}

/// Failure while parsing, serializing, or bounding a replay payload.
#[derive(Debug, Error)]
pub enum ReplayPayloadError {
    /// Adapter or scope identity was invalid.
    #[error("invalid replay identity: {0}")]
    InvalidIdentity(String),
    /// Payload exceeds the fail-closed maximum.
    #[error("replay payload is {size} bytes, exceeding the {maximum}-byte limit")]
    TooLarge {
        /// Observed serialized size.
        size: usize,
        /// Maximum permitted serialized size.
        maximum: usize,
    },
    /// JSON input could not be parsed.
    #[error("invalid replay payload JSON: {0}")]
    InvalidJson(serde_json::Error),
    /// An in-memory JSON value could not be serialized for sizing.
    #[error("could not serialize replay payload: {0}")]
    Serialization(serde_json::Error),
}

/// Result of an adapter's native replay capture attempt.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CaptureOutcome {
    /// A bounded artifact was captured.
    Captured {
        /// Captured artifact.
        artifact: NativeReplayArtifact,
    },
    /// No native state exists for this turn.
    NotAvailable,
    /// Capture was explicitly discarded.
    Discarded {
        /// Sanitized discard reason.
        reason: String,
    },
}

/// How a native replay artifact was handled during history encoding.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReplayDisposition {
    /// A compatible native artifact was replayed.
    Replayed,
    /// No artifact was attached to the history turn.
    NoArtifact,
    /// Artifact belongs to another adapter and normalized content was reconstructed.
    DiscardedForeignAdapter {
        /// Artifact adapter identity.
        found: AdapterId,
        /// Current adapter identity.
        expected: AdapterId,
    },
    /// Artifact belongs to a different provider/model/resource scope.
    DiscardedForeignScope {
        /// Artifact replay scope.
        found: NativeContextScope,
        /// Current model replay scope.
        expected: NativeContextScope,
    },
    /// Artifact could not be decoded or validated, so normalized content was reconstructed.
    DiscardedInvalidPayload {
        /// Sanitized decode or validation reason.
        reason: String,
    },
    /// Normalized assistant content was reconstructed.
    ReconstructedNormalized,
}

/// One replay decision associated with a request history entry.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReplayDecision {
    /// Index in request history.
    pub history_index: usize,
    /// Decision made for that history entry.
    pub disposition: ReplayDisposition,
}

/// Complete replay report produced during request encoding.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReplayOutcome {
    /// Ordered decision log for every assistant history entry.
    pub decisions: Vec<ReplayDecision>,
}

/// Metadata produced while validating and encoding one request.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct RequestMetadata {
    /// Complete native replay decision log.
    pub replay: ReplayOutcome,
    /// Adapter-defined safe request metadata.
    pub provider_metadata: ProviderMetadata,
}

/// Metadata available before the normalized stream is returned.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResponseHead {
    /// Successful HTTP status when the transport uses HTTP.
    pub http_status: Option<u16>,
    /// Provider request identifier from response headers.
    pub request_id: Option<String>,
    /// Safe response metadata available during stream initialization.
    pub response_metadata: ResponseMetadata,
}

/// A normalized stream with request-encoding and initial response metadata.
pub struct StreamResponse {
    /// Normalized stream parts.
    pub stream: BoxStream<'static, StreamItem>,
    /// Request metadata.
    pub request: RequestMetadata,
    /// Initial response metadata.
    pub response: ResponseHead,
}

impl StreamResponse {
    /// Creates a response with default metadata.
    #[must_use]
    pub fn new(stream: BoxStream<'static, StreamItem>) -> Self {
        Self {
            stream,
            request: RequestMetadata::default(),
            response: ResponseHead::default(),
        }
    }

    /// Replaces request metadata.
    #[must_use]
    pub fn with_request(mut self, request: RequestMetadata) -> Self {
        self.request = request;
        self
    }

    /// Replaces initial response metadata.
    #[must_use]
    pub fn with_response(mut self, response: ResponseHead) -> Self {
        self.response = response;
        self
    }
}

impl fmt::Debug for StreamResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamResponse")
            .field("stream", &"<opaque>")
            .field("request", &self.request)
            .field("response", &self.response)
            .finish()
    }
}

/// A fully collected turn and metadata from the underlying stream call.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompleteResult {
    /// Collected assistant turn.
    pub turn: CompletedTurn,
    /// Request metadata.
    pub request: RequestMetadata,
    /// Initial response metadata.
    pub response: ResponseHead,
}

/// One provider-native compaction operation over a normalized model request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompactionRequest {
    /// The normalized request whose effective provider context will be compacted.
    pub request: Request,
}

impl CompactionRequest {
    /// Creates a compaction request.
    #[must_use]
    pub fn new(request: Request) -> Self {
        Self { request }
    }

    /// Validates the request and the model's provider-native compaction declaration.
    pub fn validate_for(&self, capabilities: &ModelCapabilities) -> Result<(), ModelError> {
        capabilities.validate()?;
        if capabilities.compaction != CompactionCapability::Native {
            return Err(ModelError::unsupported(
                "provider-native compaction is not supported by this model",
            ));
        }
        self.request.validate_for(capabilities)
    }
}

/// Result of one provider-native compaction operation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompactionResult {
    /// Opaque provider-native context to attach to a subsequent [`Request`].
    pub native_context: NativeContextWindow,
    /// Provider-reported compaction token accounting.
    pub usage: Usage,
    /// Request encoding metadata.
    pub request: RequestMetadata,
    /// Initial provider response metadata.
    pub response: ResponseHead,
}

impl CompactionResult {
    /// Creates a result with empty usage and metadata.
    #[must_use]
    pub fn new(native_context: NativeContextWindow) -> Self {
        Self {
            native_context,
            usage: Usage::default(),
            request: RequestMetadata::default(),
            response: ResponseHead::default(),
        }
    }
}

/// Caller-controlled native replay handling policy.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPolicy {
    /// Do not capture or use native replay state.
    Never,
    /// Use a valid compatible artifact when present.
    #[default]
    IfValid,
    /// Require native replay behavior where the adapter supports it.
    Always,
}

struct AbortState {
    aborted: AtomicBool,
    wakers: Mutex<Vec<Waker>>,
}

/// The read side of a small runtime-neutral cancellation primitive.
#[derive(Clone)]
pub struct AbortSignal {
    state: Arc<AbortState>,
}

/// The write side paired with an [`AbortSignal`].
#[derive(Clone)]
pub struct AbortRegistration {
    state: Arc<AbortState>,
}

impl AbortSignal {
    /// Creates a signal and registration pair.
    #[must_use]
    pub fn new() -> (Self, AbortRegistration) {
        let state = Arc::new(AbortState {
            aborted: AtomicBool::new(false),
            wakers: Mutex::new(Vec::new()),
        });
        (
            Self {
                state: Arc::clone(&state),
            },
            AbortRegistration { state },
        )
    }

    /// Returns whether this signal has been aborted.
    #[must_use]
    pub fn is_aborted(&self) -> bool {
        self.state.aborted.load(Ordering::Acquire)
    }

    /// Returns a future that resolves when this signal is aborted.
    #[must_use]
    pub fn aborted(&self) -> AbortWait {
        AbortWait {
            signal: self.clone(),
        }
    }
}

impl Default for AbortSignal {
    fn default() -> Self {
        Self::new().0
    }
}

impl AbortRegistration {
    /// Aborts the paired signal and wakes each registered task outside the mutex.
    pub fn abort(&self) {
        if !self.state.aborted.swap(true, Ordering::AcqRel) {
            let wakers = {
                let mut registered = self
                    .state
                    .wakers
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                registered.drain(..).collect::<Vec<_>>()
            };
            for waker in wakers {
                waker.wake();
            }
        }
    }
}

/// A future returned by [`AbortSignal::aborted`].
pub struct AbortWait {
    signal: AbortSignal,
}

impl Future for AbortWait {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.signal.is_aborted() {
            return Poll::Ready(());
        }
        let mut registered = self
            .signal
            .state
            .wakers
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.signal.is_aborted() {
            return Poll::Ready(());
        }
        if let Some(index) = registered
            .iter()
            .position(|waker| waker.will_wake(context.waker()))
        {
            registered[index] = context.waker().clone();
        } else {
            registered.push(context.waker().clone());
        }
        Poll::Pending
    }
}

struct OpenBlock {
    slot: usize,
    text: String,
    metadata: PartMetadata,
}

struct OpenToolBlock {
    slot: usize,
    name: String,
    arguments: String,
}

struct EndedToolBlock {
    slot: usize,
    name: String,
    arguments: String,
}

/// A configured model that performs one translated provider call.
///
/// Explicit boxed futures preserve visible allocation, lifetimes, `Send`, and
/// object safety without `async_trait` or native async trait methods.
pub trait LanguageModel: Send + Sync {
    /// Returns configured provider, model, adapter, capability, and metadata information.
    fn descriptor(&self) -> &LanguageModelDescriptor;

    /// Returns explicit model capabilities.
    fn capabilities(&self) -> &ModelCapabilities {
        &self.descriptor().capabilities
    }

    /// Validates a request before the adapter performs network I/O.
    fn validate_request(&self, request: &Request) -> Result<(), ModelError> {
        request.validate_for(self.capabilities())
    }

    /// Returns whether a request is supported by this configured profile.
    fn supports_request(&self, request: &Request) -> bool {
        self.validate_request(request).is_ok()
    }

    /// Validates a provider-native compaction request before network I/O.
    fn validate_compaction(&self, request: &CompactionRequest) -> Result<(), ModelError> {
        request.validate_for(self.capabilities())
    }

    /// Returns whether provider-native compaction supports this request.
    fn supports_compaction(&self, request: &CompactionRequest) -> bool {
        self.validate_compaction(request).is_ok()
    }

    /// Starts the authoritative normalized stream for a request.
    fn stream<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>>;

    /// Drains [`LanguageModel::stream`] with strict normalized lifecycle validation.
    fn complete<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<CompleteResult, ModelError>> {
        Box::pin(async move {
            let StreamResponse {
                mut stream,
                request,
                response,
            } = self.stream(request, abort).await?;
            let mut content = Vec::<Option<AssistantPart>>::new();
            let mut text = BTreeMap::<String, OpenBlock>::new();
            let mut reasoning = BTreeMap::<String, OpenBlock>::new();
            let mut tool = BTreeMap::<String, OpenToolBlock>::new();
            let mut ended_tool = BTreeMap::<String, EndedToolBlock>::new();
            let mut finalized_tool_ids = BTreeSet::<String>::new();
            let mut tool_result_ids = BTreeSet::<String>::new();
            let mut warnings = Vec::new();
            let mut terminal = None;
            let mut in_band_error = None;
            let mut stream_started = false;
            let mut first_item = true;
            while let Some(item) =
                std::future::poll_fn(|context| stream.as_mut().poll_next(context)).await
            {
                let part = item?;
                if first_item {
                    first_item = false;
                    if !matches!(part, StreamPart::StreamStart { .. }) {
                        return Err(ModelError::invalid_response(
                            "stream must begin with stream_start",
                        ));
                    }
                }
                if terminal.is_some() {
                    return Err(ModelError::invalid_response(
                        "stream emitted a part after finish",
                    ));
                }
                if in_band_error.is_some() && !matches!(part, StreamPart::Finish { .. }) {
                    return Err(ModelError::invalid_response(
                        "in-band error was not followed immediately by finish",
                    ));
                }
                match part {
                    StreamPart::StreamStart {
                        warnings: stream_warnings,
                    } => {
                        if stream_started {
                            return Err(ModelError::invalid_response(
                                "stream emitted multiple stream_start parts",
                            ));
                        }
                        stream_started = true;
                        warnings.extend(stream_warnings);
                    }
                    StreamPart::Raw { .. } | StreamPart::ProviderEvent { .. } => {}
                    StreamPart::TextStart { id, metadata } => {
                        if text.contains_key(&id)
                            || reasoning.contains_key(&id)
                            || tool.contains_key(&id)
                            || ended_tool.contains_key(&id)
                            || finalized_tool_ids.contains(&id)
                        {
                            return Err(ModelError::invalid_response(
                                "duplicate or mismatched block start",
                            ));
                        }
                        let slot = content.len();
                        content.push(None);
                        text.insert(
                            id,
                            OpenBlock {
                                slot,
                                text: String::new(),
                                metadata,
                            },
                        );
                    }
                    StreamPart::TextDelta { id, delta, .. } => match text.get_mut(&id) {
                        Some(block) => block.text.push_str(&delta),
                        None => {
                            return Err(ModelError::invalid_response(
                                "text delta without matching start",
                            ));
                        }
                    },
                    StreamPart::TextEnd { id, .. } => match text.remove(&id) {
                        Some(block) => {
                            content[block.slot] = Some(AssistantPart::Text(TextPart {
                                text: block.text,
                                metadata: block.metadata,
                            }))
                        }
                        None => {
                            return Err(ModelError::invalid_response(
                                "text end without matching start",
                            ));
                        }
                    },
                    StreamPart::ReasoningStart { id, metadata } => {
                        if text.contains_key(&id)
                            || reasoning.contains_key(&id)
                            || tool.contains_key(&id)
                            || ended_tool.contains_key(&id)
                        {
                            return Err(ModelError::invalid_response(
                                "duplicate or mismatched block start",
                            ));
                        }
                        let slot = content.len();
                        content.push(None);
                        reasoning.insert(
                            id,
                            OpenBlock {
                                slot,
                                text: String::new(),
                                metadata,
                            },
                        );
                    }
                    StreamPart::ReasoningDelta { id, delta, .. } => match reasoning.get_mut(&id) {
                        Some(block) => block.text.push_str(&delta),
                        None => {
                            return Err(ModelError::invalid_response(
                                "reasoning delta without matching start",
                            ));
                        }
                    },
                    StreamPart::ReasoningEnd { id, .. } => match reasoning.remove(&id) {
                        Some(block) => {
                            content[block.slot] = Some(AssistantPart::Reasoning(ReasoningPart {
                                text: block.text,
                                metadata: block.metadata,
                            }))
                        }
                        None => {
                            return Err(ModelError::invalid_response(
                                "reasoning end without matching start",
                            ));
                        }
                    },
                    StreamPart::ToolCallStart { id, name, .. } => {
                        if text.contains_key(&id)
                            || reasoning.contains_key(&id)
                            || tool.contains_key(&id)
                            || ended_tool.contains_key(&id)
                        {
                            return Err(ModelError::invalid_response(
                                "duplicate or mismatched block start",
                            ));
                        }
                        let slot = content.len();
                        content.push(None);
                        tool.insert(
                            id,
                            OpenToolBlock {
                                slot,
                                name,
                                arguments: String::new(),
                            },
                        );
                    }
                    StreamPart::ToolCallDelta { id, delta, .. } => match tool.get_mut(&id) {
                        Some(block) => block.arguments.push_str(&delta),
                        None => {
                            return Err(ModelError::invalid_response(
                                "tool-call delta without matching start",
                            ));
                        }
                    },
                    StreamPart::ToolCallEnd { id, .. } => {
                        let Some(block) = tool.remove(&id) else {
                            return Err(ModelError::invalid_response(
                                "tool-call end without matching start",
                            ));
                        };
                        ended_tool.insert(
                            id,
                            EndedToolBlock {
                                slot: block.slot,
                                name: block.name,
                                arguments: block.arguments,
                            },
                        );
                    }
                    StreamPart::ToolCall { mut tool_call } => {
                        if !finalized_tool_ids.insert(tool_call.id.clone()) {
                            return Err(ModelError::invalid_response(
                                "stream emitted a duplicate finalized tool call ID",
                            ));
                        }
                        if tool.contains_key(&tool_call.id) {
                            return Err(ModelError::invalid_response(
                                "finalized tool call arrived before tool-call end",
                            ));
                        }
                        if let Some(block) = ended_tool.remove(&tool_call.id) {
                            if block.name != tool_call.name {
                                return Err(ModelError::invalid_response(
                                    "finalized tool call name does not match its stream block",
                                ));
                            }
                            if !block.arguments.is_empty() {
                                let input: JsonValue = serde_json::from_str(&block.arguments)
                                    .map_err(|_| {
                                        ModelError::invalid_response(
                                            "tool-call argument stream is not valid JSON",
                                        )
                                    })?;
                                if input != tool_call.input {
                                    return Err(ModelError::invalid_response(
                                        "finalized tool call input does not match streamed arguments",
                                    ));
                                }
                                if let Some(raw_input) = &tool_call.raw_input
                                    && raw_input != &block.arguments
                                {
                                    return Err(ModelError::invalid_response(
                                        "finalized tool call raw input does not match streamed arguments",
                                    ));
                                }
                                tool_call.raw_input = Some(block.arguments);
                            }
                            content[block.slot] = Some(AssistantPart::ToolCall(tool_call));
                        } else {
                            content.push(Some(AssistantPart::ToolCall(tool_call)));
                        }
                    }
                    StreamPart::ToolResult { tool_result } => {
                        if !tool_result_ids.insert(tool_result.tool_call_id.clone()) {
                            return Err(ModelError::invalid_response(
                                "stream emitted duplicate tool result IDs",
                            ));
                        }
                        content.push(Some(AssistantPart::ToolResult(tool_result)))
                    }
                    StreamPart::Source { source } => {
                        content.push(Some(AssistantPart::Source(source)))
                    }
                    StreamPart::File { file } => content.push(Some(AssistantPart::File(file))),
                    StreamPart::ApprovalRequested { approval } => {
                        content.push(Some(AssistantPart::ToolApproval(approval)))
                    }
                    StreamPart::Custom { part } => content.push(Some(AssistantPart::Custom(part))),
                    StreamPart::Error { error } => in_band_error = Some(error),
                    StreamPart::Finish { finish } => {
                        if !text.is_empty()
                            || !reasoning.is_empty()
                            || !tool.is_empty()
                            || !ended_tool.is_empty()
                        {
                            return Err(ModelError::invalid_response(
                                "finish emitted with unclosed or unfinalized blocks",
                            ));
                        }
                        if in_band_error.is_some() && finish.finish_reason != FinishReason::Error {
                            return Err(ModelError::invalid_response(
                                "in-band error requires finish(error)",
                            ));
                        }
                        if in_band_error.is_none() && finish.finish_reason == FinishReason::Error {
                            return Err(ModelError::invalid_response(
                                "finish(error) requires an in-band error",
                            ));
                        }
                        terminal = Some(finish);
                    }
                }
            }
            if !stream_started {
                return Err(ModelError::invalid_response(
                    "stream is missing stream_start",
                ));
            }
            let finish =
                terminal.ok_or_else(|| ModelError::unexpected_eof("stream ended before finish"))?;
            if let Some(error) = in_band_error {
                return Err(error);
            }
            for tool_call_id in &tool_result_ids {
                if !finalized_tool_ids.contains(tool_call_id) {
                    warnings.push(format!(
                        "tool result for `{tool_call_id}` has no normalized in-stream tool call"
                    ));
                }
            }
            let content = content
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    ModelError::invalid_response("stream ended with an unfilled content slot")
                })?;
            let mut turn = CompletedTurn::new(AssistantMessage::new(content), finish);
            turn.warnings = warnings;
            Ok(CompleteResult {
                turn,
                request,
                response,
            })
        })
    }

    /// Compacts a normalized request into an opaque provider-native context window.
    ///
    /// The default rejects unsupported declarations before any provider I/O and
    /// otherwise reports that the adapter has not implemented its claimed native
    /// compaction capability.
    fn compact<'a>(
        &'a self,
        request: CompactionRequest,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<CompactionResult, ModelError>> {
        let validation = self.validate_compaction(&request);
        Box::pin(async move {
            validation?;
            if abort.is_aborted() {
                return Err(ModelError::abort("native compaction was aborted")
                    .with_stage(ErrorStage::NativeContextEncode));
            }
            Err(ModelError::unsupported(
                "provider-native compaction is declared but not implemented by this adapter",
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        task::Wake,
    };

    use super::*;

    struct TestStream(VecDeque<StreamItem>);

    impl Stream for TestStream {
        type Item = StreamItem;

        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.0.pop_front())
        }
    }

    struct Scripted {
        items: Vec<StreamItem>,
        descriptor: LanguageModelDescriptor,
    }

    impl Scripted {
        fn new(parts: Vec<StreamPart>) -> Self {
            Self {
                items: parts.into_iter().map(Ok).collect(),
                descriptor: LanguageModelDescriptor::new(
                    ModelIdentity::new(ProviderId::new("test"), ModelId::new("scripted"))
                        .expect("identity"),
                    AdapterId::new("test.scripted"),
                    text_capabilities(),
                )
                .expect("descriptor"),
            }
        }
    }

    impl LanguageModel for Scripted {
        fn descriptor(&self) -> &LanguageModelDescriptor {
            &self.descriptor
        }

        fn validate_request(&self, _: &Request) -> Result<(), ModelError> {
            Ok(())
        }

        fn supports_request(&self, _: &Request) -> bool {
            true
        }

        fn stream<'a>(
            &'a self,
            _: Request,
            _: AbortSignal,
        ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
            let items = self.items.clone();
            Box::pin(async move {
                let stream: BoxStream<'static, StreamItem> =
                    Box::pin(TestStream(VecDeque::from(items)));
                Ok(StreamResponse::new(stream))
            })
        }
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => output,
            Poll::Pending => panic!("test future unexpectedly pending"),
        }
    }

    fn text_capabilities() -> ModelCapabilities {
        ModelCapabilities::conservative()
    }

    fn tool_capabilities() -> ModelCapabilities {
        let mut capabilities = text_capabilities();
        capabilities.features |= Capability::TOOL_CALLING;
        capabilities
    }

    fn native_context_scope(model_id: &str) -> NativeContextScope {
        NativeContextScope::new(
            ProviderId::new("test"),
            ModelId::new(model_id),
            ResourceId::new("test-resource").expect("resource"),
        )
        .expect("scope")
    }

    fn finish(reason: FinishReason) -> StreamPart {
        StreamPart::Finish {
            finish: Finish::new(Usage::default(), reason),
        }
    }

    fn collect(parts: Vec<StreamPart>) -> Result<CompletedTurn, ModelError> {
        let mut parts = parts;
        parts.insert(
            0,
            StreamPart::StreamStart {
                warnings: Vec::new(),
            },
        );
        block_on(Scripted::new(parts).complete(Request::new(Vec::new()), AbortSignal::default()))
            .map(|result| result.turn)
    }

    fn collect_raw(parts: Vec<StreamPart>) -> Result<CompletedTurn, ModelError> {
        block_on(Scripted::new(parts).complete(Request::new(Vec::new()), AbortSignal::default()))
            .map(|result| result.turn)
    }

    fn assert_invalid(parts: Vec<StreamPart>) {
        assert!(collect(parts).is_err_and(|error| error.is_kind(ModelErrorKind::InvalidResponse)));
    }

    #[test]
    fn stream_response_preserves_typed_request_and_response_metadata() {
        let request = RequestMetadata {
            replay: ReplayOutcome::default(),
            provider_metadata: BTreeMap::from([("request".into(), JsonValue::Bool(true))]),
        };
        let response = ResponseHead {
            http_status: Some(200),
            request_id: Some("req-1".into()),
            response_metadata: BTreeMap::from([("response".into(), JsonValue::Bool(true))]),
        };
        let stream = StreamResponse::new(Box::pin(TestStream(VecDeque::new())))
            .with_request(request.clone())
            .with_response(response.clone());
        assert_eq!(stream.request, request);
        assert_eq!(stream.response, response);
        assert!(format!("{stream:?}").contains("<opaque>"));
    }

    #[test]
    fn default_complete_preserves_stream_metadata() {
        let result = block_on(
            Scripted::new(vec![
                StreamPart::StreamStart {
                    warnings: Vec::new(),
                },
                finish(FinishReason::Stop),
            ])
            .complete(Request::new(Vec::new()), AbortSignal::default()),
        )
        .expect("complete");
        assert_eq!(result.request, RequestMetadata::default());
        assert_eq!(result.response, ResponseHead::default());
    }

    #[test]
    fn complete_result_serde_round_trip() {
        let result = CompleteResult {
            turn: CompletedTurn::new(
                AssistantMessage::new(Vec::new()),
                Finish::new(Usage::default(), FinishReason::Stop),
            ),
            request: RequestMetadata::default(),
            response: ResponseHead::default(),
        };
        let decoded: CompleteResult =
            serde_json::from_str(&serde_json::to_string(&result).expect("serialize"))
                .expect("deserialize");
        assert_eq!(decoded, result);
    }

    #[test]
    fn replay_decision_serde_round_trip_preserves_order() {
        let outcome = ReplayOutcome {
            decisions: vec![
                ReplayDecision {
                    history_index: 1,
                    disposition: ReplayDisposition::NoArtifact,
                },
                ReplayDecision {
                    history_index: 1,
                    disposition: ReplayDisposition::ReconstructedNormalized,
                },
            ],
        };
        let decoded: ReplayOutcome =
            serde_json::from_str(&serde_json::to_string(&outcome).expect("serialize"))
                .expect("deserialize");
        assert_eq!(decoded, outcome);
    }

    #[test]
    fn collector_assembles_text_blocks_by_id() {
        let turn = collect(vec![
            StreamPart::TextStart {
                id: "text-1".into(),
                metadata: None,
            },
            StreamPart::TextDelta {
                id: "text-1".into(),
                delta: "hello ".into(),
                metadata: None,
            },
            StreamPart::TextDelta {
                id: "text-1".into(),
                delta: "world".into(),
                metadata: None,
            },
            StreamPart::TextEnd {
                id: "text-1".into(),
                metadata: None,
            },
            finish(FinishReason::Stop),
        ])
        .expect("valid lifecycle");
        assert_eq!(turn.text(), "hello world");
    }

    #[test]
    fn collector_rejects_delta_without_start() {
        assert_invalid(vec![
            StreamPart::TextDelta {
                id: "missing".into(),
                delta: "x".into(),
                metadata: None,
            },
            finish(FinishReason::Stop),
        ]);
    }

    #[test]
    fn collector_rejects_duplicate_or_mismatched_start() {
        assert_invalid(vec![
            StreamPart::TextStart {
                id: "shared".into(),
                metadata: None,
            },
            StreamPart::ReasoningStart {
                id: "shared".into(),
                metadata: None,
            },
            finish(FinishReason::Stop),
        ]);
    }

    #[test]
    fn collector_rejects_mismatched_or_duplicate_end() {
        assert_invalid(vec![
            StreamPart::TextStart {
                id: "text".into(),
                metadata: None,
            },
            StreamPart::ReasoningEnd {
                id: "text".into(),
                metadata: None,
            },
            finish(FinishReason::Stop),
        ]);
    }

    #[test]
    fn collector_rejects_unclosed_blocks_at_finish() {
        assert_invalid(vec![
            StreamPart::ToolCallStart {
                id: "call".into(),
                name: "tool".into(),
                metadata: None,
            },
            finish(FinishReason::Stop),
        ]);
    }

    #[test]
    fn collector_rejects_ended_tool_block_without_finalized_call() {
        assert_invalid(vec![
            StreamPart::ToolCallStart {
                id: "call".into(),
                name: "tool".into(),
                metadata: None,
            },
            StreamPart::ToolCallDelta {
                id: "call".into(),
                delta: r#"{"value":1}"#.into(),
                metadata: None,
            },
            StreamPart::ToolCallEnd {
                id: "call".into(),
                metadata: None,
            },
            finish(FinishReason::Stop),
        ]);
    }

    #[test]
    fn collector_rejects_mismatched_finalized_tool_call_id() {
        assert_invalid(vec![
            StreamPart::ToolCallStart {
                id: "call-a".into(),
                name: "tool".into(),
                metadata: None,
            },
            StreamPart::ToolCallEnd {
                id: "call-a".into(),
                metadata: None,
            },
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call-b", "tool", JsonValue::Null),
            },
            finish(FinishReason::Stop),
        ]);
    }

    #[test]
    fn collector_pairs_finalized_tool_call_with_assembled_arguments() {
        let turn = collect(vec![
            StreamPart::ToolCallStart {
                id: "call".into(),
                name: "tool".into(),
                metadata: None,
            },
            StreamPart::ToolCallDelta {
                id: "call".into(),
                delta: r#"{"value":"#.into(),
                metadata: None,
            },
            StreamPart::ToolCallDelta {
                id: "call".into(),
                delta: r#"1}"#.into(),
                metadata: None,
            },
            StreamPart::ToolCallEnd {
                id: "call".into(),
                metadata: None,
            },
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "tool", serde_json::json!({"value": 1})),
            },
            finish(FinishReason::ToolCalls),
        ])
        .expect("paired tool call is valid");
        assert!(matches!(
            &turn.message.content[0],
            AssistantPart::ToolCall(ToolCallPart { raw_input: Some(raw), .. }) if raw == r#"{"value":1}"#
        ));
    }

    #[test]
    fn collector_rejects_genuinely_duplicate_finalized_tool_ids() {
        assert_invalid(vec![
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "tool", JsonValue::Null),
            },
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "tool", JsonValue::Null),
            },
            finish(FinishReason::ToolCalls),
        ]);
    }

    #[test]
    fn collector_rejects_sequential_tool_call_id_reuse() {
        assert_invalid(vec![
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "tool", JsonValue::Null),
            },
            StreamPart::ToolCallStart {
                id: "call".into(),
                name: "tool".into(),
                metadata: None,
            },
            finish(FinishReason::ToolCalls),
        ]);
        assert_invalid(vec![
            StreamPart::ToolCallStart {
                id: "call".into(),
                name: "tool".into(),
                metadata: None,
            },
            StreamPart::ToolCallEnd {
                id: "call".into(),
                metadata: None,
            },
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "tool", JsonValue::Null),
            },
            StreamPart::ToolCallStart {
                id: "call".into(),
                name: "tool".into(),
                metadata: None,
            },
            StreamPart::ToolCallEnd {
                id: "call".into(),
                metadata: None,
            },
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "tool", JsonValue::Null),
            },
            finish(FinishReason::ToolCalls),
        ]);
        assert_invalid(vec![
            StreamPart::ToolCallStart {
                id: "call".into(),
                name: "tool".into(),
                metadata: None,
            },
            StreamPart::ToolCallEnd {
                id: "call".into(),
                metadata: None,
            },
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "tool", JsonValue::Null),
            },
            StreamPart::ToolCallStart {
                id: "call".into(),
                name: "tool".into(),
                metadata: None,
            },
            finish(FinishReason::ToolCalls),
        ]);
    }

    #[test]
    fn collector_disambiguated_ids_round_trip_through_history_validation() {
        let turn = collect(vec![
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "tool", serde_json::json!({})),
            },
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call-1", "tool", serde_json::json!({})),
            },
            finish(FinishReason::ToolCalls),
        ])
        .expect("adapter-disambiguated call IDs are valid");
        assert_eq!(turn.message.content.len(), 2);
        Request::new(vec![
            HistoryTurn::assistant(turn),
            HistoryTurn::tool(tool_message(&["call", "call-1"])),
        ])
        .validate_for(&tool_capabilities())
        .expect("collector output must satisfy history validation");
    }

    #[test]
    fn collector_rejects_duplicate_streamed_tool_result_ids() {
        assert_invalid(vec![
            StreamPart::ToolResult {
                tool_result: ToolResultPart::new("call", ToolContent::Text("first".into())),
            },
            StreamPart::ToolResult {
                tool_result: ToolResultPart::new("call", ToolContent::Text("second".into())),
            },
            finish(FinishReason::Stop),
        ]);
    }

    #[test]
    fn collector_warns_for_hosted_tool_result_without_normalized_call() {
        let turn = collect(vec![
            StreamPart::ToolResult {
                tool_result: ToolResultPart::new(
                    "hosted-search",
                    ToolContent::Text("result".into()),
                ),
            },
            finish(FinishReason::Stop),
        ])
        .expect("hosted tool results are valid");
        assert_eq!(
            turn.warnings,
            vec!["tool result for `hosted-search` has no normalized in-stream tool call"]
        );
    }

    #[test]
    fn collector_does_not_warn_for_result_of_finalized_tool_call() {
        let turn = collect(vec![
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "tool", JsonValue::Null),
            },
            StreamPart::ToolResult {
                tool_result: ToolResultPart::new("call", ToolContent::Text("result".into())),
            },
            finish(FinishReason::Stop),
        ])
        .expect("normalized tool result is valid");
        assert!(turn.warnings.is_empty());
    }

    #[test]
    fn collector_preserves_start_order_for_interleaved_blocks() {
        let turn = collect(vec![
            StreamPart::TextStart {
                id: "a".into(),
                metadata: None,
            },
            StreamPart::ReasoningStart {
                id: "b".into(),
                metadata: None,
            },
            StreamPart::ReasoningDelta {
                id: "b".into(),
                delta: "reasoning".into(),
                metadata: None,
            },
            StreamPart::ReasoningEnd {
                id: "b".into(),
                metadata: None,
            },
            StreamPart::TextDelta {
                id: "a".into(),
                delta: "text".into(),
                metadata: None,
            },
            StreamPart::TextEnd {
                id: "a".into(),
                metadata: None,
            },
            finish(FinishReason::Stop),
        ])
        .expect("interleaved blocks are valid");
        assert!(matches!(
            &turn.message.content[..],
            [AssistantPart::Text(_), AssistantPart::Reasoning(_)]
        ));
    }

    #[test]
    fn collector_requires_one_first_stream_start() {
        assert!(matches!(
            collect_raw(vec![finish(FinishReason::Stop)]),
            Err(error) if error.is_kind(ModelErrorKind::InvalidResponse)
        ));
        assert!(matches!(
            collect_raw(vec![
                StreamPart::StreamStart {
                    warnings: Vec::new()
                },
                StreamPart::StreamStart {
                    warnings: Vec::new()
                },
                finish(FinishReason::Stop),
            ]),
            Err(error) if error.is_kind(ModelErrorKind::InvalidResponse)
        ));
        assert!(matches!(
            collect_raw(vec![
                StreamPart::Raw {
                    value: JsonValue::Null
                },
                StreamPart::StreamStart {
                    warnings: Vec::new()
                },
                finish(FinishReason::Stop),
            ]),
            Err(error) if error.is_kind(ModelErrorKind::InvalidResponse)
        ));
    }

    #[test]
    fn collector_rejects_repeated_finish() {
        assert_invalid(vec![finish(FinishReason::Stop), finish(FinishReason::Stop)]);
    }

    #[test]
    fn collector_rejects_items_after_finish() {
        assert_invalid(vec![
            finish(FinishReason::Stop),
            StreamPart::Raw {
                value: JsonValue::Null,
            },
        ]);
    }

    #[test]
    fn collector_requires_documented_in_band_error_sequence() {
        let error = ModelError::provider("broken");
        assert_invalid(vec![
            StreamPart::Error {
                error: error.clone(),
            },
            finish(FinishReason::Stop),
        ]);
        assert!(matches!(
            collect(vec![StreamPart::Error { error }]),
            Err(error) if error.is_kind(ModelErrorKind::UnexpectedEof)
        ));
    }

    #[test]
    fn collector_returns_the_in_band_error_after_valid_terminal_sequence() {
        let error = ModelError::provider("broken");
        assert_eq!(
            collect(vec![
                StreamPart::Error {
                    error: error.clone(),
                },
                finish(FinishReason::Error),
            ]),
            Err(error)
        );
    }

    #[test]
    fn model_error_constructors_set_expected_kind_stage_and_retryability() {
        let cases = [
            (
                ModelError::invalid_request("x"),
                ModelErrorKind::InvalidRequest,
                ErrorStage::RequestValidation,
                false,
            ),
            (
                ModelError::unsupported("x"),
                ModelErrorKind::Unsupported,
                ErrorStage::RequestValidation,
                false,
            ),
            (
                ModelError::invalid_response("x"),
                ModelErrorKind::InvalidResponse,
                ErrorStage::StreamFinalize,
                false,
            ),
            (
                ModelError::unexpected_eof("x"),
                ModelErrorKind::UnexpectedEof,
                ErrorStage::StreamFinalize,
                false,
            ),
            (
                ModelError::transport("x"),
                ModelErrorKind::Transport,
                ErrorStage::Connect,
                true,
            ),
            (
                ModelError::timeout("x"),
                ModelErrorKind::Timeout,
                ErrorStage::Unknown,
                true,
            ),
            (
                ModelError::rate_limited("x"),
                ModelErrorKind::RateLimited,
                ErrorStage::ResponseHeaders,
                true,
            ),
            (
                ModelError::provider("x"),
                ModelErrorKind::Provider,
                ErrorStage::ResponseHeaders,
                false,
            ),
            (
                ModelError::abort("x"),
                ModelErrorKind::Abort,
                ErrorStage::Unknown,
                false,
            ),
            (
                ModelError::replay("x"),
                ModelErrorKind::Replay,
                ErrorStage::ReplayDecode,
                false,
            ),
            (
                ModelError::native_context("x"),
                ModelErrorKind::NativeContext,
                ErrorStage::NativeContextDecode,
                false,
            ),
        ];
        for (error, kind, stage, retryable) in cases {
            assert_eq!(error.kind(), kind);
            assert_eq!(error.diagnostics.stage, stage);
            assert_eq!(error.retryable, retryable);
        }
    }

    #[test]
    fn model_error_builder_methods_preserve_existing_fields() {
        let error = ModelError::transport("safe")
            .with_http_status(503)
            .with_bytes_received(42)
            .with_vendor_code("overloaded")
            .with_request_id("req-1")
            .with_retry_after(Duration::from_millis(250))
            .with_sanitized_body(SanitizedBody::new("body"));
        assert!(error.is_kind(ModelErrorKind::Transport));
        assert_eq!(error.message, "safe");
        assert_eq!(error.diagnostics.http_status, Some(503));
        assert_eq!(error.diagnostics.bytes_received, 42);
        assert_eq!(error.diagnostics.request_id.as_deref(), Some("req-1"));
    }

    #[test]
    fn model_error_serde_round_trip_preserves_diagnostics() {
        let error = ModelError::provider("safe")
            .with_http_status(500)
            .with_sanitized_body(SanitizedBody::new("body"));
        let decoded: ModelError =
            serde_json::from_str(&serde_json::to_string(&error).expect("serialize"))
                .expect("deserialize");
        assert_eq!(decoded, error);
    }

    #[test]
    fn retry_after_serializes_as_milliseconds() {
        let error = ModelError::timeout("slow").with_retry_after(Duration::from_millis(123));
        assert_eq!(
            serde_json::to_value(error).expect("serialize")["diagnostics"]["retry_after"],
            123
        );
    }

    #[test]
    fn sanitized_body_accepts_exact_64_kib() {
        let body = SanitizedBody::new("x".repeat(SanitizedBody::MAX_BYTES));
        assert_eq!(body.len_bytes(), SanitizedBody::MAX_BYTES);
        assert!(!body.truncated());
    }

    #[test]
    fn sanitized_body_truncates_over_limit() {
        let body = SanitizedBody::new("x".repeat(SanitizedBody::MAX_BYTES + 1));
        assert_eq!(body.len_bytes(), SanitizedBody::MAX_BYTES);
        assert!(body.truncated());
    }

    #[test]
    fn sanitized_body_truncates_at_utf8_boundary() {
        let body = SanitizedBody::new("é".repeat(SanitizedBody::MAX_BYTES / 2 + 1));
        assert!(body.len_bytes() <= SanitizedBody::MAX_BYTES);
        assert!(body.text().is_char_boundary(body.text().len()));
    }

    #[test]
    fn sanitized_body_deserialization_enforces_cap() {
        let wire = serde_json::json!({"text": "x".repeat(SanitizedBody::MAX_BYTES + 1), "truncated": false});
        let body: SanitizedBody = serde_json::from_value(wire).expect("deserialize");
        assert_eq!(body.len_bytes(), SanitizedBody::MAX_BYTES);
        assert!(body.truncated());
    }

    #[test]
    fn sanitized_body_debug_redacts_text() {
        let debug = format!("{:?}", SanitizedBody::new("secret"));
        assert!(!debug.contains("secret"));
    }

    #[test]
    fn model_error_display_omits_body_status_request_id_and_vendor_code() {
        let error = ModelError::provider("safe")
            .with_http_status(500)
            .with_request_id("req-secret")
            .with_vendor_code("vendor-secret")
            .with_sanitized_body(SanitizedBody::new("body-secret"));
        let display = error.to_string();
        assert_eq!(display, "provider error: safe");
        assert!(!display.contains("secret"));
    }

    #[test]
    fn model_error_debug_does_not_expose_sanitized_body_text() {
        let debug = format!(
            "{:?}",
            ModelError::provider("safe").with_sanitized_body(SanitizedBody::new("body-secret"))
        );
        assert!(!debug.contains("body-secret"));
    }

    #[test]
    fn artifact_constructor_enforces_cap() {
        let oversized = JsonValue::String("x".repeat(NativeReplayArtifact::MAX_PAYLOAD_BYTES));
        assert!(matches!(
            NativeReplayArtifact::new(
                AdapterId::new("test"),
                native_context_scope("scripted"),
                oversized,
            ),
            Err(ReplayPayloadError::TooLarge { .. })
        ));
    }

    #[test]
    fn artifact_deserialize_enforces_cap() {
        let payload = "x".repeat(NativeReplayArtifact::MAX_PAYLOAD_BYTES);
        let scope = serde_json::to_string(&native_context_scope("scripted")).expect("scope");
        let wire = format!(r#"{{"adapter_id":"test","scope":{scope},"payload":"{payload}"}}"#);
        assert!(serde_json::from_str::<NativeReplayArtifact>(&wire).is_err());
    }

    #[test]
    fn artifact_exact_cap_is_accepted() {
        let payload = JsonValue::String("x".repeat(NativeReplayArtifact::MAX_PAYLOAD_BYTES - 2));
        let artifact = NativeReplayArtifact::new(
            AdapterId::new("test"),
            native_context_scope("scripted"),
            payload,
        )
        .expect("serialized payload is exactly at the cap");
        assert_eq!(
            serde_json::to_vec(artifact.payload())
                .expect("serialize payload")
                .len(),
            NativeReplayArtifact::MAX_PAYLOAD_BYTES
        );
    }

    #[test]
    fn role_safe_history_deserialization_rejects_invalid_content() {
        let invalid_system = serde_json::json!({
            "type": "system",
            "value": {
                "content": [{
                    "type": "tool_result",
                    "value": {
                        "tool_call_id": "call",
                        "content": {"type": "text", "value": "result"},
                        "is_error": false,
                        "metadata": null
                    }
                }],
                "provider_options": {}
            }
        });
        let invalid_user = serde_json::json!({
            "type": "user",
            "value": {
                "content": [{
                    "type": "reasoning",
                    "value": {"text": "thought", "metadata": null}
                }],
                "provider_options": {}
            }
        });
        let invalid_assistant = serde_json::json!({
            "type": "assistant",
            "value": {
                "message": {
                    "content": [{
                        "type": "unknown_assistant_part",
                        "value": {}
                    }],
                    "provider_options": {}
                },
                "finish": {
                    "usage": {}, "finish_reason": "stop", "response_metadata": {},
                    "provider_metadata": {}, "native_replay": null
                }
            }
        });
        assert!(serde_json::from_value::<HistoryTurn>(invalid_system).is_err());
        assert!(serde_json::from_value::<HistoryTurn>(invalid_user).is_err());
        assert!(serde_json::from_value::<HistoryTurn>(invalid_assistant).is_err());
    }

    fn declared_tool(name: &str) -> ToolDefinition {
        ToolDefinition::new(
            name,
            "test tool",
            JsonSchema::new(serde_json::json!({})).expect("object schema"),
        )
    }

    #[test]
    fn request_validation_rejects_named_undeclared_tool() {
        let request = Request::new(Vec::new()).with_tool_choice(ToolChoice::Tool("missing".into()));
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    #[test]
    fn request_validation_rejects_required_choice_without_tools() {
        let request = Request::new(Vec::new()).with_tool_choice(ToolChoice::Required);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    #[test]
    fn request_validation_rejects_duplicate_tool_names() {
        let request =
            Request::new(Vec::new()).with_tools(vec![declared_tool("same"), declared_tool("same")]);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    fn assistant_with_calls(ids: &[&str]) -> CompletedTurn {
        let content = ids
            .iter()
            .map(|id| AssistantPart::ToolCall(ToolCallPart::new(*id, "tool", JsonValue::Null)))
            .collect();
        CompletedTurn::new(
            AssistantMessage::new(content),
            Finish::new(Usage::default(), FinishReason::ToolCalls),
        )
    }

    fn tool_message(ids: &[&str]) -> ToolMessage {
        ToolMessage::new(
            ids.iter()
                .map(|id| ToolResultPart::new(*id, ToolContent::Text("result".into())))
                .collect(),
        )
    }

    #[test]
    fn request_validation_rejects_leading_tool_history() {
        let request = Request::new(vec![HistoryTurn::tool(tool_message(&["call"]))]);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    #[test]
    fn request_validation_rejects_empty_tool_message_after_call_less_assistant() {
        let request = Request::new(vec![
            HistoryTurn::assistant(CompletedTurn::new(
                AssistantMessage::new(Vec::new()),
                Finish::new(Usage::default(), FinishReason::Stop),
            )),
            HistoryTurn::tool(ToolMessage::new(Vec::new())),
        ]);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    #[test]
    fn request_validation_rejects_unrelated_tool_result_id() {
        let request = Request::new(vec![
            HistoryTurn::assistant(assistant_with_calls(&["call-a"])),
            HistoryTurn::tool(tool_message(&["call-b"])),
        ]);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    #[test]
    fn request_validation_rejects_duplicate_tool_result_id() {
        let request = Request::new(vec![
            HistoryTurn::assistant(assistant_with_calls(&["call"])),
            HistoryTurn::tool(tool_message(&["call", "call"])),
        ]);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    #[test]
    fn request_validation_rejects_missing_tool_result() {
        let request = Request::new(vec![
            HistoryTurn::assistant(assistant_with_calls(&["call-a", "call-b"])),
            HistoryTurn::tool(tool_message(&["call-a"])),
        ]);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    #[test]
    fn request_validation_accepts_exact_tool_result_pairing() {
        let request = Request::new(vec![
            HistoryTurn::assistant(assistant_with_calls(&["call-a", "call-b"])),
            HistoryTurn::tool(tool_message(&["call-b", "call-a"])),
        ]);
        assert!(request.validate_for(&tool_capabilities()).is_ok());
    }

    #[test]
    fn collected_provider_tool_call_and_result_round_trip_through_history() {
        let completed = collect(vec![
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call", "hosted_tool", JsonValue::Null),
            },
            StreamPart::ToolResult {
                tool_result: ToolResultPart::new("call", ToolContent::Text("result".into())),
            },
            finish(FinishReason::Stop),
        ])
        .expect("provider tool stream is valid");
        let request = Request::new(vec![HistoryTurn::assistant(completed)]);
        assert!(request.validate_for(&tool_capabilities()).is_ok());
    }

    #[test]
    fn request_validation_rejects_foreign_assistant_contained_tool_result() {
        let turn = CompletedTurn::new(
            AssistantMessage::new(vec![
                AssistantPart::ToolCall(ToolCallPart::new("call", "tool", JsonValue::Null)),
                AssistantPart::ToolResult(ToolResultPart::new(
                    "foreign",
                    ToolContent::Text("result".into()),
                )),
            ]),
            Finish::new(Usage::default(), FinishReason::Stop),
        );
        let request = Request::new(vec![HistoryTurn::assistant(turn)]);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    #[test]
    fn request_validation_rejects_result_duplicated_between_assistant_and_tool_history() {
        let turn = CompletedTurn::new(
            AssistantMessage::new(vec![
                AssistantPart::ToolCall(ToolCallPart::new("call", "tool", JsonValue::Null)),
                AssistantPart::ToolResult(ToolResultPart::new(
                    "call",
                    ToolContent::Text("contained".into()),
                )),
            ]),
            Finish::new(Usage::default(), FinishReason::Stop),
        );
        let request = Request::new(vec![
            HistoryTurn::assistant(turn),
            HistoryTurn::tool(tool_message(&["call"])),
        ]);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    #[test]
    fn request_validation_accepts_mixed_assistant_and_tool_history_results() {
        let turn = CompletedTurn::new(
            AssistantMessage::new(vec![
                AssistantPart::ToolCall(ToolCallPart::new("call-a", "tool", JsonValue::Null)),
                AssistantPart::ToolCall(ToolCallPart::new("call-b", "tool", JsonValue::Null)),
                AssistantPart::ToolResult(ToolResultPart::new(
                    "call-a",
                    ToolContent::Text("contained".into()),
                )),
            ]),
            Finish::new(Usage::default(), FinishReason::ToolCalls),
        );
        let request = Request::new(vec![
            HistoryTurn::assistant(turn),
            HistoryTurn::tool(tool_message(&["call-b"])),
        ]);
        assert!(request.validate_for(&tool_capabilities()).is_ok());
    }

    #[test]
    fn request_validation_rejects_empty_and_whitespace_tool_names() {
        for name in ["", " \t\n"] {
            let request = Request::new(Vec::new()).with_tools(vec![declared_tool(name)]);
            assert!(matches!(
                request.validate_for(&text_capabilities()),
                Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
            ));
        }
    }

    #[test]
    fn json_schema_rejects_invalid_roots_and_accepts_object_or_boolean() {
        for value in [
            JsonValue::Null,
            JsonValue::String("schema".into()),
            JsonValue::from(1),
        ] {
            assert_eq!(JsonSchema::new(value), Err(JsonSchemaError::InvalidRoot));
        }
        for value in [
            JsonValue::Null,
            JsonValue::String("schema".into()),
            JsonValue::from(1),
        ] {
            assert!(serde_json::from_value::<JsonSchema>(value).is_err());
        }
        assert!(JsonSchema::new(serde_json::json!({"type": "object"})).is_ok());
        assert!(JsonSchema::new(JsonValue::Bool(true)).is_ok());
        assert!(serde_json::from_value::<JsonSchema>(serde_json::json!({})).is_ok());
        assert!(serde_json::from_value::<JsonSchema>(JsonValue::Bool(false)).is_ok());
    }

    #[test]
    fn request_validation_requires_structured_output_capability() {
        let request = Request::new(Vec::new()).with_response_format(ResponseFormat::structured(
            JsonSchema::new(serde_json::json!({})).expect("object schema"),
        ));
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::Unsupported)
        ));
        let mut capabilities = text_capabilities();
        capabilities.features |= Capability::STRUCTURED_OUTPUT;
        assert!(request.validate_for(&capabilities).is_ok());
    }

    #[test]
    fn request_validation_rejects_non_finite_and_out_of_range_sampling() {
        let mut inference = InferenceOptions::new();
        inference.temperature = Some(f64::NAN);
        let request = Request::new(Vec::new()).with_inference(inference);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
        let mut inference = InferenceOptions::new();
        inference.top_p = Some(1.1);
        let request = Request::new(Vec::new()).with_inference(inference);
        assert!(matches!(
            request.validate_for(&text_capabilities()),
            Err(error) if error.is_kind(ModelErrorKind::InvalidRequest)
        ));
    }

    #[test]
    fn file_constructors_are_plain_file_parts() {
        let source = FileSource::Bytes(Bytes::from_static(b"media"));
        let expected = FilePart::new("application/octet-stream", source.clone());
        assert_eq!(
            FilePart::audio("application/octet-stream", source.clone()),
            expected
        );
        assert_eq!(
            FilePart::video("application/octet-stream", source.clone()),
            expected
        );
        assert_eq!(
            FilePart::image("application/octet-stream", source.clone()),
            expected
        );
        assert_eq!(
            FilePart::document("application/octet-stream", source),
            expected
        );
    }

    struct LockingWake {
        state: Arc<AbortState>,
        wakes: AtomicUsize,
    }

    impl Wake for LockingWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            let _guard = self.state.wakers.lock().expect("waker lock is available");
            self.wakes.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn abort_waiters_are_deduplicated_and_woken_outside_lock() {
        let (signal, registration) = AbortSignal::new();
        let task = Arc::new(LockingWake {
            state: Arc::clone(&signal.state),
            wakes: AtomicUsize::new(0),
        });
        let waker = Waker::from(Arc::clone(&task));
        let mut wait = Box::pin(signal.aborted());
        for _ in 0..4 {
            let mut context = Context::from_waker(&waker);
            assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));
        }
        assert_eq!(signal.state.wakers.lock().expect("waker lock").len(), 1);
        registration.abort();
        assert_eq!(task.wakes.load(Ordering::SeqCst), 1);
    }
}
