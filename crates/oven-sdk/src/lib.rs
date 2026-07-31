#![warn(missing_docs)]
//! Runtime-neutral normalized contracts for language-model providers.
//!
//! The crate owns typed model contracts, stream lifecycle, cancellation,
//! capabilities, and replay artifacts. Transport, logging, persistence, and
//! retry or fallback policy belong to adapters and calling harnesses.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
};

use bitflags::bitflags;
use bytes::Bytes;
use futures_core::Stream;
use serde::{Deserialize, Deserializer, Serialize};
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

/// A stable provider namespace.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
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
}

impl fmt::Display for ProviderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A configured model identifier.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
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
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A stable adapter identity used to decide native replay compatibility.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
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
}

impl fmt::Display for AdapterId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Normalized token accounting. Inclusive totals are never computed by adding
/// component fields.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
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

/// A declarative URL pattern directly accepted by a configured provider.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct UrlPattern {
    /// Adapter-defined pattern, such as `https://files.example/*`.
    pub pattern: String,
}

impl UrlPattern {
    /// Creates a URL pattern. It never authorizes SDK URL downloading.
    #[must_use]
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
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
    /// Declarative direct-URL support; it never enables automatic downloading.
    #[serde(rename = "supportedUrls")]
    pub supported_urls: Vec<UrlPattern>,
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
            supported_urls: Vec::new(),
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

/// One normalized content part.
///
/// Metadata is stored in every typed payload so parts remain self-contained if
/// an adapter translates or stores them independently.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content.
    Text(TextPart),
    /// Visible reasoning content.
    Reasoning(ReasoningPart),
    /// Generic MIME-typed media.
    File(FilePart),
    /// A finalized tool call.
    ToolCall(ToolCallPart),
    /// A tool result.
    ToolResult(ToolResultPart),
    /// Citation or provenance.
    Source(SourcePart),
    /// Tool approval request.
    ToolApproval(ToolApprovalPart),
    /// Namespaced extension.
    Custom(CustomPart),
}

impl ContentPart {
    /// Returns the metadata carried by this part.
    #[must_use]
    pub fn metadata(&self) -> &PartMetadata {
        match self {
            Self::Text(part) => &part.metadata,
            Self::Reasoning(part) => &part.metadata,
            Self::File(part) => &part.metadata,
            Self::ToolCall(part) => &part.metadata,
            Self::ToolResult(part) => &part.metadata,
            Self::Source(part) => &part.metadata,
            Self::ToolApproval(part) => &part.metadata,
            Self::Custom(part) => &part.metadata,
        }
    }
}

/// A role name for a conversation turn.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Role {
    /// The provider role string.
    pub role: String,
}

impl Role {
    /// Creates the system role.
    #[must_use]
    pub fn system() -> Self {
        Self::other("system")
    }

    /// Creates the user role.
    #[must_use]
    pub fn user() -> Self {
        Self::other("user")
    }

    /// Creates the assistant role.
    #[must_use]
    pub fn assistant() -> Self {
        Self::other("assistant")
    }

    /// Creates the tool role.
    #[must_use]
    pub fn tool() -> Self {
        Self::other("tool")
    }

    /// Creates a custom role. No `Role::OTHER` constant exists because custom
    /// roles must retain their actual provider string.
    #[must_use]
    pub fn other(role: impl Into<String>) -> Self {
        Self { role: role.into() }
    }
}

impl PartialEq<&str> for Role {
    fn eq(&self, other: &&str) -> bool {
        self.role == *other
    }
}

impl PartialEq<Role> for &str {
    fn eq(&self, other: &Role) -> bool {
        *self == other.role
    }
}

impl PartialEq<String> for Role {
    fn eq(&self, other: &String) -> bool {
        self.role == *other
    }
}

impl PartialEq<Role> for String {
    fn eq(&self, other: &Role) -> bool {
        *self == other.role
    }
}

/// A normalized conversation turn.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[non_exhaustive]
pub struct Turn {
    /// Role associated with the content.
    pub role: Role,
    /// Ordered content parts.
    pub content: Vec<ContentPart>,
    /// Provider options scoped to this turn.
    pub provider_options: ProviderOptions,
}

impl Turn {
    /// Creates a turn with no provider options.
    #[must_use]
    pub fn new(role: Role, content: Vec<ContentPart>) -> Self {
        Self {
            role,
            content,
            provider_options: ProviderOptions::new(),
        }
    }

    /// Replaces provider options for this turn.
    #[must_use]
    pub fn with_provider_options(mut self, provider_options: ProviderOptions) -> Self {
        self.provider_options = provider_options;
        self
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

    /// Converts generic content only when every part is system-safe.
    pub fn try_from_content(content: Vec<ContentPart>) -> Result<Self, ModelError> {
        content
            .into_iter()
            .map(|part| match part {
                ContentPart::Text(part) => Ok(SystemPart::Text(part)),
                ContentPart::Custom(part) => Ok(SystemPart::Custom(part)),
                _ => Err(ModelError::invalid_request(
                    "system messages only allow text and custom parts",
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Self::new)
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

    /// Converts generic content only when every part is user-safe.
    pub fn try_from_content(content: Vec<ContentPart>) -> Result<Self, ModelError> {
        content
            .into_iter()
            .map(|part| match part {
                ContentPart::Text(part) => Ok(InputPart::Text(part)),
                ContentPart::File(part) => Ok(InputPart::File(part)),
                ContentPart::Custom(part) => Ok(InputPart::Custom(part)),
                _ => Err(ModelError::invalid_request(
                    "user messages only allow text, file, and custom parts",
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Self::new)
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

    /// Converts generic content only when every part is assistant-safe.
    pub fn try_from_content(content: Vec<ContentPart>) -> Result<Self, ModelError> {
        content
            .into_iter()
            .map(|part| match part {
                ContentPart::Text(part) => Ok(AssistantPart::Text(part)),
                ContentPart::Reasoning(part) => Ok(AssistantPart::Reasoning(part)),
                ContentPart::ToolCall(part) => Ok(AssistantPart::ToolCall(part)),
                ContentPart::ToolResult(part) => Ok(AssistantPart::ToolResult(part)),
                ContentPart::File(part) => Ok(AssistantPart::File(part)),
                ContentPart::Source(part) => Ok(AssistantPart::Source(part)),
                ContentPart::Custom(part) => Ok(AssistantPart::Custom(part)),
                ContentPart::ToolApproval(_) => Err(ModelError::invalid_request(
                    "assistant history cannot contain approval requests",
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Self::new)
    }

    /// Converts a generic turn only when it has the assistant role and safe content.
    pub fn try_from_turn(turn: Turn) -> Result<Self, ModelError> {
        if turn.role != "assistant" {
            return Err(ModelError::invalid_request(
                "completed turns require the assistant role",
            ));
        }
        let mut message = Self::try_from_content(turn.content)?;
        message.provider_options = turn.provider_options;
        Ok(message)
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

    /// Converts a generic assistant turn while validating its role and content.
    pub fn try_from_turn(turn: Turn, finish: Finish) -> Result<Self, ModelError> {
        AssistantMessage::try_from_turn(turn).map(|message| Self::new(message, finish))
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
    pub fn validate_for(&self, capabilities: &ProviderCapabilities) -> Result<(), ModelError> {
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
        if matches!(self.response_format, ResponseFormat::Json { .. })
            && !capabilities
                .capabilities
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
        if let Some(top_p) = self.inference.top_p
            && (!top_p.is_finite() || !(0.0..=1.0).contains(&top_p))
        {
            return Err(ModelError::invalid_request(
                "top_p must be finite and between 0 and 1",
            ));
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

/// Retry facts reported by an adapter. These are not SDK retry policy.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct RetryHints {
    /// Provider-requested delay in milliseconds.
    pub retry_after_ms: Option<u64>,
    /// Provider request identifier.
    pub request_id: Option<String>,
    /// Provider-specific error code.
    pub vendor_code: Option<String>,
}

/// A structured model-error taxonomy. Harnesses own retry and fallback policy.
#[derive(Clone, Debug, Deserialize, Error, Eq, PartialEq, Serialize)]
pub enum ModelError {
    /// Transport failure.
    #[error("transport error: {message}")]
    Transport {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
    /// Timeout failure.
    #[error("timeout error: {message}")]
    Timeout {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
    /// Provider rate limit.
    #[error("rate-limited error: {message}")]
    RateLimited {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
    /// Authentication failure.
    #[error("authentication error: {message}")]
    Auth {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
    /// Authorization failure.
    #[error("permission-denied error: {message}")]
    PermissionDenied {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
    /// Invalid request.
    #[error("invalid-request error: {message}")]
    InvalidRequest {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
    /// Unsupported operation.
    #[error("unsupported error: {message}")]
    Unsupported {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
    /// EOF before mandatory semantic completion.
    #[error("unexpected EOF: {message}")]
    UnexpectedEof {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
    /// Invalid normalized or provider response.
    #[error("invalid-response error: {message}")]
    InvalidResponse {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
    /// Provider-defined failure.
    #[error("provider error: {message}")]
    Provider {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
    /// Local or provider abort.
    #[error("abort error: {message}")]
    Abort {
        /// Sanitized message.
        message: String,
        /// Factual retry hint.
        retryable: bool,
        /// Optional retry details.
        retry_hints: Option<RetryHints>,
    },
}

impl ModelError {
    /// Builds a non-retryable invalid-request error for local validation failures.
    #[must_use]
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            retryable: false,
            retry_hints: None,
        }
    }

    /// Builds a non-retryable unsupported error for capability validation failures.
    #[must_use]
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
            retryable: false,
            retry_hints: None,
        }
    }

    /// Builds a non-retryable invalid-response error for lifecycle violations.
    #[must_use]
    pub fn invalid_response(message: impl Into<String>) -> Self {
        Self::InvalidResponse {
            message: message.into(),
            retryable: false,
            retry_hints: None,
        }
    }

    /// Builds an unexpected-EOF error for a missing terminal finish.
    #[must_use]
    pub fn unexpected_eof(message: impl Into<String>) -> Self {
        Self::UnexpectedEof {
            message: message.into(),
            retryable: false,
            retry_hints: None,
        }
    }
}

bitflags! {
    /// Features supported by one configured model and wire profile.
    #[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
    pub struct Capability: u128 {
        /// Tool calls.
        const TOOL_CALLING = 1 << 0;
        /// Parallel tool calls.
        const PARALLEL_TOOLS = 1 << 1;
        /// Tool-input deltas.
        const TOOL_INPUT_DELTAS = 1 << 2;
        /// Visible reasoning.
        const REASONING = 1 << 3;
        /// Native reasoning replay.
        const REASONING_REPLAY = 1 << 4;
        /// Image input.
        const IMAGE_INPUT = 1 << 5;
        /// Document input.
        const DOCUMENT_INPUT = 1 << 6;
        /// Audio input.
        const AUDIO_INPUT = 1 << 7;
        /// Video input.
        const VIDEO_INPUT = 1 << 8;
        /// Structured output.
        const STRUCTURED_OUTPUT = 1 << 9;
        /// Prompt caching.
        const PROMPT_CACHING = 1 << 10;
        /// Usage reporting.
        const USAGE = 1 << 11;
        /// Provider-executed tools.
        const PROVIDER_TOOLS = 1 << 12;
        /// Citation output.
        const SOURCES = 1 << 13;
        /// File output.
        const FILE_OUTPUT = 1 << 14;
        /// Native replay.
        const NATIVE_REPLAY = 1 << 15;
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

/// Capabilities and limits of one configured model/profile.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProviderCapabilities {
    /// Feature bitset.
    pub capabilities: Capability,
    /// Maximum context tokens, when known.
    pub context_tokens: Option<u64>,
    /// Maximum output tokens, when known.
    pub max_output_tokens: Option<u64>,
    /// Offered cancellation semantics.
    pub cancellation: CancellationCapability,
    /// Offered native replay semantics.
    pub replay: ReplayCapability,
}

/// Identity and contract data for one configured model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LanguageModelDescriptor {
    /// Provider identity.
    pub provider_id: ProviderId,
    /// Configured model identity.
    pub model_id: ModelId,
    /// Stable wire adapter identity.
    pub adapter_id: AdapterId,
    /// Model/profile capabilities.
    pub capabilities: ProviderCapabilities,
    /// Provider-defined descriptor metadata.
    pub provider_metadata: ProviderMetadata,
}

impl LanguageModelDescriptor {
    /// Creates a descriptor with default capabilities and empty metadata.
    #[must_use]
    pub fn new(provider_id: ProviderId, model_id: ModelId, adapter_id: AdapterId) -> Self {
        Self {
            provider_id,
            model_id,
            adapter_id,
            capabilities: ProviderCapabilities::default(),
            provider_metadata: ProviderMetadata::new(),
        }
    }
}

/// A JSON fragment that can be embedded in another JSON document.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct JsonFragment(pub JsonValue);

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

/// Default normalized history encoding.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct EncodedHistory(pub Vec<JsonFragment>);

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

/// A replay artifact containing only stable adapter identity and opaque native state.
///
/// The payload is correctness state and is intentionally omitted from
/// [`fmt::Debug`] output. Construction and deserialization enforce the size cap.
#[derive(Clone, PartialEq, Serialize)]
pub struct NativeReplayArtifact {
    adapter_id: AdapterId,
    payload: JsonValue,
}

impl NativeReplayArtifact {
    /// Largest permitted serialized payload size, in bytes.
    pub const MAX_PAYLOAD_BYTES: usize = 2 * 1024 * 1024;

    /// Creates an artifact after fail-closed serialized-size validation.
    pub fn new(adapter_id: AdapterId, payload: JsonValue) -> Result<Self, ReplayPayloadError> {
        let size = serde_json::to_vec(&payload)
            .map_err(ReplayPayloadError::Serialization)?
            .len();
        if size > Self::MAX_PAYLOAD_BYTES {
            return Err(ReplayPayloadError::TooLarge {
                size,
                maximum: Self::MAX_PAYLOAD_BYTES,
            });
        }
        Ok(Self {
            adapter_id,
            payload,
        })
    }

    /// Parses a JSON payload and creates a bounded artifact.
    pub fn payload_from_str(
        adapter_id: AdapterId,
        payload: &str,
    ) -> Result<Self, ReplayPayloadError> {
        if payload.len() > Self::MAX_PAYLOAD_BYTES {
            return Err(ReplayPayloadError::TooLarge {
                size: payload.len(),
                maximum: Self::MAX_PAYLOAD_BYTES,
            });
        }
        let parsed = serde_json::from_str(payload).map_err(ReplayPayloadError::InvalidJson)?;
        Self::new(adapter_id, parsed)
    }

    /// Returns the stable adapter identity.
    #[must_use]
    pub fn adapter_id(&self) -> &AdapterId {
        &self.adapter_id
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
        struct WireArtifact {
            adapter_id: AdapterId,
            payload: JsonValue,
        }
        let wire = WireArtifact::deserialize(deserializer)?;
        Self::new(wire.adapter_id, wire.payload).map_err(serde::de::Error::custom)
    }
}

/// Failure while parsing, serializing, or bounding a replay payload.
#[derive(Debug, Error)]
pub enum ReplayPayloadError {
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
    /// Artifact could not be decoded or validated, so normalized content was reconstructed.
    DiscardedInvalidPayload {
        /// Sanitized decode or validation reason.
        reason: String,
    },
    /// Normalized assistant content was reconstructed.
    ReconstructedNormalized,
}

/// A report for one artifact discard or reconstruction decision.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DiscardReport {
    /// Index in request history.
    pub history_index: usize,
    /// Decision made for that history entry.
    pub disposition: ReplayDisposition,
}

/// Aggregate replay result returned by an adapter history encoder.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReplayOutcome {
    /// Decisions that were not ordinary compatible replay.
    pub reports: Vec<DiscardReport>,
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
    fn descriptor(&self) -> LanguageModelDescriptor;

    /// Returns declarative URL patterns directly accepted by this model.
    fn supported_urls(&self) -> Vec<UrlPattern>;

    /// Returns model/profile capabilities.
    fn capabilities(&self) -> ProviderCapabilities {
        self.descriptor().capabilities
    }

    /// Validates a request before the adapter performs network I/O.
    fn validate_request(&self, request: &Request) -> Result<(), ModelError>;

    /// Returns whether a request is supported by this configured profile.
    fn supports_request(&self, request: &Request) -> bool;

    /// Starts the authoritative normalized stream for a request.
    fn stream<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<BoxStream<'static, StreamItem>, ModelError>>;

    /// Drains [`LanguageModel::stream`] with strict normalized lifecycle validation.
    fn complete<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<CompletedTurn, ModelError>> {
        Box::pin(async move {
            let mut stream = self.stream(request, abort).await?;
            let mut content = Vec::<Option<ContentPart>>::new();
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
                            content[block.slot] = Some(ContentPart::Text(TextPart {
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
                            content[block.slot] = Some(ContentPart::Reasoning(ReasoningPart {
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
                            || finalized_tool_ids.contains(&id)
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
                            content[block.slot] = Some(ContentPart::ToolCall(tool_call));
                        } else {
                            content.push(Some(ContentPart::ToolCall(tool_call)));
                        }
                    }
                    StreamPart::ToolResult { tool_result } => {
                        if !tool_result_ids.insert(tool_result.tool_call_id.clone()) {
                            return Err(ModelError::invalid_response(
                                "stream emitted duplicate tool result IDs",
                            ));
                        }
                        content.push(Some(ContentPart::ToolResult(tool_result)))
                    }
                    StreamPart::Source { source } => {
                        content.push(Some(ContentPart::Source(source)))
                    }
                    StreamPart::File { file } => content.push(Some(ContentPart::File(file))),
                    StreamPart::ApprovalRequested { approval } => {
                        content.push(Some(ContentPart::ToolApproval(approval)))
                    }
                    StreamPart::Custom { part } => content.push(Some(ContentPart::Custom(part))),
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
            let mut turn = CompletedTurn::new(AssistantMessage::try_from_content(content)?, finish);
            turn.warnings = warnings;
            Ok(turn)
        })
    }

    /// Encodes normalized history as parsed JSON fragments by default.
    fn encode_history(&self, history: &[HistoryTurn]) -> Result<EncodedHistory, ModelError> {
        history
            .iter()
            .map(|turn| {
                serde_json::to_value(turn)
                    .map(JsonFragment)
                    .map_err(|error| ModelError::invalid_response(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(EncodedHistory)
    }

    /// Decodes the default normalized JSON-fragment history encoding.
    fn decode_history(&self, history: &EncodedHistory) -> Result<Vec<HistoryTurn>, ModelError> {
        history
            .0
            .iter()
            .map(|fragment| {
                serde_json::from_value(fragment.0.clone())
                    .map_err(|error| ModelError::invalid_response(error.to_string()))
            })
            .collect()
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
    }

    impl Scripted {
        fn new(parts: Vec<StreamPart>) -> Self {
            Self {
                items: parts.into_iter().map(Ok).collect(),
            }
        }
    }

    impl LanguageModel for Scripted {
        fn descriptor(&self) -> LanguageModelDescriptor {
            LanguageModelDescriptor::new(
                ProviderId::new("test"),
                ModelId::new("scripted"),
                AdapterId::new("test.scripted"),
            )
        }

        fn supported_urls(&self) -> Vec<UrlPattern> {
            Vec::new()
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
        ) -> BoxFuture<'a, Result<BoxStream<'static, StreamItem>, ModelError>> {
            let items = self.items.clone();
            Box::pin(async move {
                let stream: BoxStream<'static, StreamItem> =
                    Box::pin(TestStream(VecDeque::from(items)));
                Ok(stream)
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
    }

    fn collect_raw(parts: Vec<StreamPart>) -> Result<CompletedTurn, ModelError> {
        block_on(Scripted::new(parts).complete(Request::new(Vec::new()), AbortSignal::default()))
    }

    fn assert_invalid(parts: Vec<StreamPart>) {
        assert!(matches!(
            collect(parts),
            Err(ModelError::InvalidResponse { .. })
        ));
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
    fn collector_rejects_duplicate_full_call_only_ids() {
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
    fn collector_rejects_full_and_streamed_tool_call_id_reuse() {
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
    fn collector_accepts_distinct_finalized_tool_call_ids() {
        let turn = collect(vec![
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call-a", "tool", JsonValue::Null),
            },
            StreamPart::ToolCall {
                tool_call: ToolCallPart::new("call-b", "tool", JsonValue::Null),
            },
            finish(FinishReason::ToolCalls),
        ])
        .expect("distinct call IDs are valid");
        assert_eq!(turn.message.content.len(), 2);
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
            Err(ModelError::InvalidResponse { .. })
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
            Err(ModelError::InvalidResponse { .. })
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
            Err(ModelError::InvalidResponse { .. })
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
        let error = ModelError::Provider {
            message: "broken".into(),
            retryable: false,
            retry_hints: None,
        };
        assert_invalid(vec![
            StreamPart::Error {
                error: error.clone(),
            },
            finish(FinishReason::Stop),
        ]);
        assert!(matches!(
            collect(vec![StreamPart::Error { error }]),
            Err(ModelError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn collector_returns_the_in_band_error_after_valid_terminal_sequence() {
        let error = ModelError::Provider {
            message: "broken".into(),
            retryable: false,
            retry_hints: None,
        };
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
    fn artifact_constructor_enforces_cap() {
        let oversized = JsonValue::String("x".repeat(NativeReplayArtifact::MAX_PAYLOAD_BYTES));
        assert!(matches!(
            NativeReplayArtifact::new(AdapterId::new("test"), oversized),
            Err(ReplayPayloadError::TooLarge { .. })
        ));
    }

    #[test]
    fn artifact_deserialize_enforces_cap() {
        let payload = "x".repeat(NativeReplayArtifact::MAX_PAYLOAD_BYTES);
        let wire = format!(r#"{{"adapter_id":"test","payload":"{payload}"}}"#);
        assert!(serde_json::from_str::<NativeReplayArtifact>(&wire).is_err());
    }

    #[test]
    fn artifact_exact_cap_is_accepted() {
        let payload = JsonValue::String("x".repeat(NativeReplayArtifact::MAX_PAYLOAD_BYTES - 2));
        let artifact = NativeReplayArtifact::new(AdapterId::new("test"), payload)
            .expect("serialized payload is exactly at the cap");
        assert_eq!(
            serde_json::to_vec(artifact.payload())
                .expect("serialize payload")
                .len(),
            NativeReplayArtifact::MAX_PAYLOAD_BYTES
        );
    }

    #[test]
    fn completed_assistant_artifact_survives_history_encoding() {
        let artifact = NativeReplayArtifact::new(AdapterId::new("test.adapter"), JsonValue::Null)
            .expect("small artifact");
        let mut finish = Finish::new(Usage::default(), FinishReason::Stop);
        finish.native_replay = Some(artifact.clone());
        let request = Request::new(vec![HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(Vec::new()),
            finish,
        ))]);
        let request: Request =
            serde_json::from_str(&serde_json::to_string(&request).expect("serialize request"))
                .expect("deserialize request");
        let model = Scripted::new(Vec::new());
        let encoded = model
            .encode_history(&request.history)
            .expect("encode history");
        let decoded = model.decode_history(&encoded).expect("decode history");
        assert_eq!(decoded, request.history);
        assert_eq!(
            match &decoded[0] {
                HistoryTurn::Assistant(turn) => turn.finish.native_replay.as_ref(),
                _ => None,
            },
            Some(&artifact)
        );
    }

    #[test]
    fn role_safe_constructors_reject_invalid_content_and_roles() {
        let tool_result = ToolResultPart::new("call", ToolContent::Text("result".into()));
        assert!(matches!(
            SystemMessage::try_from_content(vec![ContentPart::ToolResult(tool_result)]),
            Err(ModelError::InvalidRequest { .. })
        ));
        assert!(matches!(
            UserMessage::try_from_content(vec![ContentPart::Reasoning(ReasoningPart::new(
                "thought"
            ))]),
            Err(ModelError::InvalidRequest { .. })
        ));
        assert!(matches!(
            CompletedTurn::try_from_turn(
                Turn::new(
                    Role::user(),
                    vec![ContentPart::Text(TextPart::new("not assistant"))]
                ),
                Finish::new(Usage::default(), FinishReason::Stop),
            ),
            Err(ModelError::InvalidRequest { .. })
        ));
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
                        "type": "tool_approval",
                        "value": {"tool_call_id": "call", "message": null, "metadata": null}
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
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
        ));
    }

    #[test]
    fn request_validation_rejects_required_choice_without_tools() {
        let request = Request::new(Vec::new()).with_tool_choice(ToolChoice::Required);
        assert!(matches!(
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
        ));
    }

    #[test]
    fn request_validation_rejects_duplicate_tool_names() {
        let request =
            Request::new(Vec::new()).with_tools(vec![declared_tool("same"), declared_tool("same")]);
        assert!(matches!(
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
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
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
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
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
        ));
    }

    #[test]
    fn request_validation_rejects_unrelated_tool_result_id() {
        let request = Request::new(vec![
            HistoryTurn::assistant(assistant_with_calls(&["call-a"])),
            HistoryTurn::tool(tool_message(&["call-b"])),
        ]);
        assert!(matches!(
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
        ));
    }

    #[test]
    fn request_validation_rejects_duplicate_tool_result_id() {
        let request = Request::new(vec![
            HistoryTurn::assistant(assistant_with_calls(&["call"])),
            HistoryTurn::tool(tool_message(&["call", "call"])),
        ]);
        assert!(matches!(
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
        ));
    }

    #[test]
    fn request_validation_rejects_missing_tool_result() {
        let request = Request::new(vec![
            HistoryTurn::assistant(assistant_with_calls(&["call-a", "call-b"])),
            HistoryTurn::tool(tool_message(&["call-a"])),
        ]);
        assert!(matches!(
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
        ));
    }

    #[test]
    fn request_validation_accepts_exact_tool_result_pairing() {
        let request = Request::new(vec![
            HistoryTurn::assistant(assistant_with_calls(&["call-a", "call-b"])),
            HistoryTurn::tool(tool_message(&["call-b", "call-a"])),
        ]);
        assert!(
            request
                .validate_for(&ProviderCapabilities::default())
                .is_ok()
        );
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
        assert!(
            request
                .validate_for(&ProviderCapabilities::default())
                .is_ok()
        );
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
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
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
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
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
        assert!(
            request
                .validate_for(&ProviderCapabilities::default())
                .is_ok()
        );
    }

    #[test]
    fn request_validation_rejects_empty_and_whitespace_tool_names() {
        for name in ["", " \t\n"] {
            let request = Request::new(Vec::new()).with_tools(vec![declared_tool(name)]);
            assert!(matches!(
                request.validate_for(&ProviderCapabilities::default()),
                Err(ModelError::InvalidRequest { .. })
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
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::Unsupported { .. })
        ));
        let capabilities = ProviderCapabilities {
            capabilities: Capability::STRUCTURED_OUTPUT,
            ..ProviderCapabilities::default()
        };
        assert!(request.validate_for(&capabilities).is_ok());
    }

    #[test]
    fn request_validation_rejects_non_finite_and_out_of_range_sampling() {
        let mut inference = InferenceOptions::new();
        inference.temperature = Some(f64::NAN);
        let request = Request::new(Vec::new()).with_inference(inference);
        assert!(matches!(
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
        ));
        let mut inference = InferenceOptions::new();
        inference.top_p = Some(1.1);
        let request = Request::new(Vec::new()).with_inference(inference);
        assert!(matches!(
            request.validate_for(&ProviderCapabilities::default()),
            Err(ModelError::InvalidRequest { .. })
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
