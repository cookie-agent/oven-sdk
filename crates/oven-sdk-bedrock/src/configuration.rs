//! Explicit registry-free Bedrock authentication and Converse wire settings.

use std::{future::Future, sync::Arc};

use oven_sdk::{BoxFuture, ModelError};
use reqwest::Client;

use crate::transport::BedrockTimeouts;

/// AWS credentials used for Bedrock SigV4 requests.
#[derive(Clone)]
pub struct AwsCredentials {
    /// AWS access key ID.
    pub access_key_id: String,
    /// AWS secret access key.
    pub secret_access_key: String,
    /// Optional temporary-credential session token.
    pub session_token: Option<String>,
}

impl std::fmt::Debug for AwsCredentials {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AwsCredentials")
            .field("access_key_id", &"<redacted>")
            .field("secret_access_key", &"<redacted>")
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// Caller-managed asynchronous AWS credential provider.
pub trait AwsCredentialsProvider: Send + Sync {
    /// Resolves current credentials. Implementations own caching and refresh.
    fn credentials(&self) -> BoxFuture<'static, Result<AwsCredentials, ModelError>>;
}

impl<F, Fut> AwsCredentialsProvider for F
where
    F: Fn() -> Fut + Send + Sync,
    Fut: Future<Output = Result<AwsCredentials, ModelError>> + Send + 'static,
{
    fn credentials(&self) -> BoxFuture<'static, Result<AwsCredentials, ModelError>> {
        Box::pin(self())
    }
}

/// Explicit Bedrock authentication.
#[derive(Clone)]
pub enum BedrockAuth {
    /// Fixed credentials supplied by the caller.
    Static(AwsCredentials),
    /// Caller-managed credentials resolved once per request.
    Provider(Arc<dyn AwsCredentialsProvider>),
}

impl std::fmt::Debug for BedrockAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static(_) => formatter.write_str("BedrockAuth::Static(<redacted>)"),
            Self::Provider(_) => formatter.write_str("BedrockAuth::Provider(<provider>)"),
        }
    }
}

/// Provider-specific reasoning request shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BedrockReasoningWireFormat {
    /// Reject Bedrock reasoning controls.
    Unsupported,
    /// Encode Anthropic `thinking` and `output_config` fields.
    AnthropicThinking,
    /// Encode OpenAI `reasoning_effort`.
    OpenAiReasoningEffort,
    /// Encode Bedrock `reasoningConfig`.
    BedrockReasoningConfig,
}

/// Native Bedrock structured-output wire support.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BedrockStructuredOutput {
    /// Reject structured-output requests.
    Unsupported,
    /// Encode Bedrock `outputConfig.textFormat` JSON Schema requests.
    JsonSchema,
}

/// AWS EventStream allocation limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BedrockEventStreamLimits {
    /// Maximum bytes in one declared AWS EventStream message.
    pub max_message_bytes: usize,
}

impl BedrockEventStreamLimits {
    /// Creates an explicit EventStream limit declaration.
    #[must_use]
    pub const fn new(max_message_bytes: usize) -> Self {
        Self { max_message_bytes }
    }
}

/// Explicit structural and transport settings for Bedrock Converse.
#[derive(Clone)]
pub struct BedrockConverseSettings {
    /// AWS signing region.
    pub region: String,
    /// Reasoning request wire shape.
    pub reasoning_wire_format: BedrockReasoningWireFormat,
    /// Require signatures for visible reasoning and permit redacted reasoning.
    pub signed_reasoning: bool,
    /// Structured-output wire support.
    pub structured_output: BedrockStructuredOutput,
    /// EventStream allocation limits.
    pub event_stream: BedrockEventStreamLimits,
    /// Adapter-controlled phase timeouts.
    pub timeouts: BedrockTimeouts,
    /// Optional injected HTTP client. Its connect timeout is caller-owned.
    pub client: Option<Client>,
}

impl BedrockConverseSettings {
    /// Creates explicit Bedrock Converse settings with default phase timeouts.
    #[must_use]
    pub fn new(
        region: impl Into<String>,
        reasoning_wire_format: BedrockReasoningWireFormat,
        signed_reasoning: bool,
        structured_output: BedrockStructuredOutput,
        event_stream: BedrockEventStreamLimits,
    ) -> Self {
        Self {
            region: region.into(),
            reasoning_wire_format,
            signed_reasoning,
            structured_output,
            event_stream,
            timeouts: BedrockTimeouts::default(),
            client: None,
        }
    }
}

impl std::fmt::Debug for BedrockConverseSettings {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BedrockConverseSettings")
            .field("region", &self.region)
            .field("reasoning_wire_format", &self.reasoning_wire_format)
            .field("signed_reasoning", &self.signed_reasoning)
            .field("structured_output", &self.structured_output)
            .field("event_stream", &self.event_stream)
            .field("timeouts", &self.timeouts)
            .field("client", &self.client.as_ref().map(|_| "<client>"))
            .finish()
    }
}
