//! Explicit registry-free configuration for Anthropic-compatible models.

use std::{collections::BTreeSet, fmt, sync::Arc};

use oven_sdk::{AdapterId, BoxFuture, ModelError, ResourceId, SecretString};
use reqwest::Client;

use crate::transport::AnthropicTimeouts;

/// Authentication for the direct Anthropic Messages API.
#[derive(Clone, Debug)]
pub enum AnthropicAuth {
    /// No adapter-injected authentication. Caller headers may provide authentication.
    None,
    /// Anthropic API key sent as `x-api-key` unless caller headers already authenticate.
    ApiKey(SecretString),
}

/// Authentication for a caller-selected Anthropic Messages-compatible endpoint.
#[derive(Clone, Debug)]
pub enum AnthropicCompatibleAuth {
    /// Anthropic-style API key sent as `x-api-key` unless caller headers already authenticate.
    ApiKey(SecretString),
    /// Bearer token sent as `Authorization: Bearer ...` unless caller headers already authenticate.
    Bearer(SecretString),
    /// No adapter-injected authentication. Caller headers may provide authentication.
    None,
}

/// Authentication for the MiniMax Anthropic-compatible Messages API.
#[derive(Clone, Debug)]
pub enum MiniMaxAuth {
    /// No adapter-injected authentication. Caller headers may provide authentication.
    None,
    /// MiniMax API key sent as bearer authentication unless caller headers already authenticate.
    Bearer(SecretString),
}

/// AWS credentials used to sign Claude Platform on AWS requests.
#[derive(Clone)]
pub struct AnthropicAwsCredentials {
    /// AWS access key ID.
    pub access_key_id: String,
    /// AWS secret access key.
    pub secret_access_key: SecretString,
    /// Optional temporary-credential session token.
    pub session_token: Option<SecretString>,
}

impl fmt::Debug for AnthropicAwsCredentials {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicAwsCredentials")
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
pub type AnthropicAwsCredentialProvider =
    Arc<dyn Fn() -> BoxFuture<'static, Result<AnthropicAwsCredentials, ModelError>> + Send + Sync>;

/// Authentication for Claude Platform on AWS.
#[derive(Clone)]
pub enum AnthropicAwsAuth {
    /// AWS-provisioned platform key sent as `x-api-key`.
    BearerKey(SecretString),
    /// Static AWS credentials used for SigV4.
    StaticCredentials(AnthropicAwsCredentials),
    /// Caller-managed credentials resolved once per request and used for SigV4.
    CredentialProvider(AnthropicAwsCredentialProvider),
}

impl fmt::Debug for AnthropicAwsAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::BearerKey(_) => "AnthropicAwsAuth::BearerKey(<redacted>)",
            Self::StaticCredentials(_) => "AnthropicAwsAuth::StaticCredentials(<redacted>)",
            Self::CredentialProvider(_) => "AnthropicAwsAuth::CredentialProvider(<provider>)",
        })
    }
}

/// Anthropic thinking modes accepted by one explicitly configured model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnthropicThinkingSupport {
    /// Thinking controls are unsupported.
    None,
    /// Manual extended thinking is supported.
    Extended,
    /// Adaptive thinking is supported.
    Adaptive,
    /// Both manual extended and adaptive thinking are supported.
    Both,
}

/// Explicit direct-Anthropic and Anthropic-AWS protocol behavior.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnthropicProtocolSettings {
    /// Supported thinking controls.
    pub thinking: AnthropicThinkingSupport,
    /// Whether omitted thinking controls mean thinking is active.
    pub thinking_default_active: bool,
    /// Whether an explicit disabled thinking mode is accepted.
    pub thinking_disable_allowed: bool,
    /// Open effort labels that cannot be combined with disabled thinking.
    pub thinking_disable_forbidden_efforts: BTreeSet<String>,
    /// Whether `output_config.effort` is supported.
    pub effort: bool,
    /// Whether a final assistant message may be used as a prefill.
    pub assistant_prefill: bool,
    /// Whether non-default temperature or top-p values are rejected for every request.
    pub reject_non_default_sampling: bool,
}

/// Explicit MiniMax Messages protocol behavior.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MiniMaxProtocolSettings {
    /// Whether MiniMax thinking controls are supported.
    pub thinking: bool,
    /// Whether MiniMax thinking may be disabled.
    pub thinking_disable_allowed: bool,
}

/// Adapter settings for one direct Anthropic model.
#[derive(Clone, Debug)]
pub struct AnthropicSettings {
    /// HTTP client used for requests.
    pub client: Client,
    /// Per-phase request timeouts.
    pub timeouts: AnthropicTimeouts,
    /// Explicit protocol behavior.
    pub protocol: AnthropicProtocolSettings,
    /// Optional caller discriminator combined into the internally derived native-context scope.
    pub native_context_discriminator: Option<ResourceId>,
}

/// Adapter settings for one caller-selected Anthropic Messages-compatible model.
#[derive(Clone, Debug)]
pub struct AnthropicCompatibleSettings {
    /// Caller-owned stable adapter identity.
    pub adapter_id: AdapterId,
    /// HTTP client used for requests.
    pub client: Client,
    /// Per-phase request timeouts.
    pub timeouts: AnthropicTimeouts,
    /// Explicit Anthropic Messages protocol behavior.
    pub protocol: AnthropicProtocolSettings,
    /// Optional caller discriminator combined into the internally derived native-context scope.
    pub native_context_discriminator: Option<ResourceId>,
}

/// Adapter settings for one MiniMax model.
#[derive(Clone, Debug)]
pub struct MiniMaxSettings {
    /// HTTP client used for requests.
    pub client: Client,
    /// Per-phase request timeouts.
    pub timeouts: AnthropicTimeouts,
    /// Explicit protocol behavior.
    pub protocol: MiniMaxProtocolSettings,
    /// Optional caller discriminator combined into the internally derived native-context scope.
    pub native_context_discriminator: Option<ResourceId>,
}

/// Adapter settings for one Claude Platform on AWS model.
#[derive(Clone, Debug)]
pub struct AnthropicAwsSettings {
    /// HTTP client used for requests.
    pub client: Client,
    /// Per-phase request timeouts.
    pub timeouts: AnthropicTimeouts,
    /// Explicit Claude protocol behavior.
    pub protocol: AnthropicProtocolSettings,
    /// AWS region used by SigV4.
    pub region: String,
    /// Claude Platform workspace ID sent on every request.
    pub workspace_id: String,
    /// Optional caller discriminator combined into the internally derived native-context scope.
    pub native_context_discriminator: Option<ResourceId>,
}
