//! Typed Anthropic request options.

use oven_sdk::{FilePart, Request};
use serde::{Deserialize, Serialize};

/// Anthropic thinking mode.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicThinking {
    /// Do not request thinking.
    Disabled,
    /// Extended thinking with its token budget.
    Enabled {
        /// Thinking budget.
        budget_tokens: u64,
        /// Provider-defined thinking display label, forwarded unchanged.
        display: Option<String>,
    },
    /// Adaptive thinking.
    Adaptive {
        /// Provider-defined thinking display label, forwarded unchanged.
        display: Option<String>,
    },
}

/// Prompt-cache lifetime.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicCacheTtl {
    /// Five-minute cache entry.
    FiveMinutes,
    /// One-hour cache entry.
    OneHour,
}

/// Anthropic cache-control marker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AnthropicCacheControl {
    /// Cache lifetime.
    pub ttl: AnthropicCacheTtl,
}

/// Anthropic options for a client-defined tool.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct AnthropicToolOptions {
    /// Requests Anthropic strict tool-input validation.
    pub strict: bool,
}

/// Typed Anthropic request options stored under `provider_options["anthropic"]`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct AnthropicRequestOptions {
    /// Thinking configuration.
    pub thinking: Option<AnthropicThinking>,
    /// Provider-defined effort label emitted as `output_config.effort`.
    pub effort: Option<String>,
    /// Automatic cache marker.
    pub cache_control: Option<AnthropicCacheControl>,
    /// Optional Anthropic user identifier.
    pub user_id: Option<String>,
    /// Additional requested beta identifiers.
    #[serde(default)]
    pub betas: Vec<String>,
}

/// Adds typed Anthropic options to a normalized request.
pub trait AnthropicRequestExt {
    /// Stores `options` in the Anthropic request namespace.
    fn with_anthropic_options(self, options: AnthropicRequestOptions) -> Self;
}
impl AnthropicRequestExt for Request {
    fn with_anthropic_options(mut self, options: AnthropicRequestOptions) -> Self {
        self.provider_options.insert(
            "anthropic".into(),
            serde_json::to_value(options).expect("Anthropic options are serializable"),
        );
        self
    }
}

/// MiniMax thinking wire shape.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxThinking {
    /// Requests adaptive thinking output.
    Adaptive,
    /// Requests disabled thinking where the explicit protocol settings permit it.
    Disabled,
}

/// MiniMax request options stored under `provider_options["minimax"]`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct MiniMaxRequestOptions {
    /// Thinking configuration.
    pub thinking: Option<MiniMaxThinking>,
    /// Provider-defined request admission tier, forwarded unchanged.
    pub service_tier: Option<String>,
    /// Optional MiniMax end-user identifier.
    pub user_id: Option<String>,
}

/// MiniMax media-source options stored in part metadata.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct MiniMaxMediaOptions {
    /// Provider-defined understanding detail label, forwarded unchanged.
    pub detail: Option<String>,
    /// Video frame sampling rate.
    pub fps: Option<f64>,
    /// Optional longest-side pixel limit.
    pub max_long_side_pixel: Option<u64>,
}

/// Adds typed MiniMax media options to a normalized file part.
pub trait MiniMaxMediaExt {
    /// Stores media-source options in the MiniMax metadata namespace.
    fn with_minimax_media_options(self, options: MiniMaxMediaOptions) -> Self;
}

impl MiniMaxMediaExt for FilePart {
    fn with_minimax_media_options(mut self, options: MiniMaxMediaOptions) -> Self {
        self.metadata.get_or_insert_default().insert(
            "minimax".into(),
            serde_json::to_value(options).expect("MiniMax media options are serializable"),
        );
        self
    }
}

/// Adds typed MiniMax options to normalized values.
pub trait MiniMaxRequestExt {
    /// Stores request options in the MiniMax namespace.
    fn with_minimax_options(self, options: MiniMaxRequestOptions) -> Self;
}

impl MiniMaxRequestExt for Request {
    fn with_minimax_options(mut self, options: MiniMaxRequestOptions) -> Self {
        self.provider_options.insert(
            "minimax".into(),
            serde_json::to_value(options).expect("MiniMax options are serializable"),
        );
        self
    }
}

/// Claude Platform on AWS request options stored under
/// `provider_options["anthropic_aws"]`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct AnthropicAwsRequestOptions {
    /// Provider-defined inference geography label, forwarded unchanged.
    pub inference_geo: Option<String>,
}

/// Adds typed Claude Platform on AWS options to a normalized request.
pub trait AnthropicAwsRequestExt {
    /// Stores request options in the Anthropic AWS namespace.
    fn with_anthropic_aws_options(self, options: AnthropicAwsRequestOptions) -> Self;
}

impl AnthropicAwsRequestExt for Request {
    fn with_anthropic_aws_options(mut self, options: AnthropicAwsRequestOptions) -> Self {
        self.provider_options.insert(
            "anthropic_aws".into(),
            serde_json::to_value(options).expect("Anthropic AWS options are serializable"),
        );
        self
    }
}
