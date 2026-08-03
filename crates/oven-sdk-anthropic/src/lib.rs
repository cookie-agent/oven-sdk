#![warn(missing_docs)]
// `ModelError` deliberately carries rich structured diagnostics; boxing it would make the
// public provider API inconsistent with `oven_sdk::LanguageModel` for no practical gain.
#![allow(clippy::result_large_err)]
//! Anthropic Messages API adapter for [`oven_sdk`].

/// Explicit registry-free model configuration.
pub mod config;
/// Error-envelope classification.
pub mod error;
/// Language-model implementation.
pub mod model;
/// Typed request options.
pub mod options;
/// Native replay codec.
pub mod replay;
/// Request encoding.
pub mod request;
mod signing;
/// Incremental SSE framing.
pub mod sse;
/// Stream state machine.
pub mod stream;
/// HTTP transport helpers.
pub mod transport;
/// Wire-protocol constants.
pub mod wire;

pub use config::{
    AnthropicAuth, AnthropicAwsAuth, AnthropicAwsCredentialProvider, AnthropicAwsCredentials,
    AnthropicAwsSettings, AnthropicProtocolSettings, AnthropicSettings, AnthropicThinkingSupport,
    MiniMaxAuth, MiniMaxProtocolSettings, MiniMaxSettings,
};
pub use error::classify_error;
pub use model::{AnthropicAwsModel, AnthropicModel, MiniMaxModel};
pub use options::{
    AnthropicAwsRequestExt, AnthropicAwsRequestOptions, AnthropicCacheControl, AnthropicCacheTtl,
    AnthropicRequestExt, AnthropicRequestOptions, AnthropicThinking, AnthropicToolOptions,
    MiniMaxMediaExt, MiniMaxMediaOptions, MiniMaxRequestExt, MiniMaxRequestOptions,
    MiniMaxThinking,
};
pub use transport::AnthropicTimeouts;
pub use wire::{
    ANTHROPIC_AWS_MESSAGES_ADAPTER_ID, ANTHROPIC_AWS_PROVIDER_ID, ANTHROPIC_MESSAGES_ADAPTER_ID,
    ANTHROPIC_PROVIDER_ID, MINIMAX_MESSAGES_ADAPTER_ID, MINIMAX_PROVIDER_ID,
};
