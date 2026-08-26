#![warn(missing_docs)]
// `ModelError` is intentionally rich and returned directly by public helpers.
#![allow(clippy::result_large_err)]
//! Amazon Bedrock `Converse` and `ConverseStream` adapter for oven-sdk.
//!
//! This crate uses the Bedrock Runtime wire protocol directly. It does not use
//! Anthropic's AWS Messages endpoint and it does not perform hidden retries.

mod configuration;
mod error;
/// Incremental AWS EventStream framing and CRC validation.
pub mod eventstream;
mod model;
mod options;
mod request;
/// Exact-byte AWS Signature Version 4 signing helpers.
pub mod sigv4;
mod stream;
mod transport;

pub use configuration::{
    AwsCredentials, AwsCredentialsProvider, BedrockAuth, BedrockConverseSettings,
    BedrockEventStreamLimits, BedrockReasoningWireFormat, BedrockStructuredOutput,
};
pub use error::classify_error;
pub use model::BedrockModel;
pub use options::{
    BedrockCachePoint, BedrockCacheStrategy, BedrockCacheTtl, BedrockGuardrailConfig,
    BedrockMessageCachePoint, BedrockRequestExt, BedrockRequestOptions, BedrockS3LocationOptions,
};
pub use transport::BedrockTimeouts;

/// Stable provider identity for Amazon Bedrock Runtime.
pub const BEDROCK_PROVIDER_ID: &str = "amazon.bedrock";
/// Stable replay and adapter identity for Bedrock Converse.
pub const BEDROCK_CONVERSE_ADAPTER_ID: &str = "oven.bedrock.converse";

const REPLAY_FORMAT: &str = "oven.bedrock.converse.assistant.v2";
