#![warn(missing_docs)]
// `ModelError` is the core contract's intentionally rich structured error.
#![allow(clippy::result_large_err)]
//! Google Gemini API `generateContent` and `streamGenerateContent` adapter.

mod error;
mod model;
mod options;
mod request;
/// Incremental SSE framing utilities.
pub mod sse;
mod stream;
mod transport;

pub use error::classify_error;
pub use model::{
    GoogleApiKeyAuth, GoogleGenerateContentSettings, GoogleModel, GoogleThinkingSettings,
    GoogleToolSettings,
};
pub use options::{
    GoogleProviderTool, GoogleRequestExt, GoogleRequestOptions, GoogleSafetySetting,
    GoogleThinkingConfig, GoogleToolExt, GoogleToolOptions,
};
pub use transport::GoogleTimeouts;

/// Stable provider identity for Google AI Studio Gemini requests.
pub const GOOGLE_PROVIDER_ID: &str = "google";
/// Stable replay and adapter identity for Gemini `generateContent`.
pub const GOOGLE_GENERATE_CONTENT_ADAPTER_ID: &str = "oven.google.generate-content";
