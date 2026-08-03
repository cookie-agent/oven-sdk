#![warn(missing_docs)]
// `ModelError` is the core contract's intentionally rich structured error.
#![allow(clippy::result_large_err)]
//! Google Vertex AI Gemini `generateContent` and `streamGenerateContent` adapter.

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
    GoogleVertexMediaSettings, GoogleVertexModel, GoogleVertexResource, GoogleVertexSettings,
    GoogleVertexThinkingMode, GoogleVertexToolSettings, VertexAuth, VertexTokenProvider,
    google_vertex_native_context_scope,
};
pub use options::{
    GoogleVertexProviderTool, GoogleVertexRequestExt, GoogleVertexRequestOptions,
    GoogleVertexSafetySetting, GoogleVertexThinkingConfig, GoogleVertexToolExt,
    GoogleVertexToolOptions,
};
pub use transport::GoogleVertexTimeouts;

/// Stable provider identity for Vertex-hosted Gemini.
pub const GOOGLE_VERTEX_PROVIDER_ID: &str = "google.vertex";
/// Stable replay and adapter identity for Vertex Gemini `generateContent`.
pub const GOOGLE_VERTEX_GENERATE_CONTENT_ADAPTER_ID: &str = "oven.google.vertex.generate-content";

const REPLAY_FORMAT: &str = "oven.google.vertex.generate-content.assistant.v4";
