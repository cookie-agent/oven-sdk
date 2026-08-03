//! Typed Google Vertex request options.

use oven_sdk::{Request, ToolDefinition};
use serde::{Deserialize, Serialize};

/// Vertex Gemini thinking configuration.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleVertexThinkingConfig {
    /// Explicit Vertex thinking token budget when enabled by model settings.
    pub thinking_budget: Option<i64>,
    /// Provider-defined Vertex thinking-level label, forwarded unchanged.
    pub thinking_level: Option<String>,
    /// Whether visible thought text should be included.
    pub include_thoughts: Option<bool>,
}

/// One Vertex safety setting with open provider labels.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleVertexSafetySetting {
    /// Provider-defined harm category.
    pub category: String,
    /// Provider-defined block threshold.
    pub threshold: String,
    /// Optional provider-defined evaluation method.
    pub method: Option<String>,
}

/// A provider-executed Vertex Gemini tool request.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GoogleVertexProviderTool {
    /// Google Search grounding.
    GoogleSearch,
    /// URL Context retrieval.
    UrlContext,
    /// Provider-side code execution.
    CodeExecution,
    /// Vertex RAG Store retrieval.
    VertexRagStore {
        /// RAG corpus resource name.
        rag_corpus: String,
        /// Optional nearest-neighbor count.
        top_k: Option<u32>,
    },
    /// Google Maps grounding.
    GoogleMaps,
}

/// Typed options stored under `provider_options["google_vertex"]`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleVertexRequestOptions {
    /// Top-k sampling value.
    pub top_k: Option<f64>,
    /// Additional stop sequences.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Deterministic sampling seed.
    pub seed: Option<i32>,
    /// Presence penalty.
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Thinking controls.
    pub thinking_config: Option<GoogleVertexThinkingConfig>,
    /// Provider-executed tools.
    #[serde(default)]
    pub provider_tools: Vec<GoogleVertexProviderTool>,
    /// Existing Vertex cached-content resource.
    pub cached_content: Option<String>,
    /// Explicit safety settings.
    #[serde(default)]
    pub safety_settings: Vec<GoogleVertexSafetySetting>,
    /// Provider-defined shared request admission label, forwarded as a header.
    pub shared_request_type: Option<String>,
    /// Provider-defined request type label, forwarded as a header.
    pub request_type: Option<String>,
}

/// Vertex-specific options for a client function declaration.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct GoogleVertexToolOptions {
    /// Requests the provider's validated function-calling mode.
    pub strict: bool,
}

/// Adds typed Vertex options to a normalized request.
pub trait GoogleVertexRequestExt {
    /// Stores options in the `google_vertex` namespace.
    fn with_google_vertex_options(self, options: GoogleVertexRequestOptions) -> Self;
}

impl GoogleVertexRequestExt for Request {
    fn with_google_vertex_options(mut self, options: GoogleVertexRequestOptions) -> Self {
        self.provider_options.insert(
            "google_vertex".into(),
            serde_json::to_value(options).expect("Vertex options are serializable"),
        );
        self
    }
}

/// Adds typed Vertex options to a function declaration.
pub trait GoogleVertexToolExt {
    /// Stores options in the `google_vertex` namespace.
    fn with_google_vertex_tool_options(self, options: GoogleVertexToolOptions) -> Self;
}

impl GoogleVertexToolExt for ToolDefinition {
    fn with_google_vertex_tool_options(mut self, options: GoogleVertexToolOptions) -> Self {
        self.provider_options.insert(
            "google_vertex".into(),
            serde_json::to_value(options).expect("Vertex tool options are serializable"),
        );
        self
    }
}
