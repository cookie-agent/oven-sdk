//! Typed Google Gemini request options.

use oven_sdk::{Request, ToolDefinition};
use serde::{Deserialize, Serialize};

/// Gemini thinking configuration.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleThinkingConfig {
    /// Thinking token budget for configurations that explicitly accept budgets.
    pub thinking_budget: Option<i64>,
    /// Provider-defined thinking-level label for configurations that explicitly accept levels.
    pub thinking_level: Option<String>,
    /// Whether visible thought text should be included.
    pub include_thoughts: Option<bool>,
}

/// One Gemini safety setting. Category and threshold are open provider labels.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleSafetySetting {
    /// Provider-defined harm category.
    pub category: String,
    /// Provider-defined block threshold.
    pub threshold: String,
}

/// A provider-executed Gemini tool request.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GoogleProviderTool {
    /// Google Search grounding.
    GoogleSearch,
    /// URL Context retrieval.
    UrlContext,
    /// Provider-side code execution.
    CodeExecution,
    /// Gemini File Search over existing stores.
    FileSearch {
        /// Existing File Search store resource names.
        stores: Vec<String>,
    },
    /// Google Maps grounding.
    GoogleMaps,
}

/// Typed options stored under `provider_options["google"]`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleRequestOptions {
    /// Top-k sampling value.
    pub top_k: Option<u32>,
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
    pub thinking_config: Option<GoogleThinkingConfig>,
    /// Provider-executed tools.
    #[serde(default)]
    pub provider_tools: Vec<GoogleProviderTool>,
    /// Provider-defined service tier, forwarded unchanged.
    pub service_tier: Option<String>,
    /// Existing Gemini cached-content resource.
    pub cached_content: Option<String>,
    /// Explicit safety settings.
    #[serde(default)]
    pub safety_settings: Vec<GoogleSafetySetting>,
}

/// Google-specific options for a client function declaration.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct GoogleToolOptions {
    /// Requests the provider's validated function-calling mode.
    pub strict: bool,
}

/// Adds typed Google options to a normalized request.
pub trait GoogleRequestExt {
    /// Stores options in the `google` provider namespace.
    fn with_google_options(self, options: GoogleRequestOptions) -> Self;
}

impl GoogleRequestExt for Request {
    fn with_google_options(mut self, options: GoogleRequestOptions) -> Self {
        self.provider_options.insert(
            "google".into(),
            serde_json::to_value(options).expect("Google options are serializable"),
        );
        self
    }
}

/// Adds typed Google options to a function declaration.
pub trait GoogleToolExt {
    /// Stores options in the `google` provider namespace.
    fn with_google_tool_options(self, options: GoogleToolOptions) -> Self;
}

impl GoogleToolExt for ToolDefinition {
    fn with_google_tool_options(mut self, options: GoogleToolOptions) -> Self {
        self.provider_options.insert(
            "google".into(),
            serde_json::to_value(options).expect("Google tool options are serializable"),
        );
        self
    }
}
