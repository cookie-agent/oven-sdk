//! Typed Azure OpenAI request options.

use oven_sdk::{CompactionRequest, Request};
use serde::{Deserialize, Serialize};

/// Azure OpenAI Chat Completions options.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AzureOpenAiChatOptions {
    /// Optional end-user identifier.
    pub user: Option<String>,
    /// Reasoning-effort label forwarded unchanged.
    pub reasoning_effort: Option<String>,
    /// Service-tier label forwarded unchanged.
    pub service_tier: Option<String>,
    /// Output-verbosity label forwarded unchanged.
    pub verbosity: Option<String>,
    /// Requests parallel tool calls when tools are present.
    pub parallel_tool_calls: Option<bool>,
}

/// Azure OpenAI Responses options.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AzureOpenAiResponsesOptions {
    /// Additional Responses `include` entries.
    #[serde(default)]
    pub include: Vec<String>,
    /// Optional end-user identifier.
    pub user: Option<String>,
    /// Service-tier label forwarded unchanged.
    pub service_tier: Option<String>,
    /// Text-verbosity label forwarded unchanged.
    pub verbosity: Option<String>,
    /// Reasoning-summary label forwarded unchanged.
    pub reasoning_summary: Option<String>,
    /// Reasoning-mode label forwarded unchanged.
    pub reasoning_mode: Option<String>,
    /// Truncation-mode label forwarded unchanged.
    pub truncation: Option<String>,
    /// Requests parallel tool calls when tools are present.
    pub parallel_tool_calls: Option<bool>,
}

/// Azure Responses V1 standalone compaction options.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AzureOpenAiCompactionOptions {
    /// Optional provider instructions for the compaction pass.
    pub instructions: Option<String>,
    /// Optional prompt-cache key.
    pub prompt_cache_key: Option<String>,
    /// Optional prompt-cache retention label.
    pub prompt_cache_retention: Option<String>,
    /// Optional forward-open service-tier label passed through unchanged.
    pub service_tier: Option<String>,
}

/// Current request-level Azure OpenAI provider-options envelope.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AzureOpenAiOptions {
    /// Chat Completions options.
    pub chat: Option<AzureOpenAiChatOptions>,
    /// Responses options.
    pub responses: Option<AzureOpenAiResponsesOptions>,
    /// Responses V1 standalone compaction options.
    pub compaction: Option<AzureOpenAiCompactionOptions>,
}

/// Adds typed Azure Responses V1 compaction options.
pub trait AzureOpenAiCompactionRequestExt {
    /// Stores options under `provider_options["azure_openai"].compaction`.
    fn with_azure_openai_compaction_options(self, options: AzureOpenAiCompactionOptions) -> Self;
}

impl AzureOpenAiCompactionRequestExt for CompactionRequest {
    fn with_azure_openai_compaction_options(
        mut self,
        options: AzureOpenAiCompactionOptions,
    ) -> Self {
        let mut azure = current_options(&self.request);
        azure.compaction = Some(options);
        if let Ok(value) = serde_json::to_value(azure) {
            self.request
                .provider_options
                .insert("azure_openai".into(), value);
        }
        self
    }
}

/// Adds typed Azure Chat options.
pub trait AzureOpenAiChatRequestExt {
    /// Stores options under `provider_options["azure_openai"].chat`.
    fn with_azure_openai_chat_options(self, options: AzureOpenAiChatOptions) -> Self;
}

impl AzureOpenAiChatRequestExt for Request {
    fn with_azure_openai_chat_options(mut self, options: AzureOpenAiChatOptions) -> Self {
        let mut azure = current_options(&self);
        azure.chat = Some(options);
        if let Ok(value) = serde_json::to_value(azure) {
            self.provider_options.insert("azure_openai".into(), value);
        }
        self
    }
}

/// Adds typed Azure Responses options.
pub trait AzureOpenAiResponsesRequestExt {
    /// Stores options under `provider_options["azure_openai"].responses`.
    fn with_azure_openai_responses_options(self, options: AzureOpenAiResponsesOptions) -> Self;
}

impl AzureOpenAiResponsesRequestExt for Request {
    fn with_azure_openai_responses_options(mut self, options: AzureOpenAiResponsesOptions) -> Self {
        let mut azure = current_options(&self);
        azure.responses = Some(options);
        if let Ok(value) = serde_json::to_value(azure) {
            self.provider_options.insert("azure_openai".into(), value);
        }
        self
    }
}

pub(crate) fn chat_options(
    request: &Request,
) -> Result<AzureOpenAiChatOptions, oven_sdk::ModelError> {
    decode(request).map(|options| options.chat.unwrap_or_default())
}

pub(crate) fn responses_options(
    request: &Request,
) -> Result<AzureOpenAiResponsesOptions, oven_sdk::ModelError> {
    decode(request).map(|options| options.responses.unwrap_or_default())
}

pub(crate) fn compaction_options(
    request: &CompactionRequest,
) -> Result<AzureOpenAiCompactionOptions, oven_sdk::ModelError> {
    decode(&request.request).map(|options| options.compaction.unwrap_or_default())
}

fn current_options(request: &Request) -> AzureOpenAiOptions {
    request
        .provider_options
        .get("azure_openai")
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .unwrap_or_default()
}

fn decode(request: &Request) -> Result<AzureOpenAiOptions, oven_sdk::ModelError> {
    request
        .provider_options
        .get("azure_openai")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| oven_sdk::ModelError::invalid_request("invalid Azure OpenAI options"))
        })
        .transpose()
        .map(Option::unwrap_or_default)
}
