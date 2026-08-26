//! Typed Azure OpenAI request options.

use oven_sdk::{CompactionRequest, FilePart, PartMetadata, Request, TextPart};
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
    /// Optional prompt-cache routing key, limited to 64 characters.
    pub prompt_cache_key: Option<String>,
    /// Optional prompt-cache retention policy.
    pub prompt_cache_retention: Option<AzureOpenAiPromptCacheRetention>,
    /// GPT-5.6+ prompt-cache mode and TTL controls.
    pub prompt_cache_options: Option<AzureOpenAiPromptCacheOptions>,
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
    /// Optional prompt-cache routing key, limited to 64 characters.
    pub prompt_cache_key: Option<String>,
    /// Optional prompt-cache retention policy.
    pub prompt_cache_retention: Option<AzureOpenAiPromptCacheRetention>,
    /// GPT-5.6+ prompt-cache mode and TTL controls.
    pub prompt_cache_options: Option<AzureOpenAiPromptCacheOptions>,
}

/// Azure OpenAI prompt-cache retention policy.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AzureOpenAiPromptCacheRetention {
    /// Keep cache entries in memory for the standard short-lived retention window.
    InMemory,
    /// Enable extended retention for up to 24 hours.
    #[serde(rename = "24h")]
    TwentyFourHours,
}

/// GPT-5.6+ Azure OpenAI prompt-cache controls.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AzureOpenAiPromptCacheOptions {
    /// Cache breakpoint mode.
    pub mode: AzureOpenAiPromptCacheMode,
    /// Cache TTL. The API currently accepts only 30 minutes.
    pub ttl: AzureOpenAiPromptCacheTtl,
}

/// GPT-5.6+ Azure OpenAI prompt-cache breakpoint mode.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AzureOpenAiPromptCacheMode {
    /// Add the service-managed latest-message breakpoint in addition to explicit markers.
    Implicit,
    /// Use only explicitly marked content blocks.
    Explicit,
}

/// GPT-5.6+ Azure OpenAI prompt-cache TTL.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum AzureOpenAiPromptCacheTtl {
    /// Retain written cache entries for at least 30 minutes.
    #[serde(rename = "30m")]
    ThirtyMinutes,
}

const PROMPT_CACHE_BREAKPOINT_METADATA: &str = "azure_openai.prompt_cache_breakpoint";

/// Marks normalized content for an explicit GPT-5.6+ prompt-cache write.
///
/// The API accepts at most four cache writes per request and requires at least
/// 1,024 tokens before each breakpoint. The adapter enforces the write count
/// but does not estimate tokens. Cache reads may consider the latest 50
/// breakpoints; that provider-side read window is not a request validation limit.
pub trait AzureOpenAiPromptCacheBreakpointExt {
    /// Adds an explicit prompt-cache breakpoint to this content part.
    fn with_azure_openai_prompt_cache_breakpoint(self) -> Self;
}

impl AzureOpenAiPromptCacheBreakpointExt for TextPart {
    fn with_azure_openai_prompt_cache_breakpoint(mut self) -> Self {
        set_prompt_cache_breakpoint(&mut self.metadata);
        self
    }
}

impl AzureOpenAiPromptCacheBreakpointExt for FilePart {
    fn with_azure_openai_prompt_cache_breakpoint(mut self) -> Self {
        set_prompt_cache_breakpoint(&mut self.metadata);
        self
    }
}

fn set_prompt_cache_breakpoint(metadata: &mut PartMetadata) {
    metadata
        .get_or_insert_default()
        .insert(PROMPT_CACHE_BREAKPOINT_METADATA.into(), true.into());
}

pub(crate) fn prompt_cache_breakpoint(
    metadata: &PartMetadata,
) -> Result<bool, oven_sdk::ModelError> {
    match metadata
        .as_ref()
        .and_then(|metadata| metadata.get(PROMPT_CACHE_BREAKPOINT_METADATA))
    {
        None => Ok(false),
        Some(serde_json::Value::Bool(true)) => Ok(true),
        Some(_) => Err(oven_sdk::ModelError::invalid_request(
            "invalid Azure OpenAI prompt-cache breakpoint metadata",
        )),
    }
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

pub(crate) fn response_pipeline_options(
    request: &Request,
) -> Result<(AzureOpenAiResponsesOptions, AzureOpenAiCompactionOptions), oven_sdk::ModelError> {
    decode(request).map(|options| {
        (
            options.responses.unwrap_or_default(),
            options.compaction.unwrap_or_default(),
        )
    })
}

pub(crate) fn validate_prompt_cache_key(key: Option<&str>) -> Result<(), oven_sdk::ModelError> {
    if key.is_some_and(|key| key.chars().count() > 64) {
        return Err(oven_sdk::ModelError::invalid_request(
            "Azure OpenAI prompt_cache_key must not exceed 64 characters",
        ));
    }
    Ok(())
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
