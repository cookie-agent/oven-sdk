//! Typed provider options stored in normalized request namespaces.

use oven_sdk::{CompactionRequest, Request};
use serde::{Deserialize, Serialize};

/// Official Chat Completions request options.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct OpenAiChatOptions {
    /// Optional OpenAI end-user identifier.
    pub user: Option<String>,
    /// Optional reasoning effort accepted by applicable OpenAI models.
    pub reasoning_effort: Option<String>,
    /// Optional OpenAI service-tier label, forwarded unchanged.
    pub service_tier: Option<String>,
    /// Optional OpenAI output-verbosity label, forwarded unchanged.
    pub verbosity: Option<String>,
    /// Requests parallel tool calls when tools are present.
    pub parallel_tool_calls: Option<bool>,
    /// Optional prompt-cache routing key, limited to 64 characters by OpenAI.
    pub prompt_cache_key: Option<String>,
    /// Optional prompt-cache retention policy.
    pub prompt_cache_retention: Option<OpenAiPromptCacheRetention>,
}

/// Official Responses request options.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct OpenAiResponsesOptions {
    /// Additional Responses `include` entries. Encrypted reasoning is always
    /// deduplicated into this list.
    #[serde(default)]
    pub include: Vec<String>,
    /// Optional OpenAI end-user identifier.
    pub user: Option<String>,
    /// Optional OpenAI service-tier label, forwarded unchanged.
    pub service_tier: Option<String>,
    /// Optional OpenAI text-verbosity label, forwarded unchanged.
    pub verbosity: Option<String>,
    /// Optional OpenAI reasoning-summary mode, forwarded unchanged as
    /// `reasoning.summary`.
    pub reasoning_summary: Option<String>,
    /// Optional OpenAI reasoning mode, forwarded unchanged as
    /// `reasoning.mode`.
    pub reasoning_mode: Option<String>,
    /// Optional OpenAI truncation-mode label, forwarded unchanged.
    pub truncation: Option<String>,
    /// Requests parallel tool calls when tools are present.
    pub parallel_tool_calls: Option<bool>,
    /// Optional prompt-cache routing key, limited to 64 characters by OpenAI.
    pub prompt_cache_key: Option<String>,
    /// Optional prompt-cache retention policy.
    pub prompt_cache_retention: Option<OpenAiPromptCacheRetention>,
}

/// OpenAI prompt-cache retention policy.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiPromptCacheRetention {
    /// Keep cache entries in memory for the standard short-lived retention window.
    InMemory,
    /// Enable extended retention for up to 24 hours.
    #[serde(rename = "24h")]
    TwentyFourHours,
}

/// Official standalone Responses compaction options.
///
/// OpenAI-defined labels are intentionally open strings so newly introduced
/// values can be used without a crate release.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiResponsesCompactionOptions {
    /// Optional provider instructions for the compaction pass.
    pub instructions: Option<String>,
    /// Optional prompt-cache routing key, forwarded unchanged.
    pub prompt_cache_key: Option<String>,
    /// Optional prompt-cache controls.
    pub prompt_cache_options: Option<OpenAiPromptCacheOptions>,
    /// Optional prompt-cache retention label, forwarded unchanged.
    pub prompt_cache_retention: Option<String>,
    /// Optional service-tier label, forwarded unchanged.
    pub service_tier: Option<String>,
}

/// Prompt-cache controls accepted by standalone Responses compaction.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiPromptCacheOptions {
    /// Cache breakpoint mode.
    pub mode: String,
    /// Cache TTL.
    pub ttl: String,
}

/// Current request-level official OpenAI provider-options envelope.
///
/// The endpoint-specific option structs are nested under the single
/// `provider_options["openai"]` namespace.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct OpenAiOptions {
    /// Chat Completions options, when configured.
    pub chat: Option<OpenAiChatOptions>,
    /// Responses options, when configured.
    pub responses: Option<OpenAiResponsesOptions>,
    /// Standalone Responses compaction options, when configured.
    pub compaction: Option<OpenAiResponsesCompactionOptions>,
}

/// Compatible Chat request options passed through after normalized fields.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct CompatibleChatOptions {
    /// Additional endpoint-specific request fields.
    #[serde(default)]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

/// Adds typed official Chat options.
pub trait OpenAiChatRequestExt {
    /// Stores Chat options under `provider_options["openai"].chat`.
    fn with_openai_chat_options(self, options: OpenAiChatOptions) -> Self;
    /// Stores compatible options under `provider_options["openai_compatible"]`.
    fn with_compatible_chat_options(self, options: CompatibleChatOptions) -> Self;
}

impl OpenAiChatRequestExt for Request {
    fn with_openai_chat_options(mut self, options: OpenAiChatOptions) -> Self {
        let mut openai = current_openai_options(&self);
        openai.chat = Some(options);
        self.provider_options.insert(
            "openai".into(),
            serde_json::to_value(openai).expect("OpenAI options are serializable"),
        );
        self
    }

    fn with_compatible_chat_options(mut self, options: CompatibleChatOptions) -> Self {
        self.provider_options.insert(
            "openai_compatible".into(),
            serde_json::to_value(options).expect("compatible Chat options are serializable"),
        );
        self
    }
}

/// Adds typed official Responses options.
pub trait OpenAiResponsesRequestExt {
    /// Stores Responses options under `provider_options["openai"].responses`.
    fn with_openai_responses_options(self, options: OpenAiResponsesOptions) -> Self;
}

impl OpenAiResponsesRequestExt for Request {
    fn with_openai_responses_options(mut self, options: OpenAiResponsesOptions) -> Self {
        let mut openai = current_openai_options(&self);
        openai.responses = Some(options);
        self.provider_options.insert(
            "openai".into(),
            serde_json::to_value(openai).expect("OpenAI options are serializable"),
        );
        self
    }
}

/// Adds typed official Responses standalone-compaction options.
pub trait OpenAiResponsesCompactionRequestExt {
    /// Stores options under `provider_options["openai"].compaction`.
    fn with_openai_responses_compaction_options(
        self,
        options: OpenAiResponsesCompactionOptions,
    ) -> Self;
}

impl OpenAiResponsesCompactionRequestExt for CompactionRequest {
    fn with_openai_responses_compaction_options(
        mut self,
        options: OpenAiResponsesCompactionOptions,
    ) -> Self {
        let mut openai = current_openai_options(&self.request);
        openai.compaction = Some(options);
        self.request.provider_options.insert(
            "openai".into(),
            serde_json::to_value(openai).expect("OpenAI options are serializable"),
        );
        self
    }
}

pub(crate) fn chat_options(request: &Request) -> Result<OpenAiChatOptions, oven_sdk::ModelError> {
    decode_openai(request).map(|options| options.chat.unwrap_or_default())
}

pub(crate) fn response_pipeline_options(
    request: &Request,
) -> Result<(OpenAiResponsesOptions, OpenAiResponsesCompactionOptions), oven_sdk::ModelError> {
    decode_openai(request).map(|options| {
        (
            options.responses.unwrap_or_default(),
            options.compaction.unwrap_or_default(),
        )
    })
}

pub(crate) fn compatible_options(
    request: &Request,
) -> Result<CompatibleChatOptions, oven_sdk::ModelError> {
    let options: CompatibleChatOptions = decode(
        request,
        "openai_compatible",
        "invalid compatible Chat request options",
    )?;
    const RESERVED: &[&str] = &[
        "model",
        "messages",
        "stream",
        "stream_options",
        "n",
        "modalities",
        "audio",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "reasoning_effort",
        "reasoning",
        "user",
        "service_tier",
        "verbosity",
        "parallel_tool_calls",
        "prompt_cache_key",
        "prompt_cache_retention",
        "tools",
        "tool_choice",
        "response_format",
    ];
    if let Some(key) = options
        .extra_body
        .keys()
        .find(|key| RESERVED.contains(&key.as_str()))
    {
        return Err(oven_sdk::ModelError::invalid_request(format!(
            "compatible extra_body key `{key}` is reserved"
        )));
    }
    Ok(options)
}

pub(crate) fn validate_prompt_cache_key(key: Option<&str>) -> Result<(), oven_sdk::ModelError> {
    if key.is_some_and(|key| key.chars().count() > 64) {
        return Err(oven_sdk::ModelError::invalid_request(
            "OpenAI prompt_cache_key must not exceed 64 characters",
        ));
    }
    Ok(())
}

fn decode<T: serde::de::DeserializeOwned + Default>(
    request: &Request,
    key: &str,
    message: &str,
) -> Result<T, oven_sdk::ModelError> {
    request
        .provider_options
        .get(key)
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| oven_sdk::ModelError::invalid_request(message))
        })
        .transpose()
        .map(Option::unwrap_or_default)
}

fn current_openai_options(request: &Request) -> OpenAiOptions {
    request
        .provider_options
        .get("openai")
        .map(|value| {
            serde_json::from_value(value.clone())
                .expect("existing provider_options[\"openai\"] must use OpenAiOptions")
        })
        .unwrap_or_default()
}

fn decode_openai(request: &Request) -> Result<OpenAiOptions, oven_sdk::ModelError> {
    decode(request, "openai", "invalid OpenAI request options")
}
