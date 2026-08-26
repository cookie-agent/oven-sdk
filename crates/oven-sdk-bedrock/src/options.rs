//! Typed Bedrock request options.

use std::collections::BTreeMap;

use oven_sdk::{JsonValue, Request};
use serde::{Deserialize, Serialize};

/// Bedrock guardrail request configuration.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockGuardrailConfig {
    /// Guardrail identifier.
    pub guardrail_identifier: String,
    /// Guardrail version.
    pub guardrail_version: String,
    /// Optional provider-defined trace label, forwarded unchanged.
    pub trace: Option<String>,
    /// Optional ConverseStream-only processing mode, forwarded unchanged.
    pub stream_processing_mode: Option<String>,
}

/// Optional S3 location fields attached to every encoded S3 media source.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockS3LocationOptions {
    /// Expected owning AWS account for cross-account objects.
    pub bucket_owner: Option<String>,
}

/// TTL for one Bedrock Converse cache point.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum BedrockCacheTtl {
    /// Use the standard five-minute cache lifetime.
    #[serde(rename = "5m")]
    FiveMinutes,
    /// Request the extended one-hour cache lifetime on models that support it.
    #[serde(rename = "1h")]
    OneHour,
}

/// One Bedrock Converse cache point.
///
/// Bedrock's minimum cacheable prefix varies by model, currently from 512 to
/// 4,096 tokens. This adapter does not estimate token counts or reject short
/// prefixes; Bedrock accepts those requests but does not write a cache entry.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockCachePoint {
    /// Optional cache TTL. Omission uses Bedrock's default five-minute lifetime.
    pub ttl: Option<BedrockCacheTtl>,
}

/// Cache point appended to the message produced by one normalized history turn.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockMessageCachePoint {
    /// Zero-based index in [`oven_sdk::Request::history`].
    pub history_index: usize,
    /// Cache point settings.
    #[serde(flatten)]
    pub cache_point: BedrockCachePoint,
}

/// Explicit Bedrock Converse cache-point placement strategy.
///
/// Current Converse models with manual cache points allow at most four points
/// per request. The adapter enforces that shared request ceiling without
/// inferring behavior from the model ID.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockCacheStrategy {
    /// Cache point appended after top-level system content.
    pub system: Option<BedrockCachePoint>,
    /// Cache point appended after tool specifications.
    pub tools: Option<BedrockCachePoint>,
    /// Cache points appended after selected normalized history turns.
    #[serde(default)]
    pub messages: Vec<BedrockMessageCachePoint>,
}

/// Typed options stored under `provider_options["bedrock"]`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockRequestOptions {
    /// Additional model-native request fields.
    ///
    /// Reasoning and output structural keys are reserved for typed controls.
    pub additional_model_request_fields: Option<JsonValue>,
    /// JSON Pointer paths for additional model response fields.
    #[serde(default)]
    pub additional_model_response_field_paths: Vec<String>,
    /// Provider-defined service-tier label, forwarded unchanged.
    pub service_tier: Option<String>,
    /// Provider-defined performance latency label, forwarded unchanged.
    pub performance_latency: Option<String>,
    /// Bedrock invocation-log request metadata.
    #[serde(default)]
    pub request_metadata: BTreeMap<String, String>,
    /// Guardrail configuration.
    pub guardrail: Option<BedrockGuardrailConfig>,
    /// S3 location options.
    pub s3: Option<BedrockS3LocationOptions>,
    /// Explicit cache-point placement strategy.
    pub cache: Option<BedrockCacheStrategy>,
    /// Additional stop sequences.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Provider-defined reasoning type label, forwarded unchanged.
    pub reasoning_type: Option<String>,
    /// Optional reasoning token budget.
    pub reasoning_budget_tokens: Option<u64>,
    /// Provider-defined reasoning display label, forwarded unchanged.
    pub reasoning_display: Option<String>,
    /// Provider-defined maximum reasoning effort label, forwarded unchanged.
    pub max_reasoning_effort: Option<String>,
}

/// Adds typed Bedrock options to a normalized request.
pub trait BedrockRequestExt {
    /// Stores options in the `bedrock` namespace.
    fn with_bedrock_options(self, options: BedrockRequestOptions) -> Self;
}

impl BedrockRequestExt for Request {
    fn with_bedrock_options(mut self, options: BedrockRequestOptions) -> Self {
        self.provider_options.insert(
            "bedrock".into(),
            serde_json::to_value(options).expect("Bedrock options are serializable"),
        );
        self
    }
}
