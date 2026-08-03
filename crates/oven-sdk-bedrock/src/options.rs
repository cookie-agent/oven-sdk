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
