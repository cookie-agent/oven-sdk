//! Bedrock HTTP transport helpers.

use std::time::Duration;

use oven_sdk::{AbortSignal, ErrorStage, ModelError, ResponseHead, ResponseMetadata};

/// Per-phase Bedrock transport and credential timeouts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BedrockTimeouts {
    /// Connection timeout for adapter-created clients.
    pub connect: Duration,
    /// Maximum wait for response headers.
    pub headers: Duration,
    /// Maximum wait for credential resolution.
    pub credentials: Duration,
    /// Maximum inactivity while reading a stream or error body.
    pub stream_idle: Duration,
}

impl Default for BedrockTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(60),
            headers: Duration::from_secs(300),
            credentials: Duration::from_secs(30),
            stream_idle: Duration::from_secs(300),
        }
    }
}

pub(crate) fn response_head(response: &reqwest::Response) -> ResponseHead {
    ResponseHead {
        http_status: Some(response.status().as_u16()),
        request_id: response
            .headers()
            .get("x-amzn-requestid")
            .or_else(|| response.headers().get("x-amzn-request-id"))
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned),
        response_metadata: ResponseMetadata::new(),
    }
}

pub(crate) async fn read_body(
    response: reqwest::Response,
    abort: &AbortSignal,
    idle: Duration,
    cap: usize,
) -> Result<(Vec<u8>, u64), ModelError> {
    oven_sdk::provider_support::read_bounded_body(
        response.bytes_stream(),
        abort,
        oven_sdk::provider_support::BodyReadConfig {
            cap,
            limit: oven_sdk::provider_support::BodyLimit::Truncate,
            stage: ErrorStage::ResponseBody,
            timeout_message: "Bedrock response body idle timeout",
            abort_message: "request was aborted while reading the Bedrock response body",
            read_message: "Bedrock response body read failed",
            overflow_message: "Bedrock response byte count overflowed",
        },
        tokio::time::sleep(idle),
        move |timer| timer.reset(tokio::time::Instant::now() + idle),
    )
    .await
}
