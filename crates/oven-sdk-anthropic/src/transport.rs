//! HTTP transport configuration and response helpers.

use std::time::Duration;

use oven_sdk::{AbortSignal, ErrorStage, ModelError, ResponseHead, ResponseMetadata};

/// Per-phase adapter timeouts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnthropicTimeouts {
    /// Maximum wait for successful response headers.
    pub headers: Duration,
    /// Maximum wait for an asynchronous AWS credential provider.
    pub credentials: Duration,
    /// Maximum inactivity interval while reading response bytes.
    pub stream_idle: Duration,
}
impl Default for AnthropicTimeouts {
    fn default() -> Self {
        Self {
            headers: Duration::from_secs(30),
            credentials: Duration::from_secs(30),
            stream_idle: Duration::from_secs(60),
        }
    }
}

pub(crate) async fn read_error_body(
    response: reqwest::Response,
    abort: &AbortSignal,
    idle: Duration,
) -> Result<(Vec<u8>, u64), ModelError> {
    oven_sdk::provider_support::read_bounded_body(
        response.bytes_stream(),
        abort,
        oven_sdk::provider_support::BodyReadConfig {
            cap: oven_sdk::SanitizedBody::MAX_BYTES,
            limit: oven_sdk::provider_support::BodyLimit::StopAtCap,
            stage: ErrorStage::ResponseBody,
            timeout_message: "error response body idle timeout",
            abort_message: "request was aborted while reading an error response",
            read_message: "error response body read failed",
            overflow_message: "error response body byte count overflowed",
        },
        || tokio::time::sleep(idle),
    )
    .await
}

pub(crate) fn response_head(response: &reqwest::Response) -> ResponseHead {
    ResponseHead {
        http_status: Some(response.status().as_u16()),
        request_id: response
            .headers()
            .get("request-id")
            .or_else(|| response.headers().get("x-request-id"))
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned),
        response_metadata: ResponseMetadata::new(),
    }
}
