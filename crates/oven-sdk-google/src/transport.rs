//! HTTP transport helpers.

use std::time::Duration;

use oven_sdk::{AbortSignal, ErrorStage, ModelError, ResponseHead, ResponseMetadata};

/// Per-phase Google transport timeouts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GoogleTimeouts {
    /// Connection timeout for adapter-created clients.
    pub connect: Duration,
    /// Maximum wait for response headers.
    pub headers: Duration,
    /// Maximum inactivity while reading a stream or response body.
    pub stream_idle: Duration,
}

impl Default for GoogleTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(60),
            headers: Duration::from_secs(300),
            stream_idle: Duration::from_secs(300),
        }
    }
}

pub(crate) fn response_head(response: &reqwest::Response) -> ResponseHead {
    ResponseHead {
        http_status: Some(response.status().as_u16()),
        request_id: response
            .headers()
            .get("x-request-id")
            .or_else(|| response.headers().get("request-id"))
            .or_else(|| response.headers().get("x-goog-request-id"))
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
            timeout_message: "Google response body idle timeout",
            abort_message: "request was aborted while reading the response body",
            read_message: "Google response body read failed",
            overflow_message: "Google response byte count overflowed",
        },
        tokio::time::sleep(idle),
        move |timer| timer.reset(tokio::time::Instant::now() + idle),
    )
    .await
}
