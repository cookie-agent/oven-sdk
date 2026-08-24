//! HTTP transport configuration and response helpers.

use std::time::Duration;

use oven_sdk::{
    AbortSignal, ErrorStage, ModelError, ResponseHead, ResponseMetadata, SanitizedBody,
};
use reqwest::header::HeaderMap;

/// Per-phase OpenAI adapter timeouts. There is intentionally no total stream
/// timeout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiTimeouts {
    /// Connection timeout for adapter-created clients.
    pub connect: Duration,
    /// Maximum wait for response headers.
    pub headers: Duration,
    /// Maximum inactivity while reading stream bytes.
    pub stream_idle: Duration,
}

impl Default for OpenAiTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(10),
            headers: Duration::from_secs(30),
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
            cap: SanitizedBody::MAX_BYTES + 1,
            limit: oven_sdk::provider_support::BodyLimit::Truncate,
            stage: ErrorStage::ResponseBody,
            timeout_message: "OpenAI error response body idle timeout",
            abort_message: "OpenAI error response body read was aborted",
            read_message: "OpenAI error response body read failed",
            overflow_message: "OpenAI error response body byte count overflowed",
        },
        tokio::time::sleep(idle),
        move |timer| timer.reset(tokio::time::Instant::now() + idle),
    )
    .await
}

pub(crate) async fn read_bounded_body(
    response: reqwest::Response,
    abort: &AbortSignal,
    idle: Duration,
    maximum: usize,
) -> Result<(Vec<u8>, u64), ModelError> {
    oven_sdk::provider_support::read_bounded_body(
        response.bytes_stream(),
        abort,
        oven_sdk::provider_support::BodyReadConfig {
            cap: maximum,
            limit: oven_sdk::provider_support::BodyLimit::Reject {
                message: "OpenAI response body exceeds the configured safety bound",
            },
            stage: ErrorStage::NativeContextDecode,
            timeout_message: "OpenAI response body idle timeout",
            abort_message: "OpenAI response body read was aborted",
            read_message: "OpenAI response body read failed",
            overflow_message: "OpenAI response body byte count overflowed",
        },
        tokio::time::sleep(idle),
        move |timer| timer.reset(tokio::time::Instant::now() + idle),
    )
    .await
}

pub(crate) fn response_head(
    response: &reqwest::Response,
    request_headers: &[String],
) -> ResponseHead {
    ResponseHead {
        http_status: Some(response.status().as_u16()),
        request_id: request_id(response.headers(), request_headers),
        response_metadata: ResponseMetadata::new(),
    }
}

pub(crate) fn request_id(headers: &HeaderMap, configured: &[String]) -> Option<String> {
    std::iter::once("x-request-id")
        .chain(std::iter::once("request-id"))
        .chain(configured.iter().map(String::as_str))
        .find_map(|name| headers.get(name)?.to_str().ok().map(str::to_owned))
}
