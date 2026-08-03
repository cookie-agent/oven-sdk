//! HTTP transport configuration and response helpers.

use std::time::Duration;

use futures_util::StreamExt;
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
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    let mut bytes_received = 0_u64;
    loop {
        let next = tokio::select! {
            value = tokio::time::timeout(idle, stream.next()) => value.map_err(|_| {
                ModelError::timeout("error response body idle timeout")
                    .with_stage(ErrorStage::ResponseBody)
                    .with_bytes_received(bytes_received)
            })?,
            _ = abort.aborted() => return Err(
                ModelError::abort("request was aborted while reading an error response")
                    .with_stage(ErrorStage::ResponseBody)
                    .with_bytes_received(bytes_received)
            ),
        };
        let Some(chunk) = next else {
            break;
        };
        let chunk = chunk.map_err(|_| {
            ModelError::transport("error response body read failed")
                .with_stage(ErrorStage::ResponseBody)
                .with_bytes_received(bytes_received)
        })?;
        let chunk_len = u64::try_from(chunk.len()).map_err(|_| {
            ModelError::transport("error response body byte count overflowed")
                .with_stage(ErrorStage::ResponseBody)
                .with_bytes_received(bytes_received)
        })?;
        bytes_received = bytes_received.checked_add(chunk_len).ok_or_else(|| {
            ModelError::transport("error response body byte count overflowed")
                .with_stage(ErrorStage::ResponseBody)
                .with_bytes_received(bytes_received)
        })?;
        let remaining = oven_sdk::SanitizedBody::MAX_BYTES.saturating_sub(body.len());
        let taken = chunk.len().min(remaining);
        body.extend_from_slice(&chunk[..taken]);
        if body.len() == oven_sdk::SanitizedBody::MAX_BYTES {
            break;
        }
    }
    Ok((body, bytes_received))
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
