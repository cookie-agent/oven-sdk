//! HTTP transport helpers.

use std::time::Duration;

use futures_util::StreamExt;
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
            connect: Duration::from_secs(10),
            headers: Duration::from_secs(30),
            stream_idle: Duration::from_secs(60),
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
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    let mut count = 0_u64;
    loop {
        let next = tokio::select! {
            value = tokio::time::timeout(idle, stream.next()) => value.map_err(|_| {
                ModelError::timeout("Google response body idle timeout")
                    .with_stage(ErrorStage::ResponseBody)
                    .with_bytes_received(count)
            })?,
            _ = abort.aborted() => return Err(ModelError::abort("request was aborted while reading the response body")
                .with_stage(ErrorStage::ResponseBody)
                .with_bytes_received(count)),
        };
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|_| {
            ModelError::transport("Google response body read failed")
                .with_stage(ErrorStage::ResponseBody)
                .with_bytes_received(count)
        })?;
        count = count
            .checked_add(
                u64::try_from(chunk.len())
                    .map_err(|_| ModelError::transport("Google response byte count overflowed"))?,
            )
            .ok_or_else(|| ModelError::transport("Google response byte count overflowed"))?;
        if body.len() < cap {
            let take = chunk.len().min(cap - body.len());
            body.extend_from_slice(&chunk[..take]);
        }
    }
    Ok((body, count))
}
