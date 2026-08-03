//! HTTP transport configuration and response helpers.

use std::time::Duration;

use futures_util::StreamExt;
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
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    let mut count = 0_u64;
    loop {
        let next = tokio::select! {
            value = tokio::time::timeout(idle, stream.next()) => value.map_err(|_| {
                ModelError::timeout("OpenAI error response body idle timeout")
                    .with_stage(ErrorStage::ResponseBody)
                    .with_bytes_received(count)
            })?,
            _ = abort.aborted() => {
                return Err(ModelError::abort("OpenAI error response body read was aborted")
                    .with_stage(ErrorStage::ResponseBody)
                    .with_bytes_received(count));
            }
        };
        match next {
            Some(Ok(chunk)) => {
                count = count.saturating_add(chunk.len() as u64);
                let limit = SanitizedBody::MAX_BYTES + 1;
                let remaining = limit.saturating_sub(body.len());
                body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
            }
            Some(Err(_)) => {
                return Err(
                    ModelError::transport("OpenAI error response body read failed")
                        .with_stage(ErrorStage::ResponseBody)
                        .with_bytes_received(count),
                );
            }
            None => return Ok((body, count)),
        }
    }
}

pub(crate) async fn read_bounded_body(
    response: reqwest::Response,
    abort: &AbortSignal,
    idle: Duration,
    maximum: usize,
) -> Result<(Vec<u8>, u64), ModelError> {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    let mut count = 0_u64;
    loop {
        let next = tokio::select! {
            value = tokio::time::timeout(idle, stream.next()) => value.map_err(|_| {
                ModelError::timeout("OpenAI response body idle timeout")
                    .with_stage(ErrorStage::NativeContextDecode)
                    .with_bytes_received(count)
            })?,
            _ = abort.aborted() => {
                return Err(ModelError::abort("OpenAI response body read was aborted")
                    .with_stage(ErrorStage::NativeContextDecode)
                    .with_bytes_received(count));
            }
        };
        match next {
            Some(Ok(chunk)) => {
                count = count.saturating_add(chunk.len() as u64);
                if body.len().saturating_add(chunk.len()) > maximum {
                    return Err(ModelError::invalid_response(
                        "OpenAI response body exceeds the configured safety bound",
                    )
                    .with_stage(ErrorStage::NativeContextDecode)
                    .with_bytes_received(count));
                }
                body.extend_from_slice(&chunk);
            }
            Some(Err(_)) => {
                return Err(ModelError::transport("OpenAI response body read failed")
                    .with_stage(ErrorStage::NativeContextDecode)
                    .with_bytes_received(count));
            }
            None => return Ok((body, count)),
        }
    }
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
