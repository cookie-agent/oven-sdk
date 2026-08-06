//! Runtime-neutral protocol and response-reading helpers for provider adapters.

use std::{future::Future, time::Duration};

use bytes::Bytes;
use futures_core::Stream;
use futures_util::{FutureExt, StreamExt, pin_mut, select_biased};
use http::HeaderMap;

use crate::{AbortSignal, ErrorStage, ModelError};

/// One framed server-sent event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SseEvent {
    /// Optional event name, empty when no `event:` field was supplied.
    pub name: String,
    /// Newline-joined `data:` fields.
    pub data: String,
}

/// Incremental SSE framer supporting arbitrary byte boundaries and standard line endings.
pub struct SseParser {
    bytes: Vec<u8>,
    name: String,
    data: Vec<String>,
    saw_first_line: bool,
    invalid_utf8_message: &'static str,
    clear_name_on_empty_event: bool,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new("SSE contains invalid UTF-8")
    }
}

impl SseParser {
    /// Creates a parser with the adapter-specific invalid UTF-8 diagnostic.
    #[must_use]
    pub const fn new(invalid_utf8_message: &'static str) -> Self {
        Self {
            bytes: Vec::new(),
            name: String::new(),
            data: Vec::new(),
            saw_first_line: false,
            invalid_utf8_message,
            clear_name_on_empty_event: false,
        }
    }

    /// Clears a pending event name when a data-less event is discarded.
    #[must_use]
    pub const fn clear_name_on_empty_event(mut self) -> Self {
        self.clear_name_on_empty_event = true;
        self
    }

    /// Feeds one arbitrary byte chunk and returns all completed events.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, ModelError> {
        self.bytes.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut start = 0;
        let mut index = 0;
        while index < self.bytes.len() {
            if self.bytes[index] == b'\n' || self.bytes[index] == b'\r' {
                if self.bytes[index] == b'\r' && index + 1 == self.bytes.len() {
                    break;
                }
                let end = index;
                if self.bytes[index] == b'\r' && self.bytes.get(index + 1) == Some(&b'\n') {
                    index += 1;
                }
                let line = self.bytes[start..end].to_vec();
                self.line(&line, &mut events)?;
                start = index + 1;
            }
            index += 1;
        }
        self.bytes.drain(..start);
        Ok(events)
    }

    /// Flushes a final unterminated line and pending event.
    pub fn finish(&mut self) -> Result<Vec<SseEvent>, ModelError> {
        let mut events = Vec::new();
        if self.bytes.last() == Some(&b'\r') {
            self.bytes.pop();
        }
        if !self.bytes.is_empty() {
            let line = std::mem::take(&mut self.bytes);
            self.line(&line, &mut events)?;
        }
        self.dispatch(&mut events);
        Ok(events)
    }

    fn line(&mut self, raw: &[u8], events: &mut Vec<SseEvent>) -> Result<(), ModelError> {
        let raw = if self.saw_first_line {
            raw
        } else {
            self.saw_first_line = true;
            raw.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(raw)
        };
        let line = std::str::from_utf8(raw).map_err(|_| {
            ModelError::invalid_response(self.invalid_utf8_message)
                .with_stage(ErrorStage::StreamDecode)
        })?;
        if line.is_empty() {
            self.dispatch(events);
            return Ok(());
        }
        if line.starts_with(':') {
            return Ok(());
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => self.name = value.to_owned(),
            "data" => self.data.push(value.to_owned()),
            _ => {}
        }
        Ok(())
    }

    fn dispatch(&mut self, events: &mut Vec<SseEvent>) {
        if self.data.is_empty() {
            if self.clear_name_on_empty_event {
                self.name.clear();
            }
            return;
        }
        events.push(SseEvent {
            name: std::mem::take(&mut self.name),
            data: self.data.join("\n"),
        });
        self.data.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::SseParser;

    #[test]
    fn default_parser_preserves_name_across_data_less_event() {
        let mut parser = SseParser::default();
        let events = parser.feed(b"event: foo\n\ndata: x\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "foo");
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn configured_parser_clears_name_after_data_less_event() {
        let mut parser = SseParser::default().clear_name_on_empty_event();
        let events = parser.feed(b"event: foo\n\ndata: x\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "");
    }
}

/// Behavior when a response body reaches its configured byte cap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BodyLimit {
    /// Retain at most the cap while continuing to count through EOF.
    Truncate,
    /// Retain at most the cap and stop reading as soon as it is full.
    StopAtCap,
    /// Fail when the body would exceed the cap.
    Reject {
        /// Adapter-specific diagnostic used when the cap is exceeded.
        message: &'static str,
    },
}

/// Adapter-specific diagnostics and limits for [`read_bounded_body`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BodyReadConfig {
    /// Maximum number of retained response bytes.
    pub cap: usize,
    /// Behavior when `cap` is reached.
    pub limit: BodyLimit,
    /// Error stage attached to read failures.
    pub stage: ErrorStage,
    /// Idle-timeout diagnostic.
    pub timeout_message: &'static str,
    /// Cancellation diagnostic.
    pub abort_message: &'static str,
    /// Transport-read diagnostic.
    pub read_message: &'static str,
    /// Byte-count overflow diagnostic.
    pub overflow_message: &'static str,
}

/// Reads and bounds a byte stream with adapter-supplied idle timers and diagnostics.
pub async fn read_bounded_body<S, E, T, F>(
    stream: S,
    abort: &AbortSignal,
    config: BodyReadConfig,
    mut idle_timer: F,
) -> Result<(Vec<u8>, u64), ModelError>
where
    S: Stream<Item = Result<Bytes, E>> + Send,
    T: Future<Output = ()> + Send,
    F: FnMut() -> T,
{
    let mut body = Vec::new();
    let mut count = 0_u64;
    pin_mut!(stream);
    loop {
        let mut pinned_stream = stream.as_mut();
        let next = pinned_stream.next().fuse();
        let timeout = idle_timer().fuse();
        let aborted = abort.aborted().fuse();
        pin_mut!(next, timeout, aborted);
        let next = select_biased! {
            _ = aborted => return Err(ModelError::abort(config.abort_message)
                .with_stage(config.stage)
                .with_bytes_received(count)),
            _ = timeout => return Err(ModelError::timeout(config.timeout_message)
                .with_stage(config.stage)
                .with_bytes_received(count)),
            value = next => value,
        };
        let Some(chunk) = next else {
            return Ok((body, count));
        };
        let chunk = chunk.map_err(|_| {
            ModelError::transport(config.read_message)
                .with_stage(config.stage)
                .with_bytes_received(count)
        })?;
        count = count
            .checked_add(u64::try_from(chunk.len()).map_err(|_| {
                ModelError::transport(config.overflow_message)
                    .with_stage(config.stage)
                    .with_bytes_received(count)
            })?)
            .ok_or_else(|| {
                ModelError::transport(config.overflow_message)
                    .with_stage(config.stage)
                    .with_bytes_received(count)
            })?;
        match config.limit {
            BodyLimit::Reject { message }
                if body.len().saturating_add(chunk.len()) > config.cap =>
            {
                return Err(ModelError::invalid_response(message)
                    .with_stage(config.stage)
                    .with_bytes_received(count));
            }
            BodyLimit::Reject { .. } => body.extend_from_slice(&chunk),
            BodyLimit::Truncate | BodyLimit::StopAtCap => {
                let remaining = config.cap.saturating_sub(body.len());
                body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                if config.limit == BodyLimit::StopAtCap && body.len() == config.cap {
                    return Ok((body, count));
                }
            }
        }
    }
}

/// Parses `Retry-After` seconds or HTTP-date, with optional higher-priority millisecond headers.
#[must_use]
pub fn parse_retry_after(headers: &HeaderMap, millisecond_headers: &[&str]) -> Option<Duration> {
    for name in millisecond_headers {
        if let Some(milliseconds) = headers
            .get(*name)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
        {
            return Some(Duration::from_millis(milliseconds));
        }
    }
    let value = headers.get("retry-after")?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    httpdate::parse_http_date(value)
        .ok()?
        .duration_since(std::time::SystemTime::now())
        .ok()
}
