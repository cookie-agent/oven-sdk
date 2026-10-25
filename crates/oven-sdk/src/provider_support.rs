//! Runtime-neutral protocol and response-reading helpers for provider adapters.

use std::{future::Future, pin::Pin, time::Duration};

use bytes::Bytes;
use futures_core::Stream;
use futures_util::{FutureExt, StreamExt, pin_mut, select_biased};
use http::HeaderMap;

use crate::{AbortSignal, AbortWait, ErrorStage, ModelError};

mod diagnostic_body;
mod json_event;
mod responses_native_context;
pub use diagnostic_body::{sanitize_error_body, sanitize_error_body_with_secrets};
pub use json_event::{JsonObject, KeyPresent, json_type_tag, parse_json_object};
pub use responses_native_context::{
    RESPONSES_COMPACTION_ITEM_TYPE, RESPONSES_ENCRYPTED_CONTENT_FIELD,
    RESPONSES_NATIVE_CONTEXT_WIRE_NAMES, error_names_responses_native_context,
};

/// Result of waiting for the next streaming transport item.
pub enum StreamRead<T> {
    /// Cancellation won the biased wait.
    Aborted,
    /// The idle deadline elapsed.
    TimedOut,
    /// The transport produced its next item or EOF.
    Item(T),
}

/// Persistent idle timer and abort waiter for an adapter stream.
pub struct StreamReadDeadline<T> {
    timer: Pin<Box<T>>,
    aborted: Pin<Box<AbortWait>>,
}

impl<T> StreamReadDeadline<T>
where
    T: Future<Output = ()> + Send,
{
    /// Creates persistent read synchronization state.
    pub fn new(timer: T, abort: &AbortSignal) -> Self {
        Self {
            timer: Box::pin(timer),
            aborted: Box::pin(abort.aborted()),
        }
    }

    /// Waits for one item after resetting the adapter-owned idle timer.
    pub async fn next<S, R>(
        &mut self,
        mut stream: Pin<&mut S>,
        reset: R,
    ) -> StreamRead<Option<S::Item>>
    where
        S: Stream + ?Sized,
        R: FnOnce(Pin<&mut T>),
    {
        reset(self.timer.as_mut());
        let next = stream.next().fuse();
        let timeout = self.timer.as_mut().fuse();
        let aborted = self.aborted.as_mut().fuse();
        pin_mut!(next, timeout, aborted);
        select_biased! {
            _ = aborted => StreamRead::Aborted,
            _ = timeout => StreamRead::TimedOut,
            item = next => StreamRead::Item(item),
        }
    }
}

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
    /// Carried bytes of an unterminated line; never contains `\n` or `\r`.
    bytes: Vec<u8>,
    /// The previous chunk ended in `\r`, so a leading `\n` completes that CRLF.
    skip_lf: bool,
    name: String,
    data: String,
    saw_data: bool,
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
            skip_lf: false,
            name: String::new(),
            data: String::new(),
            saw_data: false,
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
        let mut events = Vec::new();
        self.feed_into(chunk, &mut events)?;
        Ok(events)
    }

    /// Feeds one arbitrary byte chunk into a caller-owned event buffer.
    pub fn feed_into<C>(&mut self, chunk: &[u8], events: &mut C) -> Result<(), ModelError>
    where
        C: Extend<SseEvent>,
    {
        let mut position = 0;
        if std::mem::take(&mut self.skip_lf) && chunk.first() == Some(&b'\n') {
            position = 1;
        }
        // Each byte is scanned exactly once: completed lines are parsed straight
        // from the chunk, and only an unterminated tail is carried over.
        while let Some(offset) = memchr::memchr2(b'\n', b'\r', &chunk[position..]) {
            let end = position + offset;
            if self.bytes.is_empty() {
                self.line(&chunk[position..end], events)?;
            } else {
                let mut line = std::mem::take(&mut self.bytes);
                line.extend_from_slice(&chunk[position..end]);
                self.line(&line, events)?;
                line.clear();
                self.bytes = line;
            }
            position = end + 1;
            if chunk[end] == b'\r' {
                match chunk.get(position) {
                    Some(b'\n') => position += 1,
                    Some(_) => {}
                    None => self.skip_lf = true,
                }
            }
        }
        self.bytes.extend_from_slice(&chunk[position..]);
        Ok(())
    }

    /// Flushes a final unterminated line and pending event.
    pub fn finish(&mut self) -> Result<Vec<SseEvent>, ModelError> {
        let mut events = Vec::new();
        self.skip_lf = false;
        if !self.bytes.is_empty() {
            let line = std::mem::take(&mut self.bytes);
            self.line(&line, &mut events)?;
        }
        self.dispatch(&mut events);
        Ok(events)
    }

    fn line<C>(&mut self, raw: &[u8], events: &mut C) -> Result<(), ModelError>
    where
        C: Extend<SseEvent>,
    {
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
            "event" => {
                self.name.clear();
                self.name.push_str(value);
            }
            "data" => {
                if self.saw_data {
                    self.data.push('\n');
                }
                self.saw_data = true;
                self.data.push_str(value);
            }
            _ => {}
        }
        Ok(())
    }

    fn dispatch<C>(&mut self, events: &mut C)
    where
        C: Extend<SseEvent>,
    {
        if !self.saw_data {
            if self.clear_name_on_empty_event {
                self.name.clear();
            }
            return;
        }
        self.saw_data = false;
        events.extend([SseEvent {
            name: std::mem::take(&mut self.name),
            data: std::mem::take(&mut self.data),
        }]);
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

    #[test]
    fn empty_data_field_is_preserved_when_joining() {
        let mut parser = SseParser::default();
        let events = parser.feed(b"data: first\ndata:\n\n").unwrap();
        assert_eq!(
            events,
            [super::SseEvent {
                name: String::new(),
                data: "first\n".into(),
            }]
        );
    }

    #[test]
    fn empty_then_nonempty_data_fields_keep_leading_newline() {
        let mut parser = SseParser::default();
        let events = parser.feed(b"data:\ndata: x\n\n").unwrap();
        assert_eq!(
            events,
            [super::SseEvent {
                name: String::new(),
                data: "\nx".into(),
            }]
        );
    }

    /// Frames a complete body in one pass, splitting lines on CRLF, CR, or LF.
    fn reference_events(input: &[u8]) -> Vec<super::SseEvent> {
        let mut parser = SseParser::default();
        let text = std::str::from_utf8(input).unwrap();
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let mut events = parser.feed(normalized.as_bytes()).unwrap();
        events.extend(parser.finish().unwrap());
        events
    }

    fn feed_split(input: &[u8], boundaries: &[usize]) -> Vec<super::SseEvent> {
        let mut parser = SseParser::default();
        let mut events = Vec::new();
        let mut start = 0;
        for &boundary in boundaries.iter().chain([&input.len()]) {
            events.extend(parser.feed(&input[start..boundary]).unwrap());
            start = boundary;
        }
        events.extend(parser.finish().unwrap());
        events
    }

    const FRAMING_FIXTURES: &[&str] = &[
        "\u{feff}event: start\r\ndata: {\"a\":1}\r\n\r\n: keepalive\r\ndata: hé\r\ndata: second\r\n\r\n",
        "event: delta\rdata: one\rdata:\rdata: three\r\revent: tail\ndata: x\n\n",
        "data: mixed\r\n\ndata: lf\n\r\ndata: cr\r\r\ndata: unterminated",
        "data: trailing-cr\r",
        "data: a\n\r\n\r\r\n\ndata: b\n\n",
    ];

    #[test]
    fn every_two_chunk_boundary_matches_single_chunk_framing() {
        for fixture in FRAMING_FIXTURES {
            let input = fixture.as_bytes();
            let expected = reference_events(input);
            assert!(!expected.is_empty());
            for boundary in 0..=input.len() {
                assert_eq!(
                    feed_split(input, &[boundary]),
                    expected,
                    "fixture {fixture:?} split at byte {boundary}"
                );
            }
            let every_byte: Vec<usize> = (1..input.len()).collect();
            assert_eq!(feed_split(input, &every_byte), expected, "{fixture:?}");
        }
    }

    #[test]
    fn crlf_split_between_chunks_is_one_line_ending() {
        let mut parser = SseParser::default();
        let mut events = parser.feed(b"data: x\r").unwrap();
        events.extend(parser.feed(b"\n\r").unwrap());
        events.extend(parser.feed(b"\ndata: y\n\n").unwrap());
        assert_eq!(
            events,
            [
                super::SseEvent {
                    name: String::new(),
                    data: "x".into(),
                },
                super::SseEvent {
                    name: String::new(),
                    data: "y".into(),
                },
            ]
        );
    }

    #[test]
    fn long_single_line_event_survives_small_chunks() {
        let payload = "é".repeat(300_000);
        let input = format!("event: big\ndata: {payload}\r\n\r\n");
        let boundaries: Vec<usize> = (1..input.len()).step_by(1023).collect();
        let events = feed_split(input.as_bytes(), &boundaries);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "big");
        assert_eq!(events[0].data, payload);
    }

    #[test]
    fn invalid_utf8_split_across_chunks_is_rejected() {
        let mut parser = SseParser::default();
        assert!(parser.feed(b"data: \xc3").unwrap().is_empty());
        assert!(parser.feed(b"(\n\n").is_err());
    }

    #[test]
    fn fully_empty_data_event_is_dispatched() {
        let mut parser = SseParser::default();
        let events = parser.feed(b"data:\n\n").unwrap();
        assert_eq!(
            events,
            [super::SseEvent {
                name: String::new(),
                data: String::new(),
            }]
        );
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
pub async fn read_bounded_body<S, E, T, R>(
    stream: S,
    abort: &AbortSignal,
    config: BodyReadConfig,
    idle_timer: T,
    mut reset_idle: R,
) -> Result<(Vec<u8>, u64), ModelError>
where
    S: Stream<Item = Result<Bytes, E>> + Send,
    T: Future<Output = ()> + Send,
    R: FnMut(Pin<&mut T>),
{
    let mut body = Vec::new();
    let mut count = 0_u64;
    pin_mut!(stream, idle_timer);
    let aborted = abort.aborted();
    pin_mut!(aborted);
    loop {
        reset_idle(idle_timer.as_mut());
        let mut pinned_stream = stream.as_mut();
        let next = pinned_stream.next().fuse();
        let timeout = idle_timer.as_mut().fuse();
        let aborted_wait = aborted.as_mut().fuse();
        pin_mut!(next, timeout, aborted_wait);
        let next = select_biased! {
            _ = aborted_wait => return Err(ModelError::abort(config.abort_message)
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
