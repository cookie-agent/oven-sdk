//! Incremental response streaming and state normalization.

mod blocks;
pub(crate) mod state;

use std::{collections::VecDeque, time::Duration};

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, BoxStream, ErrorStage, JsonValue, ModelError, StreamItem};
#[cfg(test)]
use oven_sdk::{ReplayPolicy, StreamPart};
use reqwest::header::HeaderMap;

use crate::{error::classify_error_for, wire::Protocol};

pub(crate) struct LiveState {
    pub(crate) bytes: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    pub(crate) parser: crate::sse::Parser,
    pub(crate) state: state::State,
    pub(crate) queue: VecDeque<StreamItem>,
    pub(crate) pending_events: VecDeque<crate::sse::Event>,
    pub(crate) pending_error: Option<ModelError>,
    pub(crate) abort: AbortSignal,
    pub(crate) idle: Duration,
    pub(crate) count: u64,
    pub(crate) eof: bool,
    pub(crate) request_id: Option<String>,
    pub(crate) protocol: Protocol,
}

pub(crate) async fn early_peek(live: &mut LiveState) -> Result<(), ModelError> {
    loop {
        let had_semantic = read_live(live, true).await?;
        if had_semantic {
            return Ok(());
        }
        if live.eof {
            return Err(ModelError::unexpected_eof(
                "Messages stream ended before a semantic event",
            )
            .with_bytes_received(live.count));
        }
    }
}

pub(crate) async fn read_live(
    live: &mut LiveState,
    stop_at_first_semantic: bool,
) -> Result<bool, ModelError> {
    if !live.pending_events.is_empty() {
        return process_live_events(live, stop_at_first_semantic);
    }
    let next = tokio::select! {
        value = tokio::time::timeout(live.idle, live.bytes.next()) => value.map_err(|_| ModelError::timeout("stream idle timeout").with_stage(ErrorStage::StreamRead).with_bytes_received(live.count))?,
        _ = live.abort.aborted() => return Err(ModelError::abort("stream was aborted").with_stage(ErrorStage::StreamRead).with_bytes_received(live.count)),
    };
    let events = match next {
        Some(Ok(chunk)) => {
            let chunk_len = u64::try_from(chunk.len()).map_err(|_| {
                ModelError::transport("Messages stream byte count overflowed")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count)
            })?;
            live.count = live.count.checked_add(chunk_len).ok_or_else(|| {
                ModelError::transport("Messages stream byte count overflowed")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count)
            })?;
            live.parser.feed(&chunk)?
        }
        Some(Err(_)) => {
            return Err(ModelError::transport("Messages stream read failed")
                .with_stage(ErrorStage::StreamRead)
                .with_bytes_received(live.count));
        }
        None => {
            live.eof = true;
            live.parser.finish()?
        }
    };
    live.pending_events.extend(events);
    process_live_events(live, stop_at_first_semantic)
}
fn process_live_events(
    live: &mut LiveState,
    stop_at_first_semantic: bool,
) -> Result<bool, ModelError> {
    let mut semantic = false;
    while let Some(event) = live.pending_events.pop_front() {
        if event.data.is_empty() || event.name == "ping" {
            continue;
        }
        let value: JsonValue = serde_json::from_str(&event.data).map_err(|_| {
            ModelError::invalid_response("Messages SSE event is invalid JSON")
                .with_stage(ErrorStage::StreamDecode)
                .with_bytes_received(live.count)
        })?;
        if value.get("type").and_then(JsonValue::as_str) == Some("ping") {
            continue;
        }
        semantic = true;
        if !live.state.started
            && (event.name == "error"
                || value.get("type").and_then(JsonValue::as_str) == Some("error"))
        {
            return Err(classify_error_for(
                live.protocol,
                if value.pointer("/error/type").and_then(JsonValue::as_str)
                    == Some("overloaded_error")
                {
                    529
                } else {
                    500
                },
                event.data.as_bytes(),
                live.request_id.clone(),
                ErrorStage::StreamEvent,
                live.count,
                &HeaderMap::new(),
            ));
        }
        let mut parts = Vec::new();
        if let Err(error) = live.state.apply(&event.name, value, &mut parts, live.count) {
            live.queue.extend(parts.into_iter().map(Ok));
            live.pending_error = Some(error);
            return Ok(semantic);
        }
        if live.state.done {
            if live.pending_events.iter().any(trailing_semantic_event) {
                live.pending_events.clear();
                live.pending_error = Some(
                    ModelError::invalid_response("Messages event arrived after message_stop")
                        .with_stage(ErrorStage::StreamEvent)
                        .with_bytes_received(live.count),
                );
                live.eof = true;
                return Ok(true);
            }
            live.queue.extend(parts.into_iter().map(Ok));
            live.eof = true;
            live.pending_events.clear();
            return Ok(true);
        }
        live.queue.extend(parts.into_iter().map(Ok));
        if stop_at_first_semantic {
            return Ok(true);
        }
    }
    if live.eof && !live.state.done {
        return Err(
            ModelError::unexpected_eof("Messages stream ended before message_stop")
                .with_bytes_received(live.count),
        );
    }
    Ok(semantic)
}

fn trailing_semantic_event(event: &crate::sse::Event) -> bool {
    if event.data.is_empty() || event.name == "ping" {
        return false;
    }
    serde_json::from_str::<JsonValue>(&event.data)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .and_then(JsonValue::as_str)
                .map(str::to_owned)
        })
        .as_deref()
        != Some("ping")
}

#[cfg(test)]
fn decode_events(
    events: Vec<crate::sse::Event>,
    policy: ReplayPolicy,
    warnings: Vec<String>,
    bytes: u64,
) -> Result<Vec<StreamPart>, ModelError> {
    let mut state = state::State::new(
        policy,
        Protocol::Anthropic,
        Protocol::Anthropic.adapter_id(),
        oven_sdk::NativeContextScope::new(
            oven_sdk::ProviderId::new("anthropic"),
            oven_sdk::ModelId::new("test-model"),
            oven_sdk::ResourceId::new("test-resource").expect("valid resource"),
        )
        .expect("valid native-context scope"),
    );
    let mut parts = vec![StreamPart::StreamStart { warnings }];
    let mut first = true;
    for event in events {
        if event.data.is_empty() {
            continue;
        }
        let value: JsonValue = serde_json::from_str(&event.data).map_err(|_| {
            ModelError::invalid_response("Messages SSE event is invalid JSON")
                .with_stage(ErrorStage::StreamDecode)
                .with_bytes_received(bytes)
        })?;
        let event_type = event.name.as_str();
        if first
            && (event_type == "error"
                || value.get("type").and_then(JsonValue::as_str) == Some("error"))
        {
            return Err(classify_error_for(
                Protocol::Anthropic,
                if value.pointer("/error/type").and_then(JsonValue::as_str)
                    == Some("overloaded_error")
                {
                    529
                } else {
                    500
                },
                event.data.as_bytes(),
                None,
                ErrorStage::StreamEvent,
                bytes,
                &HeaderMap::new(),
            ));
        }
        first = false;
        state.apply(event_type, value, &mut parts, bytes)?;
    }
    if !state.done {
        return Err(
            ModelError::unexpected_eof("Messages stream ended before message_stop")
                .with_bytes_received(bytes),
        );
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_message_stop_finishes_stream() {
        let parts = decode_events(
            vec![
                crate::sse::Event {
                    name: "message_start".into(),
                    data: r#"{"type":"message_start","message":{}}"#.into(),
                },
                crate::sse::Event {
                    name: "message_stop".into(),
                    data: r#"{"type":"message_stop"}"#.into(),
                },
            ],
            ReplayPolicy::IfValid,
            Vec::new(),
            0,
        )
        .unwrap();
        assert!(matches!(parts.last(), Some(StreamPart::Finish { .. })));
    }

    #[test]
    fn one_byte_utf8_chunks_and_crlf_are_incremental() {
        let bytes = b"event: message_start\r\ndata: {\"type\":\"message_start\",\"message\":{}}\r\n\r\nevent: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n";
        let mut parser = crate::sse::Parser::default();
        let mut events = Vec::new();
        for byte in bytes {
            events.extend(parser.feed(std::slice::from_ref(byte)).unwrap());
        }
        events.extend(parser.finish().unwrap());
        let parts = decode_events(
            events,
            ReplayPolicy::IfValid,
            Vec::new(),
            bytes.len() as u64,
        )
        .unwrap();
        assert!(matches!(parts.last(), Some(StreamPart::Finish { .. })));
    }
}
