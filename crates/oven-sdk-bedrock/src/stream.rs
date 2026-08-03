//! Bedrock response normalization and strict ConverseStream state machine.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    time::Duration,
};

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AdapterId, BoxStream, ErrorStage, Finish, FinishReason, JsonValue, ModelError,
    ModelErrorKind, NativeContextScope, NativeReplayArtifact, ReplayPolicy, SourcePart, StreamItem,
    StreamPart, ToolCallPart, Usage,
};

use crate::{
    BEDROCK_CONVERSE_ADAPTER_ID, REPLAY_FORMAT,
    error::classify_stream_exception,
    eventstream::{Decoder, Message},
};

enum Block {
    Text {
        id: String,
        text: String,
    },
    Reasoning {
        id: String,
        text: String,
        signature: Option<String>,
        redacted: Option<String>,
        visible_started: bool,
    },
    Tool {
        id: String,
        name: String,
        input: String,
    },
}

pub(crate) struct State {
    policy: ReplayPolicy,
    native_context_scope: NativeContextScope,
    reasoning: bool,
    signed_reasoning: bool,
    blocks: BTreeMap<u64, Block>,
    native: BTreeMap<u64, JsonValue>,
    next_index: u64,
    stopped_indices: BTreeSet<u64>,
    call_ids: BTreeSet<String>,
    message_started: bool,
    message_stopped: bool,
    stop_reason: Option<String>,
    usage: Usage,
    response_metadata: BTreeMap<String, JsonValue>,
    provider_metadata: BTreeMap<String, JsonValue>,
    request_id: Option<String>,
    done: bool,
}

#[derive(Clone)]
pub(crate) struct StreamConfiguration {
    pub(crate) policy: ReplayPolicy,
    pub(crate) native_context_scope: NativeContextScope,
    pub(crate) reasoning: bool,
    pub(crate) signed_reasoning: bool,
}

impl State {
    pub(crate) fn new(config: StreamConfiguration) -> Self {
        Self {
            policy: config.policy,
            native_context_scope: config.native_context_scope,
            reasoning: config.reasoning,
            signed_reasoning: config.signed_reasoning,
            blocks: BTreeMap::new(),
            native: BTreeMap::new(),
            next_index: 0,
            stopped_indices: BTreeSet::new(),
            call_ids: BTreeSet::new(),
            message_started: false,
            message_stopped: false,
            stop_reason: None,
            usage: Usage::default(),
            response_metadata: BTreeMap::new(),
            provider_metadata: BTreeMap::new(),
            request_id: None,
            done: false,
        }
    }

    pub(crate) fn set_request_id(&mut self, request_id: Option<String>) {
        self.request_id = request_id.clone();
        if let Some(request_id) = request_id {
            self.response_metadata
                .insert("bedrock.request_id".into(), JsonValue::String(request_id));
        }
    }

    pub(crate) fn response_metadata(&self) -> &BTreeMap<String, JsonValue> {
        &self.response_metadata
    }

    pub(crate) fn apply(
        &mut self,
        event_type: &str,
        payload: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self.done {
            return Err(invalid_event(
                "Bedrock event arrived after terminal finish",
                bytes,
            ));
        }
        match event_type {
            "messageStart" => self.message_start(&payload, bytes),
            "contentBlockStart" => self.block_start(&payload, parts, bytes),
            "contentBlockDelta" => self.block_delta(&payload, parts, bytes),
            "contentBlockStop" => self.block_stop(&payload, parts, bytes),
            "messageStop" => self.message_stop(&payload, bytes),
            "metadata" => self.metadata(&payload, parts, bytes),
            "internalServerException"
            | "modelStreamErrorException"
            | "serviceUnavailableException"
            | "throttlingException"
            | "validationException"
            | "modelNotReadyException" => self.provider_error(event_type, &payload, parts, bytes),
            other => {
                parts.push(StreamPart::ProviderEvent {
                    name: other.to_owned(),
                    data: payload,
                });
                Ok(())
            }
        }
    }

    fn message_start(&mut self, payload: &JsonValue, bytes: u64) -> Result<(), ModelError> {
        if self.message_started || self.message_stopped || !self.blocks.is_empty() {
            return Err(invalid_event(
                "duplicate or late Bedrock messageStart",
                bytes,
            ));
        }
        if payload.get("role").and_then(JsonValue::as_str) != Some("assistant") {
            return Err(invalid_event(
                "Bedrock messageStart role must be assistant",
                bytes,
            ));
        }
        self.message_started = true;
        Ok(())
    }

    fn block_start(
        &mut self,
        payload: &JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        self.require_open_message(bytes)?;
        let index = index(payload, bytes)?;
        self.reserve_index(index, bytes)?;
        let tool = payload.pointer("/start/toolUse").ok_or_else(|| {
            invalid_event("Bedrock contentBlockStart must contain toolUse", bytes)
        })?;
        let id = required_string(tool, "toolUseId", bytes)?.to_owned();
        let name = required_string(tool, "name", bytes)?.to_owned();
        if self.call_ids.contains(&id) {
            return Err(invalid_event("Bedrock tool-use ID was reused", bytes));
        }
        parts.push(StreamPart::ToolCallStart {
            id: id.clone(),
            name: name.clone(),
            metadata: None,
        });
        self.blocks.insert(
            index,
            Block::Tool {
                id,
                name,
                input: String::new(),
            },
        );
        Ok(())
    }

    fn block_delta(
        &mut self,
        payload: &JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        self.require_open_message(bytes)?;
        let index = index(payload, bytes)?;
        if self.stopped_indices.contains(&index) {
            return Err(invalid_event(
                "Bedrock delta targeted a stopped content block",
                bytes,
            ));
        }
        let delta = payload
            .get("delta")
            .and_then(JsonValue::as_object)
            .ok_or_else(|| invalid_event("Bedrock content delta is invalid", bytes))?;
        let variants = [
            "text",
            "toolUse",
            "reasoningContent",
            "citation",
            "image",
            "toolResult",
        ]
        .iter()
        .filter(|key| delta.contains_key(**key))
        .count();
        if variants != 1 {
            return Err(invalid_event(
                "Bedrock content delta must contain exactly one union member",
                bytes,
            ));
        }
        if let Some(text) = delta.get("text").and_then(JsonValue::as_str) {
            if !self.blocks.contains_key(&index) {
                self.reserve_index(index, bytes)?;
                let id = format!("text-{index}");
                parts.push(StreamPart::TextStart {
                    id: id.clone(),
                    metadata: None,
                });
                self.blocks.insert(
                    index,
                    Block::Text {
                        id,
                        text: String::new(),
                    },
                );
            }
            let Block::Text { id, text: full } = self.blocks.get_mut(&index).ok_or_else(|| {
                invalid_event("Bedrock text delta conflicts with block type", bytes)
            })?
            else {
                return Err(invalid_event(
                    "Bedrock text delta conflicts with block type",
                    bytes,
                ));
            };
            full.push_str(text);
            if !text.is_empty() {
                parts.push(StreamPart::TextDelta {
                    id: id.clone(),
                    delta: text.to_owned(),
                    metadata: None,
                });
            }
            return Ok(());
        }
        if let Some(tool) = delta.get("toolUse") {
            let fragment = tool
                .get("input")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| invalid_event("Bedrock tool delta input is invalid", bytes))?;
            let Block::Tool { id, input, .. } = self.blocks.get_mut(&index).ok_or_else(|| {
                invalid_event("Bedrock tool delta referenced no open tool block", bytes)
            })?
            else {
                return Err(invalid_event(
                    "Bedrock tool delta conflicts with block type",
                    bytes,
                ));
            };
            input.push_str(fragment);
            if !fragment.is_empty() {
                parts.push(StreamPart::ToolCallDelta {
                    id: id.clone(),
                    delta: fragment.to_owned(),
                    metadata: None,
                });
            }
            return Ok(());
        }
        if let Some(reasoning) = delta.get("reasoningContent") {
            if !self.reasoning {
                return Err(invalid_event(
                    "Bedrock emitted reasoning without declared reasoning support",
                    bytes,
                ));
            }
            if !self.blocks.contains_key(&index) {
                self.reserve_index(index, bytes)?;
                self.blocks.insert(
                    index,
                    Block::Reasoning {
                        id: format!("reasoning-{index}"),
                        text: String::new(),
                        signature: None,
                        redacted: None,
                        visible_started: false,
                    },
                );
            }
            let Block::Reasoning {
                id,
                text,
                signature,
                redacted,
                visible_started,
            } = self.blocks.get_mut(&index).ok_or_else(|| {
                invalid_event("Bedrock reasoning delta conflicts with block type", bytes)
            })?
            else {
                return Err(invalid_event(
                    "Bedrock reasoning delta conflicts with block type",
                    bytes,
                ));
            };
            let members = ["text", "signature", "data", "redactedContent"]
                .iter()
                .filter(|key| reasoning.get(**key).is_some())
                .count();
            if members != 1 {
                return Err(invalid_event(
                    "Bedrock reasoning delta must contain exactly one union member",
                    bytes,
                ));
            }
            if let Some(value) = reasoning.get("text").and_then(JsonValue::as_str) {
                if !*visible_started {
                    parts.push(StreamPart::ReasoningStart {
                        id: id.clone(),
                        metadata: None,
                    });
                    *visible_started = true;
                }
                text.push_str(value);
                if !value.is_empty() {
                    parts.push(StreamPart::ReasoningDelta {
                        id: id.clone(),
                        delta: value.to_owned(),
                        metadata: None,
                    });
                }
            } else if let Some(value) = reasoning.get("signature").and_then(JsonValue::as_str) {
                append_once(
                    signature,
                    value,
                    "Bedrock reasoning signature changed",
                    bytes,
                )?;
            } else if let Some(value) = reasoning
                .get("data")
                .or_else(|| reasoning.get("redactedContent"))
                .and_then(JsonValue::as_str)
            {
                append_once(redacted, value, "Bedrock redacted reasoning changed", bytes)?;
            } else {
                return Err(invalid_event("Bedrock reasoning delta is invalid", bytes));
            }
            return Ok(());
        }
        if let Some(citation) = delta.get("citation") {
            let mut source = SourcePart::new();
            source.url = citation
                .pointer("/location/webLocation/url")
                .or_else(|| citation.pointer("/location/documentPage/location"))
                .and_then(JsonValue::as_str)
                .and_then(|value| value.parse().ok());
            source.title = citation
                .get("title")
                .and_then(JsonValue::as_str)
                .map(str::to_owned);
            source.excerpt = citation
                .pointer("/sourceContent/0/text")
                .and_then(JsonValue::as_str)
                .map(str::to_owned);
            source.metadata = Some(BTreeMap::from([(
                "bedrock.citation".into(),
                citation.clone(),
            )]));
            parts.push(StreamPart::Source { source });
            return Ok(());
        }
        parts.push(StreamPart::ProviderEvent {
            name: "bedrock.content_delta".into(),
            data: JsonValue::Object(delta.clone()),
        });
        Ok(())
    }

    fn block_stop(
        &mut self,
        payload: &JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        self.require_open_message(bytes)?;
        let index = index(payload, bytes)?;
        if !self.stopped_indices.insert(index) {
            return Err(invalid_event("duplicate Bedrock contentBlockStop", bytes));
        }
        let block = self.blocks.remove(&index).ok_or_else(|| {
            invalid_event("Bedrock contentBlockStop referenced no open block", bytes)
        })?;
        match block {
            Block::Text { id, text } => {
                parts.push(StreamPart::TextEnd { id, metadata: None });
                self.native.insert(index, serde_json::json!({"text":text}));
            }
            Block::Reasoning {
                id,
                text,
                signature,
                redacted,
                visible_started,
            } => {
                if visible_started {
                    parts.push(StreamPart::ReasoningEnd { id, metadata: None });
                }
                if redacted.is_some()
                    && (visible_started || !text.is_empty() || signature.is_some())
                {
                    return Err(invalid_final(
                        "Bedrock reasoning block mixed text and redacted union members",
                        bytes,
                    ));
                }
                if redacted.is_none()
                    && self.signed_reasoning
                    && signature.as_deref().is_none_or(str::is_empty)
                {
                    return Err(invalid_final(
                        "Bedrock signed reasoning block is missing its signature",
                        bytes,
                    ));
                }
                if redacted.as_deref().is_some_and(str::is_empty) {
                    return Err(invalid_final(
                        "Bedrock redacted reasoning block is empty",
                        bytes,
                    ));
                }
                if redacted.is_some() && !self.signed_reasoning {
                    return Err(invalid_final(
                        "Bedrock redacted reasoning is unsupported by this configuration",
                        bytes,
                    ));
                }
                let native = if let Some(data) = redacted {
                    serde_json::json!({"reasoningContent":{"redactedContent":data}})
                } else {
                    let mut reasoning_text = serde_json::json!({"text":text});
                    if self.signed_reasoning
                        && let Some(signature) = signature
                    {
                        reasoning_text["signature"] = JsonValue::String(signature);
                    }
                    serde_json::json!({"reasoningContent":{"reasoningText":reasoning_text}})
                };
                self.native.insert(index, native);
            }
            Block::Tool { id, name, input } => {
                parts.push(StreamPart::ToolCallEnd {
                    id: id.clone(),
                    metadata: None,
                });
                let raw = if input.is_empty() {
                    "{}".to_owned()
                } else {
                    input
                };
                let parsed: JsonValue = serde_json::from_str(&raw).map_err(|_| {
                    ModelError::new(
                        ModelErrorKind::InvalidToolInput,
                        "Bedrock tool input is invalid JSON",
                    )
                    .with_stage(ErrorStage::StreamFinalize)
                    .with_bytes_received(bytes)
                })?;
                if !parsed.is_object() {
                    return Err(ModelError::new(
                        ModelErrorKind::InvalidToolInput,
                        "Bedrock tool input must be a JSON object",
                    )
                    .with_stage(ErrorStage::StreamFinalize)
                    .with_bytes_received(bytes));
                }
                if !self.call_ids.insert(id.clone()) {
                    return Err(invalid_final("duplicate Bedrock tool-use ID", bytes));
                }
                let mut call = ToolCallPart::new(id.clone(), name.clone(), parsed.clone());
                call.provider_item_id = Some(id.clone());
                call.raw_input = Some(raw);
                parts.push(StreamPart::ToolCall { tool_call: call });
                self.native.insert(
                    index,
                    serde_json::json!({"toolUse":{"toolUseId":id,"name":name,"input":parsed}}),
                );
            }
        }
        Ok(())
    }

    fn message_stop(&mut self, payload: &JsonValue, bytes: u64) -> Result<(), ModelError> {
        self.require_open_message(bytes)?;
        if self.message_stopped || !self.blocks.is_empty() {
            return Err(invalid_event(
                "Bedrock messageStop arrived with duplicate or open blocks",
                bytes,
            ));
        }
        let reason = required_string(payload, "stopReason", bytes)?.to_owned();
        if let Some(value) = payload.get("additionalModelResponseFields") {
            self.provider_metadata.insert(
                "bedrock.additional_model_response_fields".into(),
                value.clone(),
            );
        }
        self.stop_reason = Some(reason);
        self.message_stopped = true;
        Ok(())
    }

    fn metadata(
        &mut self,
        payload: &JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if !self.message_stopped {
            return Err(invalid_event(
                "Bedrock metadata arrived before messageStop",
                bytes,
            ));
        }
        self.usage = parse_usage(payload.get("usage"), bytes)?;
        for (wire, key) in [
            ("metrics", "bedrock.metrics"),
            ("performanceConfig", "bedrock.performance_config"),
            ("serviceTier", "bedrock.service_tier"),
            ("trace", "bedrock.trace"),
        ] {
            if let Some(value) = payload.get(wire) {
                self.provider_metadata.insert(key.into(), value.clone());
            }
        }
        let reason = self.stop_reason.clone().expect("messageStop set reason");
        if matches!(
            reason.as_str(),
            "malformed_model_output" | "malformed_tool_use" | "model_context_window_exceeded"
        ) {
            let (kind, message) = match reason.as_str() {
                "malformed_tool_use" => (
                    ModelErrorKind::InvalidToolInput,
                    "Bedrock reported malformed tool use",
                ),
                "model_context_window_exceeded" => (
                    ModelErrorKind::ContextLength,
                    "Bedrock model context window was exceeded",
                ),
                _ => (
                    ModelErrorKind::InvalidResponse,
                    "Bedrock reported malformed model output",
                ),
            };
            let error = ModelError::new(kind, message)
                .with_vendor_code(reason)
                .with_stage(ErrorStage::StreamFinalize)
                .with_bytes_received(bytes);
            parts.push(StreamPart::Error { error });
            parts.push(StreamPart::Finish {
                finish: self.finish(FinishReason::Error, bytes, false)?,
            });
            self.done = true;
            return Ok(());
        }
        let finish_reason = map_finish(&reason);
        parts.push(StreamPart::Finish {
            finish: self.finish(finish_reason, bytes, true)?,
        });
        self.done = true;
        Ok(())
    }

    fn provider_error(
        &mut self,
        event_type: &str,
        payload: &JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if !self.blocks.is_empty() {
            return Err(invalid_event(
                "Bedrock exception arrived with open content blocks",
                bytes,
            ));
        }
        let error = classify_stream_exception(event_type, payload, self.request_id.clone(), bytes);
        self.terminal_provider_error(error, parts, bytes)
    }

    fn provider_message_error(
        &mut self,
        code: &str,
        message: &str,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if !self.blocks.is_empty() {
            return Err(invalid_event(
                "AWS EventStream error arrived with open content blocks",
                bytes,
            ));
        }
        let error =
            crate::error::classify_stream_error(code, message, self.request_id.clone(), bytes);
        self.terminal_provider_error(error, parts, bytes)
    }

    fn terminal_provider_error(
        &mut self,
        error: ModelError,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        parts.push(StreamPart::Error { error });
        parts.push(StreamPart::Finish {
            finish: self.finish(FinishReason::Error, bytes, false)?,
        });
        self.done = true;
        Ok(())
    }

    fn finish(
        &self,
        reason: FinishReason,
        bytes: u64,
        capture_replay: bool,
    ) -> Result<Finish, ModelError> {
        let mut finish = Finish::new(self.usage.clone(), reason);
        finish.response_metadata = self.response_metadata.clone();
        finish.provider_metadata = self.provider_metadata.clone();
        if capture_replay && self.policy != ReplayPolicy::Never {
            let content = self.native.values().cloned().collect::<Vec<_>>();
            let payload = serde_json::json!({
                "format":REPLAY_FORMAT,
                "assistant_content":content,
            });
            finish.native_replay = Some(
                NativeReplayArtifact::new(
                    AdapterId::new(BEDROCK_CONVERSE_ADAPTER_ID),
                    self.native_context_scope.clone(),
                    payload,
                )
                .map_err(|_| {
                    ModelError::replay("Bedrock native replay artifact exceeds its size limit")
                        .with_stage(ErrorStage::ReplayEncode)
                        .with_bytes_received(bytes)
                })?,
            );
        }
        Ok(finish)
    }

    fn reserve_index(&mut self, index: u64, bytes: u64) -> Result<(), ModelError> {
        if self.blocks.contains_key(&index)
            || self.stopped_indices.contains(&index)
            || index != self.next_index
        {
            return Err(invalid_event(
                "Bedrock content block indices must start at zero and increase without reuse",
                bytes,
            ));
        }
        self.next_index = self
            .next_index
            .checked_add(1)
            .ok_or_else(|| invalid_event("Bedrock block index counter overflowed", bytes))?;
        Ok(())
    }

    fn require_open_message(&self, bytes: u64) -> Result<(), ModelError> {
        if !self.message_started || self.message_stopped {
            Err(invalid_event(
                "Bedrock content event arrived outside an open message",
                bytes,
            ))
        } else {
            Ok(())
        }
    }
}

pub(crate) struct LiveState {
    pub(crate) bytes: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    pub(crate) decoder: Decoder,
    pub(crate) state: State,
    pub(crate) queue: VecDeque<StreamItem>,
    pub(crate) pending_messages: VecDeque<Message>,
    pub(crate) terminal_queue: VecDeque<StreamItem>,
    pub(crate) pending_error: Option<ModelError>,
    pub(crate) abort: AbortSignal,
    pub(crate) idle: Duration,
    pub(crate) count: u64,
    pub(crate) eof: bool,
    pub(crate) include_raw: bool,
}

pub(crate) async fn early_peek(live: &mut LiveState) -> Result<(), ModelError> {
    loop {
        if read_live(live, true).await? {
            return Ok(());
        }
        if live.eof {
            return Err(ModelError::unexpected_eof(
                "Bedrock stream ended before a semantic response",
            )
            .with_stage(ErrorStage::StreamFinalize)
            .with_bytes_received(live.count));
        }
    }
}

pub(crate) async fn read_live(
    live: &mut LiveState,
    stop_at_semantic: bool,
) -> Result<bool, ModelError> {
    if live.pending_messages.is_empty() {
        let next = tokio::select! {
            value = tokio::time::timeout(live.idle, live.bytes.next()) => value.map_err(|_| {
                ModelError::timeout("Bedrock stream idle timeout")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count)
            })?,
            _ = live.abort.aborted() => return Err(ModelError::abort("Bedrock stream was aborted")
                .with_stage(ErrorStage::StreamRead)
                .with_bytes_received(live.count)),
        };
        let messages = match next {
            Some(Ok(chunk)) => {
                live.count = live
                    .count
                    .checked_add(u64::try_from(chunk.len()).map_err(|_| {
                        ModelError::transport("Bedrock stream byte count overflowed")
                    })?)
                    .ok_or_else(|| ModelError::transport("Bedrock stream byte count overflowed"))?;
                live.decoder
                    .feed(&chunk)
                    .map_err(|error| error.with_bytes_received(live.count))?
            }
            Some(Err(_)) => {
                return Err(ModelError::transport("Bedrock stream read failed")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count));
            }
            None => {
                live.eof = true;
                live.decoder
                    .finish()
                    .map_err(|error| error.with_bytes_received(live.count))?
            }
        };
        live.pending_messages.extend(messages);
    }
    let mut semantic = false;
    while let Some(message) = live.pending_messages.pop_front() {
        if live.state.done {
            return Err(invalid_event(
                "Bedrock EventStream frame followed terminal metadata or provider error",
                live.count,
            ));
        }
        let message_type = message.string_header(":message-type").ok_or_else(|| {
            invalid_event(
                "AWS EventStream message is missing :message-type",
                live.count,
            )
        })?;
        if message_type == "error" {
            let code = message
                .string_header(":error-code")
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    invalid_event("AWS EventStream error has no :error-code", live.count)
                })?;
            let error_message = message.string_header(":error-message").ok_or_else(|| {
                invalid_event("AWS EventStream error has no :error-message", live.count)
            })?;
            let mut parts = Vec::new();
            live.state
                .provider_message_error(code, error_message, &mut parts, live.count)?;
            semantic = true;
            if live.include_raw {
                live.terminal_queue.push_back(Ok(StreamPart::Raw {
                    value: serde_json::json!({"error":{"code":code}}),
                }));
            }
            live.terminal_queue.extend(parts.into_iter().map(Ok));
            continue;
        }
        let event_type = match message_type {
            "event" => message.string_header(":event-type").ok_or_else(|| {
                invalid_event("AWS EventStream event is missing :event-type", live.count)
            })?,
            "exception" => message
                .string_header(":exception-type")
                .or_else(|| message.string_header(":event-type"))
                .ok_or_else(|| {
                    invalid_event("AWS EventStream exception has no type", live.count)
                })?,
            _ => {
                return Err(invalid_event(
                    "AWS EventStream :message-type is unsupported",
                    live.count,
                ));
            }
        };
        let payload: JsonValue = serde_json::from_slice(&message.payload).map_err(|_| {
            ModelError::invalid_response("Bedrock EventStream payload is invalid JSON")
                .with_stage(ErrorStage::StreamDecode)
                .with_bytes_received(live.count)
        })?;
        semantic = true;
        let event_payload = payload
            .get(event_type)
            .cloned()
            .unwrap_or_else(|| payload.clone());
        let mut parts = Vec::new();
        if message_type == "exception" {
            live.state
                .provider_error(event_type, &event_payload, &mut parts, live.count)?;
        } else {
            live.state
                .apply(event_type, event_payload, &mut parts, live.count)?;
        }
        if live.state.done {
            if live.include_raw {
                live.terminal_queue
                    .push_back(Ok(StreamPart::Raw { value: payload }));
            }
            live.terminal_queue.extend(parts.into_iter().map(Ok));
        } else {
            if live.include_raw {
                live.queue.push_back(Ok(StreamPart::Raw { value: payload }));
            }
            live.queue.extend(parts.into_iter().map(Ok));
        }
        if stop_at_semantic && !live.state.done {
            return Ok(true);
        }
    }
    if live.eof {
        if !live.state.done {
            return Err(ModelError::unexpected_eof(
                "Bedrock stream ended before terminal metadata",
            )
            .with_stage(ErrorStage::StreamFinalize)
            .with_bytes_received(live.count));
        }
        live.queue.append(&mut live.terminal_queue);
    }
    Ok(semantic)
}

pub(crate) fn normalize_single(
    value: JsonValue,
    config: StreamConfiguration,
    request_id: Option<String>,
    include_raw: bool,
    bytes: u64,
) -> Result<(Vec<StreamPart>, BTreeMap<String, JsonValue>), ModelError> {
    let mut state = State::new(config);
    state.set_request_id(request_id);
    let mut parts = vec![StreamPart::StreamStart {
        warnings: Vec::new(),
    }];
    if include_raw {
        parts.push(StreamPart::Raw {
            value: value.clone(),
        });
    }
    if value
        .pointer("/output/message/role")
        .and_then(JsonValue::as_str)
        != Some("assistant")
    {
        return Err(invalid_final(
            "Bedrock Converse output role must be assistant",
            bytes,
        ));
    }
    state.apply(
        "messageStart",
        serde_json::json!({"role":"assistant"}),
        &mut parts,
        bytes,
    )?;
    let content = value
        .pointer("/output/message/content")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            invalid_final("Bedrock Converse response has no assistant content", bytes)
        })?;
    for (index, block) in content.iter().enumerate() {
        let index = u64::try_from(index)
            .map_err(|_| invalid_final("Bedrock content block index overflowed", bytes))?;
        if let Some(text) = block.get("text").and_then(JsonValue::as_str) {
            state.apply(
                "contentBlockDelta",
                serde_json::json!({"contentBlockIndex":index,"delta":{"text":text}}),
                &mut parts,
                bytes,
            )?;
        } else if let Some(call) = block.get("toolUse") {
            state.apply(
                "contentBlockStart",
                serde_json::json!({"contentBlockIndex":index,"start":{"toolUse":{"toolUseId":call.get("toolUseId"),"name":call.get("name")}}}),
                &mut parts,
                bytes,
            )?;
            state.apply(
                "contentBlockDelta",
                serde_json::json!({"contentBlockIndex":index,"delta":{"toolUse":{"input":call.get("input").cloned().unwrap_or_else(|| serde_json::json!({})).to_string()}}}),
                &mut parts,
                bytes,
            )?;
        } else if let Some(reasoning) = block.get("reasoningContent") {
            if let Some(reasoning_text) = reasoning.get("reasoningText") {
                if let Some(text) = reasoning_text.get("text") {
                    state.apply("contentBlockDelta", serde_json::json!({"contentBlockIndex":index,"delta":{"reasoningContent":{"text":text}}}), &mut parts, bytes)?;
                }
                if let Some(signature) = reasoning_text.get("signature") {
                    state.apply("contentBlockDelta", serde_json::json!({"contentBlockIndex":index,"delta":{"reasoningContent":{"signature":signature}}}), &mut parts, bytes)?;
                }
            } else if let Some(data) = reasoning
                .get("redactedContent")
                .or_else(|| reasoning.pointer("/redactedReasoning/data"))
            {
                state.apply("contentBlockDelta", serde_json::json!({"contentBlockIndex":index,"delta":{"reasoningContent":{"redactedContent":data}}}), &mut parts, bytes)?;
            }
        } else if let Some(citations) = block.get("citationsContent") {
            if let Some(text) = citations.get("content").and_then(JsonValue::as_str) {
                state.apply(
                    "contentBlockDelta",
                    serde_json::json!({"contentBlockIndex":index,"delta":{"text":text}}),
                    &mut parts,
                    bytes,
                )?;
            }
            for citation in citations
                .get("citations")
                .and_then(JsonValue::as_array)
                .into_iter()
                .flatten()
            {
                state.apply(
                    "contentBlockDelta",
                    serde_json::json!({"contentBlockIndex":index,"delta":{"citation":citation}}),
                    &mut parts,
                    bytes,
                )?;
            }
        } else {
            return Err(invalid_final(
                "Bedrock Converse returned an unsupported content block",
                bytes,
            ));
        }
        state.apply(
            "contentBlockStop",
            serde_json::json!({"contentBlockIndex":index}),
            &mut parts,
            bytes,
        )?;
    }
    state.apply(
        "messageStop",
        serde_json::json!({
            "stopReason":value.get("stopReason"),
            "additionalModelResponseFields":value.get("additionalModelResponseFields")
        }),
        &mut parts,
        bytes,
    )?;
    state.apply("metadata", value, &mut parts, bytes)?;
    Ok((parts, state.response_metadata))
}

fn parse_usage(value: Option<&JsonValue>, bytes: u64) -> Result<Usage, ModelError> {
    let Some(value) = value else {
        return Ok(Usage::default());
    };
    let base = value.get("inputTokens").and_then(JsonValue::as_u64);
    let read = value
        .get("cacheReadInputTokens")
        .and_then(JsonValue::as_u64);
    let write = value
        .get("cacheWriteInputTokens")
        .and_then(JsonValue::as_u64);
    let input = [base, read, write]
        .into_iter()
        .flatten()
        .try_fold(0_u64, |total, value| total.checked_add(value))
        .ok_or_else(|| invalid_event("Bedrock input usage overflowed", bytes))?;
    Ok(Usage {
        input_tokens: base.map(|_| input),
        input_tokens_no_cache: base,
        input_tokens_cache_read: read,
        input_tokens_cache_write: write,
        output_tokens: value.get("outputTokens").and_then(JsonValue::as_u64),
        output_tokens_text: None,
        output_tokens_reasoning: None,
        raw: Some(value.clone()),
    })
}

fn map_finish(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "tool_use" => FinishReason::ToolCalls,
        "max_tokens" => FinishReason::Length,
        "guardrail_intervened" | "content_filtered" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.into()),
    }
}

fn index(payload: &JsonValue, bytes: u64) -> Result<u64, ModelError> {
    payload
        .get("contentBlockIndex")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid_event("Bedrock content block index is invalid", bytes))
}

fn required_string<'a>(value: &'a JsonValue, key: &str, bytes: u64) -> Result<&'a str, ModelError> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid_event("Bedrock event is missing a required string", bytes))
}

fn append_once(
    target: &mut Option<String>,
    value: &str,
    message: &str,
    bytes: u64,
) -> Result<(), ModelError> {
    if value.is_empty() || target.as_ref().is_some_and(|existing| existing != value) {
        return Err(invalid_event(message, bytes));
    }
    *target = Some(value.to_owned());
    Ok(())
}

fn invalid_event(message: &str, bytes: u64) -> ModelError {
    ModelError::invalid_response(message)
        .with_stage(ErrorStage::StreamEvent)
        .with_bytes_received(bytes)
}

fn invalid_final(message: &str, bytes: u64) -> ModelError {
    ModelError::invalid_response(message)
        .with_stage(ErrorStage::StreamFinalize)
        .with_bytes_received(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(model: &str) -> NativeContextScope {
        NativeContextScope::new(
            oven_sdk::ProviderId::new(crate::BEDROCK_PROVIDER_ID),
            oven_sdk::ModelId::new(model),
            oven_sdk::ResourceId::new("test-bedrock-scope").unwrap(),
        )
        .unwrap()
    }

    fn config(model: &str, reasoning: bool, signed_reasoning: bool) -> StreamConfiguration {
        StreamConfiguration {
            policy: ReplayPolicy::IfValid,
            native_context_scope: scope(model),
            reasoning,
            signed_reasoning,
        }
    }

    fn frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let mut encoded_headers = Vec::new();
        for (name, value) in headers {
            encoded_headers.push(name.len() as u8);
            encoded_headers.extend_from_slice(name.as_bytes());
            encoded_headers.push(7);
            encoded_headers.extend_from_slice(&(value.len() as u16).to_be_bytes());
            encoded_headers.extend_from_slice(value.as_bytes());
        }
        let total = 16 + encoded_headers.len() + payload.len();
        let mut frame = Vec::new();
        frame.extend_from_slice(&(total as u32).to_be_bytes());
        frame.extend_from_slice(&(encoded_headers.len() as u32).to_be_bytes());
        frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());
        frame.extend_from_slice(&encoded_headers);
        frame.extend_from_slice(payload);
        frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());
        frame
    }

    fn error_frame() -> Vec<u8> {
        frame(
            &[
                (":message-type", "error"),
                (":error-code", "ThrottlingException"),
                (":error-message", "private detail"),
            ],
            &[],
        )
    }

    fn message_start_frame() -> Vec<u8> {
        frame(
            &[(":message-type", "event"), (":event-type", "messageStart")],
            br#"{"role":"assistant"}"#,
        )
    }

    fn live(chunks: Vec<Vec<u8>>) -> LiveState {
        LiveState {
            bytes: Box::pin(futures_util::stream::iter(
                chunks
                    .into_iter()
                    .map(|chunk| Ok::<_, reqwest::Error>(bytes::Bytes::from(chunk))),
            )),
            decoder: Decoder::new(1024 * 1024),
            state: State::new(config("unknown", false, false)),
            queue: VecDeque::new(),
            pending_messages: VecDeque::new(),
            terminal_queue: VecDeque::new(),
            pending_error: None,
            abort: AbortSignal::default(),
            idle: Duration::from_secs(1),
            count: 0,
            eof: false,
            include_raw: false,
        }
    }

    #[test]
    fn text_tool_reasoning_usage_and_replay_are_strict() {
        let mut state = State::new(config("anthropic.claude-sonnet-4-6", true, true));
        let mut parts = Vec::new();
        state
            .apply(
                "messageStart",
                serde_json::json!({"role":"assistant"}),
                &mut parts,
                1,
            )
            .unwrap();
        state.apply("contentBlockDelta", serde_json::json!({"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"think"}}}), &mut parts, 2).unwrap();
        state.apply("contentBlockDelta", serde_json::json!({"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"sig"}}}), &mut parts, 3).unwrap();
        state
            .apply(
                "contentBlockStop",
                serde_json::json!({"contentBlockIndex":0}),
                &mut parts,
                4,
            )
            .unwrap();
        state.apply("contentBlockStart", serde_json::json!({"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"call","name":"lookup"}}}), &mut parts, 5).unwrap();
        state
            .apply(
                "contentBlockDelta",
                serde_json::json!({"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"x\":"}}}),
                &mut parts,
                6,
            )
            .unwrap();
        state
            .apply(
                "contentBlockDelta",
                serde_json::json!({"contentBlockIndex":1,"delta":{"toolUse":{"input":"1}"}}}),
                &mut parts,
                7,
            )
            .unwrap();
        state
            .apply(
                "contentBlockStop",
                serde_json::json!({"contentBlockIndex":1}),
                &mut parts,
                8,
            )
            .unwrap();
        state
            .apply(
                "messageStop",
                serde_json::json!({"stopReason":"tool_use"}),
                &mut parts,
                9,
            )
            .unwrap();
        state.apply("metadata", serde_json::json!({"usage":{"inputTokens":10,"cacheReadInputTokens":2,"cacheWriteInputTokens":3,"outputTokens":4}}), &mut parts, 10).unwrap();
        let StreamPart::Finish { finish } = parts.last().unwrap() else {
            panic!("finish")
        };
        assert_eq!(finish.finish_reason, FinishReason::ToolCalls);
        assert_eq!(finish.usage.input_tokens, Some(15));
        assert!(finish.native_replay.is_some());
    }

    #[test]
    fn missing_signature_malformed_tool_and_event_order_fail() {
        let mut state = State::new(config("anthropic.claude-sonnet-4-6", true, true));
        let mut parts = Vec::new();
        assert!(
            state
                .apply(
                    "contentBlockDelta",
                    serde_json::json!({"contentBlockIndex":0,"delta":{"text":"x"}}),
                    &mut parts,
                    1
                )
                .is_err()
        );
        state
            .apply(
                "messageStart",
                serde_json::json!({"role":"assistant"}),
                &mut parts,
                1,
            )
            .unwrap();
        state.apply("contentBlockDelta", serde_json::json!({"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"x"}}}), &mut parts, 2).unwrap();
        assert!(
            state
                .apply(
                    "contentBlockStop",
                    serde_json::json!({"contentBlockIndex":0}),
                    &mut parts,
                    3
                )
                .is_err()
        );
    }

    #[test]
    fn provider_exception_is_error_then_finish() {
        let mut state = State::new(config("unknown", false, false));
        let mut parts = Vec::new();
        state
            .apply(
                "throttlingException",
                serde_json::json!({"message":"slow"}),
                &mut parts,
                7,
            )
            .unwrap();
        assert!(
            matches!(parts.as_slice(), [StreamPart::Error { .. }, StreamPart::Finish { finish }] if finish.finish_reason == FinishReason::Error)
        );
    }

    #[tokio::test]
    async fn terminal_parts_are_released_only_after_clean_transport_eof() {
        let mut live = live(vec![error_frame()]);
        assert!(early_peek(&mut live).await.is_ok());
        assert!(live.queue.is_empty());
        assert!(!live.eof);
        assert!(!read_live(&mut live, false).await.unwrap());
        assert!(live.eof);
        assert!(matches!(
            live.queue.pop_front(),
            Some(Ok(StreamPart::Error { .. }))
        ));
        assert!(matches!(
            live.queue.pop_front(),
            Some(Ok(StreamPart::Finish { finish })) if finish.finish_reason == FinishReason::Error
        ));
        assert!(live.queue.is_empty());
    }

    #[tokio::test]
    async fn post_terminal_frames_fail_across_same_or_separate_chunks() {
        let mut same = error_frame();
        same.extend(message_start_frame());
        assert!(early_peek(&mut live(vec![same])).await.is_err());

        let mut split = live(vec![error_frame(), message_start_frame()]);
        assert!(early_peek(&mut split).await.is_ok());
        assert!(read_live(&mut split, false).await.is_err());
        assert_eq!(split.terminal_queue.len(), 2);

        let mut malformed = live(vec![error_frame(), vec![0; 11]]);
        assert!(early_peek(&mut malformed).await.is_ok());
        assert!(!read_live(&mut malformed, false).await.unwrap());
        assert!(read_live(&mut malformed, false).await.is_err());
        assert!(malformed.queue.is_empty());
    }
}
