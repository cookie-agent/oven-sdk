//! Anthropic stream event state machine.

use std::collections::{BTreeMap, BTreeSet};

use oven_sdk::{
    ErrorStage, Finish, FinishReason, JsonValue, ModelError, ModelErrorKind, NativeContextScope,
    NativeReplayArtifact, ReplayPolicy, StreamPart, ToolCallPart, Usage,
};

use crate::{error::classify_error_for, wire::Protocol};
use reqwest::header::HeaderMap;

use super::blocks::Block;

const MAX_CONTENT_BLOCK_INDEX: u64 = 4095;

pub(crate) struct State {
    pub(super) started: bool,
    pub(super) done: bool,
    blocks: BTreeMap<u64, Block>,
    usage: Usage,
    stop: Option<String>,
    stop_sequence: Option<String>,
    tool_ids: BTreeSet<String>,
    native: Vec<JsonValue>,
    pending_native: BTreeMap<u64, JsonValue>,
    finalized_indices: BTreeSet<u64>,
    next_expected_start_index: u64,
    policy: ReplayPolicy,
    response_metadata: std::collections::BTreeMap<String, JsonValue>,
    request_id: Option<String>,
    protocol: Protocol,
    native_context_scope: NativeContextScope,
}
impl State {
    pub(crate) fn new(
        policy: ReplayPolicy,
        protocol: Protocol,
        native_context_scope: NativeContextScope,
    ) -> Self {
        Self {
            started: false,
            done: false,
            blocks: BTreeMap::new(),
            usage: Usage::default(),
            stop: None,
            stop_sequence: None,
            tool_ids: BTreeSet::new(),
            native: Vec::new(),
            pending_native: BTreeMap::new(),
            finalized_indices: BTreeSet::new(),
            next_expected_start_index: 0,
            policy,
            response_metadata: std::collections::BTreeMap::new(),
            request_id: None,
            protocol,
            native_context_scope,
        }
    }
    pub(crate) fn response_metadata(&self) -> &std::collections::BTreeMap<String, JsonValue> {
        &self.response_metadata
    }
    pub(crate) fn set_request_id(&mut self, request_id: Option<String>) {
        if let Some(request_id) = &request_id {
            self.response_metadata.insert(
                format!("{}.request_id", self.protocol.metadata_namespace()),
                JsonValue::String(request_id.clone()),
            );
        } else {
            self.response_metadata.remove(&format!(
                "{}.request_id",
                self.protocol.metadata_namespace()
            ));
        }
        self.request_id = request_id;
    }
    pub(crate) fn apply(
        &mut self,
        event: &str,
        value: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let kind = if event.is_empty() {
            value.get("type").and_then(JsonValue::as_str).unwrap_or("")
        } else {
            event
        };
        if self.done {
            return Err(invalid_event("event after message_stop", bytes));
        }
        match kind {
            "ping" => Ok(()),
            "message_start" => {
                if self.started {
                    return Err(invalid_event("duplicate message_start", bytes));
                }
                self.started = true;
                self.usage = usage(
                    value.pointer("/message/usage").unwrap_or(&JsonValue::Null),
                    bytes,
                )?;
                self.stop = value
                    .pointer("/message/stop_reason")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                self.stop_sequence = value
                    .pointer("/message/stop_sequence")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                for (source, suffix) in [("id", "message_id"), ("model", "model")] {
                    if let Some(value) = value
                        .pointer("/message")
                        .and_then(|message| message.get(source))
                        .cloned()
                    {
                        self.response_metadata.insert(
                            format!("{}.{suffix}", self.protocol.metadata_namespace()),
                            value,
                        );
                    }
                }
                if let Some(content) = value
                    .pointer("/message/content")
                    .and_then(JsonValue::as_array)
                {
                    for (index, block) in content.iter().enumerate() {
                        let index = u64::try_from(index)
                            .map_err(|_| invalid_event("content block index overflowed", bytes))?;
                        validate_block_index(index, bytes)?;
                        let start = serde_json::json!({"index": index, "content_block": block});
                        self.apply("content_block_start", start, parts, bytes)?;
                        match block.get("type").and_then(JsonValue::as_str) {
                            Some("text") if block.get("text").and_then(JsonValue::as_str).is_some_and(|text| !text.is_empty()) => self.apply("content_block_delta", serde_json::json!({"index":index,"delta":{"type":"text_delta","text":block["text"]}}), parts, bytes)?,
                            Some("thinking") if block.get("thinking").and_then(JsonValue::as_str).is_some_and(|text| !text.is_empty()) => self.apply("content_block_delta", serde_json::json!({"index":index,"delta":{"type":"thinking_delta","thinking":block["thinking"]}}), parts, bytes)?,
                            _ => {}
                        }
                        if let Some(signature) = block.get("signature").and_then(JsonValue::as_str)
                        {
                            self.apply("content_block_delta", serde_json::json!({"index":index,"delta":{"type":"signature_delta","signature":signature}}), parts, bytes)?;
                        }
                        if block.get("type").and_then(JsonValue::as_str) == Some("tool_use") {
                            self.apply(
                                "content_block_stop",
                                serde_json::json!({"index":index}),
                                parts,
                                bytes,
                            )?;
                        }
                    }
                }
                Ok(())
            }
            "content_block_start" => {
                self.require_started(bytes)?;
                let i = value
                    .get("index")
                    .and_then(JsonValue::as_u64)
                    .ok_or_else(|| invalid_event("missing content block index", bytes))?;
                validate_block_index(i, bytes)?;
                self.reject_finalized_reuse(i, bytes)?;
                if self.blocks.contains_key(&i) {
                    return Err(invalid_event("duplicate content block index", bytes));
                }
                if i != self.next_expected_start_index {
                    return Err(invalid_event(
                        "Messages content block starts must be contiguous and ordered",
                        bytes,
                    ));
                }
                let block = value
                    .get("content_block")
                    .ok_or_else(|| invalid_event("missing content block", bytes))?;
                let ty = block.get("type").and_then(JsonValue::as_str).unwrap_or("");
                let id = format!("{ty}:{i}");
                let b = match ty {
                    "text" => {
                        parts.push(StreamPart::TextStart { id, metadata: None });
                        Block::Text {
                            text: String::new(),
                        }
                    }
                    "thinking" | "redacted_thinking" => {
                        if ty == "redacted_thinking" && self.protocol == Protocol::MiniMax {
                            return Err(invalid_event(
                                "MiniMax does not support redacted_thinking blocks",
                                bytes,
                            ));
                        }
                        let metadata = (ty == "redacted_thinking").then(|| {
                            BTreeMap::from([(
                                "anthropic.redacted".into(),
                                block.get("data").cloned().unwrap_or(JsonValue::Null),
                            )])
                        });
                        parts.push(StreamPart::ReasoningStart { id, metadata });
                        Block::Thinking {
                            redacted: ty == "redacted_thinking",
                            text: String::new(),
                            data: block.get("data").cloned(),
                            signature: None,
                        }
                    }
                    "tool_use" => {
                        let call_id = block
                            .get("id")
                            .and_then(JsonValue::as_str)
                            .filter(|v| !v.is_empty())
                            .ok_or_else(|| invalid_event("tool use is missing id", bytes))?
                            .to_owned();
                        let name = block
                            .get("name")
                            .and_then(JsonValue::as_str)
                            .filter(|v| !v.is_empty())
                            .ok_or_else(|| invalid_event("tool use is missing name", bytes))?
                            .to_owned();
                        if !self.tool_ids.insert(call_id.clone()) {
                            return Err(invalid_event("duplicate tool call id", bytes));
                        }
                        parts.push(StreamPart::ToolCallStart {
                            id: call_id.clone(),
                            name: name.clone(),
                            metadata: None,
                        });
                        let input = block
                            .get("input")
                            .filter(|v| !v.is_null() && *v != &serde_json::json!({}))
                            .map(JsonValue::to_string)
                            .unwrap_or_default();
                        if !input.is_empty() {
                            parts.push(StreamPart::ToolCallDelta {
                                id: call_id.clone(),
                                delta: input.clone(),
                                metadata: None,
                            });
                        }
                        Block::Tool {
                            id: call_id,
                            name,
                            input,
                        }
                    }
                    _ => return Err(invalid_event("unsupported Messages content block", bytes)),
                };
                self.next_expected_start_index = self
                    .next_expected_start_index
                    .checked_add(1)
                    .ok_or_else(|| invalid_event("content block start index overflowed", bytes))?;
                self.blocks.insert(i, b);
                Ok(())
            }
            "content_block_delta" => {
                let i = value
                    .get("index")
                    .and_then(JsonValue::as_u64)
                    .ok_or_else(|| invalid_event("missing content block index", bytes))?;
                validate_block_index(i, bytes)?;
                self.reject_finalized_reuse(i, bytes)?;
                let delta = value
                    .get("delta")
                    .ok_or_else(|| invalid_event("missing content block delta", bytes))?;
                let ty = delta.get("type").and_then(JsonValue::as_str).unwrap_or("");
                match self
                    .blocks
                    .get_mut(&i)
                    .ok_or_else(|| invalid_event("delta for unknown block", bytes))?
                {
                    Block::Text { text } if ty == "text_delta" => {
                        let id = format!("text:{i}");
                        let value = delta.get("text").and_then(JsonValue::as_str).unwrap_or("");
                        text.push_str(value);
                        parts.push(StreamPart::TextDelta {
                            id,
                            delta: value.into(),
                            metadata: None,
                        });
                    }
                    Block::Thinking { text, .. } if ty == "thinking_delta" => {
                        let value = delta
                            .get("thinking")
                            .and_then(JsonValue::as_str)
                            .unwrap_or("");
                        text.push_str(value);
                        parts.push(StreamPart::ReasoningDelta {
                            id: format!("thinking:{i}"),
                            delta: value.into(),
                            metadata: None,
                        });
                    }
                    Block::Thinking { signature, .. } if ty == "signature_delta" => {
                        if let Some(piece) = delta.get("signature").and_then(JsonValue::as_str) {
                            signature.get_or_insert_with(String::new).push_str(piece);
                        }
                    }
                    Block::Tool { id, input, .. } if ty == "input_json_delta" => {
                        let piece = delta
                            .get("partial_json")
                            .and_then(JsonValue::as_str)
                            .unwrap_or("");
                        input.push_str(piece);
                        if !piece.is_empty() {
                            parts.push(StreamPart::ToolCallDelta {
                                id: id.clone(),
                                delta: piece.into(),
                                metadata: None,
                            });
                        }
                    }
                    _ => return Err(invalid_event("content block delta type mismatch", bytes)),
                }
                Ok(())
            }
            "content_block_stop" => {
                let i = value
                    .get("index")
                    .and_then(JsonValue::as_u64)
                    .ok_or_else(|| invalid_event("missing content block index", bytes))?;
                validate_block_index(i, bytes)?;
                self.reject_finalized_reuse(i, bytes)?;
                match self
                    .blocks
                    .remove(&i)
                    .ok_or_else(|| invalid_event("stop for unknown block", bytes))?
                {
                    Block::Text { text } => {
                        self.push_native(i, serde_json::json!({"type":"text","text":text}), bytes)?;
                        parts.push(StreamPart::TextEnd {
                            id: format!("text:{i}"),
                            metadata: None,
                        });
                    }
                    Block::Thinking {
                        redacted,
                        text,
                        data,
                        signature,
                    } => {
                        let native = if redacted {
                            let data = data
                                .filter(|data| {
                                    data.as_str().is_some_and(|data| !data.is_empty())
                                })
                                .ok_or_else(|| {
                                    invalid_finalized_reasoning(
                                        "Anthropic redacted_thinking requires non-empty string data",
                                        bytes,
                                    )
                                })?;
                            serde_json::json!({"type":"redacted_thinking","data":data})
                        } else {
                            let signature = signature
                                .filter(|signature| !signature.is_empty())
                                .ok_or_else(|| {
                                    invalid_finalized_reasoning(
                                        "Messages thinking requires a non-empty signature",
                                        bytes,
                                    )
                                })?;
                            serde_json::json!({"type":"thinking","thinking":text,"signature":signature})
                        };
                        self.push_native(i, native, bytes)?;
                        parts.push(StreamPart::ReasoningEnd {
                            id: format!(
                                "{}:{i}",
                                if redacted {
                                    "redacted_thinking"
                                } else {
                                    "thinking"
                                }
                            ),
                            metadata: None,
                        });
                    }
                    Block::Tool { id, name, input } => {
                        let raw = if input.is_empty() { "{}".into() } else { input };
                        let parsed: JsonValue = serde_json::from_str(&raw).map_err(|_| {
                            ModelError::new(
                                ModelErrorKind::InvalidResponse,
                                "final Messages tool input is invalid JSON",
                            )
                            .with_stage(ErrorStage::StreamFinalize)
                            .with_bytes_received(bytes)
                        })?;
                        if !parsed.is_object() {
                            return Err(ModelError::new(
                                ModelErrorKind::InvalidResponse,
                                "final Messages tool input must be a JSON object",
                            )
                            .with_stage(ErrorStage::StreamFinalize)
                            .with_bytes_received(bytes));
                        }
                        let mut call = ToolCallPart::new(id, name, parsed);
                        call.raw_input = Some(raw);
                        self.push_native(i, serde_json::json!({"type":"tool_use","id":call.id,"name":call.name,"input":call.input}), bytes)?;
                        parts.push(StreamPart::ToolCallEnd {
                            id: call.id.clone(),
                            metadata: None,
                        });
                        parts.push(StreamPart::ToolCall { tool_call: call });
                    }
                }
                Ok(())
            }
            "message_delta" => {
                self.require_started(bytes)?;
                merge_usage(
                    &mut self.usage,
                    value.get("usage").unwrap_or(&JsonValue::Null),
                    bytes,
                )?;
                self.stop = value
                    .pointer("/delta/stop_reason")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                self.stop_sequence = value
                    .pointer("/delta/stop_sequence")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                Ok(())
            }
            "message_stop" => {
                self.require_started(bytes)?;
                if !self.blocks.is_empty() {
                    return Err(invalid_event(
                        "message_stop arrived before all content blocks stopped",
                        bytes,
                    ));
                }
                self.validate_native_complete(bytes)?;
                let mut finish = Finish::new(self.usage.clone(), map_stop(self.stop.as_deref()));
                finish.response_metadata = self.response_metadata.clone();
                if self.policy != ReplayPolicy::Never {
                    let payload = serde_json::json!({"format":self.protocol.replay_format(),"message":{"role":"assistant","content":self.native},"stop_reason":self.stop,"stop_sequence":self.stop_sequence});
                    finish.native_replay = Some(
                        NativeReplayArtifact::new(
                            self.protocol.adapter_id(),
                            self.native_context_scope.clone(),
                            payload,
                        )
                        .map_err(|_| {
                            ModelError::replay(
                                "Messages native replay artifact exceeds its size limit",
                            )
                            .with_stage(ErrorStage::ReplayEncode)
                        })?,
                    );
                }
                parts.push(StreamPart::Finish { finish });
                self.done = true;
                Ok(())
            }
            "error" => {
                self.require_started(bytes)?;
                let interrupted_tool = self
                    .blocks
                    .values()
                    .any(|block| matches!(block, Block::Tool { .. }));
                let open = std::mem::take(&mut self.blocks);
                for (index, block) in open {
                    match block {
                        Block::Text { .. } => parts.push(StreamPart::TextEnd {
                            id: format!("text:{index}"),
                            metadata: None,
                        }),
                        Block::Thinking { redacted, .. } => parts.push(StreamPart::ReasoningEnd {
                            id: format!(
                                "{}:{index}",
                                if redacted {
                                    "redacted_thinking"
                                } else {
                                    "thinking"
                                }
                            ),
                            metadata: None,
                        }),
                        Block::Tool { .. } => {}
                    }
                }
                if interrupted_tool {
                    self.done = true;
                    return Err(ModelError::invalid_response(
                        "Anthropic error interrupted an open tool block",
                    )
                    .with_stage(ErrorStage::StreamEvent)
                    .with_bytes_received(bytes));
                }
                let error = classify_error_for(
                    self.protocol,
                    if value.pointer("/error/type").and_then(JsonValue::as_str)
                        == Some("overloaded_error")
                    {
                        529
                    } else {
                        500
                    },
                    value.to_string().as_bytes(),
                    self.request_id.clone(),
                    ErrorStage::StreamEvent,
                    bytes,
                    &HeaderMap::new(),
                );
                parts.push(StreamPart::Error { error });
                let mut finish = Finish::new(self.usage.clone(), FinishReason::Error);
                finish.response_metadata = self.response_metadata.clone();
                parts.push(StreamPart::Finish { finish });
                self.done = true;
                Ok(())
            }
            _ => Err(invalid_event("unknown Messages stream event", bytes)),
        }
    }
    fn require_started(&self, bytes: u64) -> Result<(), ModelError> {
        if self.started {
            Ok(())
        } else {
            Err(invalid_event("event before message_start", bytes))
        }
    }
    fn push_native(&mut self, index: u64, value: JsonValue, bytes: u64) -> Result<(), ModelError> {
        validate_block_index(index, bytes)?;
        self.reject_finalized_reuse(index, bytes)?;
        self.finalized_indices.insert(index);
        self.pending_native.insert(index, value);
        loop {
            let next = u64::try_from(self.native.len()).map_err(|_| {
                invalid_finalized_index("native content length does not fit provider index", bytes)
            })?;
            let Some(value) = self.pending_native.remove(&next) else {
                break;
            };
            self.native.push(value);
        }
        Ok(())
    }

    fn reject_finalized_reuse(&self, index: u64, bytes: u64) -> Result<(), ModelError> {
        if self.finalized_indices.contains(&index) {
            Err(invalid_event(
                "Messages content block index was reused after stop",
                bytes,
            ))
        } else {
            Ok(())
        }
    }

    fn validate_native_complete(&self, bytes: u64) -> Result<(), ModelError> {
        let expected = usize::try_from(self.next_expected_start_index).map_err(|_| {
            invalid_finalized_index("provider start count does not fit usize", bytes)
        })?;
        if !self.pending_native.is_empty()
            || self.finalized_indices.len() != expected
            || self.native.len() != expected
        {
            return Err(invalid_finalized_index(
                "Messages native content is incomplete",
                bytes,
            ));
        }
        for (expected, index) in self.finalized_indices.iter().copied().enumerate() {
            let expected = u64::try_from(expected).map_err(|_| {
                invalid_finalized_index("native content length does not fit provider index", bytes)
            })?;
            if index != expected {
                return Err(invalid_finalized_index(
                    "Messages native content indices are not contiguous",
                    bytes,
                ));
            }
        }
        Ok(())
    }
}
fn validate_block_index(index: u64, bytes: u64) -> Result<(), ModelError> {
    if index > MAX_CONTENT_BLOCK_INDEX {
        Err(invalid_event(
            "Messages content block index is unreasonable",
            bytes,
        ))
    } else {
        Ok(())
    }
}
fn invalid_event(message: &str, bytes: u64) -> ModelError {
    ModelError::invalid_response(message)
        .with_stage(ErrorStage::StreamEvent)
        .with_bytes_received(bytes)
}
fn invalid_finalized_reasoning(message: &str, bytes: u64) -> ModelError {
    ModelError::invalid_response(message)
        .with_stage(ErrorStage::StreamFinalize)
        .with_bytes_received(bytes)
}
fn invalid_finalized_index(message: &str, bytes: u64) -> ModelError {
    ModelError::invalid_response(message)
        .with_stage(ErrorStage::StreamFinalize)
        .with_bytes_received(bytes)
}
fn usage(v: &JsonValue, bytes: u64) -> Result<Usage, ModelError> {
    let input = v.get("input_tokens").and_then(JsonValue::as_u64);
    let read = v.get("cache_read_input_tokens").and_then(JsonValue::as_u64);
    let write = v
        .get("cache_creation_input_tokens")
        .and_then(JsonValue::as_u64);
    let input_tokens = match (input, read, write) {
        (None, None, None) => None,
        _ => Some(
            input
                .unwrap_or(0)
                .checked_add(read.unwrap_or(0))
                .and_then(|total| total.checked_add(write.unwrap_or(0)))
                .ok_or_else(|| invalid_event("Messages input usage overflowed", bytes))?,
        ),
    };
    let output_tokens = v.get("output_tokens").and_then(JsonValue::as_u64);
    let output_tokens_reasoning = v
        .pointer("/output_tokens_details/thinking_tokens")
        .and_then(JsonValue::as_u64);
    let output_tokens_text = match (output_tokens, output_tokens_reasoning) {
        (Some(total), Some(reasoning)) => Some(
            total
                .checked_sub(reasoning)
                .ok_or_else(|| invalid_event("Messages output usage is inconsistent", bytes))?,
        ),
        _ => None,
    };
    Ok(Usage {
        input_tokens,
        input_tokens_no_cache: input,
        input_tokens_cache_read: read,
        input_tokens_cache_write: write,
        output_tokens,
        output_tokens_text,
        output_tokens_reasoning,
        raw: (!v.is_null()).then(|| v.clone()),
    })
}

fn merge_usage(current: &mut Usage, value: &JsonValue, bytes: u64) -> Result<(), ModelError> {
    if value.is_null() {
        return Ok(());
    }
    let update = usage(value, bytes)?;
    if update.input_tokens_no_cache.is_some() {
        current.input_tokens_no_cache = update.input_tokens_no_cache;
    }
    if update.input_tokens_cache_read.is_some() {
        current.input_tokens_cache_read = update.input_tokens_cache_read;
    }
    if update.input_tokens_cache_write.is_some() {
        current.input_tokens_cache_write = update.input_tokens_cache_write;
    }
    if update.input_tokens_no_cache.is_some()
        || update.input_tokens_cache_read.is_some()
        || update.input_tokens_cache_write.is_some()
    {
        current.input_tokens = Some(
            current
                .input_tokens_no_cache
                .unwrap_or(0)
                .checked_add(current.input_tokens_cache_read.unwrap_or(0))
                .and_then(|total| total.checked_add(current.input_tokens_cache_write.unwrap_or(0)))
                .ok_or_else(|| invalid_event("Messages merged input usage overflowed", bytes))?,
        );
    }
    if update.output_tokens.is_some() {
        current.output_tokens = update.output_tokens;
    }
    if update.output_tokens_text.is_some() {
        current.output_tokens_text = update.output_tokens_text;
    }
    if update.output_tokens_reasoning.is_some() {
        current.output_tokens_reasoning = update.output_tokens_reasoning;
    }
    if let (Some(total), Some(reasoning)) = (current.output_tokens, current.output_tokens_reasoning)
    {
        current.output_tokens_text =
            Some(total.checked_sub(reasoning).ok_or_else(|| {
                invalid_event("Messages merged output usage is inconsistent", bytes)
            })?);
    }
    if let Some(update) = update.raw {
        match (&mut current.raw, update) {
            (Some(JsonValue::Object(current)), JsonValue::Object(update)) => {
                current.extend(update);
            }
            (slot, update) => *slot = Some(update),
        }
    }
    Ok(())
}
fn map_stop(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn" | "stop_sequence" | "pause_turn") => FinishReason::Stop,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("max_tokens" | "model_context_window_exceeded") => FinishReason::Length,
        Some("refusal") => FinishReason::Refused,
        None => FinishReason::Unknown,
        Some(other) => FinishReason::Other(other.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_context_scope() -> NativeContextScope {
        NativeContextScope::new(
            oven_sdk::ProviderId::new("anthropic"),
            oven_sdk::ModelId::new("claude"),
            oven_sdk::ResourceId::new("test-resource").expect("valid resource"),
        )
        .expect("valid native-context scope")
    }

    #[test]
    fn malformed_final_tool_input_is_invalid_tool_input() {
        let mut state = State::new(
            ReplayPolicy::IfValid,
            Protocol::Anthropic,
            native_context_scope(),
        );
        let mut parts = Vec::new();
        for (event, value) in [
            ("message_start", serde_json::json!({"message":{}})),
            (
                "content_block_start",
                serde_json::json!({"index":0,"content_block":{"type":"tool_use","id":"call","name":"lookup"}}),
            ),
            (
                "content_block_delta",
                serde_json::json!({"index":0,"delta":{"type":"input_json_delta","partial_json":"["}}),
            ),
        ] {
            state.apply(event, value, &mut parts, 12).unwrap();
        }
        let error = state
            .apply(
                "content_block_stop",
                serde_json::json!({"index":0}),
                &mut parts,
                12,
            )
            .unwrap_err();
        assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
    }

    #[test]
    fn message_start_captures_metadata_and_initial_content() {
        let mut state = State::new(
            ReplayPolicy::IfValid,
            Protocol::Anthropic,
            native_context_scope(),
        );
        state.set_request_id(Some("req_header".into()));
        let mut parts = Vec::new();
        state.apply("message_start", serde_json::json!({"message":{"id":"msg_1","model":"claude","request_id":"req_body","content":[{"type":"text","text":"hello"},{"type":"thinking","thinking":"consider","signature":"sig"}]}}), &mut parts, 1).unwrap();
        assert_eq!(state.response_metadata()["anthropic.message_id"], "msg_1");
        assert_eq!(state.response_metadata()["anthropic.model"], "claude");
        assert_eq!(
            state.response_metadata()["anthropic.request_id"],
            "req_header"
        );
        assert!(
            parts.iter().any(
                |part| matches!(part, StreamPart::TextDelta { delta, .. } if delta == "hello")
            )
        );
        assert!(parts.iter().any(
            |part| matches!(part, StreamPart::ReasoningDelta { delta, .. } if delta == "consider")
        ));
        state
            .apply(
                "content_block_stop",
                serde_json::json!({"index":0}),
                &mut parts,
                1,
            )
            .unwrap();
        state
            .apply(
                "content_block_stop",
                serde_json::json!({"index":1}),
                &mut parts,
                1,
            )
            .unwrap();
        state
            .apply("message_stop", serde_json::json!({}), &mut parts, 1)
            .unwrap();
        let finish = parts
            .iter()
            .find_map(|part| match part {
                StreamPart::Finish { finish } => Some(finish),
                _ => None,
            })
            .unwrap();
        assert_eq!(finish.response_metadata["anthropic.message_id"], "msg_1");
        assert_eq!(finish.response_metadata["anthropic.model"], "claude");
        assert_eq!(
            finish.response_metadata["anthropic.request_id"],
            "req_header"
        );
    }

    #[test]
    fn fragmented_signature_replay_preserves_signed_reasoning_and_tool_ids() {
        let mut state = State::new(
            ReplayPolicy::IfValid,
            Protocol::Anthropic,
            native_context_scope(),
        );
        let mut parts = Vec::new();
        for (event, value) in [
            ("message_start", serde_json::json!({"message":{}})),
            (
                "content_block_start",
                serde_json::json!({"index":0,"content_block":{"type":"thinking"}}),
            ),
            (
                "content_block_delta",
                serde_json::json!({"index":0,"delta":{"type":"signature_delta","signature":"sig-"}}),
            ),
            (
                "content_block_delta",
                serde_json::json!({"index":0,"delta":{"type":"signature_delta","signature":"final"}}),
            ),
            (
                "content_block_start",
                serde_json::json!({"index":1,"content_block":{"type":"text"}}),
            ),
            (
                "content_block_start",
                serde_json::json!({"index":2,"content_block":{"type":"tool_use","id":"call_1","name":"lookup","input":{}}}),
            ),
            ("content_block_stop", serde_json::json!({"index":0})),
            ("content_block_stop", serde_json::json!({"index":1})),
            ("content_block_stop", serde_json::json!({"index":2})),
            ("message_stop", serde_json::json!({})),
        ] {
            state.apply(event, value, &mut parts, 1).unwrap();
        }
        assert!(matches!(parts.last(), Some(StreamPart::Finish { .. })));
        let finish = parts
            .into_iter()
            .find_map(|part| match part {
                StreamPart::Finish { finish } => Some(finish),
                _ => None,
            })
            .unwrap();
        let artifact = finish.native_replay.unwrap();
        assert_eq!(
            artifact
                .payload()
                .pointer("/message/content/0/signature")
                .and_then(JsonValue::as_str),
            Some("sig-final")
        );
        assert_eq!(
            artifact
                .payload()
                .pointer("/message/content/2/id")
                .and_then(JsonValue::as_str),
            Some("call_1")
        );
    }
}
