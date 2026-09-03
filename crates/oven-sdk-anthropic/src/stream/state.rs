//! Anthropic stream event state machine.

use std::collections::{BTreeMap, BTreeSet};

use oven_sdk::{
    AdapterId, ErrorStage, Finish, FinishReason, JsonValue, ModelError, ModelErrorKind,
    NativeContextScope, NativeReplayArtifact, ReplayPolicy, StreamPart, ToolCallPart, Usage,
};

use crate::{error::classify_error_for, wire::Protocol};
use reqwest::header::HeaderMap;

use super::blocks::Block;

const MAX_CONTENT_BLOCK_INDEX: u64 = 4095;

pub(crate) struct State {
    pub(super) started: bool,
    pub(super) done: bool,
    blocks: BTreeMap<u64, Block>,
    block_order: BTreeMap<u64, u64>,
    next_block_order: u64,
    usage: Usage,
    stop: Option<String>,
    stop_sequence: Option<String>,
    native: BTreeMap<u64, JsonValue>,
    tool_ids: BTreeSet<String>,
    policy: ReplayPolicy,
    response_metadata: std::collections::BTreeMap<String, JsonValue>,
    request_id: Option<String>,
    protocol: Protocol,
    adapter_id: AdapterId,
    native_context_scope: NativeContextScope,
}
impl State {
    pub(crate) fn new(
        policy: ReplayPolicy,
        protocol: Protocol,
        adapter_id: AdapterId,
        native_context_scope: NativeContextScope,
    ) -> Self {
        Self {
            started: false,
            done: false,
            blocks: BTreeMap::new(),
            block_order: BTreeMap::new(),
            next_block_order: 0,
            usage: Usage::default(),
            stop: None,
            stop_sequence: None,
            native: BTreeMap::new(),
            tool_ids: BTreeSet::new(),
            policy,
            response_metadata: std::collections::BTreeMap::new(),
            request_id: None,
            protocol,
            adapter_id,
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
                if self.blocks.contains_key(&i) {
                    self.apply(
                        "content_block_stop",
                        serde_json::json!({"index":i}),
                        parts,
                        bytes,
                    )?;
                }
                let block = value
                    .get("content_block")
                    .ok_or_else(|| invalid_event("missing content block", bytes))?;
                let ty = block.get("type").and_then(JsonValue::as_str).unwrap_or("");
                let id = format!("{ty}:{i}");
                let order = self.next_block_order;
                self.next_block_order = self.next_block_order.saturating_add(1);
                let b = match ty {
                    "text" => {
                        parts.push(StreamPart::TextStart { id, metadata: None });
                        Block::Text {
                            text: String::new(),
                        }
                    }
                    "thinking" | "redacted_thinking" => {
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
                        let provider_id = block
                            .get("id")
                            .and_then(JsonValue::as_str)
                            .filter(|v| !v.is_empty());
                        let call_id = reserve_tool_id(&mut self.tool_ids, provider_id, i);
                        let name = block
                            .get("name")
                            .and_then(JsonValue::as_str)
                            .unwrap_or_default()
                            .to_owned();
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
                    _ => {
                        parts.push(StreamPart::ProviderEvent {
                            name: format!("{}.content_block", self.protocol.metadata_namespace()),
                            data: block.clone(),
                        });
                        Block::Custom
                    }
                };
                self.block_order.insert(i, order);
                self.blocks.insert(i, b);
                Ok(())
            }
            "content_block_delta" => {
                let i = value
                    .get("index")
                    .and_then(JsonValue::as_u64)
                    .ok_or_else(|| invalid_event("missing content block index", bytes))?;
                validate_block_index(i, bytes)?;
                let delta = value
                    .get("delta")
                    .ok_or_else(|| invalid_event("missing content block delta", bytes))?;
                let ty = delta.get("type").and_then(JsonValue::as_str).unwrap_or("");
                let Some(block) = self.blocks.get_mut(&i) else {
                    parts.push(StreamPart::ProviderEvent {
                        name: format!("{}.content_block_delta", self.protocol.metadata_namespace()),
                        data: value,
                    });
                    return Ok(());
                };
                match block {
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
                    Block::Custom => parts.push(StreamPart::ProviderEvent {
                        name: format!("{}.content_block_delta", self.protocol.metadata_namespace()),
                        data: value.clone(),
                    }),
                    _ => parts.push(StreamPart::ProviderEvent {
                        name: format!("{}.content_block_delta", self.protocol.metadata_namespace()),
                        data: value.clone(),
                    }),
                }
                Ok(())
            }
            "content_block_stop" => {
                let i = value
                    .get("index")
                    .and_then(JsonValue::as_u64)
                    .ok_or_else(|| invalid_event("missing content block index", bytes))?;
                validate_block_index(i, bytes)?;
                let Some(block) = self.blocks.remove(&i) else {
                    return Ok(());
                };
                let order = self.block_order.remove(&i).unwrap_or(self.next_block_order);
                match block {
                    Block::Text { text } => {
                        self.push_native(order, serde_json::json!({"type":"text","text":text}));
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
                            let data = data.unwrap_or_else(|| JsonValue::String(String::new()));
                            serde_json::json!({"type":"redacted_thinking","data":data})
                        } else {
                            // The native schema always carries this field; a missing delta is
                            // represented as the equivalent empty signature.
                            let signature = signature.unwrap_or_default();
                            serde_json::json!({"type":"thinking","thinking":text,"signature":signature})
                        };
                        self.push_native(order, native);
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
                        self.push_native(order, serde_json::json!({"type":"tool_use","id":call.id,"name":call.name,"input":call.input}));
                        parts.push(StreamPart::ToolCallEnd {
                            id: call.id.clone(),
                            metadata: None,
                        });
                        parts.push(StreamPart::ToolCall { tool_call: call });
                    }
                    Block::Custom => parts.push(StreamPart::ProviderEvent {
                        name: format!("{}.content_block_stop", self.protocol.metadata_namespace()),
                        data: value,
                    }),
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
                let open = self.blocks.keys().copied().collect::<Vec<_>>();
                for index in open {
                    self.apply(
                        "content_block_stop",
                        serde_json::json!({"index":index}),
                        parts,
                        bytes,
                    )?;
                }
                let mut finish = Finish::new(
                    std::mem::take(&mut self.usage),
                    map_stop(self.stop.as_deref()),
                );
                finish.response_metadata = std::mem::take(&mut self.response_metadata);
                if self.policy != ReplayPolicy::Never {
                    let content = std::mem::take(&mut self.native)
                        .into_values()
                        .collect::<Vec<_>>();
                    let payload = serde_json::json!({"format":self.protocol.replay_format(),"message":{"role":"assistant","content":content},"stop_reason":self.stop.take(),"stop_sequence":self.stop_sequence.take()});
                    finish.native_replay = Some(
                        NativeReplayArtifact::new(
                            self.adapter_id.clone(),
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
                        Block::Custom => {}
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
            _ => {
                parts.push(StreamPart::ProviderEvent {
                    name: format!("{}.stream_event.{kind}", self.protocol.metadata_namespace()),
                    data: value,
                });
                Ok(())
            }
        }
    }
    fn require_started(&self, bytes: u64) -> Result<(), ModelError> {
        if self.started {
            Ok(())
        } else {
            Err(invalid_event("event before message_start", bytes))
        }
    }
    fn push_native(&mut self, order: u64, value: JsonValue) {
        self.native.insert(order, value);
    }
}

fn reserve_tool_id(
    used: &mut BTreeSet<String>,
    provider_id: Option<&str>,
    fallback: u64,
) -> String {
    let base = provider_id
        .map(str::to_owned)
        .unwrap_or_else(|| format!("google-call-{fallback}"));
    if used.insert(base.clone()) {
        return base;
    }
    for suffix in 1_u64.. {
        let candidate = format!("{base}-{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("unbounded tool ID suffix space")
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
            Protocol::Anthropic.adapter_id(),
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
            Protocol::Anthropic.adapter_id(),
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
            Protocol::Anthropic.adapter_id(),
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
