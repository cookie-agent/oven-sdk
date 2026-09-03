//! Chat Completions stream state machine.

use std::collections::BTreeMap;

use oven_sdk::{
    AdapterId, CustomPart, ErrorStage, Finish, FinishReason, JsonValue, ModelError, ModelErrorKind,
    NativeContextScope, NativeReplayArtifact, ReplayPolicy, StreamPart, ToolCallPart, Usage,
};

use crate::{configuration::ReasoningField, wire::chat::REPLAY_FORMAT};

#[derive(Default)]
struct ToolState {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    emitted: usize,
    started: bool,
}

pub(crate) struct State {
    adapter_id: AdapterId,
    scope: NativeContextScope,
    policy: ReplayPolicy,
    reasoning_field: ReasoningField,
    text_open: bool,
    reasoning_open: bool,
    text: String,
    reasoning: String,
    refusal: String,
    tools: BTreeMap<(u64, u64), ToolState>,
    usage: Usage,
    finish_reason: Option<String>,
    response_metadata: BTreeMap<String, JsonValue>,
    done: bool,
}

impl State {
    pub(crate) fn new(
        adapter_id: AdapterId,
        scope: NativeContextScope,
        policy: ReplayPolicy,
        reasoning_field: ReasoningField,
    ) -> Self {
        Self {
            adapter_id,
            scope,
            policy,
            reasoning_field,
            text_open: false,
            reasoning_open: false,
            text: String::new(),
            reasoning: String::new(),
            refusal: String::new(),
            tools: BTreeMap::new(),
            usage: Usage::default(),
            finish_reason: None,
            response_metadata: BTreeMap::new(),
            done: false,
        }
    }

    pub(crate) fn done(&self) -> bool {
        self.done
    }

    pub(crate) fn response_metadata(&self) -> &BTreeMap<String, JsonValue> {
        &self.response_metadata
    }

    pub(crate) fn apply(
        &mut self,
        value: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self.done {
            return Err(invalid_event("Chat event after terminal finish", bytes));
        }
        if value.get("error").is_some() {
            return Err(invalid_event(
                "Chat error envelope must be handled by the stream transport",
                bytes,
            ));
        }
        self.capture_metadata(&value);
        if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
            self.usage = usage_from(usage);
        }
        let choices = value
            .get("choices")
            .and_then(JsonValue::as_array)
            .cloned()
            .unwrap_or_default();
        if choices.len() > 1 {
            return Err(invalid_event(
                "multiple Chat choices are unsupported",
                bytes,
            ));
        }
        for choice in choices {
            let choice_index = choice.get("index").and_then(JsonValue::as_u64).unwrap_or(0);
            if choice_index != 0 {
                return Err(invalid_event(
                    "only Chat choice index 0 is supported",
                    bytes,
                ));
            }
            if let Some(reason) = choice.get("finish_reason").and_then(JsonValue::as_str) {
                self.finish_reason = Some(reason.to_owned());
            }
            let delta = choice.get("delta").unwrap_or(&JsonValue::Null);
            self.apply_reasoning(delta, parts);
            if let Some(content) = delta.get("content").and_then(JsonValue::as_str) {
                self.close_reasoning(parts);
                if !self.text_open {
                    self.text_open = true;
                    parts.push(StreamPart::TextStart {
                        id: "0".into(),
                        metadata: None,
                    });
                }
                self.text.push_str(content);
                parts.push(StreamPart::TextDelta {
                    id: "0".into(),
                    delta: content.into(),
                    metadata: None,
                });
            }
            if let Some(refusal) = delta.get("refusal").and_then(JsonValue::as_str) {
                self.refusal.push_str(refusal);
            }
            if let Some(calls) = delta.get("tool_calls").and_then(JsonValue::as_array) {
                self.close_reasoning(parts);
                for call in calls {
                    self.apply_tool(choice_index, call, parts, bytes)?;
                }
            }
        }
        Ok(())
    }

    fn apply_reasoning(&mut self, delta: &JsonValue, parts: &mut Vec<StreamPart>) {
        let field = match self.reasoning_field {
            ReasoningField::None => return,
            ReasoningField::ReasoningContent => "reasoning_content",
            ReasoningField::Reasoning => "reasoning",
        };
        let Some(reasoning) = delta.get(field).and_then(JsonValue::as_str) else {
            return;
        };
        if !self.reasoning_open {
            self.reasoning_open = true;
            parts.push(StreamPart::ReasoningStart {
                id: "reasoning:0".into(),
                metadata: None,
            });
        }
        self.reasoning.push_str(reasoning);
        parts.push(StreamPart::ReasoningDelta {
            id: "reasoning:0".into(),
            delta: reasoning.into(),
            metadata: None,
        });
    }

    fn apply_tool(
        &mut self,
        choice_index: u64,
        call: &JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let mut index = call.get("index").and_then(JsonValue::as_u64).unwrap_or(0);
        if let Some(id) = call.get("id").and_then(JsonValue::as_str)
            && self
                .tools
                .get(&(choice_index, index))
                .and_then(|state| state.id.as_deref())
                .is_some_and(|existing| existing != id)
        {
            index = self
                .tools
                .keys()
                .filter(|(choice, _)| *choice == choice_index)
                .map(|(_, index)| *index)
                .max()
                .unwrap_or(index)
                .saturating_add(1);
        }
        let state = self.tools.entry((choice_index, index)).or_default();
        if let Some(id) = call
            .get("id")
            .and_then(JsonValue::as_str)
            .filter(|id| !id.is_empty())
        {
            state.id = Some(id.into());
        }
        if let Some(name) = call.pointer("/function/name").and_then(JsonValue::as_str) {
            if state
                .name
                .as_deref()
                .is_some_and(|existing| existing != name)
            {
                return Err(invalid_event("Chat tool name changed", bytes));
            }
            state.name = Some(name.into());
        }
        if let Some(arguments) = call
            .pointer("/function/arguments")
            .and_then(JsonValue::as_str)
        {
            state.arguments.push_str(arguments);
        }
        if !state.started
            && let (Some(id), Some(name)) = (&state.id, &state.name)
        {
            state.started = true;
            parts.push(StreamPart::ToolCallStart {
                id: id.clone(),
                name: name.clone(),
                metadata: None,
            });
        }
        if state.started && state.emitted < state.arguments.len() {
            let delta = state.arguments[state.emitted..].to_owned();
            state.emitted = state.arguments.len();
            let id = state
                .id
                .clone()
                .ok_or_else(|| invalid_event("started Chat tool call is missing its ID", bytes))?;
            parts.push(StreamPart::ToolCallDelta {
                id,
                delta,
                metadata: None,
            });
        }
        Ok(())
    }

    pub(crate) fn finish(
        &mut self,
        done_marker: bool,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self.done {
            return Ok(());
        }
        self.close_reasoning(parts);
        if self.text_open {
            self.text_open = false;
            parts.push(StreamPart::TextEnd {
                id: "0".into(),
                metadata: None,
            });
        }
        let tools = std::mem::take(&mut self.tools);
        let mut native_calls = Vec::new();
        for (call_number, (_, tool)) in tools.into_iter().enumerate() {
            let id = tool
                .id
                .unwrap_or_else(|| format!("google-call-{call_number}"));
            let provider_name = tool.name;
            let name = provider_name.clone().unwrap_or_default();
            if !tool.started {
                parts.push(StreamPart::ToolCallStart {
                    id: id.clone(),
                    name: name.clone(),
                    metadata: None,
                });
                if !tool.arguments.is_empty() {
                    parts.push(StreamPart::ToolCallDelta {
                        id: id.clone(),
                        delta: tool.arguments.clone(),
                        metadata: None,
                    });
                }
            }
            let parsed: JsonValue = serde_json::from_str(&tool.arguments).map_err(|_| {
                invalid_finalize("final Chat tool arguments are invalid JSON", bytes)
            })?;
            if !parsed.is_object() {
                return Err(invalid_finalize(
                    "final Chat tool arguments must be a JSON object",
                    bytes,
                ));
            }
            parts.push(StreamPart::ToolCallEnd {
                id: id.clone(),
                metadata: None,
            });
            let mut call = ToolCallPart::new(id.clone(), name.clone(), parsed);
            call.raw_input = Some(tool.arguments.clone());
            parts.push(StreamPart::ToolCall { tool_call: call });
            let mut native = serde_json::json!({"id":id,"type":"function","function":{"arguments":tool.arguments}});
            if let Some(provider_name) = provider_name {
                native["function"]["name"] = provider_name.into();
            }
            native_calls.push(native);
        }
        if !self.refusal.is_empty() {
            parts.push(StreamPart::Custom {
                part: CustomPart::new("openai.refusal", JsonValue::String(self.refusal.clone())),
            });
        }
        let reason = map_finish(self.finish_reason.as_deref(), done_marker);
        let mut finish = Finish::new(self.usage.clone(), reason);
        finish.response_metadata = self.response_metadata.clone();
        if self.policy != ReplayPolicy::Never {
            let mut message = serde_json::json!({
                "role":"assistant",
                "content":if self.text.is_empty(){JsonValue::Null}else{JsonValue::String(self.text.clone())}
            });
            if !native_calls.is_empty() {
                message["tool_calls"] = JsonValue::Array(native_calls);
            }
            if !self.reasoning.is_empty() {
                match self.reasoning_field {
                    ReasoningField::ReasoningContent => {
                        message["reasoning_content"] = self.reasoning.clone().into();
                    }
                    ReasoningField::Reasoning => {
                        message["reasoning"] = self.reasoning.clone().into();
                    }
                    ReasoningField::None => {}
                }
            }
            if !self.refusal.is_empty() {
                message["refusal"] = self.refusal.clone().into();
            }
            let payload = serde_json::json!({
                "format":REPLAY_FORMAT,
                "message":message,
                "finish_reason":self.finish_reason
            });
            finish.native_replay = Some(
                NativeReplayArtifact::new(self.adapter_id.clone(), self.scope.clone(), payload)
                    .map_err(|_| {
                        ModelError::replay("OpenAI Chat replay artifact exceeds its size limit")
                            .with_stage(ErrorStage::ReplayEncode)
                    })?,
            );
        }
        parts.push(StreamPart::Finish { finish });
        self.done = true;
        Ok(())
    }

    pub(crate) fn in_band_error(
        &mut self,
        error: ModelError,
        parts: &mut Vec<StreamPart>,
    ) -> Result<(), ModelError> {
        if !self.tools.is_empty() {
            return Err(ModelError::invalid_response(
                "OpenAI Chat error interrupted an open tool call",
            )
            .with_stage(ErrorStage::StreamEvent)
            .with_bytes_received(error.diagnostics.bytes_received));
        }
        self.close_reasoning(parts);
        if self.text_open {
            self.text_open = false;
            parts.push(StreamPart::TextEnd {
                id: "0".into(),
                metadata: None,
            });
        }
        parts.push(StreamPart::Error { error });
        let mut finish = Finish::new(self.usage.clone(), FinishReason::Error);
        finish.response_metadata = self.response_metadata.clone();
        parts.push(StreamPart::Finish { finish });
        self.done = true;
        Ok(())
    }

    fn close_reasoning(&mut self, parts: &mut Vec<StreamPart>) {
        if self.reasoning_open {
            self.reasoning_open = false;
            parts.push(StreamPart::ReasoningEnd {
                id: "reasoning:0".into(),
                metadata: None,
            });
        }
    }

    fn capture_metadata(&mut self, value: &JsonValue) {
        for (source, key) in [
            ("id", "openai.response_id"),
            ("model", "openai.model"),
            ("created", "openai.created"),
            ("system_fingerprint", "openai.system_fingerprint"),
        ] {
            if let Some(value) = value.get(source).cloned().filter(|value| !value.is_null()) {
                self.response_metadata.insert(key.into(), value);
            }
        }
    }
}

fn invalid_event(message: &str, bytes: u64) -> ModelError {
    ModelError::invalid_response(message)
        .with_stage(ErrorStage::StreamEvent)
        .with_bytes_received(bytes)
}

fn invalid_finalize(message: &str, bytes: u64) -> ModelError {
    ModelError::new(ModelErrorKind::InvalidResponse, message)
        .with_stage(ErrorStage::StreamFinalize)
        .with_bytes_received(bytes)
}

fn map_finish(reason: Option<&str>, done_marker: bool) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("tool_calls" | "function_call") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        Some("cancelled" | "canceled") => FinishReason::Cancelled,
        Some(other) => FinishReason::Other(other.into()),
        None if done_marker => FinishReason::Unknown,
        None => FinishReason::Unknown,
    }
}

fn usage_from(value: &JsonValue) -> Usage {
    let output = value.get("completion_tokens").and_then(JsonValue::as_u64);
    let reasoning = value
        .pointer("/completion_tokens_details/reasoning_tokens")
        .and_then(JsonValue::as_u64);
    Usage {
        input_tokens: value.get("prompt_tokens").and_then(JsonValue::as_u64),
        input_tokens_no_cache: None,
        input_tokens_cache_read: value
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(JsonValue::as_u64),
        input_tokens_cache_write: None,
        output_tokens: output,
        output_tokens_text: output.map(|total| total.saturating_sub(reasoning.unwrap_or(0))),
        output_tokens_reasoning: reasoning,
        raw: Some(value.clone()),
    }
}
