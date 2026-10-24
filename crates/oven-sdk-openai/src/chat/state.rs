//! Chat Completions stream state machine.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};

use oven_sdk::{
    AdapterId, CustomPart, ErrorStage, Finish, FinishReason, JsonValue, ModelError,
    NativeContextScope, NativeReplayArtifact, ReplayPolicy, StreamPart, ToolCallPart, Usage,
};

use oven_sdk::provider_support::{JsonObject, KeyPresent};
use serde::Deserialize;

use crate::{configuration::ReasoningField, wire::chat::REPLAY_FORMAT};

/// One Chat Completions chunk.
///
/// The SSE loop decodes it straight from the event payload; when that typed
/// decode declines (mistyped fields, duplicate keys, arrays), [`Self::from_value`]
/// rebuilds it from a `JsonValue` with lenient `get`/`as_*` lookups, so both
/// sources feed the same state machine.
#[derive(Default, Deserialize)]
pub(crate) struct ChatChunk<'a> {
    #[serde(default)]
    pub(crate) error: KeyPresent,
    #[serde(default)]
    id: Option<JsonValue>,
    #[serde(default)]
    model: Option<JsonValue>,
    #[serde(default)]
    created: Option<JsonValue>,
    #[serde(default)]
    system_fingerprint: Option<JsonValue>,
    #[serde(default)]
    usage: Option<JsonValue>,
    #[serde(default, borrow)]
    choices: Option<Vec<JsonObject<ChatChoice<'a>>>>,
}

#[derive(Default, Deserialize)]
struct ChatChoice<'a> {
    #[serde(default)]
    index: Option<u64>,
    #[serde(default, borrow)]
    finish_reason: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    delta: Option<JsonObject<ChatDelta<'a>>>,
}

#[derive(Default, Deserialize)]
struct ChatDelta<'a> {
    #[serde(default, borrow)]
    content: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    refusal: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    reasoning_content: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    reasoning: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    tool_calls: Option<Vec<JsonObject<ChatToolDelta<'a>>>>,
}

#[derive(Default, Deserialize)]
struct ChatToolDelta<'a> {
    #[serde(default)]
    index: Option<u64>,
    #[serde(default, borrow)]
    id: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    function: Option<JsonObject<ChatFunctionDelta<'a>>>,
}

#[derive(Default, Deserialize)]
struct ChatFunctionDelta<'a> {
    #[serde(default, borrow)]
    name: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    arguments: Option<Cow<'a, str>>,
}

fn json_str<'a>(value: &'a JsonValue, key: &str) -> Option<Cow<'a, str>> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map(Cow::Borrowed)
}

impl<'a> ChatChunk<'a> {
    /// Extracts a chunk from a `JsonValue` with the adapter's lenient lookup semantics.
    pub(crate) fn from_value(value: &'a JsonValue) -> Self {
        let metadata = |key| value.get(key).filter(|value| !value.is_null()).cloned();
        let choices = value
            .get("choices")
            .and_then(JsonValue::as_array)
            .map(|choices| {
                choices
                    .iter()
                    .map(|choice| JsonObject(ChatChoice::from_value(choice)))
                    .collect()
            });
        Self {
            error: KeyPresent(value.get("error").is_some()),
            id: metadata("id"),
            model: metadata("model"),
            created: metadata("created"),
            system_fingerprint: metadata("system_fingerprint"),
            usage: metadata("usage"),
            choices,
        }
    }
}

impl<'a> ChatChoice<'a> {
    fn from_value(value: &'a JsonValue) -> Self {
        Self {
            index: value.get("index").and_then(JsonValue::as_u64),
            finish_reason: json_str(value, "finish_reason"),
            delta: value
                .get("delta")
                .map(|delta| JsonObject(ChatDelta::from_value(delta))),
        }
    }
}

impl<'a> ChatDelta<'a> {
    fn from_value(value: &'a JsonValue) -> Self {
        Self {
            content: json_str(value, "content"),
            refusal: json_str(value, "refusal"),
            reasoning_content: json_str(value, "reasoning_content"),
            reasoning: json_str(value, "reasoning"),
            tool_calls: value
                .get("tool_calls")
                .and_then(JsonValue::as_array)
                .map(|calls| {
                    calls
                        .iter()
                        .map(|call| JsonObject(ChatToolDelta::from_value(call)))
                        .collect()
                }),
        }
    }
}

impl<'a> ChatToolDelta<'a> {
    fn from_value(value: &'a JsonValue) -> Self {
        Self {
            index: value.get("index").and_then(JsonValue::as_u64),
            id: json_str(value, "id"),
            function: value.get("function").map(|function| {
                JsonObject(ChatFunctionDelta {
                    name: json_str(function, "name"),
                    arguments: json_str(function, "arguments"),
                })
            }),
        }
    }
}

#[derive(Default)]
struct ToolState {
    provider_id: Option<String>,
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    emitted: usize,
    started: bool,
    order: Option<u64>,
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
    tool_ids: BTreeSet<String>,
    next_tool_order: u64,
    /// Most recently touched tool key, used to route id-less/index-less
    /// continuation deltas.
    last_tool: Option<(u64, u64)>,
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
            tool_ids: BTreeSet::new(),
            next_tool_order: 0,
            last_tool: None,
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

    #[cfg(test)]
    pub(crate) fn apply(
        &mut self,
        value: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        self.apply_chunk(ChatChunk::from_value(&value), parts, bytes)
    }

    pub(crate) fn apply_chunk(
        &mut self,
        chunk: ChatChunk<'_>,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self.done {
            return Err(invalid_event("Chat event after terminal finish", bytes));
        }
        if chunk.error.0 {
            return Err(invalid_event(
                "Chat error envelope must be handled by the stream transport",
                bytes,
            ));
        }
        for (key, value) in [
            ("openai.response_id", chunk.id),
            ("openai.model", chunk.model),
            ("openai.created", chunk.created),
            ("openai.system_fingerprint", chunk.system_fingerprint),
        ] {
            if let Some(value) = value {
                set_metadata(&mut self.response_metadata, key, value);
            }
        }
        if let Some(usage) = chunk.usage {
            self.usage = usage_from(&usage, bytes)?;
        }
        let choices = chunk.choices.unwrap_or_default();
        if choices.len() > 1 {
            return Err(invalid_event(
                "multiple Chat choices are unsupported",
                bytes,
            ));
        }
        for JsonObject(choice) in choices {
            let choice_index = choice.index.unwrap_or(0);
            if choice_index != 0 {
                return Err(invalid_event(
                    "only Chat choice index 0 is supported",
                    bytes,
                ));
            }
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason.into_owned());
            }
            let delta = choice.delta.map(|delta| delta.0).unwrap_or_default();
            let reasoning = match self.reasoning_field {
                ReasoningField::None => None,
                ReasoningField::ReasoningContent => delta.reasoning_content.as_deref(),
                ReasoningField::Reasoning => delta.reasoning.as_deref(),
            };
            self.apply_reasoning(reasoning, parts);
            if let Some(content) = delta.content {
                // Providers send an empty `content` chunk with the role delta
                // before any text (and often before `reasoning_content`).
                // Opening the text block there would stamp it ahead of a
                // reasoning block that semantically precedes it, so committed
                // content order shows thinking after the text.
                if !content.is_empty() {
                    self.close_reasoning(parts);
                    if !self.text_open {
                        self.text_open = true;
                        parts.push(StreamPart::TextStart {
                            id: "0".into(),
                            metadata: None,
                        });
                    }
                    self.text.push_str(&content);
                    parts.push(StreamPart::TextDelta {
                        id: "0".into(),
                        delta: content.into_owned(),
                        metadata: None,
                    });
                }
            }
            if let Some(refusal) = delta.refusal {
                self.refusal.push_str(&refusal);
            }
            if let Some(calls) = delta.tool_calls
                && !calls.is_empty()
            {
                self.close_reasoning(parts);
                for JsonObject(call) in calls {
                    self.apply_tool(choice_index, call, parts, bytes)?;
                }
            }
        }
        Ok(())
    }

    fn apply_reasoning(&mut self, reasoning: Option<&str>, parts: &mut Vec<StreamPart>) {
        let Some(reasoning) = reasoning else {
            return;
        };
        // Vercel treats an empty reasoning chunk as falsy and emits nothing; do
        // not open the block for `""`. Whitespace-only chunks are non-empty and
        // still stream (whitespace semantics live in the normalized/replay
        // layer, not the raw event layer).
        if reasoning.is_empty() {
            return;
        }
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
        call: ChatToolDelta<'_>,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let explicit_index = call.index;
        let provider_id = call.id.as_deref().filter(|id| !id.is_empty());
        let function = call.function.map(|function| function.0).unwrap_or_default();
        let provider_name = function.name.as_deref();

        // Continuation routing precedence (mirrors Vercel's streaming tool-call
        // tracker): a non-empty provider id that matches an existing call for
        // this choice wins over the chunk's own index; an explicit index comes
        // next; and a delta carrying neither id nor index continues the most
        // recently touched call. The BTreeMap key remains storage-only, so
        // routing must no longer assume key == provider index.
        let mut index;
        let mut force_new_call = false;
        if let Some(id) = provider_id {
            // Ambiguity (two states sharing one provider id after `-N`
            // deduplication) resolves to the greatest key for determinism.
            let matched = self
                .tools
                .iter()
                .filter(|((choice, _), state)| {
                    *choice == choice_index && state.provider_id.as_deref() == Some(id)
                })
                .map(|(key, _)| *key)
                .next_back();
            match matched {
                Some(key) => {
                    let name_conflict = self.tools.get(&key).is_some_and(|state| {
                        provider_name.is_some_and(|name| {
                            state
                                .name
                                .as_deref()
                                .is_some_and(|existing| existing != name)
                        })
                    });
                    if name_conflict {
                        force_new_call = true;
                        index = explicit_index
                            .or_else(|| self.last_tool.map(|(_, index)| index))
                            .unwrap_or(key.1);
                    } else {
                        index = key.1;
                    }
                }
                None => {
                    // A provider id that matches nothing starts a new call;
                    // its explicit index (or 0) is only a storage hint and
                    // collides into a fresh allocation below.
                    index = explicit_index.unwrap_or(0);
                }
            }
        } else if let Some(explicit_index) = explicit_index {
            index = explicit_index;
        } else {
            index = self.last_tool.map(|(_, index)| index).unwrap_or(0);
        }

        let starts_new_call = force_new_call
            || self.tools.get(&(choice_index, index)).is_some_and(|state| {
                provider_id.is_some_and(|id| {
                    state
                        .provider_id
                        .as_deref()
                        .is_some_and(|existing| existing != id)
                        || provider_name.is_some_and(|name| {
                            state
                                .name
                                .as_deref()
                                .is_some_and(|existing| existing != name)
                        })
                })
            });
        if starts_new_call {
            index = self
                .tools
                .keys()
                .filter(|(choice, _)| *choice == choice_index)
                .map(|(_, index)| *index)
                .max()
                .unwrap_or(index)
                .saturating_add(1);
        }
        let needs_id = provider_id.is_some()
            && self
                .tools
                .get(&(choice_index, index))
                .is_none_or(|state| state.id.is_none());
        let reserved_id = needs_id.then(|| reserve_tool_id(&mut self.tool_ids, provider_id, index));
        let state = self.tools.entry((choice_index, index)).or_default();
        if let (Some(provider_id), Some(id)) = (provider_id, reserved_id) {
            state.provider_id = Some(provider_id.into());
            state.id = Some(id);
        }
        if let Some(name) = provider_name {
            if state
                .name
                .as_deref()
                .is_some_and(|existing| existing != name)
            {
                return Err(invalid_event("Chat tool name changed", bytes));
            }
            state.name = Some(name.into());
        }
        if let Some(arguments) = function.arguments {
            state.arguments.push_str(&arguments);
        }
        if !state.started
            && let (Some(id), Some(name)) = (&state.id, &state.name)
        {
            state.started = true;
            state.order = Some(self.next_tool_order);
            self.next_tool_order = self.next_tool_order.saturating_add(1);
            parts.push(StreamPart::ToolCallStart {
                id: id.clone(),
                name: name.clone(),
                metadata: None,
            });
        }
        // Withhold deltas while the entire accumulated argument text is
        // whitespace-only so zero-argument calls never open a streamed
        // argument block (collectors then take their empty-argument path).
        // Withheld whitespace flushes together with the first real content.
        if state.started
            && !state.arguments.trim().is_empty()
            && state.emitted < state.arguments.len()
        {
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
        self.last_tool = Some((choice_index, index));
        Ok(())
    }

    pub(crate) fn finish(
        &mut self,
        done_marker: bool,
        parts: &mut Vec<StreamPart>,
        _bytes: u64,
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
        for (call_number, (_, mut tool)) in tools.into_iter().enumerate() {
            let id = tool
                .id
                .unwrap_or_else(|| reserve_tool_id(&mut self.tool_ids, None, call_number as u64));
            let provider_name = tool.name;
            let name = provider_name.clone().unwrap_or_default();
            if !tool.started {
                tool.order = Some(self.next_tool_order);
                self.next_tool_order = self.next_tool_order.saturating_add(1);
                parts.push(StreamPart::ToolCallStart {
                    id: id.clone(),
                    name: name.clone(),
                    metadata: None,
                });
                if !tool.arguments.trim().is_empty() {
                    parts.push(StreamPart::ToolCallDelta {
                        id: id.clone(),
                        delta: tool.arguments.clone(),
                        metadata: None,
                    });
                }
            }
            // Three-way classification, independent of `finish_reason`:
            //   1. whitespace-only arguments are a canonical zero-argument call;
            //   2. object JSON is the happy path;
            //   3. anything else is a marked-invalid call: `input` is null and
            //      the raw provider text is preserved. `null` fails every
            //      published tool's object schema downstream, so this surfaces
            //      as a recoverable tool error rather than a fabricated call.
            let empty = tool.arguments.trim().is_empty();
            let (parsed, raw_input) = if empty {
                (serde_json::json!({}), None)
            } else {
                match serde_json::from_str::<JsonValue>(&tool.arguments) {
                    Ok(parsed) if parsed.is_object() => (parsed, Some(tool.arguments.clone())),
                    _ => (JsonValue::Null, Some(tool.arguments.clone())),
                }
            };
            // The raw-input copy above and the replay copy below each need their
            // own string; the accumulator itself moves into the replay payload.
            parts.push(StreamPart::ToolCallEnd {
                id: id.clone(),
                metadata: None,
            });
            let mut call = ToolCallPart::new(id.clone(), name, parsed);
            call.raw_input = raw_input;
            parts.push(StreamPart::ToolCall { tool_call: call });
            let native_arguments = if empty {
                "{}".to_owned()
            } else {
                tool.arguments
            };
            let mut function = serde_json::Map::new();
            function.insert("arguments".into(), native_arguments.into());
            if let Some(provider_name) = provider_name {
                function.insert("name".into(), provider_name.into());
            }
            let mut native = serde_json::Map::new();
            native.insert("id".into(), id.into());
            native.insert("type".into(), "function".into());
            native.insert("function".into(), function.into());
            let native = JsonValue::Object(native);
            native_calls.push((tool.order.unwrap_or(u64::MAX), native));
        }
        native_calls.sort_by_key(|(order, _)| *order);
        let native_calls = native_calls
            .into_iter()
            .map(|(_, call)| call)
            .collect::<Vec<_>>();
        // The turn is over: accumulated text moves into the replay payload.
        let refusal = std::mem::take(&mut self.refusal);
        if !refusal.is_empty() {
            parts.push(StreamPart::Custom {
                part: CustomPart::new("openai.refusal", JsonValue::String(refusal.clone())),
            });
        }
        let reason = map_finish(self.finish_reason.as_deref(), done_marker);
        let mut finish = Finish::new(std::mem::take(&mut self.usage), reason);
        finish.response_metadata = self.response_metadata.clone();
        if self.policy != ReplayPolicy::Never {
            let text = std::mem::take(&mut self.text);
            let reasoning = std::mem::take(&mut self.reasoning);
            let mut message = serde_json::Map::new();
            message.insert("role".into(), "assistant".into());
            message.insert(
                "content".into(),
                if text.is_empty() {
                    JsonValue::Null
                } else {
                    JsonValue::String(text)
                },
            );
            if !native_calls.is_empty() {
                message.insert("tool_calls".into(), JsonValue::Array(native_calls));
            }
            if !reasoning.is_empty() {
                match self.reasoning_field {
                    ReasoningField::ReasoningContent => {
                        message.insert("reasoning_content".into(), reasoning.into());
                    }
                    ReasoningField::Reasoning => {
                        message.insert("reasoning".into(), reasoning.into());
                    }
                    ReasoningField::None => {}
                }
            }
            if !refusal.is_empty() {
                message.insert("refusal".into(), refusal.into());
            }
            let message = JsonValue::Object(message);
            let mut payload = serde_json::Map::new();
            payload.insert("format".into(), REPLAY_FORMAT.into());
            payload.insert("message".into(), message);
            payload.insert("finish_reason".into(), self.finish_reason.clone().into());
            let payload = JsonValue::Object(payload);
            finish.native_replay = Some(
                NativeReplayArtifact::capture(self.adapter_id.clone(), self.scope.clone(), payload)
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

    pub(crate) fn saw_finish_reason(&self) -> bool {
        self.finish_reason.is_some()
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
}

/// Updates a metadata entry in place, allocating its key only on first insert.
fn set_metadata(metadata: &mut BTreeMap<String, JsonValue>, key: &str, value: JsonValue) {
    match metadata.get_mut(key) {
        Some(slot) => *slot = value,
        None => {
            metadata.insert(key.to_owned(), value);
        }
    }
}

fn invalid_event(message: &str, bytes: u64) -> ModelError {
    ModelError::invalid_response(message)
        .with_stage(ErrorStage::StreamEvent)
        .with_bytes_received(bytes)
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

fn usage_from(value: &JsonValue, bytes: u64) -> Result<Usage, ModelError> {
    let output = value.get("completion_tokens").and_then(JsonValue::as_u64);
    let reasoning = value
        .pointer("/completion_tokens_details/reasoning_tokens")
        .and_then(JsonValue::as_u64)
        .or_else(|| value.get("reasoning_tokens").and_then(JsonValue::as_u64));
    let cache_write = optional_usage_u64(
        value.pointer("/prompt_tokens_details/cache_write_tokens"),
        "Chat cache-write token usage is invalid",
        bytes,
    )?;
    let input_tokens = value.get("prompt_tokens").and_then(JsonValue::as_u64);
    let input_tokens_cache_read = value
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(JsonValue::as_u64);
    let input_tokens_no_cache = input_tokens
        .filter(|_| input_tokens_cache_read.is_some() || cache_write.is_some())
        .map(|total| {
            total
                .saturating_sub(input_tokens_cache_read.unwrap_or(0))
                .saturating_sub(cache_write.unwrap_or(0))
        });
    Ok(Usage {
        input_tokens,
        input_tokens_no_cache,
        input_tokens_cache_read,
        input_tokens_cache_write: cache_write,
        input_tokens_cache_write_by_ttl: Default::default(),
        output_tokens: output,
        output_tokens_text: output.map(|total| total.saturating_sub(reasoning.unwrap_or(0))),
        output_tokens_reasoning: reasoning,
        raw: Some(value.clone()),
    })
}

fn optional_usage_u64(
    value: Option<&JsonValue>,
    message: &str,
    bytes: u64,
) -> Result<Option<u64>, ModelError> {
    // Some OpenAI-compatible providers (GLM) send `null` on turns that wrote
    // nothing to the cache; count that as zero, not as a malformed value.
    value
        .map(|value| {
            value
                .as_u64()
                .or(value.is_null().then_some(0))
                .ok_or_else(|| invalid_event(message, bytes))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oven_sdk::{ModelId, ProviderId, ResourceId};

    fn test_state() -> State {
        State::new(
            AdapterId::new("openai-compatible"),
            test_scope(),
            ReplayPolicy::Never,
            ReasoningField::ReasoningContent,
        )
    }

    fn replay_state() -> State {
        State::new(
            AdapterId::new("openai-compatible"),
            test_scope(),
            ReplayPolicy::IfValid,
            ReasoningField::ReasoningContent,
        )
    }

    fn test_scope() -> NativeContextScope {
        NativeContextScope::new(
            ProviderId::new("test-provider"),
            ModelId::new("test-model"),
            ResourceId::new("test-resource").expect("test resource id constructs"),
        )
        .expect("test scope constructs")
    }

    fn finalized_calls(parts: &[StreamPart]) -> Vec<ToolCallPart> {
        parts
            .iter()
            .filter_map(|part| match part {
                StreamPart::ToolCall { tool_call } => Some(tool_call.clone()),
                _ => None,
            })
            .collect()
    }

    fn chunk(delta: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"choices": [{"index": 0, "delta": delta}]})
    }

    fn part_kinds(parts: &[StreamPart]) -> Vec<&'static str> {
        parts
            .iter()
            .map(|part| match part {
                StreamPart::StreamStart { .. } => "stream_start",
                StreamPart::TextStart { .. } => "text_start",
                StreamPart::TextDelta { .. } => "text_delta",
                StreamPart::TextEnd { .. } => "text_end",
                StreamPart::ReasoningStart { .. } => "reasoning_start",
                StreamPart::ReasoningDelta { .. } => "reasoning_delta",
                StreamPart::ReasoningEnd { .. } => "reasoning_end",
                StreamPart::ToolCall { .. } => "tool_call",
                StreamPart::ApprovalRequested { .. } => "approval_requested",
                StreamPart::ToolResult { .. } => "tool_result",
                StreamPart::Custom { .. } => "custom",
                StreamPart::Finish { .. } => "finish",
                _ => "other",
            })
            .collect()
    }

    /// Drives payloads like the SSE loop, optionally forcing the `JsonValue` path.
    fn drive_chunks(chunks: &[&str], typed: bool) -> (Vec<StreamPart>, usize) {
        let mut state = replay_state();
        let mut parts = Vec::new();
        let mut typed_hits = 0;
        for data in chunks {
            if typed
                && let Some(chunk) =
                    oven_sdk::provider_support::parse_json_object::<ChatChunk<'_>>(data)
            {
                typed_hits += 1;
                state.apply_chunk(chunk, &mut parts, 0).unwrap();
                continue;
            }
            let value: JsonValue = serde_json::from_str(data).unwrap();
            state
                .apply_chunk(ChatChunk::from_value(&value), &mut parts, 0)
                .unwrap();
        }
        state.finish(true, &mut parts, 0).unwrap();
        (parts, typed_hits)
    }

    #[test]
    fn typed_chunk_decode_matches_json_value_path() {
        let chunks = [
            r#"{"id":"chat_1","model":"m","created":1,"system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#,
            r#"{"id":"chat_1","choices":[{"index":0,"delta":{"reasoning_content":"think \"hard\""}}]}"#,
            r#"{"id":"chat_1","choices":[{"index":0,"delta":{"content":"hé\nllo","refusal":null}}]}"#,
            // Declined by the typed decode: mistyped content, non-object delta,
            // non-object tool call, duplicate keys.
            r#"{"choices":[{"index":0,"delta":{"content":5}}]}"#,
            r#"{"choices":[{"index":0,"delta":"text"}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[7]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"a","content":"b"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"refusal":"no"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"[1]}"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}"#,
        ];
        let (typed, typed_hits) = drive_chunks(&chunks, true);
        let (dom, dom_hits) = drive_chunks(&chunks, false);
        assert_eq!(typed_hits, 8);
        assert_eq!(dom_hits, 0);
        assert_eq!(typed, dom);
        let Some(StreamPart::Finish { finish }) = typed.last() else {
            panic!("missing finish");
        };
        assert_eq!(finish.usage.input_tokens, Some(4));
        assert_eq!(
            finish.response_metadata.get("openai.response_id"),
            Some(&serde_json::json!("chat_1"))
        );
        assert!(
            !finish
                .response_metadata
                .contains_key("openai.system_fingerprint")
        );
        assert_eq!(
            finish.native_replay.as_ref().unwrap().payload(),
            &serde_json::json!({
                "format": REPLAY_FORMAT,
                "message": {
                    "role": "assistant",
                    "content": "hé\nllob",
                    "reasoning_content": "think \"hard\"",
                    "refusal": "no",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"q\":[1]}"},
                    }],
                },
                "finish_reason": "tool_calls",
            })
        );
    }

    #[test]
    fn empty_role_chunk_does_not_open_text_before_reasoning() {
        let mut state = test_state();
        let mut parts = Vec::new();

        // Providers emit the role chunk with empty content first; reasoning
        // streams afterwards, then the actual text.
        state
            .apply(
                chunk(serde_json::json!({"role": "assistant", "content": ""})),
                &mut parts,
                0,
            )
            .expect("role chunk applies");
        state
            .apply(
                chunk(serde_json::json!({"reasoning_content": "Thinking first."})),
                &mut parts,
                0,
            )
            .expect("reasoning chunk applies");
        state
            .apply(
                chunk(serde_json::json!({"content": "Answer."})),
                &mut parts,
                0,
            )
            .expect("text chunk applies");

        assert_eq!(
            part_kinds(&parts),
            vec![
                "reasoning_start",
                "reasoning_delta",
                "reasoning_end",
                "text_start",
                "text_delta",
            ],
            "empty role content must not stamp the text block ahead of reasoning"
        );
    }

    #[test]
    fn empty_content_chunks_midstream_are_ignored() {
        let mut state = test_state();
        let mut parts = Vec::new();

        state
            .apply(
                chunk(serde_json::json!({"content": "Hello"})),
                &mut parts,
                0,
            )
            .expect("text chunk applies");
        let after_hello = parts.len();
        state
            .apply(chunk(serde_json::json!({"content": ""})), &mut parts, 0)
            .expect("empty chunk applies");
        state
            .apply(
                chunk(serde_json::json!({"content": " world"})),
                &mut parts,
                0,
            )
            .expect("second text chunk applies");

        assert_eq!(
            part_kinds(&parts),
            vec!["text_start", "text_delta", "text_delta"],
            "empty heartbeat chunks contribute no parts"
        );
        assert_eq!(parts.len(), after_hello + 1);
        assert_eq!(state.text, "Hello world");
    }

    #[test]
    fn cache_write_usage_is_optional_and_validated() {
        let present = serde_json::json!({
            "prompt_tokens_details": {"cache_write_tokens": 17}
        });
        assert_eq!(
            usage_from(&present, 0).unwrap().input_tokens_cache_write,
            Some(17)
        );

        assert_eq!(
            usage_from(&serde_json::json!({}), 0)
                .unwrap()
                .input_tokens_cache_write,
            None
        );
        assert_eq!(
            usage_from(
                &serde_json::json!({
                    "prompt_tokens": 40,
                    "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": null}
                }),
                0
            )
            .unwrap(),
            Usage {
                input_tokens: Some(40),
                input_tokens_no_cache: Some(30),
                input_tokens_cache_read: Some(10),
                input_tokens_cache_write: Some(0),
                input_tokens_cache_write_by_ttl: Default::default(),
                output_tokens: None,
                output_tokens_text: None,
                output_tokens_reasoning: None,
                raw: Some(serde_json::json!({
                    "prompt_tokens": 40,
                    "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": null}
                })),
            }
        );
        assert!(
            usage_from(
                &serde_json::json!({
                    "prompt_tokens_details": {"cache_write_tokens": -1}
                }),
                0
            )
            .is_err()
        );
    }

    #[test]
    fn whitespace_only_arguments_finalize_as_empty_object_without_delta() {
        let mut state = test_state();
        let mut parts = Vec::new();
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"index":0,"id":"call","function":{"name":"tool","arguments":"  "}}]})),
                &mut parts,
                0,
            )
            .expect("whitespace-only arguments apply");
        assert!(
            !parts
                .iter()
                .any(|part| matches!(part, StreamPart::ToolCallDelta { .. })),
            "whitespace-only arguments must not open a streamed argument block"
        );
        state.finish(false, &mut parts, 0).expect("finish");

        let calls = finalized_calls(&parts);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].input, serde_json::json!({}));
        assert!(calls[0].raw_input.is_none());
        assert!(
            !parts
                .iter()
                .any(|part| matches!(part, StreamPart::ToolCallDelta { .. }))
        );
    }

    #[test]
    fn withheld_whitespace_flushes_with_first_real_arguments() {
        let mut state = test_state();
        let mut parts = Vec::new();
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"index":0,"id":"call","function":{"name":"tool","arguments":"  "}}]})),
                &mut parts,
                0,
            )
            .expect("whitespace prefix applies");
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":1}"}}]})),
                &mut parts,
                0,
            )
            .expect("real arguments apply");
        let deltas = parts
            .iter()
            .filter_map(|part| match part {
                StreamPart::ToolCallDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(deltas, vec!["  {\"a\":1}".to_owned()]);
    }

    #[test]
    fn truncated_arguments_are_marked_invalid_for_every_finish_reason() {
        for reason in ["length", "tool_calls"] {
            let mut state = replay_state();
            let mut parts = Vec::new();
            state
                .apply(
                    serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","function":{"name":"tool","arguments":"{\"a\":"}}]},"finish_reason":reason}]}),
                    &mut parts,
                    0,
                )
                .expect("truncated arguments apply");
            state
                .finish(false, &mut parts, 0)
                .expect("truncated arguments finalize");

            assert!(
                parts
                    .iter()
                    .any(|part| matches!(part, StreamPart::ToolCallEnd { .. })),
                "every started block must close"
            );
            let calls = finalized_calls(&parts);
            assert_eq!(calls.len(), 1);
            assert!(calls[0].input.is_null());
            assert_eq!(calls[0].raw_input.as_deref(), Some("{\"a\":"));
            let finish = parts
                .iter()
                .find_map(|part| match part {
                    StreamPart::Finish { finish } => Some(finish.clone()),
                    _ => None,
                })
                .expect("finish part");
            let native = finish.native_replay.expect("replay artifact");
            assert_eq!(
                native
                    .payload()
                    .pointer("/message/tool_calls/0/function/arguments")
                    .and_then(JsonValue::as_str),
                Some("{\"a\":"),
                "native replay must keep the verbatim raw argument text for {reason}"
            );
        }
    }

    #[test]
    fn indexless_continuation_routes_to_last_touched_call() {
        let mut state = test_state();
        let mut parts = Vec::new();
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"index":0,"id":"call_0","function":{"name":"alpha","arguments":"{\"a\":"}}]})),
                &mut parts,
                0,
            )
            .expect("first call applies");
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"index":1,"id":"call_1","function":{"name":"beta","arguments":"{\"b\":"}}]})),
                &mut parts,
                0,
            )
            .expect("second call applies");
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"function":{"arguments":"1}"}}]})),
                &mut parts,
                0,
            )
            .expect("index-less continuation applies");
        state.finish(false, &mut parts, 0).expect("finish");

        let calls = finalized_calls(&parts);
        assert_eq!(calls.len(), 2);
        let alpha = calls.iter().find(|call| call.id == "call_0").unwrap();
        let beta = calls.iter().find(|call| call.id == "call_1").unwrap();
        assert!(alpha.input.is_null(), "alpha's fragment stays truncated");
        assert_eq!(alpha.raw_input.as_deref(), Some("{\"a\":"));
        assert_eq!(
            beta.input,
            serde_json::json!({"b": 1}),
            "continuation must land on the most recently touched call"
        );
    }

    #[test]
    fn provider_id_routes_across_mismatched_explicit_index() {
        let mut state = test_state();
        let mut parts = Vec::new();
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"index":0,"id":"call","function":{"name":"tool","arguments":"{\"a\":"}}]})),
                &mut parts,
                0,
            )
            .expect("first fragment applies");
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"index":5,"id":"call","function":{"arguments":"1}"}}]})),
                &mut parts,
                0,
            )
            .expect("same-id mismatched index applies");
        state.finish(false, &mut parts, 0).expect("finish");

        let calls = finalized_calls(&parts);
        assert_eq!(calls.len(), 1, "same provider id must merge into one call");
        assert_eq!(calls[0].input, serde_json::json!({"a": 1}));
    }

    #[test]
    fn unmatched_indexless_id_starts_new_call() {
        let mut state = test_state();
        let mut parts = Vec::new();
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"index":0,"id":"call_a","function":{"name":"a","arguments":"{}"}}]})),
                &mut parts,
                0,
            )
            .expect("first call applies");
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"id":"call_b","function":{"name":"b","arguments":"{}"}}]})),
                &mut parts,
                0,
            )
            .expect("index-less new id applies");
        state.finish(false, &mut parts, 0).expect("finish");

        let mut ids = finalized_calls(&parts)
            .into_iter()
            .map(|call| call.id)
            .collect::<Vec<_>>();
        ids.sort();
        assert_eq!(ids, ["call_a", "call_b"]);
    }

    #[test]
    fn duplicate_provider_id_resolves_to_greatest_key() {
        let mut state = test_state();
        let mut parts = Vec::new();
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"index":0,"id":"x","function":{"name":"a","arguments":"{\"p\":"}}]})),
                &mut parts,
                0,
            )
            .expect("first call applies");
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"index":1,"id":"x","function":{"name":"b","arguments":"{\"q\":"}}]})),
                &mut parts,
                0,
            )
            .expect("conflicting name spawns a new call sharing the provider id");
        state
            .apply(
                chunk(serde_json::json!({"tool_calls":[{"id":"x","function":{"arguments":"1}"}}]})),
                &mut parts,
                0,
            )
            .expect("ambiguous index-less continuation applies");
        state.finish(false, &mut parts, 0).expect("finish");

        let calls = finalized_calls(&parts);
        assert_eq!(calls.len(), 2);
        let first = calls.iter().find(|call| call.id == "x").unwrap();
        let second = calls.iter().find(|call| call.id == "x-1").unwrap();
        assert!(first.input.is_null(), "first call keeps its truncated text");
        assert_eq!(second.input, serde_json::json!({"q": 1}));
    }

    #[test]
    fn empty_reasoning_delta_is_ignored() {
        let mut state = test_state();
        let mut parts = Vec::new();
        state
            .apply(
                chunk(serde_json::json!({"reasoning_content":""})),
                &mut parts,
                0,
            )
            .expect("empty reasoning applies");
        assert!(parts.is_empty(), "empty reasoning must emit nothing");
        state
            .apply(chunk(serde_json::json!({"content":"x"})), &mut parts, 0)
            .expect("text applies");
        state.finish(false, &mut parts, 0).expect("finish");
        assert!(
            !parts
                .iter()
                .any(|part| matches!(part, StreamPart::ReasoningStart { .. }))
        );
    }

    #[test]
    fn empty_tool_calls_array_does_not_close_reasoning() {
        let mut state = test_state();
        let mut parts = Vec::new();
        for delta in [
            serde_json::json!({"reasoning_content":"think1"}),
            serde_json::json!({"tool_calls":[]}),
            serde_json::json!({"reasoning_content":"think2"}),
        ] {
            state
                .apply(chunk(delta), &mut parts, 0)
                .expect("delta applies");
        }
        state.finish(false, &mut parts, 0).expect("finish");

        assert_eq!(
            part_kinds(&parts),
            vec![
                "reasoning_start",
                "reasoning_delta",
                "reasoning_delta",
                "reasoning_end",
                "finish",
            ]
        );
    }
}
