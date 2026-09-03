//! Azure OpenAI Responses event state machine.

use std::collections::{BTreeMap, BTreeSet};

use oven_sdk::{
    AdapterId, CustomPart, ErrorStage, Finish, FinishReason, JsonValue, ModelError, ModelErrorKind,
    NativeContextScope, NativeReplayArtifact, ReplayCapability, ReplayPolicy, StreamPart,
    ToolCallPart, Usage,
};

use crate::{responses::replay, wire::responses::REPLAY_FORMAT};

const MAX_OUTPUT_ITEMS: usize = 128;
const MAX_CONTENT_SLOTS: usize = 128;
const MAX_SLOT_GAP: usize = 16;

#[derive(Clone, Copy)]
enum TerminalKind {
    Completed,
    Incomplete,
}

#[derive(Default)]
struct FunctionState {
    item_id: Option<String>,
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
    expected: Option<String>,
    started: bool,
    finalized: bool,
}

pub(crate) struct State {
    adapter_id: AdapterId,
    policy: ReplayPolicy,
    replay_capability: ReplayCapability,
    items: BTreeMap<usize, JsonValue>,
    functions: BTreeMap<usize, FunctionState>,
    tool_ids: BTreeSet<String>,
    finalized: BTreeSet<usize>,
    open_text: BTreeSet<String>,
    open_reasoning: BTreeSet<String>,
    text_deltas: BTreeSet<(usize, usize)>,
    refusal_deltas: BTreeSet<(usize, usize)>,
    emitted_refusals: BTreeSet<(usize, usize)>,
    reasoning_deltas: BTreeSet<(usize, String, usize)>,
    closed_reasoning: BTreeSet<(usize, String, usize)>,
    client_calls: bool,
    usage: Usage,
    response_metadata: BTreeMap<String, JsonValue>,
    done: bool,
    replay_binding: JsonValue,
    replay_scope: NativeContextScope,
}

impl State {
    pub(crate) fn new(
        adapter_id: AdapterId,
        policy: ReplayPolicy,
        replay_capability: ReplayCapability,
        replay_binding: JsonValue,
        replay_scope: NativeContextScope,
    ) -> Self {
        Self {
            adapter_id,
            policy,
            replay_capability,
            items: BTreeMap::new(),
            functions: BTreeMap::new(),
            tool_ids: BTreeSet::new(),
            finalized: BTreeSet::new(),
            open_text: BTreeSet::new(),
            open_reasoning: BTreeSet::new(),
            text_deltas: BTreeSet::new(),
            refusal_deltas: BTreeSet::new(),
            emitted_refusals: BTreeSet::new(),
            reasoning_deltas: BTreeSet::new(),
            closed_reasoning: BTreeSet::new(),
            client_calls: false,
            usage: Usage::default(),
            response_metadata: BTreeMap::new(),
            done: false,
            replay_binding,
            replay_scope,
        }
    }

    pub(crate) fn done(&self) -> bool {
        self.done
    }

    pub(crate) fn response_metadata(&self) -> &BTreeMap<String, JsonValue> {
        &self.response_metadata
    }

    fn output_index(&self, value: &JsonValue, bytes: u64) -> Result<usize, ModelError> {
        bounded_index(value, "output_index", MAX_OUTPUT_ITEMS, bytes)
    }

    pub(crate) fn apply(
        &mut self,
        value: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self.done {
            return Err(invalid_event("Responses event after terminal event", bytes));
        }
        if let Some(filters) = value
            .get("content_filters")
            .or_else(|| value.pointer("/response/content_filters"))
            .cloned()
        {
            self.response_metadata
                .insert("azure.content_filters".into(), filters.clone());
            parts.push(StreamPart::ProviderEvent {
                name: "azure.content_filters".into(),
                data: filters,
            });
        }
        let event = value.get("type").and_then(JsonValue::as_str).unwrap_or("");
        match event {
            "response.created" | "response.in_progress" => {
                self.capture_response(value.get("response").unwrap_or(&value));
                Ok(())
            }
            "response.output_item.added" => {
                let output_index = self.output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let item = value.get("item").cloned().ok_or_else(|| {
                    invalid_event("Responses item-added event is missing item", bytes)
                })?;
                self.add_item(output_index, item, parts, bytes)
            }
            "response.output_text.delta" => {
                let output_index = self.output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let content_index = event_index(&value, "content_index", bytes)?;
                let delta = value.get("delta").and_then(JsonValue::as_str).unwrap_or("");
                self.patch_text(output_index, content_index, delta, bytes)?;
                self.text_deltas.insert((output_index, content_index));
                let id = self.text_id(output_index, content_index);
                if self.open_text.insert(id.clone()) {
                    parts.push(StreamPart::TextStart {
                        id: id.clone(),
                        metadata: None,
                    });
                }
                parts.push(StreamPart::TextDelta {
                    id,
                    delta: delta.into(),
                    metadata: None,
                });
                Ok(())
            }
            "response.refusal.delta" => {
                let output_index = self.output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let content_index = event_index(&value, "content_index", bytes)?;
                let delta = value
                    .get("delta")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| {
                        invalid_event("Responses refusal delta is missing string delta", bytes)
                    })?;
                if self
                    .emitted_refusals
                    .contains(&(output_index, content_index))
                {
                    return Err(invalid_event(
                        "Responses refusal delta targeted a completed refusal",
                        bytes,
                    ));
                }
                self.patch_refusal(output_index, content_index, delta, bytes)?;
                self.refusal_deltas.insert((output_index, content_index));
                Ok(())
            }
            "response.refusal.done" => {
                let output_index = self.output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let content_index = event_index(&value, "content_index", bytes)?;
                let refusal = value
                    .get("refusal")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| {
                        invalid_event("Responses refusal done is missing string refusal", bytes)
                    })?;
                self.finish_refusal(output_index, content_index, refusal, parts, bytes)
            }
            "response.reasoning_summary_text.delta" => {
                let output_index = self.output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let summary_index = event_index(&value, "summary_index", bytes)?;
                let delta = value.get("delta").and_then(JsonValue::as_str).unwrap_or("");
                self.patch_summary(output_index, summary_index, delta, bytes)?;
                self.prepare_summary_slots(output_index, summary_index, parts, bytes)?;
                self.reasoning_delta(output_index, summary_index, "summary", delta, parts);
                Ok(())
            }
            "response.reasoning_text.delta" => {
                let output_index = self.output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let content_index = event_index(&value, "content_index", bytes)?;
                let delta = value.get("delta").and_then(JsonValue::as_str).unwrap_or("");
                self.patch_reasoning_content(output_index, content_index, delta, bytes)?;
                self.reasoning_deltas
                    .insert((output_index, "raw".into(), content_index));
                self.reasoning_delta(output_index, content_index, "raw", delta, parts);
                Ok(())
            }
            "response.function_call_arguments.delta" => {
                let output_index = self.output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let delta = value.get("delta").and_then(JsonValue::as_str).unwrap_or("");
                let function = self.functions.entry(output_index).or_default();
                start_function(function, parts);
                function.arguments.push_str(delta);
                if function.started
                    && let Some(call_id) = &function.call_id
                {
                    parts.push(StreamPart::ToolCallDelta {
                        id: call_id.clone(),
                        delta: delta.into(),
                        metadata: None,
                    });
                }
                Ok(())
            }
            "response.function_call_arguments.done" => {
                let output_index = self.output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                self.functions.entry(output_index).or_default().expected = value
                    .get("arguments")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                Ok(())
            }
            "response.output_item.done" => {
                let output_index = self.output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let item = value.get("item").cloned().ok_or_else(|| {
                    invalid_event("Responses item-done event is missing item", bytes)
                })?;
                self.finish_item(output_index, item, parts, bytes)
            }
            "response.completed" => {
                self.finish_response(value, TerminalKind::Completed, parts, bytes)
            }
            "response.incomplete" => {
                self.finish_response(value, TerminalKind::Incomplete, parts, bytes)
            }
            "response.failed" | "error" => Err(invalid_event(
                "Responses failure event must be handled by the stream transport",
                bytes,
            )),
            "response.output_text.done"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_text.done"
            | "response.content_part.added"
            | "response.content_part.done"
            | "response.queued" => Ok(()),
            _ => Ok(()),
        }
    }

    fn add_item(
        &mut self,
        output_index: usize,
        mut item: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        validate_stream_item_slots(&item, bytes)?;
        if item.get("type").and_then(JsonValue::as_str) == Some("function_call") {
            let function = self.functions.entry(output_index).or_default();
            update_function(function, &item);
            let call_id =
                ensure_function_id(function, &item, &mut self.tool_ids, output_index as u64);
            item["call_id"] = call_id.into();
            start_function(function, parts);
        }
        self.items.insert(output_index, item);
        Ok(())
    }

    fn finish_item(
        &mut self,
        output_index: usize,
        mut item: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        validate_stream_item_slots(&item, bytes)?;
        if let Some(streamed) = self.items.get(&output_index).and_then(JsonValue::as_object) {
            let target = item
                .as_object_mut()
                .ok_or_else(|| invalid_finalize("Responses output item is not an object", bytes))?;
            for (key, value) in streamed {
                if matches!(key.as_str(), "content" | "summary")
                    && !self
                        .text_deltas
                        .iter()
                        .any(|(index, _)| *index == output_index)
                    && !self
                        .refusal_deltas
                        .iter()
                        .any(|(index, _)| *index == output_index)
                    && !self
                        .reasoning_deltas
                        .iter()
                        .any(|(index, _, _)| *index == output_index)
                {
                    continue;
                }
                target.insert(key.clone(), value.clone());
            }
        }
        match item.get("type").and_then(JsonValue::as_str) {
            Some("function_call") => {
                let function = self.functions.entry(output_index).or_default();
                update_function(function, &item);
                let item_id = item
                    .get("id")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                let call_id =
                    ensure_function_id(function, &item, &mut self.tool_ids, output_index as u64);
                let name = item
                    .get("name")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let final_arguments = if function.arguments.is_empty() {
                    item.get("arguments")
                        .and_then(JsonValue::as_str)
                        .unwrap_or("{}")
                        .to_owned()
                } else {
                    function.arguments.clone()
                };
                item["call_id"] = call_id.clone().into();
                item["arguments"] = final_arguments.clone().into();
                let parsed: JsonValue = serde_json::from_str(&final_arguments).map_err(|_| {
                    invalid_finalize("final Responses tool arguments are invalid JSON", bytes)
                })?;
                if !parsed.is_object() {
                    return Err(invalid_finalize(
                        "final Responses tool arguments must be a JSON object",
                        bytes,
                    ));
                }
                function.item_id = item_id.clone();
                function.call_id = Some(call_id.clone());
                function.name = Some(name.clone());
                let started_before = function.started;
                start_function(function, parts);
                if function.arguments.is_empty() {
                    function.arguments = final_arguments.clone();
                    parts.push(StreamPart::ToolCallDelta {
                        id: call_id.clone(),
                        delta: final_arguments.clone(),
                        metadata: None,
                    });
                } else if !started_before {
                    parts.push(StreamPart::ToolCallDelta {
                        id: call_id.clone(),
                        delta: function.arguments.clone(),
                        metadata: None,
                    });
                }
                parts.push(StreamPart::ToolCallEnd {
                    id: call_id.clone(),
                    metadata: None,
                });
                let mut call = ToolCallPart::new(call_id, name, parsed);
                call.provider_item_id = item_id;
                call.raw_input = Some(final_arguments);
                parts.push(StreamPart::ToolCall { tool_call: call });
                function.finalized = true;
                self.client_calls = true;
            }
            Some("message") => self.finish_message(output_index, &item, parts),
            Some("reasoning") => self.finish_reasoning(output_index, &item, parts),
            Some(kind) => parts.push(StreamPart::Custom {
                part: CustomPart::new(format!("azure.openai.responses.{kind}"), item.clone()),
            }),
            None => {}
        }
        self.items.insert(output_index, item);
        self.finalized.insert(output_index);
        Ok(())
    }

    fn finish_response(
        &mut self,
        value: JsonValue,
        terminal_kind: TerminalKind,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let response = value
            .get("response")
            .and_then(JsonValue::as_object)
            .ok_or_else(|| {
                invalid_finalize(
                    "Responses terminal event is missing a response object",
                    bytes,
                )
            })?;
        let output = response
            .get("output")
            .and_then(JsonValue::as_array)
            .cloned()
            .unwrap_or_default();
        if output.len() > MAX_OUTPUT_ITEMS {
            return Err(invalid_finalize(
                "Responses terminal output exceeds the supported item limit",
                bytes,
            ));
        }
        let incomplete_reason = match terminal_kind {
            TerminalKind::Completed => None,
            TerminalKind::Incomplete => response
                .get("incomplete_details")
                .and_then(JsonValue::as_object)
                .and_then(|details| details.get("reason"))
                .and_then(JsonValue::as_str)
                .filter(|reason| !reason.is_empty()),
        };
        let response = JsonValue::Object(response.clone());
        let response = &response;
        self.capture_response(response);
        let mut consumed = BTreeSet::new();
        for (index, item) in output.iter().cloned().enumerate() {
            let output_index = self
                .matching_item_index(&item, &consumed)
                .or_else(|| self.available_item_index(index))
                .ok_or_else(|| {
                    invalid_finalize(
                        "Responses terminal output has no available item slot",
                        bytes,
                    )
                })?;
            consumed.insert(output_index);
            if self.finalized.contains(&output_index) {
                continue;
            } else {
                if !self.items.contains_key(&output_index) {
                    self.add_item(output_index, item.clone(), parts, bytes)?;
                }
                self.finish_item(output_index, item, parts, bytes)?;
            }
        }
        self.close_all(parts);
        if self.functions.values().any(|function| !function.finalized) {
            return Err(invalid_finalize(
                "Responses terminal event arrived before a function item was finalized",
                bytes,
            ));
        }
        if let Some(usage) = response.get("usage").filter(|value| !value.is_null()) {
            self.usage = usage_from(usage);
        }
        let finish_reason = match incomplete_reason {
            Some("max_output_tokens") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            None if self.client_calls => FinishReason::ToolCalls,
            None => FinishReason::Stop,
            Some(reason) => FinishReason::Other(reason.into()),
        };
        let mut finish = Finish::new(std::mem::take(&mut self.usage), finish_reason);
        finish.response_metadata = std::mem::take(&mut self.response_metadata);
        if self.policy != ReplayPolicy::Never {
            let items = std::mem::take(&mut self.items)
                .into_values()
                .collect::<Vec<_>>();
            let Some((items, fingerprint)) = replay::capture(&items) else {
                if self.replay_capability == ReplayCapability::Optional {
                    parts.push(StreamPart::Finish { finish });
                    self.done = true;
                    return Ok(());
                }
                return Err(invalid_finalize(
                    "Responses output contains an item unsafe for native replay",
                    bytes,
                ));
            };
            parts.push(StreamPart::Custom {
                part: CustomPart::new(
                    replay::FINGERPRINT_KIND,
                    JsonValue::String(fingerprint.clone()),
                ),
            });
            let payload = serde_json::json!({
                "format":REPLAY_FORMAT,
                "binding":self.replay_binding,
                "items":items,
                "fingerprint":fingerprint
            });
            finish.native_replay = Some(
                NativeReplayArtifact::new(
                    self.adapter_id.clone(),
                    self.replay_scope.clone(),
                    payload,
                )
                .map_err(|_| {
                    ModelError::replay("Azure Responses replay artifact exceeds its size limit")
                        .with_stage(ErrorStage::ReplayEncode)
                })?,
            );
        }
        parts.push(StreamPart::Finish { finish });
        self.done = true;
        Ok(())
    }

    fn matching_item_index(
        &self,
        terminal: &JsonValue,
        consumed: &BTreeSet<usize>,
    ) -> Option<usize> {
        self.items
            .iter()
            .find(|(index, streamed)| {
                !consumed.contains(index) && item_identity_matches(streamed, terminal)
            })
            .map(|(index, _)| *index)
    }

    fn available_item_index(&self, preferred: usize) -> Option<usize> {
        if preferred < MAX_OUTPUT_ITEMS
            && !self.items.contains_key(&preferred)
            && !self.functions.contains_key(&preferred)
        {
            return Some(preferred);
        }
        (0..MAX_OUTPUT_ITEMS)
            .find(|index| !self.items.contains_key(index) && !self.functions.contains_key(index))
    }

    fn finish_message(
        &mut self,
        output_index: usize,
        item: &JsonValue,
        parts: &mut Vec<StreamPart>,
    ) {
        self.close_text_for(output_index, parts);
        let Some(content) = item.get("content").and_then(JsonValue::as_array) else {
            return;
        };
        let item_id = item
            .get("id")
            .and_then(JsonValue::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("message-{output_index}"));
        for (content_index, content) in content.iter().enumerate() {
            match content.get("type").and_then(JsonValue::as_str) {
                Some("output_text")
                    if !self.text_deltas.contains(&(output_index, content_index)) =>
                {
                    let id = if content_index == 0 {
                        item_id.clone()
                    } else {
                        format!("{item_id}:text:{content_index}")
                    };
                    parts.push(StreamPart::TextStart {
                        id: id.clone(),
                        metadata: None,
                    });
                    parts.push(StreamPart::TextDelta {
                        id: id.clone(),
                        delta: content
                            .get("text")
                            .and_then(JsonValue::as_str)
                            .unwrap_or_default()
                            .into(),
                        metadata: None,
                    });
                    parts.push(StreamPart::TextEnd { id, metadata: None });
                }
                Some("refusal")
                    if !self
                        .emitted_refusals
                        .contains(&(output_index, content_index)) =>
                {
                    let refusal = content
                        .get("refusal")
                        .and_then(JsonValue::as_str)
                        .unwrap_or_default();
                    parts.push(StreamPart::Custom {
                        part: CustomPart::new(
                            "azure.openai.refusal",
                            JsonValue::String(refusal.into()),
                        ),
                    });
                    self.emitted_refusals.insert((output_index, content_index));
                }
                _ => {}
            }
        }
    }

    fn ensure_unfinalized(&self, output_index: usize, bytes: u64) -> Result<(), ModelError> {
        if self.finalized.contains(&output_index) {
            Err(invalid_event(
                "Responses event targeted an already finalized item",
                bytes,
            ))
        } else {
            Ok(())
        }
    }

    fn finish_reasoning(
        &mut self,
        output_index: usize,
        item: &JsonValue,
        parts: &mut Vec<StreamPart>,
    ) {
        self.close_reasoning_for(output_index, parts);
        let item_id = item
            .get("id")
            .and_then(JsonValue::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("reasoning-{output_index}"));
        if let Some(summary) = item.get("summary").and_then(JsonValue::as_array) {
            for (index, summary) in summary.iter().enumerate() {
                self.synthesize_reasoning(output_index, &item_id, "summary", index, summary, parts);
            }
        }
        if let Some(content) = item.get("content").and_then(JsonValue::as_array) {
            for (index, content) in content.iter().enumerate() {
                if content.get("type").and_then(JsonValue::as_str) == Some("reasoning_text") {
                    self.synthesize_reasoning(output_index, &item_id, "raw", index, content, parts);
                }
            }
        }
    }

    fn synthesize_reasoning(
        &self,
        output_index: usize,
        item_id: &str,
        kind: &str,
        index: usize,
        value: &JsonValue,
        parts: &mut Vec<StreamPart>,
    ) {
        if self
            .reasoning_deltas
            .contains(&(output_index, kind.into(), index))
        {
            return;
        }
        let id = format!("{item_id}:{kind}:{index}");
        parts.push(StreamPart::ReasoningStart {
            id: id.clone(),
            metadata: None,
        });
        parts.push(StreamPart::ReasoningDelta {
            id: id.clone(),
            delta: value
                .get("text")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .into(),
            metadata: None,
        });
        parts.push(StreamPart::ReasoningEnd { id, metadata: None });
    }

    pub(crate) fn in_band_error(
        &mut self,
        error: ModelError,
        parts: &mut Vec<StreamPart>,
    ) -> Result<(), ModelError> {
        if self.functions.values().any(|function| !function.finalized) {
            return Err(ModelError::invalid_response(
                "Responses error interrupted an open function call",
            )
            .with_stage(ErrorStage::StreamEvent)
            .with_bytes_received(error.diagnostics.bytes_received));
        }
        self.close_all(parts);
        parts.push(StreamPart::Error { error });
        let mut finish = Finish::new(self.usage.clone(), FinishReason::Error);
        finish.response_metadata = self.response_metadata.clone();
        parts.push(StreamPart::Finish { finish });
        self.done = true;
        Ok(())
    }

    fn reasoning_delta(
        &mut self,
        output_index: usize,
        index: usize,
        kind: &str,
        delta: &str,
        parts: &mut Vec<StreamPart>,
    ) {
        let item_id = self
            .items
            .get(&output_index)
            .and_then(|item| item.get("id"))
            .and_then(JsonValue::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("reasoning-{output_index}"));
        let id = format!("{item_id}:{kind}:{index}");
        if self.open_reasoning.insert(id.clone()) {
            parts.push(StreamPart::ReasoningStart {
                id: id.clone(),
                metadata: None,
            });
        }
        parts.push(StreamPart::ReasoningDelta {
            id,
            delta: delta.into(),
            metadata: None,
        });
    }

    fn prepare_summary_slots(
        &mut self,
        output_index: usize,
        summary_index: usize,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let item_id = self
            .items
            .get(&output_index)
            .and_then(|item| item.get("id"))
            .and_then(JsonValue::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("reasoning-{output_index}"));
        let target = (output_index, "summary".to_owned(), summary_index);
        if self.closed_reasoning.contains(&target) {
            return Err(invalid_event(
                "Responses reasoning delta targeted a closed summary slot",
                bytes,
            ));
        }
        for index in 0..summary_index {
            let key = (output_index, "summary".to_owned(), index);
            if self.reasoning_deltas.insert(key.clone()) {
                let id = format!("{item_id}:summary:{index}");
                parts.push(StreamPart::ReasoningStart {
                    id: id.clone(),
                    metadata: None,
                });
                parts.push(StreamPart::ReasoningEnd { id, metadata: None });
                self.closed_reasoning.insert(key);
            }
        }
        self.reasoning_deltas.insert(target);
        Ok(())
    }

    fn text_id(&self, output_index: usize, content_index: usize) -> String {
        let item_id = self
            .items
            .get(&output_index)
            .and_then(|item| item.get("id"))
            .and_then(JsonValue::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("message-{output_index}"));
        if content_index == 0 {
            item_id
        } else {
            format!("{item_id}:text:{content_index}")
        }
    }

    fn patch_text(
        &mut self,
        output_index: usize,
        content_index: usize,
        delta: &str,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let current_len = self
            .items
            .get(&output_index)
            .and_then(|item| item.get("content"))
            .and_then(JsonValue::as_array)
            .map_or(0, Vec::len);
        let target_len = slot_target_len(current_len, content_index, "message content", bytes)?;
        let item = self.items.entry(output_index).or_insert_with(|| {
            serde_json::json!({"type":"message","id":format!("message-{output_index}"),"role":"assistant","content":[]})
        });
        let object = item
            .as_object_mut()
            .ok_or_else(|| invalid_event("Responses message item is not an object", bytes))?;
        let content = object
            .entry("content")
            .or_insert_with(|| JsonValue::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| invalid_event("Responses message content is not an array", bytes))?;
        while content.len() < target_len {
            content.push(serde_json::json!({"type":"output_text","text":""}));
        }
        let part = content[content_index].as_object_mut().ok_or_else(|| {
            invalid_event("Responses message content part is not an object", bytes)
        })?;
        let current = match part.get("text") {
            Some(value) => value
                .as_str()
                .ok_or_else(|| invalid_event("Responses message text is not a string", bytes))?,
            None => "",
        };
        part.insert("text".into(), format!("{current}{delta}").into());
        Ok(())
    }

    fn patch_refusal(
        &mut self,
        output_index: usize,
        content_index: usize,
        delta: &str,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let current_len = self
            .items
            .get(&output_index)
            .and_then(|item| item.get("content"))
            .and_then(JsonValue::as_array)
            .map_or(0, Vec::len);
        let target_len = slot_target_len(current_len, content_index, "refusal content", bytes)?;
        let item = self.items.entry(output_index).or_insert_with(|| {
            serde_json::json!({"type":"message","id":format!("message-{output_index}"),"role":"assistant","content":[]})
        });
        let object = item
            .as_object_mut()
            .ok_or_else(|| invalid_event("Responses refusal item is not an object", bytes))?;
        let content = object
            .entry("content")
            .or_insert_with(|| JsonValue::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| invalid_event("Responses refusal content is not an array", bytes))?;
        while content.len() < target_len {
            content.push(serde_json::json!({"type":"refusal","refusal":""}));
        }
        let part = content[content_index]
            .as_object_mut()
            .ok_or_else(|| invalid_event("Responses refusal part is not an object", bytes))?;
        if part.get("type").and_then(JsonValue::as_str) != Some("refusal") {
            return Err(invalid_event(
                "Responses refusal delta targeted a non-refusal content part",
                bytes,
            ));
        }
        let current = part
            .get("refusal")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| invalid_event("Responses refusal is not a string", bytes))?;
        part.insert("refusal".into(), format!("{current}{delta}").into());
        Ok(())
    }

    fn finish_refusal(
        &mut self,
        output_index: usize,
        content_index: usize,
        refusal: &str,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self
            .emitted_refusals
            .contains(&(output_index, content_index))
        {
            return Err(invalid_event(
                "duplicate Responses refusal done event",
                bytes,
            ));
        }
        let streamed = self
            .items
            .get(&output_index)
            .and_then(|item| item_refusal(item, content_index));
        if self.refusal_deltas.contains(&(output_index, content_index)) && streamed != Some(refusal)
        {
            return Err(invalid_finalize(
                "Responses refusal done differs from streamed refusal",
                bytes,
            ));
        }
        if !self.refusal_deltas.contains(&(output_index, content_index)) {
            match streamed {
                Some(existing) if existing != refusal => {
                    return Err(invalid_finalize(
                        "Responses refusal done differs from the authoritative item",
                        bytes,
                    ));
                }
                Some(_) => {}
                None => self.patch_refusal(output_index, content_index, refusal, bytes)?,
            }
        }
        parts.push(StreamPart::Custom {
            part: CustomPart::new("azure.openai.refusal", JsonValue::String(refusal.into())),
        });
        self.emitted_refusals.insert((output_index, content_index));
        Ok(())
    }

    fn patch_summary(
        &mut self,
        output_index: usize,
        summary_index: usize,
        delta: &str,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let current_len = self
            .items
            .get(&output_index)
            .and_then(|item| item.get("summary"))
            .and_then(JsonValue::as_array)
            .map_or(0, Vec::len);
        let target_len = slot_target_len(current_len, summary_index, "reasoning summary", bytes)?;
        let item = self.items.entry(output_index).or_insert_with(|| {
            serde_json::json!({"type":"reasoning","id":format!("reasoning-{output_index}"),"summary":[]})
        });
        let object = item
            .as_object_mut()
            .ok_or_else(|| invalid_event("Responses reasoning item is not an object", bytes))?;
        let summary = object
            .entry("summary")
            .or_insert_with(|| JsonValue::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| invalid_event("Responses reasoning summary is not an array", bytes))?;
        while summary.len() < target_len {
            summary.push(serde_json::json!({"type":"summary_text","text":""}));
        }
        let part = summary[summary_index].as_object_mut().ok_or_else(|| {
            invalid_event("Responses reasoning summary part is not an object", bytes)
        })?;
        let current = match part.get("text") {
            Some(value) => value.as_str().ok_or_else(|| {
                invalid_event("Responses reasoning summary text is not a string", bytes)
            })?,
            None => "",
        };
        part.insert("text".into(), format!("{current}{delta}").into());
        Ok(())
    }

    fn patch_reasoning_content(
        &mut self,
        output_index: usize,
        content_index: usize,
        delta: &str,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let current_len = self
            .items
            .get(&output_index)
            .and_then(|item| item.get("content"))
            .and_then(JsonValue::as_array)
            .map_or(0, Vec::len);
        let target_len = slot_target_len(current_len, content_index, "reasoning content", bytes)?;
        let item = self.items.entry(output_index).or_insert_with(|| {
            serde_json::json!({"type":"reasoning","id":format!("reasoning-{output_index}"),"content":[]})
        });
        let object = item
            .as_object_mut()
            .ok_or_else(|| invalid_event("Responses reasoning item is not an object", bytes))?;
        let content = object
            .entry("content")
            .or_insert_with(|| JsonValue::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| invalid_event("Responses reasoning content is not an array", bytes))?;
        while content.len() < target_len {
            content.push(serde_json::json!({"type":"reasoning_text","text":""}));
        }
        let part = content[content_index].as_object_mut().ok_or_else(|| {
            invalid_event("Responses reasoning content part is not an object", bytes)
        })?;
        let current = match part.get("text") {
            Some(value) => value.as_str().ok_or_else(|| {
                invalid_event("Responses reasoning content text is not a string", bytes)
            })?,
            None => "",
        };
        part.insert("text".into(), format!("{current}{delta}").into());
        Ok(())
    }

    fn close_text_for(&mut self, output_index: usize, parts: &mut Vec<StreamPart>) {
        let prefix = self.text_id(output_index, 0);
        let ids = self
            .open_text
            .iter()
            .filter(|id| *id == &prefix || id.starts_with(&format!("{prefix}:text:")))
            .cloned()
            .collect::<Vec<_>>();
        for id in ids {
            self.open_text.remove(&id);
            parts.push(StreamPart::TextEnd { id, metadata: None });
        }
    }

    fn close_reasoning_for(&mut self, output_index: usize, parts: &mut Vec<StreamPart>) {
        let item_id = self
            .items
            .get(&output_index)
            .and_then(|item| item.get("id"))
            .and_then(JsonValue::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("reasoning-{output_index}"));
        let ids = self
            .open_reasoning
            .iter()
            .filter(|id| id.starts_with(&format!("{item_id}:")))
            .cloned()
            .collect::<Vec<_>>();
        for id in ids {
            self.open_reasoning.remove(&id);
            parts.push(StreamPart::ReasoningEnd { id, metadata: None });
        }
    }

    fn close_all(&mut self, parts: &mut Vec<StreamPart>) {
        for id in std::mem::take(&mut self.open_text) {
            parts.push(StreamPart::TextEnd { id, metadata: None });
        }
        for id in std::mem::take(&mut self.open_reasoning) {
            parts.push(StreamPart::ReasoningEnd { id, metadata: None });
        }
    }

    fn capture_response(&mut self, response: &JsonValue) {
        for (source, key) in [
            ("id", "azure.response_id"),
            ("model", "azure.model"),
            ("created_at", "azure.created_at"),
            ("service_tier", "azure.service_tier"),
        ] {
            if let Some(value) = response
                .get(source)
                .cloned()
                .filter(|value| !value.is_null())
            {
                self.response_metadata.insert(key.into(), value);
            }
        }
    }
}

fn event_index(value: &JsonValue, field: &str, bytes: u64) -> Result<usize, ModelError> {
    bounded_index(value, field, MAX_CONTENT_SLOTS, bytes)
}

fn bounded_index(
    value: &JsonValue,
    field: &str,
    limit: usize,
    bytes: u64,
) -> Result<usize, ModelError> {
    let raw = value
        .get(field)
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid_event(&format!("Responses event is missing {field}"), bytes))?;
    let index = usize::try_from(raw).map_err(|_| {
        invalid_event(
            &format!("Responses {field} does not fit the local index type"),
            bytes,
        )
    })?;
    if index >= limit {
        return Err(invalid_event(
            &format!("Responses {field} exceeds the supported slot limit"),
            bytes,
        ));
    }
    Ok(index)
}

fn slot_target_len(
    current_len: usize,
    index: usize,
    label: &str,
    bytes: u64,
) -> Result<usize, ModelError> {
    if current_len > MAX_CONTENT_SLOTS {
        return Err(invalid_event(
            &format!("Responses {label} exceeds the supported slot limit"),
            bytes,
        ));
    }
    let target_len = index
        .checked_add(1)
        .ok_or_else(|| invalid_event(&format!("Responses {label} index overflow"), bytes))?;
    if target_len > MAX_CONTENT_SLOTS {
        return Err(invalid_event(
            &format!("Responses {label} exceeds the supported slot limit"),
            bytes,
        ));
    }
    let gap = if target_len > current_len {
        target_len.checked_sub(current_len).ok_or_else(|| {
            invalid_event(&format!("Responses {label} gap arithmetic overflow"), bytes)
        })?
    } else {
        0
    };
    if gap > MAX_SLOT_GAP {
        return Err(invalid_event(
            &format!("Responses {label} index gap is too large"),
            bytes,
        ));
    }
    Ok(target_len)
}

fn validate_stream_item_slots(item: &JsonValue, bytes: u64) -> Result<(), ModelError> {
    for field in ["content", "summary"] {
        if item
            .get(field)
            .and_then(JsonValue::as_array)
            .is_some_and(|values| values.len() > MAX_CONTENT_SLOTS)
        {
            return Err(invalid_event(
                "Responses streamed item exceeds the supported slot limit",
                bytes,
            ));
        }
    }
    Ok(())
}

fn item_refusal(item: &JsonValue, index: usize) -> Option<&str> {
    item.get("content")
        .and_then(JsonValue::as_array)
        .and_then(|values| values.get(index))
        .filter(|value| value.get("type").and_then(JsonValue::as_str) == Some("refusal"))
        .and_then(|value| value.get("refusal"))
        .and_then(JsonValue::as_str)
}

fn update_function(function: &mut FunctionState, item: &JsonValue) {
    if let Some(value) = item.get("id").and_then(JsonValue::as_str) {
        function.item_id = Some(value.into());
    }
    if let Some(value) = item.get("name").and_then(JsonValue::as_str) {
        function.name = Some(value.into());
    }
}

fn ensure_function_id(
    function: &mut FunctionState,
    item: &JsonValue,
    used: &mut BTreeSet<String>,
    fallback: u64,
) -> String {
    if let Some(id) = &function.call_id {
        return id.clone();
    }
    let provider_id = item
        .get("call_id")
        .and_then(JsonValue::as_str)
        .filter(|value| !value.is_empty());
    let id = reserve_tool_id(used, provider_id, fallback);
    function.call_id = Some(id.clone());
    id
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

fn item_identity_matches(streamed: &JsonValue, terminal: &JsonValue) -> bool {
    if let (Some(streamed), Some(terminal)) = (
        streamed.get("id").and_then(JsonValue::as_str),
        terminal.get("id").and_then(JsonValue::as_str),
    ) {
        return !streamed.is_empty() && streamed == terminal;
    }
    if let (Some(streamed), Some(terminal)) = (
        streamed.get("call_id").and_then(JsonValue::as_str),
        terminal.get("call_id").and_then(JsonValue::as_str),
    ) {
        return !streamed.is_empty() && streamed == terminal;
    }
    streamed.get("type") == terminal.get("type")
        && streamed.get("content") == terminal.get("content")
        && streamed.get("summary") == terminal.get("summary")
}

fn start_function(function: &mut FunctionState, parts: &mut Vec<StreamPart>) {
    if function.started {
        return;
    }
    let Some(call_id) = function.call_id.clone().filter(|value| !value.is_empty()) else {
        return;
    };
    let name = function.name.clone().unwrap_or_default();
    function.started = true;
    parts.push(StreamPart::ToolCallStart {
        id: call_id,
        name,
        metadata: None,
    });
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

fn usage_from(value: &JsonValue) -> Usage {
    let output = value.get("output_tokens").and_then(JsonValue::as_u64);
    let reasoning = value
        .pointer("/output_tokens_details/reasoning_tokens")
        .and_then(JsonValue::as_u64);
    Usage {
        input_tokens: value.get("input_tokens").and_then(JsonValue::as_u64),
        input_tokens_no_cache: None,
        input_tokens_cache_read: value
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(JsonValue::as_u64),
        input_tokens_cache_write: None,
        output_tokens: output,
        output_tokens_text: output.map(|total| total.saturating_sub(reasoning.unwrap_or(0))),
        output_tokens_reasoning: reasoning,
        raw: Some(value.clone()),
    }
}
