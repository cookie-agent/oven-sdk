//! OpenAI Responses event state machine.

use std::collections::{BTreeMap, BTreeSet};

use oven_sdk::{
    AdapterId, ErrorStage, Finish, FinishReason, JsonValue, ModelError, ModelErrorKind,
    NativeContextScope, NativeReplayArtifact, ReplayPolicy, StreamPart, ToolCallPart, Usage,
};

use crate::responses::replay;
use crate::wire::responses::REPLAY_FORMAT;

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
    scope: NativeContextScope,
    policy: ReplayPolicy,
    items: BTreeMap<u64, JsonValue>,
    functions: BTreeMap<u64, FunctionState>,
    tool_ids: BTreeSet<String>,
    finalized: BTreeSet<u64>,
    open_text: BTreeSet<String>,
    open_reasoning: BTreeSet<String>,
    text_deltas: BTreeSet<(u64, u64)>,
    reasoning_deltas: BTreeSet<(u64, String, u64)>,
    closed_reasoning: BTreeSet<(u64, String, u64)>,
    client_calls: bool,
    usage: Usage,
    response_metadata: BTreeMap<String, JsonValue>,
    status: Option<String>,
    incomplete_details: JsonValue,
    done: bool,
}

impl State {
    pub(crate) fn new(
        adapter_id: AdapterId,
        scope: NativeContextScope,
        policy: ReplayPolicy,
    ) -> Self {
        Self {
            adapter_id,
            scope,
            policy,
            items: BTreeMap::new(),
            functions: BTreeMap::new(),
            tool_ids: BTreeSet::new(),
            finalized: BTreeSet::new(),
            open_text: BTreeSet::new(),
            open_reasoning: BTreeSet::new(),
            text_deltas: BTreeSet::new(),
            reasoning_deltas: BTreeSet::new(),
            closed_reasoning: BTreeSet::new(),
            client_calls: false,
            usage: Usage::default(),
            response_metadata: BTreeMap::new(),
            status: None,
            incomplete_details: JsonValue::Null,
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
            return Err(invalid_event("Responses event after terminal event", bytes));
        }
        let event = value.get("type").and_then(JsonValue::as_str).unwrap_or("");
        match event {
            "response.created" | "response.in_progress" => {
                self.capture_response(value.get("response").unwrap_or(&value));
                Ok(())
            }
            "response.output_item.added" => {
                let output_index = output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let item = value.get("item").cloned().ok_or_else(|| {
                    invalid_event("Responses item-added event is missing item", bytes)
                })?;
                self.add_item(output_index, item, parts);
                Ok(())
            }
            "response.output_text.delta" => {
                let output_index = output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let content_index = value
                    .get("content_index")
                    .and_then(JsonValue::as_u64)
                    .unwrap_or(0);
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
            "response.reasoning_summary_text.delta" => {
                let output_index = output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let summary_index = value
                    .get("summary_index")
                    .and_then(JsonValue::as_u64)
                    .unwrap_or(0);
                let delta = value.get("delta").and_then(JsonValue::as_str).unwrap_or("");
                self.patch_summary(output_index, summary_index, delta, bytes)?;
                self.prepare_summary_slots(output_index, summary_index, parts, bytes)?;
                self.reasoning_delta(output_index, summary_index, "summary", delta, parts);
                Ok(())
            }
            "response.reasoning_text.delta" => {
                let output_index = output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let content_index = value
                    .get("content_index")
                    .and_then(JsonValue::as_u64)
                    .unwrap_or(0);
                let delta = value.get("delta").and_then(JsonValue::as_str).unwrap_or("");
                self.patch_reasoning_content(output_index, content_index, delta, bytes)?;
                self.reasoning_deltas
                    .insert((output_index, "raw".into(), content_index));
                self.reasoning_delta(output_index, content_index, "raw", delta, parts);
                Ok(())
            }
            "response.function_call_arguments.delta" => {
                let output_index = output_index(&value, bytes)?;
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
                let output_index = output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                self.functions.entry(output_index).or_default().expected = value
                    .get("arguments")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                Ok(())
            }
            "response.output_item.done" => {
                let output_index = output_index(&value, bytes)?;
                self.ensure_unfinalized(output_index, bytes)?;
                let item = value.get("item").cloned().ok_or_else(|| {
                    invalid_event("Responses item-done event is missing item", bytes)
                })?;
                self.finish_item(output_index, item, parts, bytes)
            }
            "response.completed" => self.finish_response(value, None, parts, bytes),
            "response.incomplete" => {
                let reason = value
                    .pointer("/response/incomplete_details/reason")
                    .or_else(|| value.pointer("/incomplete_details/reason"))
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
                self.finish_response(value, reason, parts, bytes)
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
            | "response.refusal.delta"
            | "response.refusal.done"
            | "response.queued" => Ok(()),
            _ => Ok(()),
        }
    }

    fn add_item(&mut self, output_index: u64, mut item: JsonValue, parts: &mut Vec<StreamPart>) {
        if item.get("type").and_then(JsonValue::as_str) == Some("function_call") {
            self.normalize_function_id(output_index, &mut item);
            let function = self
                .functions
                .get_mut(&output_index)
                .expect("function added");
            start_function(function, parts);
        }
        self.items.insert(output_index, item);
    }

    fn finish_item(
        &mut self,
        output_index: u64,
        mut item: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if item.get("type").and_then(JsonValue::as_str) == Some("function_call") {
            self.normalize_function_id(output_index, &mut item);
        }
        self.validate_streamed_item(output_index, &item, bytes)?;
        match item.get("type").and_then(JsonValue::as_str) {
            Some("function_call") => {
                let item_id = required_function_field(&item, "id", bytes)?;
                let call_id = required_function_field(&item, "call_id", bytes)?;
                let name = required_function_field(&item, "name", bytes)?;
                let final_arguments = required_function_field(&item, "arguments", bytes)?;
                let function = self.functions.entry(output_index).or_default();
                update_function(function, &item);
                if function
                    .expected
                    .as_deref()
                    .is_some_and(|expected| expected != final_arguments)
                {
                    return Err(invalid_finalize(
                        "Responses done and authoritative tool arguments differ",
                        bytes,
                    ));
                }
                if !function.arguments.is_empty() && function.arguments != final_arguments {
                    return Err(invalid_finalize(
                        "Responses streamed and authoritative tool arguments differ",
                        bytes,
                    ));
                }
                let parsed: JsonValue = serde_json::from_str(&final_arguments).map_err(|_| {
                    invalid_finalize("final Responses tool arguments are invalid JSON", bytes)
                })?;
                if !parsed.is_object() {
                    return Err(invalid_finalize(
                        "final Responses tool arguments must be a JSON object",
                        bytes,
                    ));
                }
                function.item_id = Some(item_id.clone());
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
                call.provider_item_id = Some(item_id);
                call.raw_input = Some(final_arguments);
                parts.push(StreamPart::ToolCall { tool_call: call });
                function.finalized = true;
                self.client_calls = true;
            }
            Some("message") => self.finish_message(output_index, &item, parts),
            Some("reasoning") => self.finish_reasoning(output_index, &item, parts),
            _ => {}
        }
        self.items.insert(output_index, item);
        self.finalized.insert(output_index);
        Ok(())
    }

    fn finish_response(
        &mut self,
        value: JsonValue,
        incomplete_reason: Option<String>,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let response = value.get("response").unwrap_or(&value);
        self.capture_response(response);
        let output = response
            .get("output")
            .and_then(JsonValue::as_array)
            .cloned()
            .unwrap_or_default();
        let output_len = output.len() as u64;
        if self
            .items
            .keys()
            .chain(self.functions.keys())
            .any(|index| *index >= output_len)
        {
            return Err(invalid_finalize(
                "Responses terminal output omitted a streamed item",
                bytes,
            ));
        }
        for (index, mut item) in output.iter().cloned().enumerate() {
            let output_index = index as u64;
            if item.get("type").and_then(JsonValue::as_str) == Some("function_call") {
                self.normalize_function_id(output_index, &mut item);
            }
            if self.finalized.contains(&output_index) {
                let finalized = self.items.get(&output_index).ok_or_else(|| {
                    invalid_finalize("Responses finalized item state is missing", bytes)
                })?;
                validate_finalized_item(finalized, &item, bytes)?;
            } else {
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
        self.status = response
            .get("status")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
        self.incomplete_details = response
            .get("incomplete_details")
            .cloned()
            .unwrap_or(JsonValue::Null);
        let finish_reason = match incomplete_reason.as_deref() {
            Some("max_output_tokens") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            Some(other) => FinishReason::Other(other.into()),
            None if self.client_calls => FinishReason::ToolCalls,
            None => FinishReason::Stop,
        };
        let mut finish = Finish::new(std::mem::take(&mut self.usage), finish_reason);
        finish.response_metadata = std::mem::take(&mut self.response_metadata);
        if self.policy != ReplayPolicy::Never {
            let items = std::mem::take(&mut self.items)
                .into_values()
                .collect::<Vec<_>>();
            let payload = serde_json::json!({
                "format":REPLAY_FORMAT,
                "items":items,
                "store":false,
                "status":self.status.take(),
                "incomplete_details":std::mem::take(&mut self.incomplete_details)
            });
            finish.native_replay = Some(
                NativeReplayArtifact::new(self.adapter_id.clone(), self.scope.clone(), payload)
                    .map_err(|_| {
                        ModelError::replay(
                            "OpenAI Responses replay artifact exceeds its size limit",
                        )
                        .with_stage(ErrorStage::ReplayEncode)
                    })?,
            );
        }
        parts.push(StreamPart::Finish { finish });
        self.done = true;
        Ok(())
    }

    fn normalize_function_id(&mut self, output_index: u64, item: &mut JsonValue) {
        let function = self.functions.entry(output_index).or_default();
        update_function(function, item);
        let call_id = if let Some(call_id) = &function.call_id {
            Some(call_id.clone())
        } else if let Some(provider_id) = item
            .get("call_id")
            .and_then(JsonValue::as_str)
            .filter(|value| !value.is_empty())
        {
            let call_id = reserve_tool_id(&mut self.tool_ids, provider_id);
            function.call_id = Some(call_id.clone());
            Some(call_id)
        } else {
            None
        };
        if let Some(call_id) = call_id {
            item["call_id"] = call_id.into();
        }
    }

    fn finish_message(&mut self, output_index: u64, item: &JsonValue, parts: &mut Vec<StreamPart>) {
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
            let content_index = content_index as u64;
            if content.get("type").and_then(JsonValue::as_str) != Some("output_text")
                || self.text_deltas.contains(&(output_index, content_index))
            {
                continue;
            }
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
    }

    fn validate_streamed_item(
        &self,
        output_index: u64,
        authoritative: &JsonValue,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let Some(streamed) = self.items.get(&output_index) else {
            return Ok(());
        };
        validate_optional_string(streamed, authoritative, "type", bytes)?;
        validate_optional_string(streamed, authoritative, "id", bytes)?;
        match authoritative.get("type").and_then(JsonValue::as_str) {
            Some("message") => {
                for (_, item_index) in self
                    .text_deltas
                    .iter()
                    .filter(|(index, _)| *index == output_index)
                {
                    if item_text(streamed, "content", *item_index)
                        != item_text(authoritative, "content", *item_index)
                    {
                        return Err(invalid_finalize(
                            "Responses streamed and authoritative message text differ",
                            bytes,
                        ));
                    }
                }
            }
            Some("reasoning") => {
                for (_, kind, item_index) in self
                    .reasoning_deltas
                    .iter()
                    .filter(|(index, _, _)| *index == output_index)
                {
                    let field = if kind == "summary" {
                        "summary"
                    } else {
                        "content"
                    };
                    if item_text(streamed, field, *item_index)
                        != item_text(authoritative, field, *item_index)
                    {
                        return Err(invalid_finalize(
                            "Responses streamed and authoritative reasoning text differ",
                            bytes,
                        ));
                    }
                }
            }
            Some("function_call") => {
                validate_optional_string(streamed, authoritative, "call_id", bytes)?;
                validate_optional_string(streamed, authoritative, "name", bytes)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn ensure_unfinalized(&self, output_index: u64, bytes: u64) -> Result<(), ModelError> {
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
        output_index: u64,
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
                self.synthesize_reasoning(
                    output_index,
                    &item_id,
                    "summary",
                    index as u64,
                    summary,
                    parts,
                );
            }
        }
        if let Some(content) = item.get("content").and_then(JsonValue::as_array) {
            for (index, content) in content.iter().enumerate() {
                if content.get("type").and_then(JsonValue::as_str) == Some("reasoning_text") {
                    self.synthesize_reasoning(
                        output_index,
                        &item_id,
                        "raw",
                        index as u64,
                        content,
                        parts,
                    );
                }
            }
        }
        if self.policy != ReplayPolicy::Never
            && let (Some(item_id), Some(encrypted)) = (
                item.get("id").and_then(JsonValue::as_str),
                item.get("encrypted_content").and_then(JsonValue::as_str),
            )
            && !item_id.is_empty()
            && !encrypted.is_empty()
        {
            parts.push(StreamPart::Custom {
                part: replay::continuation_part(item_id, encrypted),
            });
        }
    }

    fn synthesize_reasoning(
        &self,
        output_index: u64,
        item_id: &str,
        kind: &str,
        index: u64,
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
        output_index: u64,
        index: u64,
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
        output_index: u64,
        summary_index: u64,
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

    fn text_id(&self, output_index: u64, content_index: u64) -> String {
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
        output_index: u64,
        content_index: u64,
        delta: &str,
        bytes: u64,
    ) -> Result<(), ModelError> {
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
        while content.len() <= content_index as usize {
            content.push(serde_json::json!({"type":"output_text","text":""}));
        }
        let part = content[content_index as usize]
            .as_object_mut()
            .ok_or_else(|| {
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

    fn patch_summary(
        &mut self,
        output_index: u64,
        summary_index: u64,
        delta: &str,
        bytes: u64,
    ) -> Result<(), ModelError> {
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
        while summary.len() <= summary_index as usize {
            summary.push(serde_json::json!({"type":"summary_text","text":""}));
        }
        let part = summary[summary_index as usize]
            .as_object_mut()
            .ok_or_else(|| {
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
        output_index: u64,
        content_index: u64,
        delta: &str,
        bytes: u64,
    ) -> Result<(), ModelError> {
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
        while content.len() <= content_index as usize {
            content.push(serde_json::json!({"type":"reasoning_text","text":""}));
        }
        let part = content[content_index as usize]
            .as_object_mut()
            .ok_or_else(|| {
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

    fn close_text_for(&mut self, output_index: u64, parts: &mut Vec<StreamPart>) {
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

    fn close_reasoning_for(&mut self, output_index: u64, parts: &mut Vec<StreamPart>) {
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
            ("id", "openai.response_id"),
            ("model", "openai.model"),
            ("created_at", "openai.created_at"),
            ("service_tier", "openai.service_tier"),
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

fn output_index(value: &JsonValue, _bytes: u64) -> Result<u64, ModelError> {
    Ok(value
        .get("output_index")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0))
}

fn item_text<'a>(item: &'a JsonValue, field: &str, index: u64) -> Option<&'a str> {
    item.get(field)
        .and_then(JsonValue::as_array)
        .and_then(|values| values.get(index as usize))
        .and_then(|value| value.get("text"))
        .and_then(JsonValue::as_str)
}

fn validate_optional_string(
    streamed: &JsonValue,
    authoritative: &JsonValue,
    field: &str,
    bytes: u64,
) -> Result<(), ModelError> {
    if let Some(streamed) = streamed.get(field).and_then(JsonValue::as_str)
        && authoritative.get(field).and_then(JsonValue::as_str) != Some(streamed)
    {
        return Err(invalid_finalize(
            "Responses streamed and authoritative item identity differ",
            bytes,
        ));
    }
    Ok(())
}

fn validate_finalized_item(
    finalized: &JsonValue,
    authoritative: &JsonValue,
    bytes: u64,
) -> Result<(), ModelError> {
    let kind = finalized.get("type").and_then(JsonValue::as_str);
    if kind != authoritative.get("type").and_then(JsonValue::as_str) {
        return Err(invalid_finalize(
            "Responses done and terminal item types differ",
            bytes,
        ));
    }
    let matches = match kind {
        Some("message") => {
            selected_item(finalized, &["type", "id", "role", "content"])
                == selected_item(authoritative, &["type", "id", "role", "content"])
        }
        Some("reasoning") => {
            selected_item(
                finalized,
                &["type", "id", "summary", "content", "encrypted_content"],
            ) == selected_item(
                authoritative,
                &["type", "id", "summary", "content", "encrypted_content"],
            )
        }
        Some("function_call") => {
            selected_item(finalized, &["type", "id", "call_id", "name", "arguments"])
                == selected_item(
                    authoritative,
                    &["type", "id", "call_id", "name", "arguments"],
                )
        }
        _ => finalized == authoritative,
    };
    if matches {
        Ok(())
    } else {
        Err(invalid_finalize(
            "Responses done and terminal authoritative items differ",
            bytes,
        ))
    }
}

fn selected_item(item: &JsonValue, fields: &[&str]) -> JsonValue {
    JsonValue::Object(
        fields
            .iter()
            .filter_map(|field| {
                item.get(*field)
                    .cloned()
                    .map(|value| ((*field).into(), value))
            })
            .collect(),
    )
}

fn update_function(function: &mut FunctionState, item: &JsonValue) {
    if let Some(value) = item.get("id").and_then(JsonValue::as_str) {
        function.item_id = Some(value.into());
    }
    if let Some(value) = item.get("name").and_then(JsonValue::as_str) {
        function.name = Some(value.into());
    }
}

fn reserve_tool_id(used: &mut BTreeSet<String>, provider_id: &str) -> String {
    let base = provider_id.to_owned();
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

fn required_function_field(
    item: &JsonValue,
    field: &str,
    bytes: u64,
) -> Result<String, ModelError> {
    item.get(field)
        .and_then(JsonValue::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            invalid_finalize(
                &format!("Responses function call is missing non-empty {field}"),
                bytes,
            )
        })
}

fn start_function(function: &mut FunctionState, parts: &mut Vec<StreamPart>) {
    if function.started {
        return;
    }
    let Some(_) = function
        .item_id
        .as_deref()
        .filter(|value| !value.is_empty())
    else {
        return;
    };
    let Some(call_id) = function.call_id.clone().filter(|value| !value.is_empty()) else {
        return;
    };
    let Some(name) = function.name.clone().filter(|value| !value.is_empty()) else {
        return;
    };
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

pub(crate) fn usage_from(value: &JsonValue) -> Usage {
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
