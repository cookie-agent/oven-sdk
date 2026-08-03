//! Vertex Gemini response normalization and live SSE handling.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    time::Duration,
};

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, BoxStream, CustomPart, ErrorStage, FilePart, FileSource, Finish, FinishReason,
    JsonValue, ModelError, ModelErrorKind, NativeContextScope, NativeReplayArtifact, ReplayPolicy,
    SourcePart, StreamItem, StreamPart, ToolCallPart, Usage,
};
use reqwest::header::HeaderMap;
use url::Url;

use crate::{GOOGLE_VERTEX_GENERATE_CONTENT_ADAPTER_ID, REPLAY_FORMAT, error::classify_error};

struct PartialCall {
    provider_id: Option<String>,
    name: String,
    accumulator: JsonAccumulator,
    thought_signature: Option<JsonValue>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PathSegment {
    Key(String),
    Index(usize),
}

struct StackEntry {
    segment: Option<PathSegment>,
    is_array: bool,
    child_count: usize,
}

struct JsonAccumulator {
    value: JsonValue,
    text: String,
    stack: Vec<StackEntry>,
    open_string_path: Option<Vec<PathSegment>>,
    closed_containers: BTreeSet<Vec<PathSegment>>,
}

impl JsonAccumulator {
    fn new() -> Self {
        Self {
            value: serde_json::json!({}),
            text: String::new(),
            stack: Vec::new(),
            open_string_path: None,
            closed_containers: BTreeSet::new(),
        }
    }

    fn process(&mut self, arg: &JsonValue, bytes: u64) -> Result<String, ModelError> {
        let path = arg
            .get("jsonPath")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| invalid_event("Vertex partial argument is missing jsonPath", bytes))?;
        let path = parse_path(path, bytes)?;
        let value = partial_value(arg, bytes)?;
        let will_continue = match arg.get("willContinue") {
            Some(value) => Some(value.as_bool().ok_or_else(|| {
                invalid_event(
                    "Vertex partial argument willContinue must be a boolean",
                    bytes,
                )
            })?),
            None => None,
        };

        let Some(value) = value else {
            if self.open_string_path.as_ref() == Some(&path) && will_continue != Some(true) {
                self.open_string_path = None;
                self.text.push('"');
                return Ok("\"".into());
            }
            return Err(invalid_event(
                "Vertex partial argument has no supported scalar value",
                bytes,
            ));
        };
        if will_continue == Some(true) && !value.is_string() {
            return Err(invalid_event(
                "Vertex partial argument willContinue is valid only for strings",
                bytes,
            ));
        }

        if let Some(existing) = nested_value(&self.value, &path) {
            let (JsonValue::String(_), JsonValue::String(fragment)) = (existing, &value) else {
                return Err(invalid_event(
                    "Vertex partial argument path was assigned conflicting values",
                    bytes,
                ));
            };
            if self.open_string_path.as_ref() != Some(&path) {
                return Err(invalid_event(
                    "Vertex string continuation targeted a closed argument path",
                    bytes,
                ));
            }
            let mut delta = escaped_string_contents(fragment);
            append_nested_string(&mut self.value, &path, fragment, bytes)?;
            if will_continue != Some(true) {
                delta.push('"');
                self.open_string_path = None;
            }
            self.text.push_str(&delta);
            return Ok(delta);
        }

        if self.open_string_path.is_some() {
            return Err(invalid_event(
                "Vertex partial arguments changed path before a string fragment completed",
                bytes,
            ));
        }

        let delta = self.emit_navigation(&path, arg, &value, bytes)?;
        insert_nested_value(&mut self.value, &path, value, bytes)?;
        self.text.push_str(&delta);
        Ok(delta)
    }

    fn finalize(&mut self, bytes: u64) -> Result<(JsonValue, String, String), ModelError> {
        let mut closing = String::new();
        if self.open_string_path.take().is_some() {
            closing.push('"');
        }
        if self.stack.is_empty() {
            closing.push_str("{}");
        } else {
            closing.push_str(&self.close_down_to(1));
            closing.push('}');
        }
        self.text.push_str(&closing);
        let parsed: JsonValue = serde_json::from_str(&self.text).map_err(|_| {
            invalid_final(
                "Vertex partial function arguments did not compose into valid JSON",
                bytes,
            )
        })?;
        if parsed != self.value {
            return Err(invalid_final(
                "Vertex partial function argument text did not match accumulated values",
                bytes,
            ));
        }
        Ok((parsed, closing, self.text.clone()))
    }

    fn emit_navigation(
        &mut self,
        target: &[PathSegment],
        arg: &JsonValue,
        value: &JsonValue,
        bytes: u64,
    ) -> Result<String, ModelError> {
        let mut fragment = self.ensure_root();
        let target_container = &target[..target.len() - 1];
        for depth in 1..=target_container.len() {
            if self.closed_containers.contains(&target_container[..depth]) {
                return Err(invalid_event(
                    "Vertex partial argument attempted to reopen a completed JSON container",
                    bytes,
                ));
            }
        }
        let common_depth = self.common_stack_depth(target_container);
        fragment.push_str(&self.close_down_to(common_depth));
        fragment.push_str(&self.open_down_to(
            target_container,
            target.last().expect("path is non-empty"),
            bytes,
        )?);
        fragment.push_str(&self.emit_leaf(
            target.last().expect("path is non-empty"),
            arg,
            value,
            bytes,
        )?);
        Ok(fragment)
    }

    fn ensure_root(&mut self) -> String {
        if self.stack.is_empty() {
            self.stack.push(StackEntry {
                segment: None,
                is_array: false,
                child_count: 0,
            });
            "{".into()
        } else {
            String::new()
        }
    }

    fn common_stack_depth(&self, target: &[PathSegment]) -> usize {
        let mut common = 0;
        for (entry, segment) in self.stack.iter().skip(1).zip(target) {
            if entry.segment.as_ref() == Some(segment) {
                common += 1;
            } else {
                break;
            }
        }
        common + 1
    }

    fn close_down_to(&mut self, target_depth: usize) -> String {
        let mut fragment = String::new();
        while self.stack.len() > target_depth {
            let path = self
                .stack
                .iter()
                .skip(1)
                .filter_map(|entry| entry.segment.clone())
                .collect::<Vec<_>>();
            self.closed_containers.insert(path);
            let entry = self.stack.pop().expect("stack depth checked");
            fragment.push(if entry.is_array { ']' } else { '}' });
        }
        fragment
    }

    fn open_down_to(
        &mut self,
        target: &[PathSegment],
        leaf: &PathSegment,
        bytes: u64,
    ) -> Result<String, ModelError> {
        let mut fragment = String::new();
        let start = self.stack.len() - 1;
        for index in start..target.len() {
            let segment = &target[index];
            let parent = self.stack.last_mut().expect("root is open");
            fragment.push_str(&member_prefix(parent, segment, bytes)?);
            let next = target.get(index + 1).unwrap_or(leaf);
            let is_array = matches!(next, PathSegment::Index(_));
            fragment.push(if is_array { '[' } else { '{' });
            self.stack.push(StackEntry {
                segment: Some(segment.clone()),
                is_array,
                child_count: 0,
            });
        }
        Ok(fragment)
    }

    fn emit_leaf(
        &mut self,
        leaf: &PathSegment,
        arg: &JsonValue,
        value: &JsonValue,
        bytes: u64,
    ) -> Result<String, ModelError> {
        let container = self.stack.last_mut().expect("root is open");
        let mut fragment = member_prefix(container, leaf, bytes)?;
        let encoded = serde_json::to_string(value)
            .map_err(|_| invalid_event("Vertex partial argument could not be encoded", bytes))?;
        if value.is_string() && arg.get("willContinue").and_then(JsonValue::as_bool) == Some(true) {
            fragment.push_str(
                encoded
                    .strip_suffix('"')
                    .expect("JSON string ends in quote"),
            );
            self.open_string_path = Some(
                self.stack
                    .iter()
                    .skip(1)
                    .filter_map(|entry| entry.segment.clone())
                    .chain(std::iter::once(leaf.clone()))
                    .collect(),
            );
        } else {
            fragment.push_str(&encoded);
        }
        Ok(fragment)
    }
}

fn member_prefix(
    container: &mut StackEntry,
    segment: &PathSegment,
    bytes: u64,
) -> Result<String, ModelError> {
    let mut fragment = String::new();
    if container.child_count > 0 {
        fragment.push(',');
    }
    match (container.is_array, segment) {
        (false, PathSegment::Key(key)) => {
            fragment.push_str(
                &serde_json::to_string(key).map_err(|_| {
                    invalid_event("Vertex JSONPath key could not be encoded", bytes)
                })?,
            );
            fragment.push(':');
        }
        (true, PathSegment::Index(index)) if *index == container.child_count => {}
        (true, PathSegment::Index(_)) => {
            return Err(invalid_event(
                "Vertex partial argument array indices must be contiguous and ordered",
                bytes,
            ));
        }
        _ => {
            return Err(invalid_event(
                "Vertex partial argument JSONPath conflicts with its container type",
                bytes,
            ));
        }
    }
    container.child_count += 1;
    Ok(fragment)
}

pub(crate) struct State {
    policy: ReplayPolicy,
    native_context_scope: NativeContextScope,
    stream_function_call_arguments: bool,
    native_parts: Vec<JsonValue>,
    partial_calls: BTreeMap<String, PartialCall>,
    usage: Usage,
    response_metadata: BTreeMap<String, JsonValue>,
    provider_metadata: BTreeMap<String, JsonValue>,
    emitted_sources: BTreeSet<String>,
    call_ids: BTreeSet<String>,
    block_counter: u64,
    call_counter: u64,
    done: bool,
}

impl State {
    pub(crate) fn new(
        policy: ReplayPolicy,
        native_context_scope: NativeContextScope,
        stream_function_call_arguments: bool,
    ) -> Self {
        Self {
            policy,
            native_context_scope,
            stream_function_call_arguments,
            native_parts: Vec::new(),
            partial_calls: BTreeMap::new(),
            usage: Usage::default(),
            response_metadata: BTreeMap::new(),
            provider_metadata: BTreeMap::new(),
            emitted_sources: BTreeSet::new(),
            call_ids: BTreeSet::new(),
            block_counter: 0,
            call_counter: 0,
            done: false,
        }
    }

    pub(crate) fn response_metadata(&self) -> &BTreeMap<String, JsonValue> {
        &self.response_metadata
    }

    pub(crate) fn set_request_id(&mut self, request_id: Option<String>) {
        if let Some(request_id) = request_id {
            self.response_metadata.insert(
                "google_vertex.request_id".into(),
                JsonValue::String(request_id),
            );
        }
    }

    pub(crate) fn apply(
        &mut self,
        value: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if self.done {
            return Err(invalid_event(
                "Vertex response arrived after terminal finish",
                bytes,
            ));
        }
        if value.get("error").is_some() {
            let open_ids = self.partial_calls.keys().cloned().collect::<Vec<_>>();
            for id in open_ids {
                self.finalize_partial_call(&id, parts, bytes)?;
            }
            let error = classify_error(
                value
                    .pointer("/error/code")
                    .and_then(JsonValue::as_u64)
                    .and_then(|value| u16::try_from(value).ok())
                    .unwrap_or(500),
                value.to_string().as_bytes(),
                self.response_metadata
                    .get("google_vertex.request_id")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned),
                ErrorStage::StreamEvent,
                bytes,
                &HeaderMap::new(),
            );
            parts.push(StreamPart::Error { error });
            let mut finish = Finish::new(self.usage.clone(), FinishReason::Error);
            finish.response_metadata = self.response_metadata.clone();
            finish.provider_metadata = self.provider_metadata.clone();
            parts.push(StreamPart::Finish { finish });
            self.done = true;
            return Ok(());
        }
        self.capture_metadata(&value);
        if let Some(usage) = value.get("usageMetadata") {
            self.usage = parse_usage(usage, bytes)?;
        }
        let candidates = value.get("candidates").and_then(JsonValue::as_array);
        let candidate = candidates.and_then(|values| values.first());
        if candidates.is_some_and(|values| values.len() > 1) {
            self.provider_metadata.insert(
                "google_vertex.ignored_candidate_count".into(),
                JsonValue::from(candidates.map_or(0, |values| values.len() - 1)),
            );
        }
        if let Some(candidate) = candidate {
            self.capture_candidate_metadata(candidate);
            if let Some(native) = candidate
                .pointer("/content/parts")
                .and_then(JsonValue::as_array)
            {
                for native_part in native {
                    self.apply_part(native_part.clone(), parts, bytes)?;
                }
            }
            self.emit_sources(candidate, parts);
            if let Some(reason) = candidate.get("finishReason").and_then(JsonValue::as_str) {
                self.finish(reason, parts, bytes)?;
            }
        } else if let Some(reason) = value
            .pointer("/promptFeedback/blockReason")
            .and_then(JsonValue::as_str)
        {
            self.provider_metadata.insert(
                "google_vertex.prompt_feedback".into(),
                value
                    .get("promptFeedback")
                    .cloned()
                    .unwrap_or(JsonValue::Null),
            );
            self.finish(reason, parts, bytes)?;
        }
        Ok(())
    }

    fn capture_metadata(&mut self, value: &JsonValue) {
        for (wire, key) in [
            ("responseId", "google_vertex.response_id"),
            ("modelVersion", "google_vertex.model_version"),
        ] {
            if let Some(value) = value.get(wire).cloned() {
                self.response_metadata.insert(key.into(), value);
            }
        }
        if let Some(value) = value.get("promptFeedback").cloned() {
            self.provider_metadata
                .insert("google_vertex.prompt_feedback".into(), value);
        }
    }

    fn capture_candidate_metadata(&mut self, candidate: &JsonValue) {
        for (wire, key) in [
            ("groundingMetadata", "google_vertex.grounding_metadata"),
            ("urlContextMetadata", "google_vertex.url_context_metadata"),
            ("safetyRatings", "google_vertex.safety_ratings"),
            ("finishMessage", "google_vertex.finish_message"),
        ] {
            if let Some(value) = candidate.get(wire).cloned() {
                self.provider_metadata.insert(key.into(), value);
            }
        }
    }

    fn apply_part(
        &mut self,
        native: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if let Some(text) = native.get("text").and_then(JsonValue::as_str) {
            let text = text.to_owned();
            let thought = native.get("thought").and_then(JsonValue::as_bool) == Some(true);
            if !text.is_empty() {
                let id = format!(
                    "{}:{}",
                    if thought { "reasoning" } else { "text" },
                    self.block_counter
                );
                self.block_counter = self
                    .block_counter
                    .checked_add(1)
                    .ok_or_else(|| invalid_event("Vertex block counter overflowed", bytes))?;
                if thought {
                    parts.push(StreamPart::ReasoningStart {
                        id: id.clone(),
                        metadata: None,
                    });
                    parts.push(StreamPart::ReasoningDelta {
                        id: id.clone(),
                        delta: text,
                        metadata: None,
                    });
                    parts.push(StreamPart::ReasoningEnd { id, metadata: None });
                } else {
                    parts.push(StreamPart::TextStart {
                        id: id.clone(),
                        metadata: None,
                    });
                    parts.push(StreamPart::TextDelta {
                        id: id.clone(),
                        delta: text,
                        metadata: None,
                    });
                    parts.push(StreamPart::TextEnd { id, metadata: None });
                }
            }
            self.native_parts.push(native);
            return Ok(());
        }
        if let Some(call) = native.get("functionCall") {
            if self.stream_function_call_arguments
                && (call.get("partialArgs").is_some()
                    || call.get("willContinue").and_then(JsonValue::as_bool) == Some(true)
                    || (call.get("name").is_none()
                        && call.get("args").is_none()
                        && !self.partial_calls.is_empty()))
            {
                return self.apply_partial_call(&native, parts, bytes);
            }
            return self.emit_full_call(native, parts, bytes);
        }
        if native.get("executableCode").is_some() || native.get("toolCall").is_some() {
            parts.push(StreamPart::Custom {
                part: CustomPart::new("google_vertex.server_tool_call", native.clone()),
            });
            self.native_parts.push(native);
            return Ok(());
        }
        if native.get("codeExecutionResult").is_some() || native.get("toolResponse").is_some() {
            parts.push(StreamPart::Custom {
                part: CustomPart::new("google_vertex.server_tool_result", native.clone()),
            });
            self.native_parts.push(native);
            return Ok(());
        }
        if let Some(data) = native.get("inlineData") {
            let media_type = data
                .get("mimeType")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| invalid_event("Vertex inline output is missing MIME type", bytes))?;
            let bytes_value = data
                .get("data")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| invalid_event("Vertex inline output is missing data", bytes))?;
            let decoded =
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, bytes_value)
                    .map_err(|_| invalid_event("Vertex inline output is invalid base64", bytes))?;
            parts.push(StreamPart::File {
                file: FilePart::new(media_type, FileSource::Bytes(decoded.into())),
            });
            self.native_parts.push(native);
            return Ok(());
        }
        if native.get("thoughtSignature").is_some() {
            self.native_parts.push(native);
            return Ok(());
        }
        Err(invalid_event(
            "Vertex returned an unsupported content part",
            bytes,
        ))
    }

    fn emit_full_call(
        &mut self,
        mut native: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let call = native
            .get_mut("functionCall")
            .expect("caller checked functionCall");
        let name = call
            .get("name")
            .and_then(JsonValue::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid_final("Vertex function call is missing a name", bytes))?
            .to_owned();
        let id = call
            .get("id")
            .and_then(JsonValue::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let local_id = id.clone().unwrap_or_else(|| self.next_call_id());
        let input = call
            .get("args")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        self.emit_call((local_id, id), name, input, native, parts, bytes)
    }

    fn apply_partial_call(
        &mut self,
        native: &JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let call = native
            .get("functionCall")
            .expect("caller checked functionCall");
        let supplied_id = call
            .get("id")
            .and_then(JsonValue::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let name = call
            .get("name")
            .and_then(JsonValue::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let id = match (supplied_id.as_ref(), name.as_ref()) {
            (Some(id), _) => id.clone(),
            (None, Some(_)) => self.next_call_id(),
            (None, None) if self.partial_calls.len() == 1 => {
                self.partial_calls.keys().next().cloned().expect("one key")
            }
            (None, None) => {
                return Err(invalid_event(
                    "Vertex partial function continuation is ambiguous without an ID",
                    bytes,
                ));
            }
        };
        if let Some(name) = name {
            if self.partial_calls.contains_key(&id) {
                return Err(invalid_event(
                    "Vertex partial function call ID was reused",
                    bytes,
                ));
            }
            parts.push(StreamPart::ToolCallStart {
                id: id.clone(),
                name: name.clone(),
                metadata: None,
            });
            self.partial_calls.insert(
                id.clone(),
                PartialCall {
                    provider_id: supplied_id,
                    name,
                    accumulator: JsonAccumulator::new(),
                    thought_signature: native.get("thoughtSignature").cloned(),
                },
            );
        }
        let partial = self.partial_calls.get_mut(&id).ok_or_else(|| {
            invalid_event("Vertex partial arguments referenced no open call", bytes)
        })?;
        if let Some(signature) = native.get("thoughtSignature") {
            match &partial.thought_signature {
                Some(existing) if existing != signature => {
                    return Err(invalid_event(
                        "Vertex partial function call changed thought signature",
                        bytes,
                    ));
                }
                None => partial.thought_signature = Some(signature.clone()),
                _ => {}
            }
        }
        if let Some(values) = call.get("partialArgs").and_then(JsonValue::as_array) {
            for value in values {
                let delta = partial.accumulator.process(value, bytes)?;
                if !delta.is_empty() {
                    parts.push(StreamPart::ToolCallDelta {
                        id: id.clone(),
                        delta,
                        metadata: None,
                    });
                }
            }
        }
        let continues = call.get("willContinue").and_then(JsonValue::as_bool) == Some(true)
            || call
                .get("partialArgs")
                .and_then(JsonValue::as_array)
                .is_some_and(|values| {
                    values.iter().any(|value| {
                        value.get("willContinue").and_then(JsonValue::as_bool) == Some(true)
                    })
                });
        if continues {
            return Ok(());
        }
        self.finalize_partial_call(&id, parts, bytes)
    }

    fn finalize_partial_call(
        &mut self,
        id: &str,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let mut partial = self.partial_calls.remove(id).ok_or_else(|| {
            invalid_event("Vertex partial arguments referenced no open call", bytes)
        })?;
        let (input, closing_delta, raw) = partial.accumulator.finalize(bytes)?;
        if !closing_delta.is_empty() {
            parts.push(StreamPart::ToolCallDelta {
                id: id.to_owned(),
                delta: closing_delta,
                metadata: None,
            });
        }
        let mut native = serde_json::json!({
            "functionCall":{"name":partial.name,"args":input.clone()}
        });
        if let Some(provider_id) = &partial.provider_id {
            native["functionCall"]["id"] = JsonValue::String(provider_id.clone());
        }
        if let Some(signature) = partial.thought_signature {
            native["thoughtSignature"] = signature;
        }
        parts.push(StreamPart::ToolCallEnd {
            id: id.to_owned(),
            metadata: None,
        });
        if !self.call_ids.insert(id.to_owned()) {
            return Err(invalid_final(
                "Vertex function call ID was duplicated",
                bytes,
            ));
        }
        let mut tool_call = ToolCallPart::new(
            id,
            native["functionCall"]["name"].as_str().unwrap_or_default(),
            input,
        );
        tool_call.provider_item_id = partial.provider_id;
        tool_call.raw_input = Some(raw);
        parts.push(StreamPart::ToolCall { tool_call });
        self.native_parts.push(native);
        Ok(())
    }

    fn emit_call(
        &mut self,
        ids: (String, Option<String>),
        name: String,
        input: JsonValue,
        native: JsonValue,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let (id, provider_id) = ids;
        if !input.is_object() {
            return Err(ModelError::new(
                ModelErrorKind::InvalidToolInput,
                "Vertex function arguments must be a JSON object",
            )
            .with_stage(ErrorStage::StreamFinalize)
            .with_bytes_received(bytes));
        }
        if !self.call_ids.insert(id.clone()) {
            return Err(invalid_final(
                "Vertex function call ID was duplicated",
                bytes,
            ));
        }
        let raw = input.to_string();
        parts.push(StreamPart::ToolCallStart {
            id: id.clone(),
            name: name.clone(),
            metadata: None,
        });
        parts.push(StreamPart::ToolCallDelta {
            id: id.clone(),
            delta: raw.clone(),
            metadata: None,
        });
        parts.push(StreamPart::ToolCallEnd {
            id: id.clone(),
            metadata: None,
        });
        let mut tool_call = ToolCallPart::new(id, name, input);
        tool_call.provider_item_id = provider_id;
        tool_call.raw_input = Some(raw);
        parts.push(StreamPart::ToolCall { tool_call });
        self.native_parts.push(native);
        Ok(())
    }

    fn next_call_id(&mut self) -> String {
        let id = format!("vertex-call-{}", self.call_counter);
        self.call_counter = self.call_counter.saturating_add(1);
        id
    }

    fn emit_sources(&mut self, candidate: &JsonValue, parts: &mut Vec<StreamPart>) {
        let Some(chunks) = candidate
            .pointer("/groundingMetadata/groundingChunks")
            .and_then(JsonValue::as_array)
        else {
            return;
        };
        for chunk in chunks {
            let Some((key, source)) = normalize_source(chunk) else {
                continue;
            };
            if self.emitted_sources.insert(key) {
                parts.push(StreamPart::Source { source });
            }
        }
    }

    fn finish(
        &mut self,
        reason: &str,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        if matches!(
            reason,
            "MALFORMED_FUNCTION_CALL" | "UNEXPECTED_TOOL_CALL" | "MISSING_THOUGHT_SIGNATURE"
        ) {
            return self.finish_provider_error(reason, parts, bytes);
        }
        if !self.partial_calls.is_empty() {
            return Err(invalid_final(
                "Vertex stream finished with incomplete function arguments",
                bytes,
            ));
        }
        let has_calls = !self.call_ids.is_empty();
        let mut finish = Finish::new(self.usage.clone(), map_finish(reason, has_calls));
        finish.response_metadata = self.response_metadata.clone();
        finish.provider_metadata = self.provider_metadata.clone();
        if self.policy != ReplayPolicy::Never {
            let payload = serde_json::json!({
                "format":REPLAY_FORMAT,
                "content":{"role":"model","parts":self.native_parts},
            });
            finish.native_replay = Some(
                NativeReplayArtifact::new(
                    oven_sdk::AdapterId::new(GOOGLE_VERTEX_GENERATE_CONTENT_ADAPTER_ID),
                    self.native_context_scope.clone(),
                    payload,
                )
                .map_err(|_| {
                    ModelError::replay("Vertex native replay artifact exceeds its size limit")
                        .with_stage(ErrorStage::ReplayEncode)
                        .with_bytes_received(bytes)
                })?,
            );
        }
        parts.push(StreamPart::Finish { finish });
        self.done = true;
        Ok(())
    }

    fn finish_provider_error(
        &mut self,
        reason: &str,
        parts: &mut Vec<StreamPart>,
        bytes: u64,
    ) -> Result<(), ModelError> {
        let open_ids = self.partial_calls.keys().cloned().collect::<Vec<_>>();
        for id in open_ids {
            self.finalize_partial_call(&id, parts, bytes)?;
        }
        let (kind, message) = match reason {
            "MALFORMED_FUNCTION_CALL" => (
                ModelErrorKind::InvalidToolInput,
                "Vertex returned a malformed function call",
            ),
            "UNEXPECTED_TOOL_CALL" => (
                ModelErrorKind::InvalidResponse,
                "Vertex returned an unexpected tool call",
            ),
            "MISSING_THOUGHT_SIGNATURE" => (
                ModelErrorKind::InvalidResponse,
                "Vertex reported a missing thought signature",
            ),
            _ => unreachable!("caller checked provider error finish reason"),
        };
        let error = ModelError::new(kind, message)
            .with_vendor_code(reason)
            .with_stage(ErrorStage::StreamFinalize)
            .with_bytes_received(bytes);
        parts.push(StreamPart::Error { error });
        let mut finish = Finish::new(self.usage.clone(), FinishReason::Error);
        finish.response_metadata = self.response_metadata.clone();
        finish.provider_metadata = self.provider_metadata.clone();
        parts.push(StreamPart::Finish { finish });
        self.done = true;
        Ok(())
    }
}

fn normalize_source(chunk: &JsonValue) -> Option<(String, SourcePart)> {
    let mut output = SourcePart::new();
    let kind;
    if let Some(source) = chunk.get("web") {
        kind = "web";
        output.url = source
            .get("uri")
            .and_then(JsonValue::as_str)
            .and_then(|value| Url::parse(value).ok());
        output.title = source
            .get("title")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
    } else if let Some(source) = chunk.get("maps") {
        kind = "maps";
        output.url = source
            .get("uri")
            .and_then(JsonValue::as_str)
            .and_then(|value| Url::parse(value).ok());
        output.title = source
            .get("title")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
        output.id = source
            .get("placeId")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
    } else if let Some(source) = chunk.get("retrievedContext") {
        kind = "retrieved_context";
        output.url = source
            .get("uri")
            .and_then(JsonValue::as_str)
            .and_then(|value| Url::parse(value).ok());
        output.title = source
            .get("title")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
        output.excerpt = source
            .get("text")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
        output.id = source
            .get("documentName")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
    } else {
        let source = chunk.get("image")?;
        kind = "image";
        output.url = source
            .get("sourceUri")
            .and_then(JsonValue::as_str)
            .and_then(|value| Url::parse(value).ok());
        output.id = source
            .get("imageUri")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
    }
    let mut metadata = BTreeMap::new();
    metadata.insert("google_vertex.grounding_chunk".into(), chunk.clone());
    output.metadata = Some(metadata);
    let key = format!("{kind}:{chunk}");
    Some((key, output))
}

pub(crate) struct LiveState {
    pub(crate) bytes: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    pub(crate) parser: crate::sse::Parser,
    pub(crate) state: State,
    pub(crate) queue: VecDeque<StreamItem>,
    pub(crate) pending_events: VecDeque<crate::sse::Event>,
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
                "Vertex stream ended before a semantic response",
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
    if live.pending_events.is_empty() {
        let next = tokio::select! {
            value = tokio::time::timeout(live.idle, live.bytes.next()) => value.map_err(|_| {
                ModelError::timeout("Vertex stream idle timeout")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count)
            })?,
            _ = live.abort.aborted() => return Err(ModelError::abort("Vertex stream was aborted")
                .with_stage(ErrorStage::StreamRead)
                .with_bytes_received(live.count)),
        };
        let events = match next {
            Some(Ok(chunk)) => {
                live.count = live
                    .count
                    .checked_add(u64::try_from(chunk.len()).map_err(|_| {
                        ModelError::transport("Vertex stream byte count overflowed")
                    })?)
                    .ok_or_else(|| ModelError::transport("Vertex stream byte count overflowed"))?;
                live.parser.feed_events(&chunk)?
            }
            Some(Err(_)) => {
                return Err(ModelError::transport("Vertex stream read failed")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count));
            }
            None => {
                live.eof = true;
                live.parser.finish_events()?
            }
        };
        live.pending_events.extend(events);
    }
    let mut semantic = false;
    while let Some(event) = live.pending_events.pop_front() {
        if event.data.is_empty() || event.name == "ping" {
            continue;
        }
        let value: JsonValue = serde_json::from_str(&event.data).map_err(|_| {
            ModelError::invalid_response("Vertex SSE event is invalid JSON")
                .with_stage(ErrorStage::StreamDecode)
                .with_bytes_received(live.count)
        })?;
        semantic = true;
        let mut parts = Vec::new();
        if let Err(error) = live.state.apply(value.clone(), &mut parts, live.count) {
            live.queue.extend(parts.into_iter().map(Ok));
            live.pending_error = Some(error);
            return Ok(true);
        }
        let done = live.state.done;
        if done
            && live
                .pending_events
                .iter()
                .any(|event| !event.data.is_empty() && event.name != "ping")
        {
            live.pending_events.clear();
            live.eof = true;
            return Err(invalid_event(
                "Vertex event arrived after terminal finish",
                live.count,
            ));
        }
        if live.include_raw {
            live.queue.push_back(Ok(StreamPart::Raw { value }));
        }
        live.queue.extend(parts.into_iter().map(Ok));
        if done {
            live.pending_events.clear();
            live.eof = true;
            return Ok(true);
        }
        if stop_at_semantic {
            return Ok(true);
        }
    }
    if live.eof && !live.state.done {
        return Err(ModelError::unexpected_eof(
            "Vertex stream ended before a terminal finish reason",
        )
        .with_stage(ErrorStage::StreamFinalize)
        .with_bytes_received(live.count));
    }
    Ok(semantic)
}

pub(crate) fn normalize_single(
    value: JsonValue,
    policy: ReplayPolicy,
    native_context_scope: NativeContextScope,
    request_id: Option<String>,
    include_raw: bool,
    bytes: u64,
) -> Result<(Vec<StreamPart>, BTreeMap<String, JsonValue>), ModelError> {
    let mut state = State::new(policy, native_context_scope, false);
    state.set_request_id(request_id);
    let mut parts = vec![StreamPart::StreamStart {
        warnings: Vec::new(),
    }];
    if include_raw {
        parts.push(StreamPart::Raw {
            value: value.clone(),
        });
    }
    state.apply(value, &mut parts, bytes)?;
    if !state.done {
        return Err(ModelError::unexpected_eof(
            "Vertex generateContent response had no terminal finish reason",
        )
        .with_stage(ErrorStage::StreamFinalize)
        .with_bytes_received(bytes));
    }
    Ok((parts, state.response_metadata))
}

fn parse_path(path: &str, bytes: u64) -> Result<Vec<PathSegment>, ModelError> {
    let Some(raw) = path.strip_prefix("$.") else {
        return Err(invalid_event(
            "Vertex partial argument JSONPath must start with `$.`",
            bytes,
        ));
    };
    if raw.is_empty() {
        return Err(invalid_event(
            "Vertex partial argument path is empty",
            bytes,
        ));
    }
    let chars = raw.chars().collect::<Vec<_>>();
    let mut index = 0;
    let mut segments = Vec::new();
    while index < chars.len() {
        let start = index;
        while index < chars.len() && !matches!(chars[index], '.' | '[') {
            index += 1;
        }
        if start == index {
            return Err(invalid_event(
                "Vertex partial argument JSONPath contains an empty property",
                bytes,
            ));
        }
        segments.push(PathSegment::Key(chars[start..index].iter().collect()));
        while index < chars.len() && chars[index] == '[' {
            index += 1;
            let number_start = index;
            while index < chars.len() && chars[index].is_ascii_digit() {
                index += 1;
            }
            if number_start == index || chars.get(index) != Some(&']') {
                return Err(invalid_event(
                    "Vertex partial argument JSONPath has an invalid array index",
                    bytes,
                ));
            }
            let number = chars[number_start..index]
                .iter()
                .collect::<String>()
                .parse::<usize>()
                .map_err(|_| {
                    invalid_event("Vertex partial argument array index is too large", bytes)
                })?;
            segments.push(PathSegment::Index(number));
            index += 1;
        }
        if index < chars.len() {
            if chars[index] != '.' {
                return Err(invalid_event(
                    "Vertex partial argument JSONPath contains unsupported syntax",
                    bytes,
                ));
            }
            index += 1;
            if index == chars.len() {
                return Err(invalid_event(
                    "Vertex partial argument JSONPath cannot end with a dot",
                    bytes,
                ));
            }
        }
    }
    Ok(segments)
}

fn partial_value(arg: &JsonValue, bytes: u64) -> Result<Option<JsonValue>, ModelError> {
    let object = arg
        .as_object()
        .ok_or_else(|| invalid_event("Vertex partial argument must be a JSON object", bytes))?;
    let fields = ["stringValue", "numberValue", "boolValue", "nullValue"];
    let present = fields
        .iter()
        .filter(|field| object.contains_key(**field))
        .count();
    if present > 1 {
        return Err(invalid_event(
            "Vertex partial argument contains multiple scalar values",
            bytes,
        ));
    }
    if let Some(value) = object.get("stringValue") {
        return value
            .as_str()
            .map(|value| Some(JsonValue::String(value.to_owned())))
            .ok_or_else(|| invalid_event("Vertex stringValue must be a string", bytes));
    }
    if let Some(value) = object.get("numberValue") {
        return value
            .is_number()
            .then(|| Some(value.clone()))
            .ok_or_else(|| invalid_event("Vertex numberValue must be a number", bytes));
    }
    if let Some(value) = object.get("boolValue") {
        return value
            .as_bool()
            .map(|value| Some(JsonValue::Bool(value)))
            .ok_or_else(|| invalid_event("Vertex boolValue must be a boolean", bytes));
    }
    if object.contains_key("nullValue") {
        return Ok(Some(JsonValue::Null));
    }
    Ok(None)
}

fn nested_value<'a>(root: &'a JsonValue, path: &[PathSegment]) -> Option<&'a JsonValue> {
    let mut current = root;
    for segment in path {
        current = match segment {
            PathSegment::Key(key) => current.as_object()?.get(key)?,
            PathSegment::Index(index) => current.as_array()?.get(*index)?,
        };
    }
    Some(current)
}

fn append_nested_string(
    root: &mut JsonValue,
    path: &[PathSegment],
    fragment: &str,
    bytes: u64,
) -> Result<(), ModelError> {
    let mut current = root;
    for segment in path {
        current = match segment {
            PathSegment::Key(key) => current
                .as_object_mut()
                .and_then(|object| object.get_mut(key))
                .ok_or_else(|| invalid_event("Vertex continuation path disappeared", bytes))?,
            PathSegment::Index(index) => current
                .as_array_mut()
                .and_then(|array| array.get_mut(*index))
                .ok_or_else(|| invalid_event("Vertex continuation index disappeared", bytes))?,
        };
    }
    let mut combined = current
        .as_str()
        .ok_or_else(|| invalid_event("Vertex continuation target is not a string", bytes))?
        .to_owned();
    combined.push_str(fragment);
    *current = JsonValue::String(combined);
    Ok(())
}

fn insert_nested_value(
    current: &mut JsonValue,
    path: &[PathSegment],
    value: JsonValue,
    bytes: u64,
) -> Result<(), ModelError> {
    let (segment, rest) = path.split_first().expect("path is non-empty");
    let last = rest.is_empty();
    match segment {
        PathSegment::Key(key) => {
            let object = current.as_object_mut().ok_or_else(|| {
                invalid_event(
                    "Vertex partial argument path conflicts with a non-object value",
                    bytes,
                )
            })?;
            if last {
                if object.contains_key(key) {
                    return Err(invalid_event(
                        "Vertex partial argument path was assigned more than once",
                        bytes,
                    ));
                }
                object.insert(key.clone(), value);
                return Ok(());
            }
            let next_is_array = matches!(rest[0], PathSegment::Index(_));
            let child = object.entry(key.clone()).or_insert_with(|| {
                if next_is_array {
                    JsonValue::Array(Vec::new())
                } else {
                    serde_json::json!({})
                }
            });
            if child.is_array() != next_is_array || (!next_is_array && !child.is_object()) {
                return Err(invalid_event(
                    "Vertex partial argument path conflicts with an existing container",
                    bytes,
                ));
            }
            insert_nested_value(child, rest, value, bytes)
        }
        PathSegment::Index(index) => {
            let array = current.as_array_mut().ok_or_else(|| {
                invalid_event(
                    "Vertex partial argument index conflicts with a non-array value",
                    bytes,
                )
            })?;
            if *index > array.len() {
                return Err(invalid_event(
                    "Vertex partial argument array indices must not contain gaps",
                    bytes,
                ));
            }
            if last {
                if *index < array.len() {
                    return Err(invalid_event(
                        "Vertex partial argument array index was assigned more than once",
                        bytes,
                    ));
                }
                array.push(value);
                return Ok(());
            }
            let next_is_array = matches!(rest[0], PathSegment::Index(_));
            if *index == array.len() {
                array.push(if next_is_array {
                    JsonValue::Array(Vec::new())
                } else {
                    serde_json::json!({})
                });
            }
            let child = &mut array[*index];
            if child.is_array() != next_is_array || (!next_is_array && !child.is_object()) {
                return Err(invalid_event(
                    "Vertex partial argument index conflicts with an existing container",
                    bytes,
                ));
            }
            insert_nested_value(child, rest, value, bytes)
        }
    }
}

fn escaped_string_contents(value: &str) -> String {
    let encoded = serde_json::to_string(value).expect("strings are JSON serializable");
    encoded[1..encoded.len() - 1].to_owned()
}

fn parse_usage(value: &JsonValue, bytes: u64) -> Result<Usage, ModelError> {
    let input = value.get("promptTokenCount").and_then(JsonValue::as_u64);
    let cached = value
        .get("cachedContentTokenCount")
        .and_then(JsonValue::as_u64);
    let text = value
        .get("candidatesTokenCount")
        .and_then(JsonValue::as_u64);
    let reasoning = value.get("thoughtsTokenCount").and_then(JsonValue::as_u64);
    let no_cache = match (input, cached) {
        (Some(total), Some(cached)) => Some(
            total
                .checked_sub(cached)
                .ok_or_else(|| invalid_event("Vertex input usage is inconsistent", bytes))?,
        ),
        (Some(total), None) => Some(total),
        _ => None,
    };
    let output = match (text, reasoning) {
        (None, None) => value
            .get("totalTokenCount")
            .and_then(JsonValue::as_u64)
            .and_then(|total| input.and_then(|input| total.checked_sub(input))),
        _ => Some(
            text.unwrap_or(0)
                .checked_add(reasoning.unwrap_or(0))
                .ok_or_else(|| invalid_event("Vertex output usage overflowed", bytes))?,
        ),
    };
    Ok(Usage {
        input_tokens: input,
        input_tokens_no_cache: no_cache,
        input_tokens_cache_read: cached,
        input_tokens_cache_write: None,
        output_tokens: output,
        output_tokens_text: text,
        output_tokens_reasoning: reasoning,
        raw: Some(value.clone()),
    })
}

fn map_finish(reason: &str, has_calls: bool) -> FinishReason {
    match reason {
        "STOP" if has_calls => FinishReason::ToolCalls,
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" => {
            FinishReason::ContentFilter
        }
        "MALFORMED_FUNCTION_CALL" | "UNEXPECTED_TOOL_CALL" | "MISSING_THOUGHT_SIGNATURE" => {
            FinishReason::Error
        }
        other => FinishReason::Other(other.into()),
    }
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

    fn native_context_scope() -> NativeContextScope {
        NativeContextScope::new(
            oven_sdk::ProviderId::new("google.vertex"),
            oven_sdk::ModelId::new("future-model"),
            oven_sdk::ResourceId::new(
                "projects/p/locations/global/publishers/google/models/resource-model",
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn partial_function_arguments_finalize_without_guessing_across_calls() {
        let mut state = State::new(ReplayPolicy::IfValid, native_context_scope(), true);
        let mut parts = Vec::new();
        state.apply(serde_json::json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":"c1","name":"lookup","partialArgs":[{"jsonPath":"$.city","stringValue":"Paris","willContinue":false}],"willContinue":false}}]}}]}), &mut parts, 1).unwrap();
        state
            .apply(
                serde_json::json!({"candidates":[{"finishReason":"STOP"}],"usageMetadata":{}}),
                &mut parts,
                2,
            )
            .unwrap();
        assert!(parts.iter().any(|part| matches!(part, StreamPart::ToolCall { tool_call } if tool_call.input == serde_json::json!({"city":"Paris"}))));
        assert!(matches!(parts.last(), Some(StreamPart::Finish { .. })));
    }

    #[test]
    fn ambiguous_idless_partial_continuation_is_rejected() {
        let mut state = State::new(ReplayPolicy::IfValid, native_context_scope(), true);
        let mut parts = Vec::new();
        for (id, name) in [("a", "one"), ("b", "two")] {
            state.apply(serde_json::json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":id,"name":name,"willContinue":true}}]}}]}), &mut parts, 1).unwrap();
        }
        let error = state.apply(serde_json::json!({"candidates":[{"content":{"parts":[{"functionCall":{"partialArgs":[{"jsonPath":"$.x","stringValue":"y"}],"willContinue":true}}]}}]}), &mut parts, 2).unwrap_err();
        assert_eq!(error.diagnostics.stage, ErrorStage::StreamEvent);
    }

    #[test]
    fn accumulator_composes_nested_arrays_and_string_fragments_incrementally() {
        let mut accumulator = JsonAccumulator::new();
        let fragments = [
            serde_json::json!({"jsonPath":"$.recipe.name","stringValue":"Las","willContinue":true}),
            serde_json::json!({"jsonPath":"$.recipe.name","stringValue":"agna"}),
            serde_json::json!({"jsonPath":"$.recipe.ingredients[0].name","stringValue":"Pasta"}),
            serde_json::json!({"jsonPath":"$.recipe.ingredients[0].quantity","numberValue":2}),
            serde_json::json!({"jsonPath":"$.recipe.ingredients[1].name","stringValue":"Sauce"}),
            serde_json::json!({"jsonPath":"$.ready","boolValue":true}),
            serde_json::json!({"jsonPath":"$.note","nullValue":null}),
        ];
        let mut streamed = String::new();
        for fragment in &fragments {
            let delta = accumulator.process(fragment, 1).unwrap();
            assert!(!delta.is_empty());
            streamed.push_str(&delta);
        }
        let (value, closing, raw) = accumulator.finalize(2).unwrap();
        streamed.push_str(&closing);
        assert_eq!(streamed, raw);
        assert_eq!(serde_json::from_str::<JsonValue>(&raw).unwrap(), value);
        assert_eq!(
            value,
            serde_json::json!({
                "recipe":{
                    "name":"Lasagna",
                    "ingredients":[
                        {"name":"Pasta","quantity":2},
                        {"name":"Sauce"}
                    ]
                },
                "ready":true,
                "note":null
            })
        );
    }

    #[test]
    fn empty_terminal_partial_arg_closes_an_open_string_without_inventing_content() {
        let mut accumulator = JsonAccumulator::new();
        let first = accumulator
            .process(
                &serde_json::json!({
                    "jsonPath":"$.location","stringValue":"New Delhi","willContinue":true
                }),
                1,
            )
            .unwrap();
        let terminal = accumulator
            .process(&serde_json::json!({"jsonPath":"$.location"}), 2)
            .unwrap();
        let (value, closing, raw) = accumulator.finalize(3).unwrap();
        assert_eq!(format!("{first}{terminal}{closing}"), raw);
        assert_eq!(value, serde_json::json!({"location":"New Delhi"}));
    }

    #[test]
    fn accumulator_rejects_conflicts_gaps_reopened_paths_and_invalid_typed_values() {
        let cases = [
            vec![
                serde_json::json!({"jsonPath":"$.value","numberValue":1}),
                serde_json::json!({"jsonPath":"$.value.child","numberValue":2}),
            ],
            vec![serde_json::json!({"jsonPath":"$.items[1]","numberValue":1})],
            vec![
                serde_json::json!({"jsonPath":"$.closed.value","numberValue":1}),
                serde_json::json!({"jsonPath":"$.other","numberValue":2}),
                serde_json::json!({"jsonPath":"$.closed.next","numberValue":3}),
            ],
            vec![serde_json::json!({
                "jsonPath":"$.value","stringValue":"x","numberValue":1
            })],
            vec![serde_json::json!({
                "jsonPath":"$.value","numberValue":1,"willContinue":true
            })],
            vec![serde_json::json!({
                "jsonPath":"$.value","stringValue":"x","willContinue":"yes"
            })],
            vec![serde_json::json!({"jsonPath":"items[0]","numberValue":1})],
        ];
        for case in cases {
            let mut accumulator = JsonAccumulator::new();
            let mut error = None;
            for fragment in case {
                if let Err(value) = accumulator.process(&fragment, 1) {
                    error = Some(value);
                    break;
                }
            }
            assert!(error.is_some());
        }
    }

    #[test]
    fn parallel_partial_calls_interleave_by_explicit_id_and_emit_early_deltas() {
        let mut state = State::new(ReplayPolicy::IfValid, native_context_scope(), true);
        let mut parts = Vec::new();
        state
            .apply(
                serde_json::json!({"candidates":[{"content":{"parts":[
                    {"functionCall":{"id":"a","name":"first","willContinue":true}},
                    {"functionCall":{"id":"b","name":"second","willContinue":true}}
                ]}}]}),
                &mut parts,
                1,
            )
            .unwrap();
        state
            .apply(
                serde_json::json!({"candidates":[{"content":{"parts":[
                    {"functionCall":{"id":"a","partialArgs":[{"jsonPath":"$.text","stringValue":"hel","willContinue":true}],"willContinue":true}},
                    {"functionCall":{"id":"b","partialArgs":[{"jsonPath":"$.items[0].value","numberValue":1}],"willContinue":true}}
                ]}}]}),
                &mut parts,
                2,
            )
            .unwrap();
        assert!(parts.iter().any(|part| matches!(part, StreamPart::ToolCallDelta { id, delta, .. } if id == "a" && delta.contains("hel"))));
        assert!(parts.iter().any(|part| matches!(part, StreamPart::ToolCallDelta { id, delta, .. } if id == "b" && delta.contains("items"))));
        assert!(
            !parts
                .iter()
                .any(|part| matches!(part, StreamPart::ToolCall { .. }))
        );

        state
            .apply(
                serde_json::json!({"candidates":[{"content":{"parts":[
                    {"functionCall":{"id":"a","partialArgs":[{"jsonPath":"$.text","stringValue":"lo"}]}},
                    {"functionCall":{"id":"b","partialArgs":[{"jsonPath":"$.items[1].value","numberValue":2}]}}
                ]}}]}),
                &mut parts,
                3,
            )
            .unwrap();
        let calls = parts
            .iter()
            .filter_map(|part| match part {
                StreamPart::ToolCall { tool_call } => Some((&tool_call.id, &tool_call.input)),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(calls[&"a".to_owned()], &serde_json::json!({"text":"hello"}));
        assert_eq!(
            calls[&"b".to_owned()],
            &serde_json::json!({"items":[{"value":1},{"value":2}]})
        );
    }

    #[test]
    fn all_documented_grounding_chunk_forms_become_sources() {
        let (parts, _) = normalize_single(
            serde_json::json!({
                "candidates":[{
                    "content":{"parts":[{"text":"answer"}]},
                    "groundingMetadata":{"groundingChunks":[
                        {"web":{"uri":"https://example.com/web","title":"Web"}},
                        {"maps":{"uri":"https://maps.example/place","title":"Place","placeId":"place-1"}},
                        {"retrievedContext":{"uri":"gs://bucket/doc","title":"Doc","text":"excerpt","documentName":"projects/p/locations/global/collections/c/dataStores/d/branches/b/documents/1","ragChunk":{"text":"rag"}}},
                        {"image":{"sourceUri":"https://example.com/page","imageUri":"https://example.com/image.jpg"}}
                    ]},
                    "finishReason":"STOP"
                }]
            }),
            ReplayPolicy::IfValid,
            native_context_scope(),
            None,
            false,
            1,
        )
        .unwrap();
        let sources = parts
            .iter()
            .filter_map(|part| match part {
                StreamPart::Source { source } => Some(source),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(sources.len(), 4);
        assert_eq!(sources[0].title.as_deref(), Some("Web"));
        assert_eq!(sources[1].id.as_deref(), Some("place-1"));
        assert_eq!(sources[2].excerpt.as_deref(), Some("excerpt"));
        assert_eq!(
            sources[3].id.as_deref(),
            Some("https://example.com/image.jpg")
        );
        assert!(sources.iter().all(|source| source.metadata.is_some()));
    }

    #[tokio::test]
    async fn semantic_event_after_finish_in_same_chunk_is_fatal_without_finish() {
        let bytes: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>> = Box::pin(
            futures_util::stream::iter(vec![Ok(bytes::Bytes::from_static(
                b"data: {\"candidates\":[{\"finishReason\":\"STOP\"}]}\n\ndata: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"late\"}]}}]}\n\n",
            ))]),
        );
        let mut live = LiveState {
            bytes,
            parser: crate::sse::Parser::default(),
            state: State::new(ReplayPolicy::IfValid, native_context_scope(), false),
            queue: VecDeque::new(),
            pending_events: VecDeque::new(),
            pending_error: None,
            abort: AbortSignal::default(),
            idle: Duration::from_secs(1),
            count: 0,
            eof: false,
            include_raw: true,
        };
        let error = early_peek(&mut live).await.unwrap_err();
        assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
        assert!(
            !live
                .queue
                .iter()
                .any(|part| matches!(part, Ok(StreamPart::Finish { .. })))
        );
    }
}
