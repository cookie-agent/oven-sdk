//! Gemini response normalization and live SSE handling.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    time::Duration,
};

#[cfg(test)]
use oven_sdk::AbortSignal;
use oven_sdk::{
    BoxStream, CustomPart, ErrorStage, FilePart, FileSource, Finish, FinishReason, JsonValue,
    ModelError, ModelErrorKind, NativeContextScope, NativeReplayArtifact, ReplayPolicy, SourcePart,
    StreamItem, StreamPart, ToolCallPart, Usage,
};
use reqwest::header::HeaderMap;
use url::Url;

use crate::{GOOGLE_GENERATE_CONTENT_ADAPTER_ID, error::classify_error};

pub(crate) struct State {
    policy: ReplayPolicy,
    native_context_scope: NativeContextScope,
    native_parts: Vec<JsonValue>,
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
    pub(crate) fn new(policy: ReplayPolicy, native_context_scope: NativeContextScope) -> Self {
        Self {
            policy,
            native_context_scope,
            native_parts: Vec::new(),
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
            self.response_metadata
                .insert("google.request_id".into(), JsonValue::String(request_id));
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
                "Gemini response arrived after terminal finish",
                bytes,
            ));
        }
        if value.get("error").is_some() {
            let error = classify_error(
                value
                    .pointer("/error/code")
                    .and_then(JsonValue::as_u64)
                    .and_then(|value| u16::try_from(value).ok())
                    .unwrap_or(500),
                value.to_string().as_bytes(),
                self.response_metadata
                    .get("google.request_id")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned),
                ErrorStage::StreamEvent,
                bytes,
                &HeaderMap::new(),
            );
            parts.push(StreamPart::Error { error });
            parts.push(StreamPart::Finish {
                finish: Finish::new(self.usage.clone(), FinishReason::Error),
            });
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
                "google.ignored_candidate_count".into(),
                JsonValue::from(candidates.map_or(0, |v| v.len() - 1)),
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
                "google.prompt_feedback".into(),
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
            ("responseId", "google.response_id"),
            ("modelVersion", "google.model_version"),
        ] {
            if let Some(value) = value.get(wire).cloned() {
                self.response_metadata.insert(key.into(), value);
            }
        }
        if let Some(value) = value.get("promptFeedback").cloned() {
            self.provider_metadata
                .insert("google.prompt_feedback".into(), value);
        }
    }

    fn capture_candidate_metadata(&mut self, candidate: &JsonValue) {
        for (wire, key) in [
            ("groundingMetadata", "google.grounding_metadata"),
            ("urlContextMetadata", "google.url_context_metadata"),
            ("safetyRatings", "google.safety_ratings"),
            ("finishMessage", "google.finish_message"),
        ] {
            if let Some(value) = candidate.get(wire).cloned() {
                self.provider_metadata.insert(key.into(), value);
            }
        }
    }

    fn apply_part(
        &mut self,
        mut native: JsonValue,
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
                    .ok_or_else(|| invalid_event("Gemini block counter overflowed", bytes))?;
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
        if let Some(call) = native.get_mut("functionCall") {
            let name = call
                .get("name")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_owned();
            let provider_id = call
                .get("id")
                .and_then(JsonValue::as_str)
                .filter(|value| !value.is_empty());
            let id = reserve_tool_id(&mut self.call_ids, provider_id, self.call_counter);
            self.call_counter = self.call_counter.saturating_add(1);
            let input = call
                .get("args")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            if !input.is_object() {
                return Err(ModelError::new(
                    ModelErrorKind::InvalidToolInput,
                    "Gemini function arguments must be a JSON object",
                )
                .with_stage(ErrorStage::StreamFinalize)
                .with_bytes_received(bytes));
            }
            call["id"] = JsonValue::String(id.clone());
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
            tool_call.raw_input = Some(raw);
            parts.push(StreamPart::ToolCall { tool_call });
            self.native_parts.push(native);
            return Ok(());
        }
        if native.get("executableCode").is_some() || native.get("toolCall").is_some() {
            let normalized = normalize_server_tool_call(&native, bytes)?;
            parts.push(StreamPart::Custom {
                part: CustomPart::new("google.server_tool_call", normalized),
            });
            self.native_parts.push(native);
            return Ok(());
        }
        if native.get("codeExecutionResult").is_some() || native.get("toolResponse").is_some() {
            let normalized = normalize_server_tool_result(&native, bytes)?;
            parts.push(StreamPart::Custom {
                part: CustomPart::new("google.server_tool_result", normalized),
            });
            self.native_parts.push(native);
            return Ok(());
        }
        if let Some(data) = native.get("inlineData") {
            let media_type = data
                .get("mimeType")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| invalid_event("Gemini inline output is missing MIME type", bytes))?;
            let bytes_value = data
                .get("data")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| invalid_event("Gemini inline output is missing data", bytes))?;
            let decoded =
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, bytes_value)
                    .map_err(|_| invalid_event("Gemini inline output is invalid base64", bytes))?;
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
        self.native_parts.push(native);
        Ok(())
    }

    fn emit_sources(&mut self, candidate: &JsonValue, parts: &mut Vec<StreamPart>) {
        let Some(chunks) = candidate
            .pointer("/groundingMetadata/groundingChunks")
            .and_then(JsonValue::as_array)
        else {
            return;
        };
        for chunk in chunks {
            let Some((identity, source)) = normalize_source(chunk) else {
                continue;
            };
            if self.emitted_sources.insert(identity) {
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
        let has_calls = !self.call_ids.is_empty();
        let mut finish = Finish::new(
            std::mem::take(&mut self.usage),
            map_finish(reason, has_calls),
        );
        finish.response_metadata = self.response_metadata.clone();
        finish.provider_metadata = std::mem::take(&mut self.provider_metadata);
        if self.policy != ReplayPolicy::Never {
            let payload = serde_json::json!({
                "role":"model",
                "parts":std::mem::take(&mut self.native_parts),
            });
            finish.native_replay = Some(
                NativeReplayArtifact::new(
                    oven_sdk::AdapterId::new(GOOGLE_GENERATE_CONTENT_ADAPTER_ID),
                    self.native_context_scope.clone(),
                    payload,
                )
                .map_err(|_| {
                    ModelError::replay("Google native replay artifact exceeds its size limit")
                        .with_stage(ErrorStage::ReplayEncode)
                        .with_bytes_received(bytes)
                })?,
            );
        }
        parts.push(StreamPart::Finish { finish });
        self.done = true;
        Ok(())
    }
}

fn normalize_server_tool_call(native: &JsonValue, bytes: u64) -> Result<JsonValue, ModelError> {
    if let Some(call) = native.get("toolCall") {
        let tool_type = call
            .get("toolType")
            .and_then(JsonValue::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                invalid_event("Gemini server tool call is missing a tool type", bytes)
            })?;
        let mut normalized = serde_json::json!({
            "toolCall": {
                "toolType": tool_type,
                "args": call.get("args").cloned().unwrap_or_else(|| serde_json::json!({})),
            }
        });
        if let Some(id) = optional_server_tool_id(call, "call", bytes)? {
            normalized["toolCall"]["id"] = JsonValue::String(id.to_owned());
        }
        if let Some(tool_name) = call.get("toolName").and_then(JsonValue::as_str) {
            normalized["toolCall"]["toolName"] = JsonValue::String(tool_name.to_owned());
        }
        return Ok(normalized);
    }
    let code = native
        .get("executableCode")
        .ok_or_else(|| invalid_event("Gemini server tool call is invalid", bytes))?;
    let language = code
        .get("language")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| invalid_event("Gemini executable code is missing a language", bytes))?;
    let source = code
        .get("code")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| invalid_event("Gemini executable code is missing code", bytes))?;
    let mut normalized = serde_json::json!({
        "executableCode":{"language":language,"code":source}
    });
    if let Some(id) = code.get("id").and_then(JsonValue::as_str) {
        normalized["executableCode"]["id"] = JsonValue::String(id.to_owned());
    }
    Ok(normalized)
}

fn normalize_server_tool_result(native: &JsonValue, bytes: u64) -> Result<JsonValue, ModelError> {
    if let Some(response) = native.get("toolResponse") {
        let tool_type = response
            .get("toolType")
            .and_then(JsonValue::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                invalid_event("Gemini server tool response is missing a tool type", bytes)
            })?;
        let mut normalized = serde_json::json!({
            "toolResponse": {
                "toolType": tool_type,
                "response": response.get("response").cloned().unwrap_or_else(|| serde_json::json!({})),
            }
        });
        if let Some(id) = optional_server_tool_id(response, "response", bytes)? {
            normalized["toolResponse"]["id"] = JsonValue::String(id.to_owned());
        }
        return Ok(normalized);
    }
    let result = native
        .get("codeExecutionResult")
        .ok_or_else(|| invalid_event("Gemini server tool result is invalid", bytes))?;
    let outcome = result
        .get("outcome")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| invalid_event("Gemini code result is missing an outcome", bytes))?;
    let output = result
        .get("output")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    let mut normalized =
        serde_json::json!({"codeExecutionResult":{"outcome":outcome,"output":output}});
    if let Some(id) = result.get("id").and_then(JsonValue::as_str) {
        normalized["codeExecutionResult"]["id"] = JsonValue::String(id.to_owned());
    }
    Ok(normalized)
}

fn optional_server_tool_id<'a>(
    value: &'a JsonValue,
    kind: &str,
    bytes: u64,
) -> Result<Option<&'a str>, ModelError> {
    match value.get("id") {
        None => Ok(None),
        Some(JsonValue::String(id)) if id.is_empty() => Ok(None),
        Some(JsonValue::String(id)) => Ok(Some(id)),
        Some(_) => Err(invalid_event(
            &format!("Gemini server tool {kind} ID must be a string"),
            bytes,
        )),
    }
}

fn normalize_source(chunk: &JsonValue) -> Option<(String, SourcePart)> {
    if let Some(web) = chunk.get("web") {
        return normalize_url_source("web", web, "uri", None);
    }
    if let Some(maps) = chunk.get("maps") {
        let mut source = source_base(maps);
        source.id = maps
            .get("placeId")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
        source.excerpt = maps
            .get("text")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
        if let Some(uri) = maps.get("uri").and_then(JsonValue::as_str) {
            match Url::parse(uri) {
                Ok(url) => source.url = Some(url),
                Err(_) if source.id.is_none() => source.id = Some(uri.to_owned()),
                Err(_) => {}
            }
        }
        let identity = source_identity("maps", &source)?;
        return Some((identity, source));
    }
    if let Some(image) = chunk.get("image") {
        let mut result = normalize_url_source("image", image, "sourceUri", None)?;
        let mut google = BTreeMap::new();
        for (wire, key) in [("imageUri", "image_uri"), ("domain", "domain")] {
            if let Some(value) = image.get(wire).and_then(JsonValue::as_str) {
                google.insert(key.to_owned(), JsonValue::String(value.to_owned()));
            }
        }
        if !google.is_empty() {
            let mut metadata = BTreeMap::new();
            metadata.insert(
                "google".into(),
                serde_json::to_value(google).expect("source metadata is serializable"),
            );
            result.1.metadata = Some(metadata);
        }
        return Some(result);
    }
    let context = chunk.get("retrievedContext")?;
    let mut source = source_base(context);
    source.id = context
        .get("mediaId")
        .and_then(JsonValue::as_str)
        .map(str::to_owned);
    source.excerpt = context
        .get("text")
        .and_then(JsonValue::as_str)
        .map(str::to_owned);
    if let Some(uri) = context.get("uri").and_then(JsonValue::as_str) {
        if let Ok(url) = Url::parse(uri)
            && matches!(url.scheme(), "http" | "https")
        {
            source.url = Some(url);
        } else {
            if source.id.is_none() {
                source.id = Some(uri.to_owned());
            }
            source.media_type = Some(media_type_for_document_uri(uri).into());
        }
    } else if let Some(store) = context.get("fileSearchStore").and_then(JsonValue::as_str) {
        if source.id.is_none() {
            source.id = Some(store.to_owned());
        }
        source.media_type = Some("application/octet-stream".into());
    } else if source.id.is_some() {
        source.media_type = Some("application/octet-stream".into());
    }
    let mut google = BTreeMap::new();
    if let Some(store) = context.get("fileSearchStore").and_then(JsonValue::as_str) {
        google.insert(
            "file_search_store".to_owned(),
            JsonValue::String(store.to_owned()),
        );
    }
    if let Some(page) = context.get("pageNumber").and_then(JsonValue::as_i64) {
        google.insert("page_number".to_owned(), JsonValue::from(page));
    }
    if let Some(metadata) = safe_custom_metadata(context.get("customMetadata")) {
        google.insert("custom_metadata".to_owned(), JsonValue::Array(metadata));
    }
    if !google.is_empty() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "google".into(),
            serde_json::to_value(google).expect("source metadata is serializable"),
        );
        source.metadata = Some(metadata);
    }
    let identity = source_identity("retrieved_context", &source)?;
    Some((identity, source))
}

fn normalize_url_source(
    kind: &str,
    value: &JsonValue,
    uri_field: &str,
    media_type: Option<&str>,
) -> Option<(String, SourcePart)> {
    let uri = value.get(uri_field).and_then(JsonValue::as_str)?;
    let mut source = source_base(value);
    source.media_type = media_type.map(str::to_owned);
    match Url::parse(uri) {
        Ok(url) => source.url = Some(url),
        Err(_) => source.id = Some(uri.to_owned()),
    }
    Some((source_identity(kind, &source)?, source))
}

fn source_base(value: &JsonValue) -> SourcePart {
    let mut source = SourcePart::new();
    source.title = value
        .get("title")
        .and_then(JsonValue::as_str)
        .map(str::to_owned);
    source
}

fn source_identity(kind: &str, source: &SourcePart) -> Option<String> {
    source
        .id
        .as_ref()
        .map(|id| format!("{kind}:id:{id}"))
        .or_else(|| source.url.as_ref().map(|url| format!("{kind}:url:{url}")))
        .or_else(|| {
            source
                .title
                .as_ref()
                .map(|title| format!("{kind}:title:{title}"))
        })
}

fn safe_custom_metadata(value: Option<&JsonValue>) -> Option<Vec<JsonValue>> {
    let values = value?.as_array()?;
    let output = values
        .iter()
        .filter_map(|value| {
            let object = value.as_object()?;
            let key = object.get("key")?.as_str()?;
            let mut safe = serde_json::Map::new();
            safe.insert("key".into(), JsonValue::String(key.to_owned()));
            if let Some(value) = object.get("stringValue").and_then(JsonValue::as_str) {
                safe.insert("stringValue".into(), JsonValue::String(value.to_owned()));
            }
            if let Some(value) = object.get("numericValue").and_then(JsonValue::as_f64) {
                safe.insert("numericValue".into(), JsonValue::from(value));
            }
            if let Some(values) = object
                .get("stringListValue")
                .and_then(|value| value.get("values"))
                .and_then(JsonValue::as_array)
            {
                let values = values
                    .iter()
                    .filter_map(JsonValue::as_str)
                    .map(|value| JsonValue::String(value.to_owned()))
                    .collect::<Vec<_>>();
                safe.insert(
                    "stringListValue".into(),
                    serde_json::json!({"values":values}),
                );
            }
            Some(JsonValue::Object(safe))
        })
        .collect::<Vec<_>>();
    (!output.is_empty()).then_some(output)
}

fn media_type_for_document_uri(uri: &str) -> &'static str {
    let lower = uri.to_ascii_lowercase();
    if lower.ends_with(".pdf") {
        "application/pdf"
    } else if lower.ends_with(".txt") {
        "text/plain"
    } else if lower.ends_with(".docx") {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    } else if lower.ends_with(".doc") {
        "application/msword"
    } else if lower.ends_with(".md") || lower.ends_with(".markdown") {
        "text/markdown"
    } else {
        "application/octet-stream"
    }
}

pub(crate) struct LiveState {
    pub(crate) bytes: BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    pub(crate) parser: crate::sse::Parser,
    pub(crate) state: State,
    pub(crate) queue: VecDeque<StreamItem>,
    pub(crate) pending_events: VecDeque<crate::sse::Event>,
    pub(crate) pending_error: Option<ModelError>,
    pub(crate) deadline: oven_sdk::provider_support::StreamReadDeadline<tokio::time::Sleep>,
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
                "Gemini stream ended before a semantic response",
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
        let next = match live
            .deadline
            .next(live.bytes.as_mut(), |timer| {
                timer.reset(tokio::time::Instant::now() + live.idle);
            })
            .await
        {
            oven_sdk::provider_support::StreamRead::Aborted => {
                return Err(ModelError::abort("Gemini stream was aborted")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count));
            }
            oven_sdk::provider_support::StreamRead::TimedOut => {
                return Err(ModelError::timeout("Gemini stream idle timeout")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count));
            }
            oven_sdk::provider_support::StreamRead::Item(value) => value,
        };
        match next {
            Some(Ok(chunk)) => {
                live.count = live
                    .count
                    .checked_add(u64::try_from(chunk.len()).map_err(|_| {
                        ModelError::transport("Gemini stream byte count overflowed")
                    })?)
                    .ok_or_else(|| ModelError::transport("Gemini stream byte count overflowed"))?;
                live.parser
                    .feed_events_into(&chunk, &mut live.pending_events)?;
            }
            Some(Err(_)) => {
                return Err(ModelError::transport("Gemini stream read failed")
                    .with_stage(ErrorStage::StreamRead)
                    .with_bytes_received(live.count));
            }
            None => {
                live.eof = true;
                live.pending_events.extend(live.parser.finish_events()?);
            }
        }
    }
    let mut semantic = false;
    while let Some(event) = live.pending_events.pop_front() {
        if event.data.is_empty() || event.name == "ping" {
            continue;
        }
        let value: JsonValue = serde_json::from_str(&event.data).map_err(|_| {
            ModelError::invalid_response("Gemini SSE event is invalid JSON")
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
                "Gemini event arrived after terminal finish",
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
            "Gemini stream ended before a terminal finish reason",
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
    let mut state = State::new(policy, native_context_scope);
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
            "Gemini generateContent response had no terminal finish reason",
        )
        .with_stage(ErrorStage::StreamFinalize)
        .with_bytes_received(bytes));
    }
    Ok((parts, state.response_metadata))
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
                .ok_or_else(|| invalid_event("Gemini input usage is inconsistent", bytes))?,
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
                .ok_or_else(|| invalid_event("Gemini output usage overflowed", bytes))?,
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
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY"
        | "OTHER"
            if reason != "OTHER" =>
        {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn native_context_scope(model: &str) -> NativeContextScope {
        NativeContextScope::new(
            oven_sdk::ProviderId::new("google"),
            oven_sdk::ModelId::new(model),
            oven_sdk::ResourceId::new(format!("ai-studio:models/{model}")).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn text_reasoning_tool_usage_and_replay_are_normalized() {
        let value = serde_json::json!({
            "responseId":"r1",
            "modelVersion":"gemini-2.5-flash",
            "candidates":[{"content":{"parts":[
                {"text":"think","thought":true,"thoughtSignature":"sig"},
                {"text":"answer"},
                {"functionCall":{"id":"c1","name":"lookup","args":{"q":"x"}},"thoughtSignature":"sig2"}
            ]},"finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":3,"candidatesTokenCount":4,"thoughtsTokenCount":2}
        });
        let (parts, metadata) = normalize_single(
            value,
            ReplayPolicy::IfValid,
            native_context_scope("gemini-2.5-flash"),
            None,
            false,
            100,
        )
        .unwrap();
        assert_eq!(metadata["google.response_id"], "r1");
        let finish = parts
            .iter()
            .find_map(|part| match part {
                StreamPart::Finish { finish } => Some(finish),
                _ => None,
            })
            .unwrap();
        assert_eq!(finish.finish_reason, FinishReason::ToolCalls);
        assert_eq!(finish.usage.input_tokens_no_cache, Some(7));
        assert_eq!(finish.usage.output_tokens, Some(6));
        assert_eq!(
            finish
                .native_replay
                .as_ref()
                .unwrap()
                .payload()
                .pointer("/parts/0/thoughtSignature")
                .and_then(JsonValue::as_str),
            Some("sig")
        );
    }

    #[test]
    fn missing_finish_is_not_success() {
        let error = normalize_single(
            serde_json::json!({"candidates":[{"content":{"parts":[{"text":"x"}]}}]}),
            ReplayPolicy::IfValid,
            native_context_scope("gemini-2.5-flash"),
            None,
            false,
            1,
        )
        .unwrap_err();
        assert_eq!(error.kind, ModelErrorKind::UnexpectedEof);
    }

    #[test]
    fn server_tool_custom_parts_are_safe_and_all_source_forms_are_normalized() {
        let value = serde_json::json!({
            "candidates":[{
                "content":{"parts":[
                    {
                        "toolCall":{"toolType":"GOOGLE_SEARCH_WEB","toolName":"google_search","args":{"q":"rust"},"id":"s1","opaque":"drop"},
                        "thoughtSignature":"opaque-signature",
                        "partMetadata":{"opaque":true}
                    },
                    {
                        "executableCode":{"language":"PYTHON","code":"print(1)","id":"code-1","opaque":"drop"},
                        "thoughtSignature":"opaque-code-signature"
                    },
                    {
                        "codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1","id":"code-1","opaque":"drop"},
                        "thoughtSignature":"opaque-code-result-signature"
                    },
                    {
                        "toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"items":[]},"id":"s1","opaque":"drop"},
                        "thoughtSignature":"opaque-result-signature"
                    }
                ]},
                "groundingMetadata":{"groundingChunks":[
                    {"web":{"uri":"https://example.com/web","title":"Web"}},
                    {"maps":{"uri":"https://maps.example/place","title":"Place","text":"Map excerpt","placeId":"place-1"}},
                    {"image":{"sourceUri":"https://example.com/source","imageUri":"https://example.com/image.png","title":"Image","domain":"example.com"}},
                    {"retrievedContext":{"uri":"https://example.com/document","title":"URL document","text":"Excerpt"}},
                    {"retrievedContext":{"uri":"gs://bucket/report.pdf","title":"PDF","text":"PDF excerpt"}},
                    {"retrievedContext":{"fileSearchStore":"fileSearchStores/store-1","title":"File Search","text":"Stored excerpt"}},
                    {"retrievedContext":{
                        "mediaId":"fileSearchStores/store-1/media/blob-1",
                        "title":"Media result",
                        "pageNumber":7,
                        "customMetadata":[
                            {"key":"category","stringValue":"manual","opaque":"drop"},
                            {"key":"score","numericValue":0.75},
                            {"key":"tags","stringListValue":{"values":["rust","sdk"],"opaque":"drop"}}
                        ]
                    }}
                ]},
                "finishReason":"STOP"
            }]
        });
        let (parts, _) = normalize_single(
            value,
            ReplayPolicy::IfValid,
            native_context_scope("gemini-3.5-flash"),
            None,
            false,
            100,
        )
        .unwrap();
        let custom = parts
            .iter()
            .filter_map(|part| match part {
                StreamPart::Custom { part } => Some(part),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(custom.len(), 4);
        assert_eq!(
            custom[0].data,
            serde_json::json!({"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","toolName":"google_search","args":{"q":"rust"},"id":"s1"}})
        );
        assert_eq!(custom[1].data["executableCode"]["id"], "code-1");
        assert_eq!(custom[2].data["codeExecutionResult"]["id"], "code-1");
        assert!(!custom[0].data.to_string().contains("thoughtSignature"));
        assert!(!custom[0].data.to_string().contains("opaque"));
        assert!(custom.iter().all(|part| {
            !part.data.to_string().contains("thoughtSignature")
                && !part.data.to_string().contains("opaque")
        }));

        let finish = parts
            .iter()
            .find_map(|part| match part {
                StreamPart::Finish { finish } => Some(finish),
                _ => None,
            })
            .unwrap();
        let replay = finish.native_replay.as_ref().unwrap().payload();
        assert_eq!(
            replay
                .pointer("/parts/0/thoughtSignature")
                .and_then(JsonValue::as_str),
            Some("opaque-signature")
        );
        assert_eq!(
            replay
                .pointer("/parts/1/thoughtSignature")
                .and_then(JsonValue::as_str),
            Some("opaque-code-signature")
        );

        let sources = parts
            .iter()
            .filter_map(|part| match part {
                StreamPart::Source { source } => Some(source),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(sources.len(), 7);
        assert_eq!(sources[1].id.as_deref(), Some("place-1"));
        assert_eq!(sources[1].excerpt.as_deref(), Some("Map excerpt"));
        assert_eq!(
            sources[2].url.as_ref().unwrap().as_str(),
            "https://example.com/source"
        );
        assert_eq!(
            sources[2].metadata.as_ref().unwrap().get("google").unwrap()["image_uri"],
            "https://example.com/image.png"
        );
        assert_eq!(sources[4].id.as_deref(), Some("gs://bucket/report.pdf"));
        assert_eq!(sources[4].media_type.as_deref(), Some("application/pdf"));
        assert_eq!(sources[5].id.as_deref(), Some("fileSearchStores/store-1"));
        assert_eq!(
            sources[6].id.as_deref(),
            Some("fileSearchStores/store-1/media/blob-1")
        );
        let metadata = &sources[6].metadata.as_ref().unwrap()["google"];
        assert_eq!(metadata["page_number"], 7);
        assert_eq!(metadata["custom_metadata"][0]["key"], "category");
        assert_eq!(
            metadata["custom_metadata"][2]["stringListValue"]["values"][1],
            "sdk"
        );
        assert!(!metadata.to_string().contains("opaque"));
    }

    #[test]
    fn optional_server_tool_ids_are_preserved_only_when_nonempty() {
        let value = serde_json::json!({
            "candidates":[{
                "content":{"parts":[
                    {"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"q":"missing"}}},
                    {"toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"ok":true}}},
                    {"toolCall":{"toolType":"URL_CONTEXT","args":{"url":"https://example.com"},"id":""}},
                    {"toolResponse":{"toolType":"URL_CONTEXT","response":{"ok":true},"id":""}},
                    {"toolCall":{"toolType":"FILE_SEARCH","args":{"query":"guide"},"id":"call-exact"}},
                    {"toolResponse":{"toolType":"FILE_SEARCH","response":{"matches":1},"id":"response-exact"}}
                ]},
                "groundingMetadata":{"groundingChunks":[{
                    "retrievedContext":{
                        "mediaId":"fileSearchStores/store-1/media/blob-idless",
                        "pageNumber":2
                    }
                }]},
                "finishReason":"STOP"
            }]
        });
        let (parts, _) = normalize_single(
            value,
            ReplayPolicy::IfValid,
            native_context_scope("gemini-3.5-flash"),
            None,
            false,
            100,
        )
        .unwrap();
        assert!(!parts.iter().any(|part| {
            matches!(
                part,
                StreamPart::ToolCallStart { .. }
                    | StreamPart::ToolCallDelta { .. }
                    | StreamPart::ToolCallEnd { .. }
                    | StreamPart::ToolCall { .. }
            )
        }));
        let custom = parts
            .iter()
            .filter_map(|part| match part {
                StreamPart::Custom { part } => Some(part),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(custom.len(), 6);
        for (index, field) in [
            (0, "toolCall"),
            (1, "toolResponse"),
            (2, "toolCall"),
            (3, "toolResponse"),
        ] {
            assert!(custom[index].data[field].get("id").is_none());
        }
        assert_eq!(custom[4].data["toolCall"]["id"], "call-exact");
        assert_eq!(custom[5].data["toolResponse"]["id"], "response-exact");

        let source = parts
            .iter()
            .find_map(|part| match part {
                StreamPart::Source { source } => Some(source),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            source.id.as_deref(),
            Some("fileSearchStores/store-1/media/blob-idless")
        );
        let metadata = source.metadata.as_ref().unwrap()["google"].to_string();
        assert!(!metadata.contains("toolCall"));
        assert!(!metadata.contains("toolResponse"));
        assert!(!metadata.contains("call-exact"));

        let finish = parts
            .iter()
            .find_map(|part| match part {
                StreamPart::Finish { finish } => Some(finish),
                _ => None,
            })
            .unwrap();
        let replay = finish.native_replay.as_ref().unwrap().payload();
        assert!(replay.pointer("/parts/0/toolCall/id").is_none());
        assert!(replay.pointer("/parts/1/toolResponse/id").is_none());
        assert_eq!(replay["parts"][2]["toolCall"]["id"], "");
        assert_eq!(replay["parts"][3]["toolResponse"]["id"], "");
        assert_eq!(replay["parts"][4]["toolCall"]["id"], "call-exact");
        assert_eq!(replay["parts"][5]["toolResponse"]["id"], "response-exact");
    }

    #[test]
    fn client_calls_accept_duplicate_ids_missing_names_and_unknown_parts() {
        let value = serde_json::json!({
            "candidates":[{
                "content":{"parts":[
                    {"functionCall":{"id":"same","args":{}}},
                    {"functionCall":{"id":"same","name":"lookup","args":{}}},
                    {"futurePart":{"opaque":true}}
                ]},
                "finishReason":"STOP"
            }]
        });
        let (parts, _) = normalize_single(
            value,
            ReplayPolicy::IfValid,
            native_context_scope("gemini-3.5-flash"),
            None,
            false,
            100,
        )
        .unwrap();
        let calls = parts
            .iter()
            .filter_map(|part| match part {
                StreamPart::ToolCall { tool_call } => Some(tool_call),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "same");
        assert_eq!(calls[1].id, "same-1");
        assert_eq!(calls[0].name, "");
        let finish = parts
            .iter()
            .find_map(|part| match part {
                StreamPart::Finish { finish } => Some(finish),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            finish.native_replay.as_ref().unwrap().payload()["parts"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
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
            state: State::new(
                ReplayPolicy::IfValid,
                native_context_scope("gemini-2.5-flash"),
            ),
            queue: VecDeque::new(),
            pending_events: VecDeque::new(),
            pending_error: None,
            deadline: oven_sdk::provider_support::StreamReadDeadline::new(
                tokio::time::sleep(Duration::from_secs(1)),
                &AbortSignal::default(),
            ),
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
