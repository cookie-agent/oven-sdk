//! Standalone Responses compaction wire validation and native-context codec.

use std::collections::{BTreeMap, BTreeSet};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use oven_sdk::{
    AdapterId, ErrorStage, JsonValue, ModelError, NativeContextScope, NativeContextWindow,
    ResponseMetadata, Usage,
};
use sha2::{Digest, Sha256};

use crate::{options::OpenAiResponsesCompactionOptions, wire::responses::NATIVE_CONTEXT_FORMAT};

pub(crate) const MAX_COMPACTION_REQUEST_BYTES: usize = NativeContextWindow::MAX_PAYLOAD_BYTES;
pub(crate) const MAX_COMPACTION_RESPONSE_BYTES: usize =
    NativeContextWindow::MAX_PAYLOAD_BYTES + 64 * 1024;

pub(crate) struct DecodedCompaction {
    pub(crate) native_context: NativeContextWindow,
    pub(crate) usage: Usage,
    pub(crate) response_metadata: ResponseMetadata,
}

pub(crate) fn validate_request_size(size: usize) -> Result<(), ModelError> {
    if size > MAX_COMPACTION_REQUEST_BYTES {
        return Err(ModelError::invalid_request(
            "OpenAI compaction request body exceeds the 32 MiB safety bound",
        )
        .with_stage(ErrorStage::NativeContextEncode));
    }
    Ok(())
}

pub(crate) fn validate_response_length(length: Option<u64>) -> Result<(), ModelError> {
    if length.is_some_and(|length| length > MAX_COMPACTION_RESPONSE_BYTES as u64) {
        return Err(ModelError::invalid_response(
            "OpenAI Responses compaction response exceeds the bounded response limit",
        )
        .with_stage(ErrorStage::NativeContextDecode));
    }
    Ok(())
}

pub(crate) fn validate_options(
    options: &OpenAiResponsesCompactionOptions,
) -> Result<(), ModelError> {
    validate_optional_instructions(options.instructions.as_deref())?;
    validate_optional_text(
        "prompt_cache_key",
        options.prompt_cache_key.as_deref(),
        1024,
    )?;
    validate_optional_text(
        "prompt_cache_retention",
        options.prompt_cache_retention.as_deref(),
        64,
    )?;
    validate_optional_text("service_tier", options.service_tier.as_deref(), 64)?;
    if let Some(cache) = &options.prompt_cache_options
        && (!matches!(cache.mode.as_str(), "implicit" | "explicit") || cache.ttl != "30m")
    {
        return Err(ModelError::invalid_request(
            "OpenAI compaction prompt-cache options require implicit/explicit mode and 30m TTL",
        ));
    }
    Ok(())
}

pub(crate) fn native_input(window: &NativeContextWindow) -> Result<Vec<JsonValue>, ModelError> {
    let root = window.payload().as_object().ok_or_else(|| {
        ModelError::native_context("OpenAI native context payload must be an object")
            .with_stage(ErrorStage::NativeContextDecode)
    })?;
    exact_keys(root, &["fingerprint", "format", "output"]).ok_or_else(|| {
        ModelError::native_context("OpenAI native context payload fields are not canonical")
            .with_stage(ErrorStage::NativeContextDecode)
    })?;
    if root.get("format").and_then(JsonValue::as_str) != Some(NATIVE_CONTEXT_FORMAT) {
        return Err(
            ModelError::native_context("OpenAI native context format is unsupported")
                .with_stage(ErrorStage::NativeContextDecode),
        );
    }
    let output = root
        .get("output")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            ModelError::native_context("OpenAI native context output is invalid")
                .with_stage(ErrorStage::NativeContextDecode)
        })?
        .clone();
    validate_output(&output).ok_or_else(|| {
        ModelError::native_context("OpenAI native context output is not canonical")
            .with_stage(ErrorStage::NativeContextDecode)
    })?;
    let expected = output_fingerprint(&output).ok_or_else(|| {
        ModelError::native_context("could not verify OpenAI native context")
            .with_stage(ErrorStage::NativeContextDecode)
    })?;
    if root.get("fingerprint").and_then(JsonValue::as_str) != Some(expected.as_str()) {
        return Err(
            ModelError::native_context("OpenAI native context fingerprint is invalid")
                .with_stage(ErrorStage::NativeContextDecode),
        );
    }
    Ok(output)
}

pub(crate) fn decode_response(
    value: JsonValue,
    adapter_id: AdapterId,
    scope: NativeContextScope,
    bytes: u64,
) -> Result<DecodedCompaction, ModelError> {
    let root = value
        .as_object()
        .ok_or_else(|| invalid("OpenAI compaction response must be an object", bytes))?;
    exact_keys(root, &["created_at", "id", "object", "output", "usage"])
        .ok_or_else(|| invalid("OpenAI compaction response fields are not canonical", bytes))?;
    require_bounded_string(root.get("id"), 1024)
        .ok_or_else(|| invalid("OpenAI compaction response ID is invalid", bytes))?;
    if root.get("object").and_then(JsonValue::as_str) != Some("response.compaction") {
        return Err(invalid(
            "OpenAI compaction response object is invalid",
            bytes,
        ));
    }
    root.get("created_at")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid("OpenAI compaction response timestamp is invalid", bytes))?;
    let output = root
        .get("output")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| invalid("OpenAI compaction output must be an array", bytes))?
        .clone();
    validate_output(&output).ok_or_else(|| {
        invalid(
            "OpenAI compaction output is not a canonical user-message/compaction window",
            bytes,
        )
    })?;
    let usage = parse_usage(
        root.get("usage")
            .ok_or_else(|| invalid("OpenAI compaction usage is missing", bytes))?,
        bytes,
    )?;
    let fingerprint = output_fingerprint(&output)
        .ok_or_else(|| invalid("could not fingerprint OpenAI compaction output", bytes))?;
    let payload = serde_json::json!({
        "format": NATIVE_CONTEXT_FORMAT,
        "fingerprint": fingerprint,
        "output": output,
    });
    let native_context = NativeContextWindow::new(adapter_id, scope, payload).map_err(|_| {
        invalid(
            "OpenAI compaction output exceeds the native-context limit",
            bytes,
        )
    })?;
    let mut response_metadata = BTreeMap::new();
    for (source, target) in [
        ("id", "openai.compaction_id"),
        ("created_at", "openai.created_at"),
        ("object", "openai.object"),
    ] {
        if let Some(value) = root.get(source).cloned() {
            response_metadata.insert(target.into(), value);
        }
    }
    Ok(DecodedCompaction {
        native_context,
        usage,
        response_metadata,
    })
}

fn validate_output(output: &[JsonValue]) -> Option<()> {
    if output.is_empty() {
        return None;
    }
    let mut ids = BTreeSet::new();
    for (index, item) in output.iter().enumerate() {
        let item_type = item.get("type")?.as_str()?;
        if index.checked_add(1) == Some(output.len()) {
            validate_compaction_item(item, &mut ids)?;
        } else if item_type == "message" {
            validate_user_message(item, &mut ids)?;
        } else {
            return None;
        }
    }
    Some(())
}

fn validate_user_message(item: &JsonValue, ids: &mut BTreeSet<String>) -> Option<()> {
    let object = item.as_object()?;
    exact_subset(object, &["content", "id", "role", "status", "type"])?;
    if object.get("type")?.as_str()? != "message" || object.get("role")?.as_str()? != "user" {
        return None;
    }
    validate_optional_id(object.get("id"), ids)?;
    if object.get("status").is_some_and(|value| {
        !matches!(
            value.as_str(),
            Some("completed" | "incomplete" | "in_progress")
        )
    }) {
        return None;
    }
    let content = object.get("content")?.as_array()?;
    if content.is_empty() {
        return None;
    }
    for part in content {
        validate_user_content(part)?;
    }
    Some(())
}

fn validate_user_content(part: &JsonValue) -> Option<()> {
    let object = part.as_object()?;
    match object.get("type")?.as_str()? {
        "input_text" => {
            exact_subset(object, &["prompt_cache_breakpoint", "text", "type"])?;
            object.get("text")?.as_str()?;
            validate_prompt_cache_breakpoint(object.get("prompt_cache_breakpoint"))?;
        }
        "input_image" => {
            exact_subset(
                object,
                &[
                    "detail",
                    "file_id",
                    "image_url",
                    "prompt_cache_breakpoint",
                    "type",
                ],
            )?;
            exactly_one_nonempty(object, &["image_url", "file_id"])?;
            if object.get("detail").is_some_and(|value| {
                !matches!(value.as_str(), Some("auto" | "low" | "high" | "original"))
            }) {
                return None;
            }
            validate_prompt_cache_breakpoint(object.get("prompt_cache_breakpoint"))?;
        }
        "input_file" => {
            exact_subset(
                object,
                &[
                    "detail",
                    "file_data",
                    "file_id",
                    "file_url",
                    "filename",
                    "prompt_cache_breakpoint",
                    "type",
                ],
            )?;
            exactly_one_nonempty(object, &["file_data", "file_url", "file_id"])?;
            if object
                .get("filename")
                .is_some_and(|value| value.as_str().is_none_or(str::is_empty))
                || object
                    .get("detail")
                    .is_some_and(|value| !matches!(value.as_str(), Some("auto" | "low" | "high")))
            {
                return None;
            }
            validate_prompt_cache_breakpoint(object.get("prompt_cache_breakpoint"))?;
        }
        _ => return None,
    }
    Some(())
}

fn validate_compaction_item(item: &JsonValue, ids: &mut BTreeSet<String>) -> Option<()> {
    let object = item.as_object()?;
    exact_subset(object, &["created_by", "encrypted_content", "id", "type"])?;
    if object.get("type")?.as_str()? != "compaction" {
        return None;
    }
    let id = require_bounded_string(object.get("id"), 1024)?;
    if !ids.insert(id.to_owned()) {
        return None;
    }
    require_bounded_string(
        object.get("encrypted_content"),
        NativeContextWindow::MAX_PAYLOAD_BYTES,
    )?;
    if object.get("created_by").is_some_and(|value| {
        !value
            .as_str()
            .is_some_and(|value| !value.is_empty() && value.len() <= 1024)
    }) {
        return None;
    }
    Some(())
}

fn validate_optional_id(value: Option<&JsonValue>, ids: &mut BTreeSet<String>) -> Option<()> {
    if let Some(value) = value {
        let id = require_bounded_string(Some(value), 1024)?;
        if !ids.insert(id.to_owned()) {
            return None;
        }
    }
    Some(())
}

fn exactly_one_nonempty(object: &serde_json::Map<String, JsonValue>, names: &[&str]) -> Option<()> {
    (names
        .iter()
        .filter(|name| {
            object
                .get(**name)
                .and_then(JsonValue::as_str)
                .is_some_and(|value| !value.is_empty())
        })
        .count()
        == 1)
        .then_some(())
}

fn require_bounded_string(value: Option<&JsonValue>, maximum: usize) -> Option<&str> {
    value
        .and_then(JsonValue::as_str)
        .filter(|value| !value.is_empty() && value.len() <= maximum)
}

fn parse_usage(value: &JsonValue, bytes: u64) -> Result<Usage, ModelError> {
    let input = value
        .get("input_tokens")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid("OpenAI compaction input token usage is invalid", bytes))?;
    let output = value
        .get("output_tokens")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid("OpenAI compaction output token usage is invalid", bytes))?;
    let total = value
        .get("total_tokens")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid("OpenAI compaction total token usage is invalid", bytes))?;
    if input.checked_add(output) != Some(total) {
        return Err(invalid(
            "OpenAI compaction token totals are inconsistent",
            bytes,
        ));
    }
    let cached = value
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    let reasoning = value
        .pointer("/output_tokens_details/reasoning_tokens")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    if cached > input || reasoning > output {
        return Err(invalid(
            "OpenAI compaction token details are inconsistent",
            bytes,
        ));
    }
    Ok(Usage {
        input_tokens: Some(input),
        input_tokens_no_cache: None,
        input_tokens_cache_read: Some(cached),
        input_tokens_cache_write: None,
        output_tokens: Some(output),
        output_tokens_text: Some(output - reasoning),
        output_tokens_reasoning: Some(reasoning),
        raw: Some(value.clone()),
    })
}

fn output_fingerprint(output: &[JsonValue]) -> Option<String> {
    let encoded = serde_json::to_vec(output).ok()?;
    Some(URL_SAFE_NO_PAD.encode(Sha256::digest(encoded)))
}

fn validate_optional_text(
    name: &str,
    value: Option<&str>,
    maximum: usize,
) -> Result<(), ModelError> {
    if value.is_some_and(|value| {
        value.trim().is_empty() || value.len() > maximum || value.chars().any(char::is_control)
    }) {
        return Err(ModelError::invalid_request(format!(
            "OpenAI compaction {name} is invalid"
        )));
    }
    Ok(())
}

fn validate_optional_instructions(value: Option<&str>) -> Result<(), ModelError> {
    const MAXIMUM: usize = 1_048_576;
    if value.is_some_and(|value| {
        value.trim().is_empty()
            || value.len() > MAXIMUM
            || value
                .chars()
                .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
    }) {
        return Err(ModelError::invalid_request(
            "OpenAI compaction instructions are invalid",
        ));
    }
    Ok(())
}

fn validate_prompt_cache_breakpoint(value: Option<&JsonValue>) -> Option<()> {
    let Some(value) = value else {
        return Some(());
    };
    let object = value.as_object()?;
    exact_keys(object, &["mode"])?;
    (object.get("mode")?.as_str()? == "explicit").then_some(())
}

fn exact_keys(object: &serde_json::Map<String, JsonValue>, expected: &[&str]) -> Option<()> {
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    (actual == expected).then_some(())
}

fn exact_subset(object: &serde_json::Map<String, JsonValue>, allowed: &[&str]) -> Option<()> {
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    object
        .keys()
        .all(|key| allowed.contains(key.as_str()))
        .then_some(())
}

fn invalid(message: &str, bytes: u64) -> ModelError {
    ModelError::invalid_response(message)
        .with_stage(ErrorStage::NativeContextDecode)
        .with_bytes_received(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_and_response_body_bounds_are_inclusive() {
        assert!(validate_request_size(MAX_COMPACTION_REQUEST_BYTES).is_ok());
        assert!(validate_request_size(MAX_COMPACTION_REQUEST_BYTES + 1).is_err());
        assert!(validate_response_length(Some(MAX_COMPACTION_RESPONSE_BYTES as u64)).is_ok());
        assert!(validate_response_length(Some(MAX_COMPACTION_RESPONSE_BYTES as u64 + 1)).is_err());
    }

    #[test]
    fn prompt_cache_breakpoint_requires_the_exact_explicit_shape() {
        for invalid in [
            serde_json::json!({
                "type":"input_text",
                "text":"x",
                "prompt_cache_breakpoint":"explicit"
            }),
            serde_json::json!({
                "type":"input_text",
                "text":"x",
                "prompt_cache_breakpoint":{"mode":"implicit"}
            }),
            serde_json::json!({
                "type":"input_text",
                "text":"x",
                "prompt_cache_breakpoint":{"mode":"explicit","extra":true}
            }),
        ] {
            assert!(validate_user_content(&invalid).is_none());
        }
        assert!(
            validate_user_content(&serde_json::json!({
                "type":"input_text",
                "text":"x",
                "prompt_cache_breakpoint":{"mode":"explicit"}
            }))
            .is_some()
        );
    }
}
