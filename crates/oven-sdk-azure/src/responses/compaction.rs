//! Azure Responses V1 standalone native compaction.

use std::{collections::BTreeSet, time::Duration};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use oven_sdk::{
    AbortSignal, CompactionRequest, ErrorStage, JsonValue, LanguageModelDescriptor, ModelError,
    NativeContextScope, NativeContextWindow, Usage,
};
use sha2::{Digest, Sha256};

use crate::{
    options::{AzureOpenAiCompactionOptions, compaction_options},
    wire::responses::COMPACTION_FORMAT,
};

use super::request::{self, EncodedInput};

pub(crate) const MAX_COMPACTION_OUTPUT_ITEMS: usize = 128;
pub(crate) const MAX_COMPACTION_CONTENT_ITEMS: usize = 128;
pub(crate) const MAX_COMPACTION_REQUEST_BYTES: usize = NativeContextWindow::MAX_PAYLOAD_BYTES;
pub(crate) const MAX_COMPACTION_RESPONSE_BYTES: usize =
    NativeContextWindow::MAX_PAYLOAD_BYTES + 64 * 1024;

pub(crate) struct EncodedCompaction {
    pub(crate) body: JsonValue,
    pub(crate) request: oven_sdk::RequestMetadata,
}

pub(crate) fn validate(
    request_value: &CompactionRequest,
    descriptor: &LanguageModelDescriptor,
    replay_binding: &JsonValue,
    native_scope: &NativeContextScope,
) -> Result<(), ModelError> {
    request_value.validate_for(&descriptor.capabilities)?;
    request::validate_request(
        &request_value.request,
        &descriptor.capabilities,
        descriptor,
        Some(native_scope),
    )?;
    validate_options(&compaction_options(request_value)?)?;
    let encoded = request::encode_input(
        &request_value.request,
        descriptor,
        descriptor.capabilities.replay.policy,
        replay_binding,
        native_scope,
    )?;
    validate_nonempty_input(&encoded.input)
}

pub(crate) fn encode(
    request_value: &CompactionRequest,
    descriptor: &LanguageModelDescriptor,
    replay_binding: &JsonValue,
    native_scope: &NativeContextScope,
) -> Result<EncodedCompaction, ModelError> {
    let options = compaction_options(request_value)?;
    validate_options(&options)?;
    let EncodedInput {
        input,
        replay,
        warnings: _,
    } = request::encode_input(
        &request_value.request,
        descriptor,
        descriptor.capabilities.replay.policy,
        replay_binding,
        native_scope,
    )?;
    validate_nonempty_input(&input)?;
    let mut body = serde_json::json!({
        "model": descriptor.identity.model_id.as_str(),
        "input": input,
    });
    insert_option(&mut body, "instructions", options.instructions);
    insert_option(&mut body, "prompt_cache_key", options.prompt_cache_key);
    insert_option(
        &mut body,
        "prompt_cache_retention",
        options.prompt_cache_retention,
    );
    insert_option(&mut body, "service_tier", options.service_tier);
    let size = serde_json::to_vec(&body)
        .map_err(|_| {
            ModelError::invalid_request("could not encode Azure compaction request")
                .with_stage(ErrorStage::NativeContextEncode)
        })?
        .len();
    if size > MAX_COMPACTION_REQUEST_BYTES {
        return Err(ModelError::invalid_request(
            "Azure compaction request exceeds the 32 MiB encoded limit",
        )
        .with_stage(ErrorStage::NativeContextEncode));
    }
    Ok(EncodedCompaction {
        body,
        request: oven_sdk::RequestMetadata {
            replay,
            provider_metadata: oven_sdk::ProviderMetadata::new(),
        },
    })
}

pub(crate) fn parse_response(
    value: JsonValue,
    descriptor: &LanguageModelDescriptor,
    native_scope: &NativeContextScope,
    bytes: u64,
) -> Result<(NativeContextWindow, Usage), ModelError> {
    let root = value
        .as_object()
        .ok_or_else(|| invalid("Azure compaction response must be an object", bytes))?;
    exact_keys(root, &["created_at", "id", "object", "output", "usage"])
        .ok_or_else(|| invalid("Azure compaction response fields are not canonical", bytes))?;
    require_bounded_string(root.get("id"), 1024)
        .ok_or_else(|| invalid("Azure compaction response ID is invalid", bytes))?;
    if root.get("object").and_then(JsonValue::as_str) != Some("response.compaction") {
        return Err(invalid(
            "Azure compaction response object is invalid",
            bytes,
        ));
    }
    root.get("created_at")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid("Azure compaction response timestamp is invalid", bytes))?;
    let output = root
        .get("output")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| invalid("Azure compaction output must be an array", bytes))?
        .clone();
    validate_output(&output).ok_or_else(|| {
        invalid(
            "Azure compaction output is not a canonical user-message/compaction window",
            bytes,
        )
    })?;
    let usage = parse_usage(
        root.get("usage")
            .ok_or_else(|| invalid("Azure compaction usage is missing", bytes))?,
        bytes,
    )?;
    let fingerprint = output_fingerprint(&output)
        .ok_or_else(|| invalid("could not fingerprint Azure compaction output", bytes))?;
    let payload = serde_json::json!({
        "format": COMPACTION_FORMAT,
        "fingerprint": fingerprint,
        "output": output,
    });
    let window =
        NativeContextWindow::new(descriptor.adapter_id.clone(), native_scope.clone(), payload)
            .map_err(|_| {
                invalid(
                    "Azure compaction output exceeds the native-context limit",
                    bytes,
                )
            })?;
    Ok((window, usage))
}

pub(crate) fn decode_window(window: &NativeContextWindow) -> Result<Vec<JsonValue>, ModelError> {
    let root = window.payload().as_object().ok_or_else(|| {
        ModelError::native_context("Azure native context payload must be an object")
    })?;
    exact_keys(root, &["fingerprint", "format", "output"]).ok_or_else(|| {
        ModelError::native_context("Azure native context payload fields are not canonical")
    })?;
    if root.get("format").and_then(JsonValue::as_str) != Some(COMPACTION_FORMAT) {
        return Err(ModelError::native_context(
            "Azure native context format is unsupported",
        ));
    }
    let output = root
        .get("output")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| ModelError::native_context("Azure native context output is invalid"))?
        .clone();
    validate_output(&output).ok_or_else(|| {
        ModelError::native_context("Azure native context output is not canonical")
    })?;
    let expected = output_fingerprint(&output)
        .ok_or_else(|| ModelError::native_context("could not verify Azure native context"))?;
    if root.get("fingerprint").and_then(JsonValue::as_str) != Some(expected.as_str()) {
        return Err(ModelError::native_context(
            "Azure native context fingerprint is invalid",
        ));
    }
    Ok(output)
}

pub(crate) async fn read_body(
    response: reqwest::Response,
    abort: &AbortSignal,
    idle: Duration,
) -> Result<(Vec<u8>, u64), ModelError> {
    oven_sdk::provider_support::read_bounded_body(
        response.bytes_stream(),
        abort,
        oven_sdk::provider_support::BodyReadConfig {
            cap: MAX_COMPACTION_RESPONSE_BYTES,
            limit: oven_sdk::provider_support::BodyLimit::Reject {
                message: "Azure compaction response exceeds the bounded response limit",
            },
            stage: ErrorStage::NativeContextDecode,
            timeout_message: "Azure compaction response body idle timeout",
            abort_message: "Azure compaction response body read was aborted",
            read_message: "Azure compaction response body read failed",
            overflow_message: "Azure compaction response body byte count overflowed",
        },
        tokio::time::sleep(idle),
        move |timer| timer.reset(tokio::time::Instant::now() + idle),
    )
    .await
}

fn validate_options(options: &AzureOpenAiCompactionOptions) -> Result<(), ModelError> {
    validate_optional_text("instructions", options.instructions.as_deref(), 1_048_576)?;
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
    if options
        .prompt_cache_retention
        .as_deref()
        .is_some_and(|value| !matches!(value, "in_memory" | "24h"))
    {
        return Err(ModelError::invalid_request(
            "Azure compaction prompt_cache_retention must be in_memory or 24h",
        ));
    }
    validate_optional_text("service_tier", options.service_tier.as_deref(), 64)?;
    Ok(())
}

fn validate_nonempty_input(input: &[JsonValue]) -> Result<(), ModelError> {
    if input.is_empty() {
        return Err(ModelError::invalid_request(
            "Azure standalone compaction requires at least one encoded local input item",
        )
        .with_stage(ErrorStage::NativeContextEncode));
    }
    Ok(())
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
            "Azure compaction {name} is invalid"
        )));
    }
    Ok(())
}

fn insert_option(body: &mut JsonValue, name: &str, value: Option<String>) {
    if let Some(value) = value {
        body[name] = value.into();
    }
}

fn validate_output(output: &[JsonValue]) -> Option<()> {
    if output.is_empty() || output.len() > MAX_COMPACTION_OUTPUT_ITEMS {
        return None;
    }
    let mut ids = BTreeSet::new();
    for (index, item) in output.iter().enumerate() {
        let item_type = item.get("type")?.as_str()?;
        if index + 1 == output.len() {
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
    if content.is_empty() || content.len() > MAX_COMPACTION_CONTENT_ITEMS {
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
            exact_keys(object, &["text", "type"])?;
            object.get("text")?.as_str()?;
        }
        "input_image" => {
            exact_subset(object, &["detail", "file_id", "image_url", "type"])?;
            exactly_one_nonempty(object, &["image_url", "file_id"])?;
            if object.get("detail").is_some_and(|value| {
                !matches!(value.as_str(), Some("auto" | "low" | "high" | "original"))
            }) {
                return None;
            }
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
                    "type",
                ],
            )?;
            exactly_one_nonempty(object, &["file_data", "file_url", "file_id"])?;
            optional_bounded_string(object.get("filename"), 1024)?;
            if object
                .get("detail")
                .is_some_and(|value| !matches!(value.as_str(), Some("auto" | "low" | "high")))
            {
                return None;
            }
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
    optional_bounded_string(object.get("created_by"), 1024)?;
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

fn optional_bounded_string(value: Option<&JsonValue>, maximum: usize) -> Option<()> {
    match value {
        None => Some(()),
        Some(value) => require_bounded_string(Some(value), maximum).map(|_| ()),
    }
}

fn parse_usage(value: &JsonValue, bytes: u64) -> Result<Usage, ModelError> {
    let input = value
        .get("input_tokens")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid("Azure compaction input token usage is invalid", bytes))?;
    let output = value
        .get("output_tokens")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid("Azure compaction output token usage is invalid", bytes))?;
    let total = value
        .get("total_tokens")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid("Azure compaction total token usage is invalid", bytes))?;
    if input.checked_add(output) != Some(total) {
        return Err(invalid(
            "Azure compaction token totals are inconsistent",
            bytes,
        ));
    }
    let cached = value
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    let cache_write = value
        .pointer("/input_tokens_details/cache_write_tokens")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    let reasoning = value
        .pointer("/output_tokens_details/reasoning_tokens")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    let cached_total = cached.checked_add(cache_write).ok_or_else(|| {
        invalid(
            "Azure compaction input cache token details overflowed",
            bytes,
        )
    })?;
    let input_without_cache = input.checked_sub(cached_total).ok_or_else(|| {
        invalid(
            "Azure compaction input cache token details are inconsistent",
            bytes,
        )
    })?;
    if reasoning > output {
        return Err(invalid(
            "Azure compaction token details are inconsistent",
            bytes,
        ));
    }
    Ok(Usage {
        input_tokens: Some(input),
        input_tokens_no_cache: Some(input_without_cache),
        input_tokens_cache_read: Some(cached),
        input_tokens_cache_write: Some(cache_write),
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
