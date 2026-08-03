//! Current private Responses output replay codec.

use std::collections::BTreeSet;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use oven_sdk::{AssistantPart, JsonValue, NativeReplayArtifact};
use sha2::{Digest, Sha256};

use crate::wire::responses::REPLAY_FORMAT;

pub(crate) const FINGERPRINT_KIND: &str = "azure.openai.responses.replay_fingerprint";

pub(crate) fn capture(items: &[JsonValue]) -> Option<(Vec<JsonValue>, String)> {
    let sanitized = items
        .iter()
        .map(sanitize_item)
        .collect::<Option<Vec<_>>>()?;
    validate_item_relations(&sanitized)?;
    let fingerprint = fingerprint(&sanitized)?;
    Some((sanitized, fingerprint))
}

pub(crate) fn decode(
    artifact: &NativeReplayArtifact,
    normalized: &[AssistantPart],
    replay_binding: &JsonValue,
) -> Option<Vec<JsonValue>> {
    let root = artifact.payload().as_object()?;
    exact_keys(root, &["binding", "fingerprint", "format", "items"])?;
    if root.get("format")?.as_str()? != REPLAY_FORMAT || root.get("binding")? != replay_binding {
        return None;
    }
    let items = root.get("items")?.as_array()?.clone();
    let exact = items
        .iter()
        .map(sanitize_item)
        .collect::<Option<Vec<_>>>()?;
    if exact != items {
        return None;
    }
    validate_item_relations(&items)?;
    let artifact_fingerprint = root.get("fingerprint")?.as_str()?;
    if fingerprint(&items)?.as_str() != artifact_fingerprint
        || normalized_fingerprint(normalized)? != artifact_fingerprint
        || semantic_items(&items) != semantic_normalized(normalized)
    {
        return None;
    }
    Some(items)
}

fn sanitize_item(item: &JsonValue) -> Option<JsonValue> {
    match item.get("type").and_then(JsonValue::as_str)? {
        "message" => sanitize_message(item),
        "reasoning" => sanitize_reasoning(item),
        "function_call" => sanitize_function_call(item),
        _ => None,
    }
}

fn sanitize_message(item: &JsonValue) -> Option<JsonValue> {
    let object = item.as_object()?;
    if object.get("role")?.as_str()? != "assistant" {
        return None;
    }
    let content = object
        .get("content")?
        .as_array()?
        .iter()
        .map(|part| match part.get("type").and_then(JsonValue::as_str)? {
            "output_text" => Some(serde_json::json!({
                "type":"output_text",
                "text":part.get("text")?.as_str()?
            })),
            "refusal" => Some(serde_json::json!({
                "type":"refusal",
                "refusal":part.get("refusal")?.as_str()?
            })),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let mut sanitized = serde_json::json!({
        "type":"message",
        "role":"assistant",
        "content":content
    });
    if let Some(id) = object.get("id").and_then(JsonValue::as_str) {
        if id.is_empty() {
            return None;
        }
        sanitized["id"] = id.into();
    }
    Some(sanitized)
}

fn sanitize_reasoning(item: &JsonValue) -> Option<JsonValue> {
    let object = item.as_object()?;
    let id = object.get("id")?.as_str()?;
    let encrypted = object.get("encrypted_content")?.as_str()?;
    if id.is_empty() || encrypted.is_empty() {
        return None;
    }
    let summary = sanitize_text_parts(object.get("summary")?, "summary_text")?;
    let mut sanitized = serde_json::json!({
        "type":"reasoning",
        "id":id,
        "summary":summary,
        "encrypted_content":encrypted
    });
    if let Some(content) = object.get("content") {
        sanitized["content"] = JsonValue::Array(sanitize_text_parts(content, "reasoning_text")?);
    }
    Some(sanitized)
}

fn sanitize_text_parts(value: &JsonValue, expected_type: &str) -> Option<Vec<JsonValue>> {
    value
        .as_array()?
        .iter()
        .map(|part| {
            if part.get("type")?.as_str()? != expected_type {
                return None;
            }
            Some(serde_json::json!({
                "type":expected_type,
                "text":part.get("text")?.as_str()?
            }))
        })
        .collect()
}

fn sanitize_function_call(item: &JsonValue) -> Option<JsonValue> {
    let object = item.as_object()?;
    let id = object.get("id")?.as_str()?;
    let call_id = object.get("call_id")?.as_str()?;
    let name = object.get("name")?.as_str()?;
    let arguments = object.get("arguments")?.as_str()?;
    if id.is_empty()
        || call_id.is_empty()
        || name.is_empty()
        || !serde_json::from_str::<JsonValue>(arguments)
            .ok()
            .is_some_and(|value| value.is_object())
    {
        return None;
    }
    Some(serde_json::json!({
        "type":"function_call",
        "id":id,
        "call_id":call_id,
        "name":name,
        "arguments":arguments
    }))
}

fn fingerprint(items: &[JsonValue]) -> Option<String> {
    let encoded = serde_json::to_vec(items).ok()?;
    Some(URL_SAFE_NO_PAD.encode(Sha256::digest(encoded)))
}

fn validate_item_relations(items: &[JsonValue]) -> Option<()> {
    let mut item_ids = BTreeSet::new();
    let mut call_ids = BTreeSet::new();
    for item in items {
        if let Some(id) = item.get("id").and_then(JsonValue::as_str)
            && !item_ids.insert(id)
        {
            return None;
        }
        if item.get("type").and_then(JsonValue::as_str) == Some("function_call") {
            let call_id = item.get("call_id")?.as_str()?;
            if !call_ids.insert(call_id) {
                return None;
            }
        }
    }
    Some(())
}

fn normalized_fingerprint(normalized: &[AssistantPart]) -> Option<&str> {
    let mut values = normalized.iter().filter_map(|part| match part {
        AssistantPart::Custom(part) if part.kind == FINGERPRINT_KIND => part.data.as_str(),
        _ => None,
    });
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn semantic_items(items: &[JsonValue]) -> JsonValue {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut refusals = Vec::new();
    let mut calls = Vec::new();
    for item in items {
        match item.get("type").and_then(JsonValue::as_str) {
            Some("message") => {
                for part in item
                    .get("content")
                    .and_then(JsonValue::as_array)
                    .into_iter()
                    .flatten()
                {
                    match part.get("type").and_then(JsonValue::as_str) {
                        Some("output_text") => {
                            text.push_str(
                                part.get("text").and_then(JsonValue::as_str).unwrap_or(""),
                            );
                        }
                        Some("refusal") => refusals.push(
                            part.get("refusal")
                                .and_then(JsonValue::as_str)
                                .unwrap_or("")
                                .to_owned(),
                        ),
                        _ => {}
                    }
                }
            }
            Some("reasoning") => {
                for field in ["summary", "content"] {
                    for part in item
                        .get(field)
                        .and_then(JsonValue::as_array)
                        .into_iter()
                        .flatten()
                    {
                        reasoning
                            .push_str(part.get("text").and_then(JsonValue::as_str).unwrap_or(""));
                    }
                }
            }
            Some("function_call") => calls.push(serde_json::json!({
                "id":item.get("call_id"),
                "provider_item_id":item.get("id"),
                "name":item.get("name"),
                "arguments":item.get("arguments")
            })),
            _ => {}
        }
    }
    serde_json::json!({"text":text,"reasoning":reasoning,"refusals":refusals,"calls":calls})
}

fn semantic_normalized(normalized: &[AssistantPart]) -> JsonValue {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut refusals = Vec::new();
    let mut calls = Vec::new();
    for part in normalized {
        match part {
            AssistantPart::Text(part) => text.push_str(&part.text),
            AssistantPart::Reasoning(part) => reasoning.push_str(&part.text),
            AssistantPart::ToolCall(call) => calls.push(serde_json::json!({
                "id":call.id,
                "provider_item_id":call.provider_item_id,
                "name":call.name,
                "arguments":call.raw_input.clone().unwrap_or_else(|| call.input.to_string())
            })),
            AssistantPart::Custom(part) if part.kind == "azure.openai.refusal" => {
                if let Some(value) = part.data.as_str() {
                    refusals.push(value.to_owned());
                }
            }
            _ => {}
        }
    }
    serde_json::json!({"text":text,"reasoning":reasoning,"refusals":refusals,"calls":calls})
}

fn exact_keys(object: &serde_json::Map<String, JsonValue>, expected: &[&str]) -> Option<()> {
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    (actual == expected).then_some(())
}
