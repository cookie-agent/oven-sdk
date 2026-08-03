//! Current private Responses output replay codec.

use std::collections::BTreeSet;

use oven_sdk::{AssistantPart, JsonValue, NativeReplayArtifact};

use crate::{configuration::sha256_hex, wire::responses::REPLAY_FORMAT};

const CONTINUATION_KIND: &str = "openai.responses.reasoning_continuation";

pub(crate) fn decode(
    artifact: &NativeReplayArtifact,
    normalized: &[AssistantPart],
) -> Option<Vec<JsonValue>> {
    let root = artifact.payload();
    exact_keys(
        root,
        &["format", "items", "store", "status", "incomplete_details"],
        &["format", "items", "store", "status", "incomplete_details"],
    )?;
    if root.get("format")?.as_str()? != REPLAY_FORMAT || root.get("store")?.as_bool()? {
        return None;
    }
    if !matches!(root.get("status")?.as_str()?, "completed" | "incomplete") {
        return None;
    }
    let items = root.get("items")?.as_array()?.clone();
    (semantic_items(&items)? == semantic_normalized(normalized)?).then_some(items)
}

fn semantic_items(items: &[JsonValue]) -> Option<Vec<JsonValue>> {
    let mut semantic = Vec::new();
    for item in items {
        match item.get("type")?.as_str()? {
            "message" => message_semantics(item, &mut semantic)?,
            "reasoning" => reasoning_semantics(item, &mut semantic)?,
            "function_call" => function_semantics(item, &mut semantic)?,
            _ => return None,
        }
    }
    Some(semantic)
}

fn message_semantics(item: &JsonValue, semantic: &mut Vec<JsonValue>) -> Option<()> {
    exact_keys(
        item,
        &["type", "id", "status", "role", "content"],
        &["type", "id", "role", "content"],
    )?;
    non_empty_string(item, "id")?;
    if item.get("role")?.as_str()? != "assistant" {
        return None;
    }
    completed_status(item)?;
    for part in item.get("content")?.as_array()? {
        exact_keys(part, &["type", "text"], &["type", "text"])?;
        if part.get("type")?.as_str()? != "output_text" {
            return None;
        }
        semantic.push(serde_json::json!({
            "type":"text",
            "text":part.get("text")?.as_str()?
        }));
    }
    Some(())
}

fn reasoning_semantics(item: &JsonValue, semantic: &mut Vec<JsonValue>) -> Option<()> {
    exact_keys(
        item,
        &[
            "type",
            "id",
            "status",
            "summary",
            "content",
            "encrypted_content",
        ],
        &["type", "id", "summary", "encrypted_content"],
    )?;
    let item_id = non_empty_string(item, "id")?;
    completed_status(item)?;
    let encrypted = non_empty_string(item, "encrypted_content")?;
    for part in item.get("summary")?.as_array()? {
        exact_keys(part, &["type", "text"], &["type", "text"])?;
        if part.get("type")?.as_str()? != "summary_text" {
            return None;
        }
        semantic.push(serde_json::json!({
            "type":"reasoning",
            "text":part.get("text")?.as_str()?
        }));
    }
    if let Some(content) = item.get("content") {
        for part in content.as_array()? {
            exact_keys(part, &["type", "text"], &["type", "text"])?;
            if part.get("type")?.as_str()? != "reasoning_text" {
                return None;
            }
            semantic.push(serde_json::json!({
                "type":"reasoning",
                "text":part.get("text")?.as_str()?
            }));
        }
    }
    semantic.push(continuation_semantic(item_id, encrypted));
    Some(())
}

fn function_semantics(item: &JsonValue, semantic: &mut Vec<JsonValue>) -> Option<()> {
    exact_keys(
        item,
        &["type", "id", "status", "call_id", "name", "arguments"],
        &["type", "id", "call_id", "name", "arguments"],
    )?;
    completed_status(item)?;
    semantic.push(serde_json::json!({
        "type":"tool_call",
        "id":non_empty_string(item, "call_id")?,
        "provider_item_id":non_empty_string(item, "id")?,
        "name":non_empty_string(item, "name")?,
        "arguments":non_empty_string(item, "arguments")?
    }));
    Some(())
}

fn semantic_normalized(normalized: &[AssistantPart]) -> Option<Vec<JsonValue>> {
    normalized
        .iter()
        .map(|part| match part {
            AssistantPart::Text(part) => Some(serde_json::json!({
                "type":"text",
                "text":part.text
            })),
            AssistantPart::Reasoning(part) => Some(serde_json::json!({
                "type":"reasoning",
                "text":part.text
            })),
            AssistantPart::ToolCall(call) => Some(serde_json::json!({
                "type":"tool_call",
                "id":call.id,
                "provider_item_id":call.provider_item_id,
                "name":call.name,
                "arguments":call.raw_input.clone().unwrap_or_else(|| call.input.to_string())
            })),
            AssistantPart::Custom(part) if part.kind == CONTINUATION_KIND => {
                exact_keys(
                    &part.data,
                    &["item_id", "encrypted_sha256"],
                    &["item_id", "encrypted_sha256"],
                )?;
                Some(serde_json::json!({
                    "type":"reasoning_continuation",
                    "item_id":non_empty_string(&part.data, "item_id")?,
                    "encrypted_sha256":non_empty_string(&part.data, "encrypted_sha256")?
                }))
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn continuation_part(item_id: &str, encrypted: &str) -> oven_sdk::CustomPart {
    oven_sdk::CustomPart::new(
        CONTINUATION_KIND,
        serde_json::json!({
            "item_id":item_id,
            "encrypted_sha256":sha256_hex(encrypted.as_bytes())
        }),
    )
}

fn continuation_semantic(item_id: &str, encrypted: &str) -> JsonValue {
    serde_json::json!({
        "type":"reasoning_continuation",
        "item_id":item_id,
        "encrypted_sha256":sha256_hex(encrypted.as_bytes())
    })
}

fn exact_keys(value: &JsonValue, allowed: &[&str], required: &[&str]) -> Option<()> {
    let object = value.as_object()?;
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    if object.keys().any(|key| !allowed.contains(key.as_str()))
        || required.iter().any(|key| !object.contains_key(*key))
    {
        return None;
    }
    Some(())
}

fn non_empty_string<'a>(value: &'a JsonValue, field: &str) -> Option<&'a str> {
    value.get(field)?.as_str().filter(|value| !value.is_empty())
}

fn completed_status(item: &JsonValue) -> Option<()> {
    if item
        .get("status")
        .is_some_and(|status| status.as_str() != Some("completed"))
    {
        return None;
    }
    Some(())
}
