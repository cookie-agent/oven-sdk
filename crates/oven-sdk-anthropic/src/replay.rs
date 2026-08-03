//! Native Anthropic assistant replay codec.

use oven_sdk::{AssistantPart, JsonValue, NativeReplayArtifact};

use crate::wire::Protocol;

/// Decodes a matching current-format artifact.
pub(crate) fn decode(
    artifact: &NativeReplayArtifact,
    normalized: &[AssistantPart],
    protocol: Protocol,
) -> Result<Vec<JsonValue>, &'static str> {
    let root = artifact.payload();
    if root.get("format").and_then(JsonValue::as_str) != Some(protocol.replay_format()) {
        return Err("Messages replay format is invalid");
    }
    if root.pointer("/message/role").and_then(JsonValue::as_str) != Some("assistant")
        || root.get("stop_reason").is_none()
        || root.get("stop_sequence").is_none()
    {
        return Err("Messages replay envelope is incomplete");
    }
    let content = root
        .pointer("/message/content")
        .and_then(JsonValue::as_array)
        .ok_or("Messages replay content is invalid")?
        .clone();
    if !authoritative_reasoning_is_valid(&content, protocol) {
        return Err("Messages replay contains unsigned or invalid provider reasoning");
    }
    let expected = normalized
        .iter()
        .filter_map(crate::request::assistant_semantic_block)
        .collect::<Vec<_>>();
    if semantic_content(&content) != semantic_content(&expected) {
        return Err("Messages replay payload did not match normalized content");
    }
    Ok(content)
}

fn authoritative_reasoning_is_valid(content: &[JsonValue], protocol: Protocol) -> bool {
    content.iter().all(
        |block| match block.get("type").and_then(JsonValue::as_str) {
            Some("thinking") => {
                block.get("thinking").and_then(JsonValue::as_str).is_some()
                    && block
                        .get("signature")
                        .and_then(JsonValue::as_str)
                        .is_some_and(|signature| !signature.is_empty())
            }
            Some("redacted_thinking") => {
                protocol != Protocol::MiniMax
                    && block
                        .get("data")
                        .and_then(JsonValue::as_str)
                        .is_some_and(|data| !data.is_empty())
            }
            _ => true,
        },
    )
}
fn semantic_content(content: &[JsonValue]) -> Vec<JsonValue> {
    content
        .iter()
        .cloned()
        .map(|mut value| {
            if value.get("type").and_then(JsonValue::as_str) == Some("thinking")
                && let Some(object) = value.as_object_mut()
            {
                object.remove("signature");
            }
            value
        })
        .collect()
}
