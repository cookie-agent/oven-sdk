//! Current private Responses output replay codec.

use std::borrow::Cow;

use crate::{AssistantPart, JsonValue, NativeReplayArtifact};

use super::{sha256_hex, sha256_json_hex};
const REPLAY_FORMAT: &str = "oven.openai.responses.output.v1";

const CONTINUATION_KIND: &str = "openai.responses.reasoning_continuation";
const MESSAGE_CONTINUATION_KIND: &str = "openai.responses.message_continuation";

pub(crate) fn decode<'a>(
    artifact: &'a NativeReplayArtifact,
    normalized: &[AssistantPart],
) -> Option<&'a [JsonValue]> {
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
    let items = root.get("items")?.as_array()?;
    let semantic = semantic_items(items)?;
    matches_normalized(&semantic, normalized).then_some(items.as_slice())
}

/// Replay-relevant meaning of one native item or normalized part. Native items
/// and normalized history must produce the same sequence for replay to apply.
#[derive(Debug, PartialEq)]
enum Semantic<'a> {
    Text(&'a str),
    Reasoning(&'a str),
    ToolCall {
        id: &'a str,
        provider_item_id: Option<&'a str>,
        name: &'a str,
        arguments: Cow<'a, str>,
    },
    ReasoningContinuation {
        item_id: &'a str,
        encrypted_sha256: Cow<'a, str>,
    },
    MessageContinuation {
        item_sha256: Cow<'a, str>,
    },
}

fn semantic_items(items: &[JsonValue]) -> Option<Vec<Semantic<'_>>> {
    let mut semantic = Vec::with_capacity(items.len() + 2);
    for item in items {
        match item.get("type")?.as_str()? {
            "message" => message_semantics(item, &mut semantic)?,
            "reasoning" => reasoning_semantics(item, &mut semantic)?,
            "function_call" => function_semantics(item, &mut semantic)?,
            // Server-side compaction state: opaque, with no normalized part.
            "compaction" => exact_keys(
                item,
                &["type", "id", "encrypted_content", "created_by"],
                &["type", "id", "encrypted_content"],
            )
            .and_then(|()| non_empty_string(item, "encrypted_content").map(drop))?,
            _ => return None,
        }
    }
    Some(semantic)
}

fn message_semantics<'a>(item: &'a JsonValue, semantic: &mut Vec<Semantic<'a>>) -> Option<()> {
    exact_keys(
        item,
        &["type", "id", "status", "role", "content", "phase"],
        &["type", "id", "role", "content"],
    )?;
    non_empty_string(item, "id")?;
    if item.get("role")?.as_str()? != "assistant" {
        return None;
    }
    completed_status(item)?;
    match item.get("phase") {
        None | Some(JsonValue::Null) => {}
        Some(JsonValue::String(phase))
            if matches!(phase.as_str(), "commentary" | "final_answer") => {}
        Some(_) => return None,
    }
    let mut has_metadata = item.get("phase").is_some();
    for part in item.get("content")?.as_array()? {
        exact_keys(
            part,
            &["type", "text", "annotations", "logprobs"],
            &["type", "text"],
        )?;
        if part.get("type")?.as_str()? != "output_text" {
            return None;
        }
        if let Some(annotations) = part.get("annotations") {
            annotations.as_array()?;
            has_metadata = true;
        }
        if let Some(logprobs) = part.get("logprobs") {
            if !logprobs.is_null() {
                logprobs.as_array()?;
            }
            has_metadata = true;
        }
        semantic.push(Semantic::Text(part.get("text")?.as_str()?));
    }
    if has_metadata {
        semantic.push(Semantic::MessageContinuation {
            item_sha256: Cow::Owned(sha256_json_hex(item)?),
        });
    }
    Some(())
}

fn reasoning_semantics<'a>(item: &'a JsonValue, semantic: &mut Vec<Semantic<'a>>) -> Option<()> {
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
        semantic.push(Semantic::Reasoning(part.get("text")?.as_str()?));
    }
    if let Some(content) = item.get("content") {
        for part in content.as_array()? {
            exact_keys(part, &["type", "text"], &["type", "text"])?;
            if part.get("type")?.as_str()? != "reasoning_text" {
                return None;
            }
            semantic.push(Semantic::Reasoning(part.get("text")?.as_str()?));
        }
    }
    semantic.push(Semantic::ReasoningContinuation {
        item_id,
        encrypted_sha256: Cow::Owned(sha256_hex(encrypted.as_bytes())),
    });
    Some(())
}

fn function_semantics<'a>(item: &'a JsonValue, semantic: &mut Vec<Semantic<'a>>) -> Option<()> {
    // `namespace` is present on a call to a tool loaded by `additional_tools`.
    exact_keys(
        item,
        &[
            "type",
            "id",
            "status",
            "call_id",
            "name",
            "arguments",
            "namespace",
        ],
        &["type", "id", "call_id", "name", "arguments"],
    )?;
    if let Some(namespace) = item.get("namespace") {
        namespace
            .as_str()
            .filter(|namespace| !namespace.is_empty())?;
    }
    completed_status(item)?;
    semantic.push(Semantic::ToolCall {
        id: non_empty_string(item, "call_id")?,
        provider_item_id: Some(non_empty_string(item, "id")?),
        name: non_empty_string(item, "name")?,
        arguments: Cow::Borrowed(non_empty_string(item, "arguments")?),
    });
    Some(())
}

/// Walks normalized parts against the native semantics without copying text.
/// Any part outside the replayable vocabulary, or a malformed continuation
/// witness, rejects the artifact.
fn matches_normalized(semantic: &[Semantic<'_>], normalized: &[AssistantPart]) -> bool {
    let mut expected = semantic.iter();
    for part in normalized {
        let actual = match part {
            AssistantPart::Text(part) => Semantic::Text(&part.text),
            AssistantPart::Reasoning(part) => Semantic::Reasoning(&part.text),
            AssistantPart::ToolCall(call) => Semantic::ToolCall {
                id: &call.id,
                provider_item_id: call.provider_item_id.as_deref(),
                name: &call.name,
                arguments: call
                    .raw_input
                    .as_deref()
                    .map_or_else(|| Cow::Owned(call.input.to_string()), Cow::Borrowed),
            },
            AssistantPart::Custom(part) if part.kind == CONTINUATION_KIND => {
                let witness = (|| {
                    exact_keys(
                        &part.data,
                        &["item_id", "encrypted_sha256"],
                        &["item_id", "encrypted_sha256"],
                    )?;
                    Some(Semantic::ReasoningContinuation {
                        item_id: non_empty_string(&part.data, "item_id")?,
                        encrypted_sha256: Cow::Borrowed(non_empty_string(
                            &part.data,
                            "encrypted_sha256",
                        )?),
                    })
                })();
                let Some(witness) = witness else {
                    return false;
                };
                witness
            }
            AssistantPart::Custom(part) if part.kind == MESSAGE_CONTINUATION_KIND => {
                let witness = (|| {
                    exact_keys(&part.data, &["item_sha256"], &["item_sha256"])?;
                    Some(Semantic::MessageContinuation {
                        item_sha256: Cow::Borrowed(non_empty_string(&part.data, "item_sha256")?),
                    })
                })();
                let Some(witness) = witness else {
                    return false;
                };
                witness
            }
            _ => return false,
        };
        if expected.next() != Some(&actual) {
            return false;
        }
    }
    expected.next().is_none()
}

fn exact_keys(value: &JsonValue, allowed: &[&str], required: &[&str]) -> Option<()> {
    let object = value.as_object()?;
    if object.keys().any(|key| !allowed.contains(&key.as_str()))
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
