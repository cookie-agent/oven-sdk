//! Current private Chat assistant replay codec.

use oven_sdk::{AssistantPart, JsonValue, NativeReplayArtifact};

use crate::configuration::ReasoningField;
use crate::wire::chat::REPLAY_FORMAT;

pub(crate) fn decode(
    artifact: &NativeReplayArtifact,
    normalized: &[AssistantPart],
    reasoning_field: ReasoningField,
) -> Option<JsonValue> {
    let root = artifact.payload();
    if root.get("format")?.as_str()? != REPLAY_FORMAT {
        return None;
    }
    let message = root.get("message")?.clone();
    if message.get("role")?.as_str()? != "assistant" {
        return None;
    }
    (semantic_message(&message, reasoning_field) == semantic_normalized(normalized))
        .then_some(message)
}

fn semantic_message(message: &JsonValue, reasoning_field: ReasoningField) -> JsonValue {
    let text = message
        .get("content")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    let reasoning = message
        .get(match reasoning_field {
            ReasoningField::None | ReasoningField::ReasoningContent => "reasoning_content",
            ReasoningField::Reasoning => "reasoning",
        })
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    let refusal = message
        .get("refusal")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    let tools = message
        .get("tool_calls")
        .and_then(JsonValue::as_array)
        .map(|calls| {
            calls
                .iter()
                .map(|call| {
                    serde_json::json!({
                        "id": call.get("id"),
                        "name": call.pointer("/function/name"),
                        "arguments": call.pointer("/function/arguments")
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    serde_json::json!({"text":text,"reasoning":reasoning,"refusal":refusal,"tools":tools})
}

fn semantic_normalized(normalized: &[AssistantPart]) -> JsonValue {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut refusal = String::new();
    let mut tools = Vec::new();
    for part in normalized {
        match part {
            AssistantPart::Text(part) => text.push_str(&part.text),
            AssistantPart::Reasoning(part) => reasoning.push_str(&part.text),
            AssistantPart::ToolCall(call) => tools.push(serde_json::json!({
                "id": call.id,
                "name": call.name,
                "arguments": call.raw_input.clone().unwrap_or_else(|| call.input.to_string())
            })),
            AssistantPart::Custom(part) if part.kind == "openai.refusal" => {
                if let Some(value) = part.data.as_str() {
                    refusal.push_str(value);
                }
            }
            _ => {}
        }
    }
    serde_json::json!({"text":text,"reasoning":reasoning,"refusal":refusal,"tools":tools})
}
