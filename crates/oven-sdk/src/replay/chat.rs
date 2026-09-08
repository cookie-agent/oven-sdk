//! Current private Chat assistant replay codec.

use std::collections::BTreeSet;

use crate::{AssistantPart, JsonValue, NativeReplayArtifact};

/// Explicit target wire field for visible, non-encrypted Chat reasoning.
#[derive(Clone, Copy)]
pub enum ChatReasoningField {
    /// Target does not encode visible reasoning.
    None,
    /// Target encodes `reasoning_content`.
    ReasoningContent,
    /// Target encodes `reasoning`.
    Reasoning,
}

/// Validates current OpenAI/Azure Chat envelopes and emits supported Chat fields.
pub fn decode(
    artifact: &NativeReplayArtifact,
    normalized: &[AssistantPart],
    target_field: ChatReasoningField,
) -> Option<JsonValue> {
    let root = artifact.payload().as_object()?;
    match root.get("format")?.as_str()? {
        "oven.azure.openai.chat.assistant.v4" => {
            exact_keys(root, &["binding", "format", "message"])?;
            let binding = root.get("binding")?.as_object()?;
            exact_keys(binding, &["version", "sha256"])?;
            binding.get("version")?.as_str()?;
            binding.get("sha256")?.as_str()?;
        }
        "oven.openai.chat.assistant.v1" => {
            exact_subset(root, &["format", "message", "finish_reason"])?
        }
        _ => return None,
    }
    let mut message = root.get("message")?.clone();
    let reasoning_field = if message.get("reasoning").is_some() {
        ChatReasoningField::Reasoning
    } else if message.get("reasoning_content").is_some() {
        ChatReasoningField::ReasoningContent
    } else {
        ChatReasoningField::None
    };
    validate_message(&message, reasoning_field)?;
    if semantic_message(&message, reasoning_field) != semantic_normalized(normalized) {
        return None;
    }
    let object = message.as_object_mut()?;
    let reasoning = object
        .remove("reasoning")
        .or_else(|| object.remove("reasoning_content"));
    if let Some(reasoning) = reasoning {
        match target_field {
            ChatReasoningField::None => {}
            ChatReasoningField::ReasoningContent => {
                object.insert("reasoning_content".into(), reasoning);
            }
            ChatReasoningField::Reasoning => {
                object.insert("reasoning".into(), reasoning);
            }
        }
    }
    Some(message)
}

fn validate_message(message: &JsonValue, reasoning_field: ChatReasoningField) -> Option<()> {
    let object = message.as_object()?;
    let mut allowed = vec!["content", "refusal", "role", "tool_calls"];
    match reasoning_field {
        ChatReasoningField::None => {}
        ChatReasoningField::ReasoningContent => allowed.push("reasoning_content"),
        ChatReasoningField::Reasoning => allowed.push("reasoning"),
    }
    exact_subset(object, &allowed)?;
    if object.get("role")?.as_str()? != "assistant" {
        return None;
    }
    if !object
        .get("content")
        .is_some_and(|value| value.is_null() || value.is_string())
    {
        return None;
    }
    for field in ["refusal", "reasoning_content", "reasoning"] {
        if object.get(field).is_some_and(|value| !value.is_string()) {
            return None;
        }
    }
    if let Some(calls) = object.get("tool_calls") {
        let calls = calls.as_array()?;
        for call in calls {
            let call = call.as_object()?;
            exact_keys(call, &["function", "id", "type"])?;
            let id = call.get("id")?.as_str()?;
            if id.is_empty() || call.get("type")?.as_str()? != "function" {
                return None;
            }
            let function = call.get("function")?.as_object()?;
            exact_subset(function, &["arguments", "name"])?;
            if function.get("name").is_some_and(|value| !value.is_string()) {
                return None;
            }
            let arguments = function.get("arguments")?.as_str()?;
            if !serde_json::from_str::<JsonValue>(arguments)
                .ok()
                .is_some_and(|value| value.is_object())
            {
                return None;
            }
        }
    }
    Some(())
}

fn semantic_message(message: &JsonValue, reasoning_field: ChatReasoningField) -> JsonValue {
    let text = message
        .get("content")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    let reasoning = message
        .get(match reasoning_field {
            ChatReasoningField::None | ChatReasoningField::ReasoningContent => "reasoning_content",
            ChatReasoningField::Reasoning => "reasoning",
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
                        "name": call.pointer("/function/name").and_then(JsonValue::as_str).unwrap_or_default(),
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
            AssistantPart::Custom(part)
                if matches!(
                    part.kind.as_str(),
                    "azure.openai.refusal" | "openai.refusal"
                ) =>
            {
                if let Some(value) = part.data.as_str() {
                    refusal.push_str(value);
                }
            }
            _ => {}
        }
    }
    serde_json::json!({"text":text,"reasoning":reasoning,"refusal":refusal,"tools":tools})
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
