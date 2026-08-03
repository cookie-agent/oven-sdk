//! Current private Chat assistant replay codec.

use std::collections::BTreeSet;

use oven_sdk::{AssistantPart, JsonValue, NativeReplayArtifact};

use crate::configuration::AzureReasoningField;
use crate::wire::chat::REPLAY_FORMAT;

pub(crate) fn decode(
    artifact: &NativeReplayArtifact,
    normalized: &[AssistantPart],
    reasoning_field: AzureReasoningField,
    replay_binding: &JsonValue,
) -> Option<JsonValue> {
    let root = artifact.payload().as_object()?;
    exact_keys(root, &["binding", "format", "message"])?;
    if root.get("format")?.as_str()? != REPLAY_FORMAT || root.get("binding")? != replay_binding {
        return None;
    }
    let message = root.get("message")?.clone();
    validate_message(&message, reasoning_field)?;
    (semantic_message(&message, reasoning_field) == semantic_normalized(normalized))
        .then_some(message)
}

fn validate_message(message: &JsonValue, reasoning_field: AzureReasoningField) -> Option<()> {
    let object = message.as_object()?;
    let mut allowed = vec!["content", "refusal", "role", "tool_calls"];
    match reasoning_field {
        AzureReasoningField::None => {}
        AzureReasoningField::ReasoningContent => allowed.push("reasoning_content"),
        AzureReasoningField::Reasoning => allowed.push("reasoning"),
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
        let mut ids = BTreeSet::new();
        for call in calls {
            let call = call.as_object()?;
            exact_keys(call, &["function", "id", "type"])?;
            let id = call.get("id")?.as_str()?;
            if id.is_empty() || !ids.insert(id) || call.get("type")?.as_str()? != "function" {
                return None;
            }
            let function = call.get("function")?.as_object()?;
            exact_keys(function, &["arguments", "name"])?;
            if function.get("name")?.as_str()?.is_empty() {
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

fn semantic_message(message: &JsonValue, reasoning_field: AzureReasoningField) -> JsonValue {
    let text = message
        .get("content")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    let reasoning = message
        .get(match reasoning_field {
            AzureReasoningField::None | AzureReasoningField::ReasoningContent => {
                "reasoning_content"
            }
            AzureReasoningField::Reasoning => "reasoning",
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
            AssistantPart::Custom(part) if part.kind == "azure.openai.refusal" => {
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
