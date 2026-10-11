//! Current private Chat assistant replay codec.

use std::{borrow::Cow, collections::BTreeSet};

use crate::{AssistantPart, JsonValue, NativeReplayArtifact, is_semantic_text};

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

/// A validated replayed Chat assistant message, borrowed from its artifact.
///
/// Serializes as the native message with visible reasoning moved to the
/// target's reasoning field (or dropped when the target has none).
#[derive(Clone, Copy, Debug)]
pub struct ChatReplayMessage<'a> {
    message: &'a serde_json::Map<String, JsonValue>,
    reasoning: Option<(&'static str, &'a JsonValue)>,
}

static EMPTY_STRING: JsonValue = JsonValue::String(String::new());

impl ChatReplayMessage<'_> {
    /// Returns whether the emitted message carries a reasoning field.
    #[must_use]
    pub fn has_reasoning(&self) -> bool {
        self.reasoning.is_some()
    }

    /// Emits an empty `reasoning_content` when the message carries no reasoning.
    #[must_use]
    pub fn with_empty_reasoning_content(mut self) -> Self {
        if self.reasoning.is_none() {
            self.reasoning = Some(("reasoning_content", &EMPTY_STRING));
        }
        self
    }
}

impl serde::Serialize for ChatReplayMessage<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        for (key, value) in self.message {
            if !matches!(key.as_str(), "reasoning" | "reasoning_content") {
                map.serialize_entry(key, value)?;
            }
        }
        if let Some((field, value)) = self.reasoning {
            map.serialize_entry(field, value)?;
        }
        map.end()
    }
}

/// Validates current OpenAI/Azure Chat envelopes and emits supported Chat fields.
pub fn decode<'a>(
    artifact: &'a NativeReplayArtifact,
    normalized: &[AssistantPart],
    target_field: ChatReasoningField,
) -> Option<ChatReplayMessage<'a>> {
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
    let message = root.get("message")?;
    let reasoning_field = if message.get("reasoning").is_some() {
        ChatReasoningField::Reasoning
    } else if message.get("reasoning_content").is_some() {
        ChatReasoningField::ReasoningContent
    } else {
        ChatReasoningField::None
    };
    validate_message(message, reasoning_field)?;
    if !semantic_message(message, reasoning_field).matches(normalized) {
        return None;
    }
    let message = message.as_object()?;
    let reasoning = message
        .get("reasoning")
        .or_else(|| message.get("reasoning_content"))
        .and_then(|reasoning| match target_field {
            ChatReasoningField::None => None,
            ChatReasoningField::ReasoningContent => Some(("reasoning_content", reasoning)),
            ChatReasoningField::Reasoning => Some(("reasoning", reasoning)),
        });
    Some(ChatReplayMessage { message, reasoning })
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

/// Replay-relevant meaning of a Chat assistant message.
struct Semantic<'a> {
    text: &'a str,
    reasoning: &'a str,
    refusal: &'a str,
    tools: Vec<(&'a str, &'a str, &'a str)>,
}

impl Semantic<'_> {
    /// Compares against normalized parts without copying their text.
    fn matches(&self, normalized: &[AssistantPart]) -> bool {
        let mut text = Cow::Borrowed("");
        let mut reasoning = Cow::Borrowed("");
        let mut refusal = Cow::Borrowed("");
        let mut tools = self.tools.iter();
        for part in normalized {
            match part {
                // Whitespace-only text parts carry no semantic content
                // (see `is_semantic_text`).
                AssistantPart::Text(part) if !is_semantic_text(&part.text) => {}
                AssistantPart::Text(part) => push(&mut text, &part.text),
                AssistantPart::Reasoning(part) => push(&mut reasoning, &part.text),
                AssistantPart::ToolCall(call) => {
                    let arguments = call
                        .raw_input
                        .as_deref()
                        .map_or_else(|| Cow::Owned(call.input.to_string()), Cow::Borrowed);
                    if tools.next()
                        != Some(&(call.id.as_str(), call.name.as_str(), arguments.as_ref()))
                    {
                        return false;
                    }
                }
                AssistantPart::Custom(part)
                    if matches!(
                        part.kind.as_str(),
                        "azure.openai.refusal" | "openai.refusal"
                    ) =>
                {
                    if let Some(value) = part.data.as_str() {
                        push(&mut refusal, value);
                    }
                }
                _ => {}
            }
        }
        tools.next().is_none()
            && text == self.text
            && reasoning == self.reasoning
            && refusal == self.refusal
    }
}

fn push<'a>(run: &mut Cow<'a, str>, text: &'a str) {
    if run.is_empty() {
        *run = Cow::Borrowed(text);
    } else {
        run.to_mut().push_str(text);
    }
}

/// Reads the semantic fields of a message that passed `validate_message`.
fn semantic_message(message: &JsonValue, reasoning_field: ChatReasoningField) -> Semantic<'_> {
    let text = message
        .get("content")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // Whitespace-only text is not semantic content (see `is_semantic_text`),
    // even though the native artifact may preserve the provider's raw bytes.
    let text = if is_semantic_text(text) { text } else { "" };
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
                    (
                        call.get("id")
                            .and_then(JsonValue::as_str)
                            .unwrap_or_default(),
                        call.pointer("/function/name")
                            .and_then(JsonValue::as_str)
                            .unwrap_or_default(),
                        call.pointer("/function/arguments")
                            .and_then(JsonValue::as_str)
                            .unwrap_or_default(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    Semantic {
        text,
        reasoning,
        refusal,
        tools,
    }
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
