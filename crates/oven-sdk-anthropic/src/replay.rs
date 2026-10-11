//! Native Anthropic assistant replay codec.

use oven_sdk::{AssistantPart, JsonValue, ModelError, ModelId, NativeReplayArtifact, ToolCallPart};

use crate::wire::Protocol;

/// A validated native content block, borrowed from the replay payload.
pub(crate) type NativeBlock<'a> = &'a serde_json::Map<String, JsonValue>;

/// Decodes a matching current-format artifact, borrowing its content blocks.
pub(crate) fn decode<'a>(
    artifact: &'a NativeReplayArtifact,
    normalized: &[AssistantPart],
    _protocol: Protocol,
) -> Result<&'a [JsonValue], &'static str> {
    let root = artifact.payload();
    if !matches!(
        root.get("format").and_then(JsonValue::as_str),
        Some(
            "oven.anthropic.messages.assistant.v3"
                | "oven.anthropic.aws.messages.assistant.v3"
                | "oven.minimax.messages.assistant.v3"
        )
    ) {
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
        .ok_or("Messages replay content is invalid")?;
    if !authoritative_reasoning_is_well_formed(content) {
        return Err("Messages replay contains invalid provider reasoning");
    }
    if !semantic_content_matches(content, normalized) {
        return Err("Messages replay payload did not match normalized content");
    }
    Ok(content)
}

fn authoritative_reasoning_is_well_formed(content: &[JsonValue]) -> bool {
    content.iter().all(
        |block| match block.get("type").and_then(JsonValue::as_str) {
            Some("thinking") => {
                block.get("thinking").and_then(JsonValue::as_str).is_some()
                    && block.get("signature").is_none_or(JsonValue::is_string)
            }
            Some("redacted_thinking") => block.get("data").is_some(),
            _ => true,
        },
    )
}

/// The semantic block a normalized part must correspond to in the native payload.
enum Expected<'a> {
    Thinking(&'a str),
    RedactedThinking(&'a JsonValue),
    Text(&'a str),
    ToolUse(&'a ToolCallPart),
}

impl<'a> Expected<'a> {
    fn from_part(part: &'a AssistantPart) -> Option<Self> {
        match part {
            AssistantPart::Reasoning(reasoning) => Some(
                match reasoning
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("anthropic.redacted"))
                {
                    Some(data) => Self::RedactedThinking(data),
                    None => Self::Thinking(&reasoning.text),
                },
            ),
            AssistantPart::Text(text) if !text.text.is_empty() => Some(Self::Text(&text.text)),
            AssistantPart::ToolCall(call) => Some(Self::ToolUse(call)),
            _ => None,
        }
    }

    /// Exact JSON equality with the normalized block, ignoring only a thinking `signature`.
    fn matches(&self, block: &JsonValue) -> bool {
        let Some(block) = block.as_object() else {
            return false;
        };
        let str_field = |key: &str, expected: &str| {
            block.get(key).and_then(JsonValue::as_str) == Some(expected)
        };
        match self {
            Self::Thinking(thinking) => {
                str_field("type", "thinking")
                    && str_field("thinking", thinking)
                    && block.len() == 2 + usize::from(block.contains_key("signature"))
            }
            Self::RedactedThinking(data) => {
                block.len() == 2
                    && str_field("type", "redacted_thinking")
                    && block.get("data") == Some(*data)
            }
            Self::Text(text) => {
                block.len() == 2 && str_field("type", "text") && str_field("text", text)
            }
            Self::ToolUse(call) => {
                block.len() == 4
                    && str_field("type", "tool_use")
                    && str_field("id", &call.id)
                    && str_field("name", &call.name)
                    && block.get("input").is_some_and(|input| {
                        if call.input.is_object() {
                            *input == call.input
                        } else {
                            input.as_object().is_some_and(|input| {
                                input.len() == 1
                                    && input.get("rawInvalidInput") == Some(&call.input)
                            })
                        }
                    })
            }
        }
    }
}

fn semantic_content_matches(content: &[JsonValue], normalized: &[AssistantPart]) -> bool {
    let mut blocks = content.iter();
    for expected in normalized.iter().filter_map(Expected::from_part) {
        if !blocks.next().is_some_and(|block| expected.matches(block)) {
            return false;
        }
    }
    blocks.next().is_none()
}

/// Applies Messages block eligibility to validated replay blocks without copying them.
///
/// Mirrors `oven_sdk::replay::eligible_blocks` with `BlockFormat::Messages`: thinking needs
/// target reasoning support, redacted thinking additionally needs the same known wire model,
/// and a tool continuation cannot silently lose reasoning state.
pub(crate) fn eligible_blocks<'a>(
    artifact: &NativeReplayArtifact,
    target: &ModelId,
    reasoning_supported: bool,
    blocks: &'a [JsonValue],
    warnings: &mut Vec<String>,
) -> Result<Vec<NativeBlock<'a>>, ModelError> {
    let encrypted_allowed = reasoning_supported && artifact.source_wire_model_id() == Some(target);
    let mut omitted = false;
    let mut has_calls = false;
    let mut eligible = Vec::with_capacity(blocks.len());
    for block in blocks {
        let object = block
            .as_object()
            .ok_or_else(|| ModelError::replay("native replay block must be an object"))?;
        let (reasoning, encrypted) = match object.get("type").and_then(JsonValue::as_str) {
            Some("text") => (false, false),
            Some("tool_use") => {
                has_calls = true;
                (false, false)
            }
            Some("thinking") => (true, false),
            Some("redacted_thinking") => (true, true),
            _ => {
                return Err(ModelError::replay(
                    "target has no explicit support for this custom Messages replay block",
                ));
            }
        };
        if reasoning && !reasoning_supported || encrypted && !encrypted_allowed {
            omitted = true;
        } else {
            eligible.push(object);
        }
    }
    if omitted {
        if has_calls {
            return Err(ModelError::replay(
                "native tool continuation cannot omit ineligible reasoning state",
            ));
        }
        warnings.push("native reasoning excluded: target support or equal known wire model_id is required for encrypted reasoning".into());
    }
    Ok(eligible)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use oven_sdk::{ReasoningPart, TextPart};
    use serde_json::json;

    use super::*;

    fn parts() -> Vec<AssistantPart> {
        let mut redacted = ReasoningPart::new("");
        redacted.metadata = Some(BTreeMap::from([(
            "anthropic.redacted".into(),
            json!("opaque"),
        )]));
        vec![
            AssistantPart::Reasoning(ReasoningPart::new("think")),
            AssistantPart::Reasoning(redacted),
            AssistantPart::Text(TextPart::new("")),
            AssistantPart::Text(TextPart::new("answer")),
            AssistantPart::ToolCall(ToolCallPart::new("id-1", "lookup", json!({"q":1}))),
            AssistantPart::ToolCall(ToolCallPart::new("id-2", "lookup", json!([1]))),
        ]
    }

    fn content() -> Vec<JsonValue> {
        vec![
            json!({"type":"thinking","thinking":"think","signature":"sig"}),
            json!({"type":"redacted_thinking","data":"opaque"}),
            json!({"type":"text","text":"answer"}),
            json!({"type":"tool_use","id":"id-1","name":"lookup","input":{"q":1}}),
            json!({"type":"tool_use","id":"id-2","name":"lookup","input":{"rawInvalidInput":[1]}}),
        ]
    }

    #[test]
    fn semantic_comparison_ignores_only_thinking_signatures() {
        assert!(semantic_content_matches(&content(), &parts()));
        let mut unsigned = content();
        unsigned[0].as_object_mut().unwrap().remove("signature");
        assert!(semantic_content_matches(&unsigned, &parts()));
    }

    #[test]
    fn semantic_comparison_rejects_any_other_difference() {
        let mutations: [fn(&mut Vec<JsonValue>); 8] = [
            |content| content[1]["signature"] = json!("sig"),
            |content| content[2]["citations"] = json!([]),
            |content| content[2]["text"] = json!("answer "),
            |content| content[3]["input"] = json!({"q":2}),
            |content| content[4]["input"] = json!([1]),
            |content| content[0]["thinking"] = json!("other"),
            |content| {
                content.pop();
            },
            |content| content.push(json!({"type":"text","text":"extra"})),
        ];
        for mutate in mutations {
            let mut changed = content();
            mutate(&mut changed);
            assert!(!semantic_content_matches(&changed, &parts()));
        }
    }
}
