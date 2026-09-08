//! Validated replay codecs shared by targets with the same wire-block semantics.

mod azure_responses;
mod chat;
mod continuation;
mod openai_responses;

pub use chat::{ChatReasoningField, decode as chat_message};
pub use continuation::{
    AZURE_RESPONSES_CONTINUATION, VERTEX_REQUIRED_SIGNATURE, azure_responses_continuation,
    has_required_vertex_signature, mark_required_vertex_signature, vertex_signatures_match,
};

/// Validates the shared standard-parts subset of Gemini and Vertex envelopes.
/// Provider-specific parts require their owning codec instead of guessed conversion.
pub fn generate_content_parts(
    artifact: &NativeReplayArtifact,
    normalized: &[AssistantPart],
) -> Option<Vec<JsonValue>> {
    let root = artifact.payload().as_object()?;
    let vertex = root.get("format").and_then(JsonValue::as_str)
        == Some("oven.google.vertex.generate-content.assistant.v4");
    let content = if vertex {
        if root.len() != 2 {
            return None;
        }
        root.get("content")?.as_object()?
    } else {
        if root.contains_key("format") {
            return None;
        }
        root
    };
    if content.len() != 2 || content.get("role")?.as_str()? != "model" {
        return None;
    }
    let parts = content.get("parts")?.as_array()?.clone();
    let mut native = Vec::new();
    for part in &parts {
        let object = part.as_object()?;
        if !object.keys().all(|key| {
            matches!(
                key.as_str(),
                "text" | "thought" | "thoughtSignature" | "functionCall"
            )
        }) || object
            .get("thought")
            .is_some_and(|value| !value.is_boolean())
            || object
                .get("thoughtSignature")
                .is_some_and(|value| !value.is_string())
        {
            return None;
        }
        if let Some(text) = object.get("text") {
            let text = text.as_str()?;
            if object.contains_key("functionCall") {
                return None;
            }
            if vertex && text.is_empty() && object.contains_key("thoughtSignature") {
                continue;
            }
            native.push(serde_json::json!({"kind": if object.get("thought").and_then(JsonValue::as_bool) == Some(true) { "reasoning" } else { "text" }, "text": text}));
        } else {
            let call = object.get("functionCall")?.as_object()?;
            if !call
                .keys()
                .all(|key| matches!(key.as_str(), "id" | "name" | "args"))
            {
                return None;
            }
            let id = match call.get("id") {
                Some(value) => JsonValue::String(value.as_str()?.into()),
                None if vertex => JsonValue::Null,
                None => JsonValue::String(String::new()),
            };
            let args = call
                .get("args")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            if !args.is_object() {
                return None;
            }
            native.push(serde_json::json!({"kind":"call","id":id,"name":call.get("name")?.as_str()?,"args":args}));
        }
    }
    let expected = normalized.iter().filter_map(|part| match part {
        AssistantPart::Text(part) => Some(serde_json::json!({"kind":"text","text":part.text})),
        AssistantPart::Reasoning(part) => Some(serde_json::json!({"kind":"reasoning","text":part.text})),
        AssistantPart::ToolCall(call) => Some(serde_json::json!({"kind":"call","id":if vertex { serde_json::to_value(&call.provider_item_id).ok()? } else { JsonValue::String(call.id.clone()) },"name":call.name,"args":call.input})),
        _ => None,
    }).collect::<Vec<_>>();
    (native == expected && vertex_signatures_match(&parts, normalized)).then_some(parts)
}

pub use azure_responses::{
    FINGERPRINT_KIND as AZURE_RESPONSES_FINGERPRINT_KIND, capture as azure_responses_capture,
};

use crate::{AssistantPart, JsonValue, ModelError, ModelId, NativeReplayArtifact};

/// Validates an entire source envelope before extracting Responses wire items.
/// Origin adapter and routing provenance are not eligibility gates.
pub fn responses_items(
    artifact: &NativeReplayArtifact,
    normalized: &[AssistantPart],
) -> Option<Vec<JsonValue>> {
    match artifact.payload().get("format")?.as_str()? {
        "oven.openai.responses.output.v1" => openai_responses::decode(artifact, normalized),
        "oven.azure.openai.responses.output.v4" => azure_responses::decode(artifact, normalized),
        _ => None,
    }
}

/// Filters encrypted reasoning only after source integrity/semantic validation.
/// Function-call continuations cannot silently lose their associated encrypted state.
pub fn eligible_responses_items(
    artifact: &NativeReplayArtifact,
    target: &ModelId,
    reasoning_supported: bool,
    mut items: Vec<JsonValue>,
    warnings: &mut Vec<String>,
) -> Result<Vec<JsonValue>, ModelError> {
    let same_model = artifact.source_wire_model_id() == Some(target);
    let before = items.len();
    items.retain(|item| {
        item.get("type").and_then(JsonValue::as_str) != Some("reasoning")
            || reasoning_supported
                && (item
                    .get("encrypted_content")
                    .and_then(JsonValue::as_str)
                    .is_none_or(str::is_empty)
                    || same_model)
    });
    if items.len() != before {
        if items
            .iter()
            .any(|item| item.get("type").and_then(JsonValue::as_str) == Some("function_call"))
        {
            return Err(ModelError::replay(
                "reasoning continuation cannot omit ineligible native reasoning state",
            ));
        }
        warnings.push("native reasoning excluded: target format support or equal known wire model_id is required for encrypted reasoning".into());
    }
    Ok(items)
}

fn sha256_hex(value: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(value))
}

/// Native array formats whose blocks have been validated by their source codec.
#[derive(Clone, Copy)]
pub enum BlockFormat {
    /// Anthropic Messages content blocks; signatures and redacted data are distinct.
    Messages,
    /// Bedrock Converse content blocks.
    Converse {
        /// Whether the target explicitly supports redacted Converse reasoning.
        redacted_reasoning: bool,
    },
    /// Gemini/Vertex parts; thought signatures carry opaque encrypted thought state.
    GenerateContent {
        /// Explicit target support for provider-tool native part formats.
        provider_tools: bool,
    },
}

/// Applies block-specific eligibility without using origin/routing fingerprints.
pub fn eligible_blocks(
    artifact: &NativeReplayArtifact,
    target: &ModelId,
    format: BlockFormat,
    reasoning_supported: bool,
    mut blocks: Vec<JsonValue>,
    warnings: &mut Vec<String>,
) -> Result<Vec<JsonValue>, ModelError> {
    let encrypted_allowed = reasoning_supported
        && artifact.source_wire_model_id() == Some(target)
        && !matches!(
            format,
            BlockFormat::Converse {
                redacted_reasoning: false
            }
        );
    let mut omitted = false;
    let mut has_calls = false;
    let mut eligible = Vec::new();
    for mut block in blocks.drain(..) {
        let object = block
            .as_object()
            .ok_or_else(|| ModelError::replay("native replay block must be an object"))?;
        let (reasoning, encrypted) = match format {
            BlockFormat::Messages => match block.get("type").and_then(JsonValue::as_str) {
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
            },
            BlockFormat::Converse { .. } => {
                if object.len() != 1
                    || !object
                        .keys()
                        .all(|key| matches!(key.as_str(), "text" | "toolUse" | "reasoningContent"))
                {
                    return Err(ModelError::replay(
                        "unsupported custom Converse replay block",
                    ));
                }
                has_calls |= block.get("toolUse").is_some();
                (
                    block.get("reasoningContent").is_some(),
                    block.pointer("/reasoningContent/redactedContent").is_some(),
                )
            }
            BlockFormat::GenerateContent { provider_tools } => {
                if [
                    "text",
                    "functionCall",
                    "toolCall",
                    "toolResponse",
                    "executableCode",
                    "codeExecutionResult",
                ]
                .iter()
                .filter(|key| object.contains_key(**key))
                .count()
                    != 1
                {
                    return Err(ModelError::replay("ambiguous generateContent native part"));
                }
                let server_part = [
                    "toolCall",
                    "toolResponse",
                    "executableCode",
                    "codeExecutionResult",
                ]
                .iter()
                .any(|key| object.contains_key(*key));
                if !object.keys().all(|key| {
                    matches!(
                        key.as_str(),
                        "text" | "thought" | "thoughtSignature" | "functionCall"
                    ) || provider_tools
                        && matches!(
                            key.as_str(),
                            "toolCall" | "toolResponse" | "executableCode" | "codeExecutionResult"
                        )
                }) {
                    return Err(ModelError::replay(
                        "target has no explicit support for these custom generateContent replay fields",
                    ));
                }
                if object.get("text").is_some_and(|value| !value.is_string())
                    || object
                        .get("thought")
                        .is_some_and(|value| !value.is_boolean())
                    || object
                        .get("thoughtSignature")
                        .is_some_and(|value| !value.is_string())
                    || object.contains_key("text") && object.contains_key("functionCall")
                {
                    return Err(ModelError::replay("invalid generateContent replay part"));
                }
                if let Some(call) = object.get("functionCall") {
                    let call = call
                        .as_object()
                        .ok_or_else(|| ModelError::replay("invalid native functionCall"))?;
                    if !call
                        .keys()
                        .all(|key| matches!(key.as_str(), "id" | "name" | "args"))
                        || !call.get("name").is_some_and(JsonValue::is_string)
                        || call.get("id").is_some_and(|value| !value.is_string())
                        || call.get("args").is_some_and(|value| !value.is_object())
                    {
                        return Err(ModelError::replay("invalid native functionCall fields"));
                    }
                }
                has_calls |= block.get("functionCall").is_some() || server_part;
                if block.get("text").is_none()
                    && block.get("functionCall").is_none()
                    && !(provider_tools && server_part)
                {
                    return Err(ModelError::replay(
                        "target has no explicit support for this custom generateContent replay block",
                    ));
                }
                if !encrypted_allowed && block.get("thoughtSignature").is_some() {
                    omitted = true;
                    block
                        .as_object_mut()
                        .expect("validated native part")
                        .remove("thoughtSignature");
                }
                (
                    block.get("thought").and_then(JsonValue::as_bool) == Some(true),
                    false,
                )
            }
        };
        if reasoning && !reasoning_supported || encrypted && !encrypted_allowed {
            omitted = true;
        } else {
            eligible.push(block);
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
    use super::*;
    use crate::{AdapterId, NativeContextScope, ProviderId, ResourceId};

    fn artifact(known_source: bool) -> NativeReplayArtifact {
        let scope = NativeContextScope::new(
            ProviderId::new("source"),
            ModelId::new("same-model"),
            ResourceId::new("source-provenance").unwrap(),
        )
        .unwrap();
        if known_source {
            NativeReplayArtifact::capture(
                AdapterId::new("source-adapter"),
                scope,
                serde_json::json!({}),
            )
            .unwrap()
        } else {
            NativeReplayArtifact::new(
                AdapterId::new("source-adapter"),
                scope,
                serde_json::json!({}),
            )
            .unwrap()
        }
    }

    #[test]
    fn signed_and_encrypted_messages_have_distinct_eligibility() {
        let blocks = vec![
            serde_json::json!({"type":"text","text":"portable"}),
            serde_json::json!({"type":"thinking","thinking":"visible","signature":"signed-state"}),
            serde_json::json!({"type":"redacted_thinking","data":"encrypted-state"}),
        ];
        for (known, target, expected) in [
            (true, "same-model", 3),
            (true, "different-model", 2),
            (false, "same-model", 2),
        ] {
            let filtered = eligible_blocks(
                &artifact(known),
                &ModelId::new(target),
                BlockFormat::Messages,
                true,
                blocks.clone(),
                &mut vec![],
            )
            .unwrap();
            assert_eq!(filtered.len(), expected);
            assert_eq!(filtered[1]["signature"], "signed-state");
        }
    }

    #[test]
    fn filtered_encrypted_tool_continuation_and_unknown_custom_semantics_fail_closed() {
        let blocks = vec![
            serde_json::json!({"type":"redacted_thinking","data":"opaque"}),
            serde_json::json!({"type":"tool_use","id":"call","name":"inspect","input":{}}),
        ];
        assert!(
            eligible_blocks(
                &artifact(true),
                &ModelId::new("different"),
                BlockFormat::Messages,
                true,
                blocks,
                &mut vec![]
            )
            .is_err()
        );
        assert!(
            eligible_blocks(
                &artifact(true),
                &ModelId::new("same-model"),
                BlockFormat::Messages,
                true,
                vec![serde_json::json!({"type":"future-custom","opaque":true})],
                &mut vec![]
            )
            .is_err()
        );
    }
}
