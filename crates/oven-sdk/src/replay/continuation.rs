use crate::{AssistantPart, CustomPart, JsonValue, ToolCallPart};

/// Evidence that a captured Vertex call needs its original thought signature.
pub const VERTEX_REQUIRED_SIGNATURE: &str = "google_vertex.required_thought_signature";

/// Evidence of encrypted Azure Responses state, not an artifact-integrity marker.
pub const AZURE_RESPONSES_CONTINUATION: &str = "azure.openai.responses.reasoning_continuation";

/// Records a non-secret witness only for a call with actual opaque signature data.
pub fn mark_required_vertex_signature(call: &mut ToolCallPart, native: &JsonValue) {
    if let Some(signature) = native
        .get("thoughtSignature")
        .and_then(JsonValue::as_str)
        .filter(|value| !value.is_empty())
    {
        call.metadata.get_or_insert_with(Default::default).insert(
            VERTEX_REQUIRED_SIGNATURE.into(),
            JsonValue::String(super::sha256_hex(signature.as_bytes())),
        );
    }
}

/// Whether normalized history records a required Vertex continuation.
pub fn has_required_vertex_signature(parts: &[AssistantPart]) -> bool {
    parts.iter().any(|part| matches!(part, AssistantPart::ToolCall(call) if call.metadata.as_ref().is_some_and(|metadata| metadata.contains_key(VERTEX_REQUIRED_SIGNATURE))))
}

/// Checks witnesses against the corresponding validated native function-call parts.
pub fn vertex_signatures_match(parts: &[JsonValue], normalized: &[AssistantPart]) -> bool {
    let mut calls = parts
        .iter()
        .filter(|part| part.get("functionCall").is_some());
    for call in normalized.iter().filter_map(|part| match part {
        AssistantPart::ToolCall(call) => Some(call),
        _ => None,
    }) {
        let native = calls.next();
        if let Some(witness) = call
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get(VERTEX_REQUIRED_SIGNATURE))
        {
            let signature = native
                .and_then(|part| part.get("thoughtSignature"))
                .and_then(JsonValue::as_str)
                .filter(|value| !value.is_empty());
            if !signature.is_some_and(|signature| {
                witness.as_str() == Some(super::sha256_hex(signature.as_bytes()).as_str())
            }) {
                return false;
            }
        }
    }
    true
}

/// Produces an encrypted-state witness without retaining plaintext credentials or ciphertext.
pub fn azure_responses_continuation(item: &JsonValue) -> Option<CustomPart> {
    if item.get("type")?.as_str()? != "reasoning" {
        return None;
    }
    let encrypted = item
        .get("encrypted_content")?
        .as_str()
        .filter(|value| !value.is_empty())?;
    Some(CustomPart::new(
        AZURE_RESPONSES_CONTINUATION,
        serde_json::json!({
            "item_id": item.get("id").and_then(JsonValue::as_str),
            "encrypted_sha256": super::sha256_hex(encrypted.as_bytes()),
        }),
    ))
}
