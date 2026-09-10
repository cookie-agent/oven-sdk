pub(crate) use oven_sdk::replay::responses_items as decode;

const MESSAGE_CONTINUATION_KIND: &str = "openai.responses.message_continuation";

pub(crate) fn message_continuation_part(
    item: &oven_sdk::JsonValue,
) -> Option<oven_sdk::CustomPart> {
    let has_metadata = item.get("phase").is_some()
        || item
            .get("content")
            .and_then(oven_sdk::JsonValue::as_array)
            .is_some_and(|content| {
                content
                    .iter()
                    .any(|part| part.get("annotations").is_some() || part.get("logprobs").is_some())
            });
    has_metadata.then(|| {
        let encoded = serde_json::to_vec(item).expect("Responses message item is serializable");
        oven_sdk::CustomPart::new(
            MESSAGE_CONTINUATION_KIND,
            serde_json::json!({"item_sha256": crate::configuration::sha256_hex(&encoded)}),
        )
    })
}

pub(crate) fn continuation_part(item_id: &str, encrypted: &str) -> oven_sdk::CustomPart {
    oven_sdk::CustomPart::new(
        "openai.responses.reasoning_continuation",
        serde_json::json!({
            "item_id": item_id,
            "encrypted_sha256": crate::configuration::sha256_hex(encrypted.as_bytes()),
        }),
    )
}
