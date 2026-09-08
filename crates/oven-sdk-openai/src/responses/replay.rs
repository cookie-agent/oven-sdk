pub(crate) use oven_sdk::replay::responses_items as decode;

pub(crate) fn continuation_part(item_id: &str, encrypted: &str) -> oven_sdk::CustomPart {
    oven_sdk::CustomPart::new(
        "openai.responses.reasoning_continuation",
        serde_json::json!({
            "item_id": item_id,
            "encrypted_sha256": crate::configuration::sha256_hex(encrypted.as_bytes()),
        }),
    )
}
