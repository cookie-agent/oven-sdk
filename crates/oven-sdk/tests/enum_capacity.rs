//! Regression tests for the fixed normalized enum surfaces.
//!
//! `StreamPart` has twenty variants and `FinishReason` has ten normalized
//! variants plus `Other`.
//! Their tagged serde representation is intentionally safe in `Vec` history and
//! stream captures: these tests round-trip the last declared variant in each
//! largest enum, guarding the discriminant boundary as variants evolve.

use oven_sdk::{FinishReason, StreamPart};

#[test]
fn stream_part_custom_variant_round_trips_in_a_vector() {
    // StreamPart's current highest declared discriminant is Custom (20 variants).
    let encoded =
        r#"[{"type":"custom","part":{"kind":"test.custom","data":null,"metadata":null}}]"#;
    let values: Vec<StreamPart> = serde_json::from_str(encoded).expect("stream parts deserialize");
    let encoded = serde_json::to_string(&values).expect("stream parts serialize");
    let decoded: Vec<StreamPart> =
        serde_json::from_str(&encoded).expect("stream parts deserialize");
    assert_eq!(decoded, values);
}

#[test]
fn finish_reason_other_round_trips_without_contradictory_fields() {
    // FinishReason's final variant carries its only provider-specific value.
    let value = FinishReason::other("provider_specific_stop");
    let encoded = serde_json::to_string(&value).expect("finish reason serializes");
    let decoded: FinishReason = serde_json::from_str(&encoded).expect("finish reason deserializes");
    assert_eq!(decoded, value);
}
