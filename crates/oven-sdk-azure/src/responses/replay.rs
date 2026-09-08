pub(crate) use oven_sdk::replay::{
    AZURE_RESPONSES_FINGERPRINT_KIND as FINGERPRINT_KIND, azure_responses_capture as capture,
};

pub(crate) fn decode(
    artifact: &oven_sdk::NativeReplayArtifact,
    normalized: &[oven_sdk::AssistantPart],
    _replay_binding: &oven_sdk::JsonValue,
) -> Option<Vec<oven_sdk::JsonValue>> {
    oven_sdk::replay::responses_items(artifact, normalized)
}
