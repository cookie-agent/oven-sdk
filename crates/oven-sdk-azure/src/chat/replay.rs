pub(crate) fn decode(
    artifact: &oven_sdk::NativeReplayArtifact,
    normalized: &[oven_sdk::AssistantPart],
    field: crate::configuration::AzureReasoningField,
    _binding: &oven_sdk::JsonValue,
) -> Option<oven_sdk::JsonValue> {
    use oven_sdk::replay::ChatReasoningField;
    oven_sdk::replay::chat_message(
        artifact,
        normalized,
        match field {
            crate::configuration::AzureReasoningField::None => ChatReasoningField::None,
            crate::configuration::AzureReasoningField::Reasoning => ChatReasoningField::Reasoning,
            crate::configuration::AzureReasoningField::ReasoningContent => {
                ChatReasoningField::ReasoningContent
            }
        },
    )
}
