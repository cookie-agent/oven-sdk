pub(crate) fn decode<'a>(
    artifact: &'a oven_sdk::NativeReplayArtifact,
    normalized: &[oven_sdk::AssistantPart],
    field: crate::configuration::ReasoningField,
) -> Option<oven_sdk::replay::ChatReplayMessage<'a>> {
    use oven_sdk::replay::ChatReasoningField;
    oven_sdk::replay::chat_message(
        artifact,
        normalized,
        match field {
            crate::configuration::ReasoningField::None => ChatReasoningField::None,
            crate::configuration::ReasoningField::Reasoning => ChatReasoningField::Reasoning,
            crate::configuration::ReasoningField::ReasoningContent => {
                ChatReasoningField::ReasoningContent
            }
        },
    )
}
