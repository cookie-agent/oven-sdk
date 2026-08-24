use std::collections::BTreeMap;

use bytes::Bytes;
use oven_sdk::{
    AssistantMessage, AssistantPart, CancellationCapability, Capability, CompactionCapability,
    FilePart, FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions, InputPart,
    MediaCapabilities, MediaInputSupport, MediaSourceSupport, Modalities, Modality,
    ModelCapabilities, ModelErrorKind, ModelLimits, ReasoningPart, ReplayCapability,
    ReplayDeclaration, ReplayPolicy, Request, ResponseFormat, TextPart, ToolContent,
    ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};

fn capabilities(features: Capability) -> ModelCapabilities {
    ModelCapabilities {
        features,
        limits: ModelLimits::new(Some(16_384), Some(12_000), Some(4_096)),
        modalities: Modalities::new([Modality::text()], [Modality::text()]),
        media: MediaCapabilities::default(),
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        replay: ReplayDeclaration {
            policy: ReplayPolicy::Never,
            capability: ReplayCapability::Unsupported,
            reasoning: false,
        },
    }
}

fn object_schema() -> oven_sdk::JsonSchema {
    oven_sdk::JsonSchema::new(serde_json::json!({"type":"object"})).expect("schema")
}

#[test]
fn sampling_and_output_controls_require_explicit_features() {
    let mut temperature = InferenceOptions::new();
    temperature.temperature = Some(0.5);
    let mut top_p = InferenceOptions::new();
    top_p.top_p = Some(0.9);
    let mut max_output_tokens = InferenceOptions::new();
    max_output_tokens.max_output_tokens = Some(100);
    let mut reasoning = InferenceOptions::new();
    reasoning.reasoning_effort = Some("medium".into());
    let cases = [
        (Capability::TEMPERATURE, temperature),
        (Capability::TOP_P, top_p),
        (Capability::MAX_OUTPUT_TOKENS, max_output_tokens),
        (Capability::REASONING, reasoning),
    ];
    for (feature, inference) in cases {
        let request = Request::new(Vec::new()).with_inference(inference);
        assert!(matches!(
            request.validate_for(&capabilities(Capability::empty())),
            Err(error) if error.kind() == ModelErrorKind::Unsupported
        ));
        assert!(request.validate_for(&capabilities(feature)).is_ok());
    }
}

#[test]
fn output_limit_is_bounded_by_the_declaration() {
    let mut inference = InferenceOptions::new();
    inference.max_output_tokens = Some(4_097);
    let request = Request::new(Vec::new()).with_inference(inference);
    assert!(matches!(
        request.validate_for(&capabilities(Capability::MAX_OUTPUT_TOKENS)),
        Err(error) if error.kind() == ModelErrorKind::InvalidRequest
    ));
}

#[test]
fn tools_and_structured_output_require_explicit_features() {
    let tool_request = Request::new(Vec::new()).with_tools(vec![ToolDefinition::new(
        "lookup",
        "lookup",
        object_schema(),
    )]);
    assert!(matches!(
        tool_request.validate_for(&capabilities(Capability::empty())),
        Err(error) if error.kind() == ModelErrorKind::Unsupported
    ));
    assert!(
        tool_request
            .validate_for(&capabilities(Capability::TOOL_CALLING))
            .is_ok()
    );

    let structured =
        Request::new(Vec::new()).with_response_format(ResponseFormat::structured(object_schema()));
    assert!(matches!(
        structured.validate_for(&capabilities(Capability::empty())),
        Err(error) if error.kind() == ModelErrorKind::Unsupported
    ));
    assert!(
        structured
            .validate_for(&capabilities(Capability::STRUCTURED_OUTPUT))
            .is_ok()
    );
}

#[test]
fn media_requires_matching_modality_mime_and_source_declarations() {
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("inspect")),
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(Bytes::from_static(b"png")),
        )),
    ]))]);
    let mut declared = capabilities(Capability::empty());
    assert!(matches!(
        request.validate_for(&declared),
        Err(error) if error.kind() == ModelErrorKind::Unsupported
    ));

    declared.modalities.input.insert(Modality::image());
    declared.media.input.insert(
        Modality::image(),
        MediaInputSupport::new(["image/jpeg".to_owned()], MediaSourceSupport::INLINE_BYTES)
            .expect("media"),
    );
    assert!(matches!(
        request.validate_for(&declared),
        Err(error) if error.kind() == ModelErrorKind::Unsupported
    ));

    declared.media.input.insert(
        Modality::image(),
        MediaInputSupport::new(["image/*".to_owned()], MediaSourceSupport::URL).expect("media"),
    );
    assert!(matches!(
        request.validate_for(&declared),
        Err(error) if error.kind() == ModelErrorKind::Unsupported
    ));

    declared.media.input.insert(
        Modality::image(),
        MediaInputSupport::new(["image/*".to_owned()], MediaSourceSupport::INLINE_BYTES)
            .expect("media"),
    );
    assert!(request.validate_for(&declared).is_ok());
}

#[test]
fn arbitrary_open_modality_media_rules_are_operational() {
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("inspect")),
        InputPart::File(FilePart::new(
            "application/x-chemical-structure",
            FileSource::Text("H2O".into()),
        )),
    ]))]);
    let chemical = Modality::new("chemical-structure").expect("open modality");
    let mut declared = capabilities(Capability::empty());
    declared.modalities.input.insert(chemical.clone());
    declared.media.input.insert(
        chemical,
        MediaInputSupport::new(
            ["application/x-chemical-*".to_owned()],
            MediaSourceSupport::INLINE_TEXT,
        )
        .expect("media"),
    );
    assert!(request.validate_for(&declared).is_ok());
}

#[test]
fn media_in_tool_results_is_validated_too() {
    let result = ToolResultPart::new(
        "call",
        ToolContent::Mixed(vec![oven_sdk::ContentValue::File(FilePart::document(
            "application/pdf",
            FileSource::Bytes(Bytes::from_static(b"pdf")),
        ))]),
    );
    let assistant = oven_sdk::CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::ToolCall(oven_sdk::ToolCallPart::new(
            "call",
            "tool",
            serde_json::Value::Null,
        ))]),
        Finish::new(Default::default(), FinishReason::ToolCalls),
    );
    let request = Request::new(vec![
        HistoryTurn::assistant(assistant),
        HistoryTurn::tool(ToolMessage::new(vec![result])),
    ]);
    let mut declared = capabilities(Capability::TOOL_CALLING);
    declared.modalities.input.insert(Modality::pdf());
    declared.media.input.insert(
        Modality::pdf(),
        MediaInputSupport::new(
            ["application/pdf".to_owned()],
            MediaSourceSupport::INLINE_BYTES,
        )
        .expect("media"),
    );
    assert!(request.validate_for(&declared).is_ok());
}

#[test]
fn assistant_reasoning_history_requires_reasoning_support() {
    let turn = oven_sdk::CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::Reasoning(ReasoningPart::new(
            "thought",
        ))]),
        Finish::new(Default::default(), FinishReason::Stop),
    );
    let request = Request::new(vec![HistoryTurn::assistant(turn)]);
    assert!(matches!(
        request.validate_for(&capabilities(Capability::empty())),
        Err(error) if error.kind() == ModelErrorKind::Unsupported
    ));
    assert!(
        request
            .validate_for(&capabilities(Capability::REASONING))
            .is_ok()
    );
}

#[test]
fn capability_declarations_reject_dependency_mismatches() {
    for features in [Capability::PARALLEL_TOOLS, Capability::TOOL_INPUT_DELTAS] {
        assert!(capabilities(features).validate().is_err());
    }
    let mut declared = capabilities(Capability::REASONING);
    declared.replay = ReplayDeclaration {
        policy: ReplayPolicy::Never,
        capability: ReplayCapability::Required,
        reasoning: true,
    };
    assert!(declared.validate().is_err());
}

#[test]
fn request_validation_rechecks_mutated_capability_consistency() {
    let mut declared = capabilities(Capability::TOOL_CALLING | Capability::PARALLEL_TOOLS);
    declared.features.remove(Capability::TOOL_CALLING);

    assert!(matches!(
        Request::new(Vec::new()).validate_for(&declared),
        Err(error) if error.kind() == ModelErrorKind::InvalidRequest
    ));
}

#[test]
fn replay_policy_capability_matrix_is_complete() {
    for policy in [
        ReplayPolicy::Never,
        ReplayPolicy::IfValid,
        ReplayPolicy::Always,
    ] {
        for capability in [
            ReplayCapability::Unsupported,
            ReplayCapability::Optional,
            ReplayCapability::Required,
        ] {
            let mut declared = capabilities(Capability::empty());
            declared.replay = ReplayDeclaration {
                policy,
                capability,
                reasoning: false,
            };
            let expected_valid = matches!(
                (policy, capability),
                (ReplayPolicy::Never, ReplayCapability::Unsupported)
                    | (ReplayPolicy::IfValid, ReplayCapability::Optional)
                    | (ReplayPolicy::IfValid, ReplayCapability::Required)
                    | (ReplayPolicy::Always, ReplayCapability::Required)
            );
            assert_eq!(
                declared.validate().is_ok(),
                expected_valid,
                "unexpected matrix result for {policy:?}/{capability:?}"
            );
        }
    }
}

#[test]
fn model_capabilities_serde_preserves_complete_media_content() {
    let mut declared = capabilities(Capability::TEMPERATURE);
    declared.modalities.input.insert(Modality::image());
    declared.media = MediaCapabilities {
        input: BTreeMap::from([(
            Modality::image(),
            MediaInputSupport::new(
                ["image/png".to_owned()],
                MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
            )
            .expect("media"),
        )]),
    };
    let encoded = serde_json::to_string(&declared).expect("serialize");
    let decoded: ModelCapabilities = serde_json::from_str(&encoded).expect("deserialize");
    assert_eq!(decoded, declared);
}
