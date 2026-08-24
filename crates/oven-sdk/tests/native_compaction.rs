use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, BoxFuture, CompactionCapability,
    CompactionRequest, LanguageModel, LanguageModelDescriptor, ModelCapabilities, ModelError,
    ModelErrorKind, ModelId, ModelIdentity, NativeContextScope, NativeContextWindow,
    NativeReplayArtifact, ProviderId, Request, ResourceId, StreamResponse, ToolApprovalPart,
};

fn scope() -> NativeContextScope {
    NativeContextScope::new(
        ProviderId::new("test"),
        ModelId::new("model"),
        ResourceId::new("resource").expect("resource"),
    )
    .expect("scope")
}

fn window() -> NativeContextWindow {
    NativeContextWindow::new(
        AdapterId::new("test.adapter"),
        scope(),
        serde_json::json!({"secret":"opaque-native-context"}),
    )
    .expect("window")
}

struct DefaultModel {
    descriptor: LanguageModelDescriptor,
}

impl LanguageModel for DefaultModel {
    fn descriptor(&self) -> &LanguageModelDescriptor {
        &self.descriptor
    }

    fn stream<'a>(
        &'a self,
        _: Request,
        _: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>> {
        Box::pin(async { Err(ModelError::unsupported("not used")) })
    }
}

#[test]
fn native_context_window_is_bounded_redacted_and_current_shape_only() {
    let context = window();
    let debug = format!("{context:?}");
    assert!(!debug.contains("opaque-native-context"));
    assert_eq!(context.adapter_id(), &AdapterId::new("test.adapter"));
    assert_eq!(context.scope(), &scope());

    let value = serde_json::to_value(&context).expect("serialize");
    let decoded: NativeContextWindow = serde_json::from_value(value.clone()).expect("deserialize");
    assert_eq!(decoded, context);

    let mut unknown = value.clone();
    unknown["legacy"] = serde_json::json!(true);
    assert!(serde_json::from_value::<NativeContextWindow>(unknown).is_err());
    assert!(
        serde_json::from_value::<NativeContextWindow>(serde_json::json!({
            "adapter":"test.adapter",
            "provider_id":"test",
            "model_id":"model",
            "resource_id":"resource",
            "context":{}
        }))
        .is_err()
    );

    let oversized = format!("\"{}\"", "x".repeat(NativeContextWindow::MAX_PAYLOAD_BYTES));
    assert!(
        NativeContextWindow::payload_from_str(AdapterId::new("test.adapter"), scope(), &oversized)
            .is_err()
    );

    let exact_payload =
        serde_json::Value::String("x".repeat(NativeContextWindow::MAX_PAYLOAD_BYTES - 2));
    assert!(
        NativeContextWindow::new(AdapterId::new("test.adapter"), scope(), exact_payload).is_ok()
    );
    let oversized_value = serde_json::json!({
        "adapter_id":"test.adapter",
        "scope": {
            "provider_id":"test",
            "model_id":"model",
            "resource_id":"resource"
        },
        "payload":"x".repeat(NativeContextWindow::MAX_PAYLOAD_BYTES)
    });
    assert!(serde_json::from_value::<NativeContextWindow>(oversized_value).is_err());
}

#[test]
fn native_replay_artifact_rejects_unknown_and_legacy_top_level_shapes() {
    let artifact = NativeReplayArtifact::new(
        AdapterId::new("test.adapter"),
        scope(),
        serde_json::json!({"replay":"current"}),
    )
    .expect("artifact");
    let value = serde_json::to_value(&artifact).expect("serialize");
    let decoded: NativeReplayArtifact =
        serde_json::from_value(value.clone()).expect("deserialize current shape");
    assert_eq!(decoded, artifact);

    let mut unknown = value;
    unknown["legacy"] = serde_json::json!(true);
    assert!(serde_json::from_value::<NativeReplayArtifact>(unknown).is_err());

    assert!(
        serde_json::from_value::<NativeReplayArtifact>(serde_json::json!({
            "adapter_id":"test.adapter",
            "payload":{"replay":"legacy"}
        }))
        .is_err()
    );
}

#[test]
fn compaction_capability_field_is_required_by_serde() {
    let capabilities = ModelCapabilities::conservative();
    let mut value = serde_json::to_value(capabilities).expect("serialize");
    value.as_object_mut().expect("object").remove("compaction");
    let error = serde_json::from_value::<ModelCapabilities>(value).expect_err("required field");
    assert!(error.to_string().contains("missing field `compaction`"));
}

#[test]
fn request_and_compaction_validation_cover_the_complete_capability_matrix() {
    for capability in [
        CompactionCapability::Unsupported,
        CompactionCapability::Native,
    ] {
        let mut capabilities = ModelCapabilities::conservative();
        capabilities.compaction = capability;
        let plain = Request::new(Vec::new());
        let native = plain.clone().with_native_context(window());
        let compaction = CompactionRequest::new(plain.clone());

        assert!(plain.validate_for(&capabilities).is_ok());
        assert_eq!(
            native.validate_for(&capabilities).is_ok(),
            capability == CompactionCapability::Native
        );
        assert_eq!(
            compaction.validate_for(&capabilities).is_ok(),
            capability == CompactionCapability::Native
        );
    }
}

#[test]
fn language_model_defaults_are_object_safe_and_reject_unsupported_compaction() {
    let model = DefaultModel {
        descriptor: LanguageModelDescriptor::new(
            ModelIdentity::new(ProviderId::new("test"), ModelId::new("model")).expect("identity"),
            AdapterId::new("test.adapter"),
            ModelCapabilities::conservative(),
        )
        .expect("descriptor"),
    };
    let object: &dyn LanguageModel = &model;
    let request = Request::new(Vec::new());
    assert!(object.validate_request(&request).is_ok());
    assert!(object.supports_request(&request));

    let compaction = CompactionRequest::new(request);
    assert!(matches!(
        object.validate_compaction(&compaction),
        Err(error) if error.kind() == ModelErrorKind::Unsupported
    ));
    assert!(!object.supports_compaction(&compaction));
}

#[test]
fn native_context_error_uses_the_native_context_kind_and_decode_stage() {
    let error = ModelError::native_context("bad context");
    assert_eq!(error.kind(), ModelErrorKind::NativeContext);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::NativeContextDecode
    );
}

#[test]
fn request_serde_preserves_tool_approval_parts() {
    let turn = oven_sdk::CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::ToolApproval(ToolApprovalPart::new(
            "call-1",
        ))]),
        oven_sdk::Finish::new(Default::default(), oven_sdk::FinishReason::Stop),
    );
    let request = Request::new(vec![oven_sdk::HistoryTurn::assistant(turn)]);
    let encoded = serde_json::to_vec(&request).expect("serialize");
    let decoded: Request = serde_json::from_slice(&encoded).expect("deserialize");
    assert_eq!(decoded, request);
    assert!(matches!(
        &decoded.history[0],
        oven_sdk::HistoryTurn::Assistant(turn)
            if matches!(turn.message.content[0], AssistantPart::ToolApproval(_))
    ));
}
