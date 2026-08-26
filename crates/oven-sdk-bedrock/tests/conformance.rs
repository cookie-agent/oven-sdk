mod support;

use oven_sdk::{
    AbortSignal, AdapterId, Capability, CompactionCapability, CompactionRequest, HistoryTurn,
    LanguageModel, MediaInputSupport, MediaSourceSupport, Modality, NativeContextScope,
    NativeReplayArtifact, Request,
};
use oven_sdk_bedrock::{BedrockAuth, BedrockModel};
use oven_sdk_conformance::{
    assert_capability_honesty, assert_compaction_unsupported_before_io, assert_complete_drain,
    assert_declaration_honesty, assert_foreign_replay_is_reported,
    assert_foreign_replay_scope_is_reported, assert_invalid_replay_reconstructs,
    assert_media_honesty, assert_model_id_independence, assert_replay_round_trip,
    assert_stream_lifecycle,
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

#[tokio::test]
async fn bedrock_passes_lifecycle_complete_and_capability_conformance() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("ok")))
        .expect(4)
        .mount(&server)
        .await;
    let model = support::model(
        &server.uri(),
        "anthropic.claude-sonnet-4-6",
        support::FixtureKind::SignedReasoning,
    );
    assert_declaration_honesty(&model).unwrap();
    assert_capability_honesty(&model).unwrap();
    assert_media_honesty(&model).unwrap();
    assert_stream_lifecycle(&model, Request::new(Vec::new()))
        .await
        .unwrap();
    assert_complete_drain(&model, Request::new(Vec::new()))
        .await
        .unwrap();
    assert_compaction_unsupported_before_io(
        &model,
        CompactionRequest::new(Request::new(Vec::new())),
    )
    .await
    .unwrap();
    let result = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "ok");
}

#[tokio::test]
async fn bedrock_passes_native_replay_conformance_sequences() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("ok")))
        .mount(&server)
        .await;
    let model = support::model(
        &server.uri(),
        "anthropic.claude-sonnet-4-6",
        support::FixtureKind::SignedReasoning,
    );
    let turn = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    assert_replay_round_trip(
        &model,
        model.native_context_scope(),
        Request::new(vec![HistoryTurn::assistant(turn.clone())]),
    )
    .await
    .unwrap();

    let mut invalid = turn.clone();
    invalid.finish.native_replay = Some(
        NativeReplayArtifact::new(
            AdapterId::new("oven.bedrock.converse"),
            model.native_context_scope().clone(),
            serde_json::Value::String("garbage".into()),
        )
        .unwrap(),
    );
    assert_invalid_replay_reconstructs(
        &model,
        model.native_context_scope(),
        Request::new(vec![HistoryTurn::assistant(invalid)]),
    )
    .await
    .unwrap();

    let mut foreign_scope = turn.clone();
    let original_payload = foreign_scope
        .finish
        .native_replay
        .as_ref()
        .expect("captured replay")
        .payload()
        .clone();
    foreign_scope.finish.native_replay = Some(
        NativeReplayArtifact::new(
            AdapterId::new("oven.bedrock.converse"),
            NativeContextScope::new(
                oven_sdk::ProviderId::new(oven_sdk_bedrock::BEDROCK_PROVIDER_ID),
                oven_sdk::ModelId::new("different-model"),
                oven_sdk::ResourceId::new("different-resource").unwrap(),
            )
            .unwrap(),
            original_payload,
        )
        .unwrap(),
    );
    assert_foreign_replay_scope_is_reported(
        &model,
        model.native_context_scope(),
        Request::new(vec![HistoryTurn::assistant(foreign_scope)]),
    )
    .await
    .unwrap();

    let mut foreign = turn;
    foreign.finish.native_replay = Some(
        NativeReplayArtifact::new(
            AdapterId::new("foreign.adapter"),
            model.native_context_scope().clone(),
            serde_json::json!({"opaque":true}),
        )
        .unwrap(),
    );
    assert_foreign_replay_is_reported(
        &model,
        model.native_context_scope(),
        Request::new(vec![HistoryTurn::assistant(foreign)]),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn bedrock_model_ids_do_not_select_behavior() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("ok")))
        .mount(&server)
        .await;
    let endpoint = server.uri();
    let anthropic_looking = support::model(
        &endpoint,
        "anthropic.this-name-must-not-route",
        support::FixtureKind::UnsignedReasoning,
    );
    let neutral = support::model(
        &endpoint,
        "opaque-provider-resource",
        support::FixtureKind::UnsignedReasoning,
    );
    assert_model_id_independence(&anthropic_looking, &neutral)
        .await
        .unwrap();
}

#[test]
fn bedrock_constructor_enforces_adapter_declaration_and_media_ceilings() {
    let endpoint = "https://bedrock-runtime.us-east-1.amazonaws.com";
    let mut config = support::config(
        endpoint,
        "opaque",
        support::FixtureKind::Text,
        BedrockAuth::Static(support::credentials()),
    );
    config.model.capabilities.features |= Capability::PROVIDER_TOOLS;
    assert!(BedrockModel::new(config).is_err());

    let mut native_compaction = support::config(
        endpoint,
        "opaque",
        support::FixtureKind::Text,
        BedrockAuth::Static(support::credentials()),
    );
    native_compaction.model.capabilities.compaction = CompactionCapability::Native;
    assert!(BedrockModel::new(native_compaction).is_err());

    let mut audio = support::config(
        endpoint,
        "opaque",
        support::FixtureKind::Text,
        BedrockAuth::Static(support::credentials()),
    );
    audio
        .model
        .capabilities
        .modalities
        .input
        .insert(Modality::audio());
    audio.model.capabilities.media.input.insert(
        Modality::audio(),
        MediaInputSupport::new(["audio/mpeg".into()], MediaSourceSupport::INLINE_BYTES).unwrap(),
    );
    assert!(BedrockModel::new(audio).is_err());

    let mut bmp = support::config(
        endpoint,
        "opaque",
        support::FixtureKind::MediaTools,
        BedrockAuth::Static(support::credentials()),
    );
    bmp.model
        .capabilities
        .media
        .input
        .get_mut(&Modality::image())
        .unwrap()
        .media_types
        .push("image/bmp".into());
    assert!(BedrockModel::new(bmp).is_err());

    let mut provider_reference = support::config(
        endpoint,
        "opaque",
        support::FixtureKind::MediaTools,
        BedrockAuth::Static(support::credentials()),
    );
    provider_reference
        .model
        .capabilities
        .media
        .input
        .get_mut(&Modality::image())
        .unwrap()
        .sources |= MediaSourceSupport::PROVIDER_REFERENCE;
    assert!(BedrockModel::new(provider_reference).is_err());

    let mut generic_url = support::config(
        endpoint,
        "opaque",
        support::FixtureKind::MediaTools,
        BedrockAuth::Static(support::credentials()),
    );
    generic_url
        .model
        .capabilities
        .media
        .input
        .get_mut(&Modality::image())
        .unwrap()
        .sources |= MediaSourceSupport::URL;
    assert!(BedrockModel::new(generic_url).is_err());

    let mut inline_text_image = support::config(
        endpoint,
        "opaque",
        support::FixtureKind::MediaTools,
        BedrockAuth::Static(support::credentials()),
    );
    inline_text_image
        .model
        .capabilities
        .media
        .input
        .get_mut(&Modality::image())
        .unwrap()
        .sources |= MediaSourceSupport::INLINE_TEXT;
    assert!(BedrockModel::new(inline_text_image).is_err());

    let mut missing_rules = support::config(
        endpoint,
        "opaque",
        support::FixtureKind::Text,
        BedrockAuth::Static(support::credentials()),
    );
    missing_rules
        .model
        .capabilities
        .modalities
        .input
        .insert(Modality::image());
    assert!(BedrockModel::new(missing_rules).is_err());

    let mut document = support::config(
        endpoint,
        "opaque",
        support::FixtureKind::Text,
        BedrockAuth::Static(support::credentials()),
    );
    document
        .model
        .capabilities
        .modalities
        .input
        .insert(Modality::pdf());
    document.model.capabilities.media.input.insert(
        Modality::pdf(),
        MediaInputSupport::new(["application/pdf".into()], MediaSourceSupport::INLINE_BYTES)
            .unwrap(),
    );
    assert!(BedrockModel::new(document).is_err());

    let mut image_output = support::config(
        endpoint,
        "opaque",
        support::FixtureKind::Text,
        BedrockAuth::Static(support::credentials()),
    );
    image_output
        .model
        .capabilities
        .modalities
        .output
        .insert(Modality::image());
    assert!(BedrockModel::new(image_output).is_err());
}
