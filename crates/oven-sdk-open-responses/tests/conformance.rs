mod common;

use oven_sdk::{
    Capability, CompactionCapability, CompactionRequest, HistoryTurn, InferenceOptions,
    LanguageModel, ModelErrorKind, NativeReplayArtifact, Request,
};
use oven_sdk_conformance::{
    CapabilityProbe, ModelIdIndependenceProbe, ToolResultFileKind, ToolResultFilePolicy,
    assert_capability_honesty_with, assert_compaction_unsupported_before_io, assert_complete_drain,
    assert_declaration_honesty, assert_invalid_replay_reconstructs, assert_media_honesty,
    assert_model_id_independence_with, assert_replay_artifact, assert_replay_round_trip,
    assert_tool_result_file_policy,
};
use oven_sdk_open_responses::OpenResponsesModel;
use wiremock::MockServer;

#[tokio::test]
async fn generic_and_hugging_face_profiles_pass_core_04_conformance() {
    let server = MockServer::start().await;
    common::mount(&server, common::text_stream("ok")).await;
    let first = common::generic_model(&server, "opaque-a");
    let second = common::generic_model(&server, "opaque-b");
    let hugging_face = common::hugging_face_model(&server, "org/model:provider");
    assert_declaration_honesty(&first).unwrap();
    for kind in [ToolResultFileKind::Image, ToolResultFileKind::Pdf] {
        assert_tool_result_file_policy(&first, kind, ToolResultFilePolicy::Encode).unwrap();
    }
    let mut output = InferenceOptions::new();
    output.max_output_tokens = Some(16);
    assert_capability_honesty_with(
        &first,
        [CapabilityProbe {
            capability: Capability::MAX_OUTPUT_TOKENS,
            name: "MAX_OUTPUT_TOKENS",
            request: Request::new(Vec::new()).with_inference(output),
        }],
    )
    .unwrap();
    assert_media_honesty(&first).unwrap();
    let mut output = InferenceOptions::new();
    output.max_output_tokens = Some(16);
    assert_model_id_independence_with(
        &first,
        &second,
        [ModelIdIndependenceProbe::new(
            "MAX_OUTPUT_TOKENS",
            Request::new(Vec::new()).with_inference(output),
        )],
    )
    .await
    .unwrap();
    assert_declaration_honesty(&hugging_face).unwrap();
    assert_media_honesty(&hugging_face).unwrap();
    assert_tool_result_file_policy(
        &hugging_face,
        ToolResultFileKind::Image,
        ToolResultFilePolicy::Encode,
    )
    .unwrap();
    let completed = assert_complete_drain(&first, Request::new(Vec::new()))
        .await
        .unwrap();
    let scope = completed
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .scope()
        .clone();
    let payload = completed
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .payload()
        .clone();
    assert_eq!(payload["format"], "open.responses.items.v2");
    assert_eq!(
        payload["binding"]["version"],
        "open.responses.native_context_scope.v2"
    );
    assert_replay_artifact(first.descriptor(), &scope, &completed.turn).unwrap();
    assert_replay_round_trip(
        &first,
        &scope,
        Request::new(vec![HistoryTurn::assistant(completed.turn.clone())]),
    )
    .await
    .unwrap();

    let mut old_payload = payload;
    old_payload["format"] = "open.responses.items.v1".into();
    let mut old_turn = completed.turn;
    old_turn.finish.native_replay = Some(
        NativeReplayArtifact::new(
            first.descriptor().adapter_id.clone(),
            scope.clone(),
            old_payload,
        )
        .unwrap(),
    );
    assert_invalid_replay_reconstructs(
        &first,
        &scope,
        Request::new(vec![HistoryTurn::assistant(old_turn)]),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn native_compaction_is_explicitly_unsupported_for_both_transports() {
    let server = MockServer::start().await;
    let generic = common::generic_model(&server, "opaque");
    let hugging_face = common::hugging_face_model(&server, "org/model:provider");
    let request = CompactionRequest::new(Request::new(Vec::new()));
    assert_compaction_unsupported_before_io(&generic, request.clone())
        .await
        .unwrap();
    assert_compaction_unsupported_before_io(&hugging_face, request)
        .await
        .unwrap();
    assert!(server.received_requests().await.unwrap().is_empty());

    for mut config in [
        common::generic_config(&server, "opaque"),
        common::hugging_face_config(&server, "org/model:provider"),
    ] {
        config.model.capabilities.compaction = CompactionCapability::Native;
        let error = match OpenResponsesModel::new(config) {
            Ok(_) => panic!("native compaction declaration unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ModelErrorKind::InvalidRequest);
    }
}
