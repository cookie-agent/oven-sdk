mod common;

use oven_sdk::{
    AdapterId, CompactionRequest, HistoryTurn, NativeContextScope, NativeReplayArtifact, Request,
    ResourceId,
};
use oven_sdk_conformance::{
    ToolResultFileKind, ToolResultFilePolicy, assert_compaction_unsupported_before_io,
    assert_complete_drain, assert_declaration_honesty, assert_foreign_replay_is_reported,
    assert_foreign_replay_scope_is_reported, assert_invalid_replay_reconstructs,
    assert_media_honesty, assert_model_id_independence, assert_replay_artifact,
    assert_replay_round_trip, assert_tool_result_file_policy,
};
use wiremock::MockServer;

#[tokio::test]
async fn core_04_declaration_media_lifecycle_compaction_and_replay_conformance() {
    let server = MockServer::start().await;
    common::mount(&server, common::text_stream("ok")).await;
    let first = common::model(&server, "opaque-a");
    let second = common::model(&server, "opaque-b");
    assert_declaration_honesty(&first).unwrap();
    assert_media_honesty(&first).unwrap();
    assert_tool_result_file_policy(
        &first,
        ToolResultFileKind::Image,
        ToolResultFilePolicy::Reject,
    )
    .unwrap();
    assert_compaction_unsupported_before_io(
        &first,
        CompactionRequest::new(Request::new(Vec::new())),
    )
    .await
    .unwrap();
    assert_model_id_independence(&first, &second).await.unwrap();
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
    assert_replay_artifact(first.descriptor(), &scope, &completed.turn).unwrap();
    assert_replay_round_trip(
        &first,
        &scope,
        Request::new(vec![HistoryTurn::assistant(completed.turn.clone())]),
    )
    .await
    .unwrap();

    let original_payload = completed
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .payload()
        .clone();
    let mut invalid_turn = completed.turn.clone();
    invalid_turn.finish.native_replay = Some(
        NativeReplayArtifact::new(
            first.descriptor().adapter_id.clone(),
            scope.clone(),
            serde_json::json!({"invalid":true}),
        )
        .unwrap(),
    );
    assert_invalid_replay_reconstructs(
        &first,
        &scope,
        Request::new(vec![HistoryTurn::assistant(invalid_turn)]),
    )
    .await
    .unwrap();

    let mut foreign_adapter_turn = completed.turn.clone();
    foreign_adapter_turn.finish.native_replay = Some(
        NativeReplayArtifact::new(
            AdapterId::new("foreign.adapter"),
            scope.clone(),
            original_payload.clone(),
        )
        .unwrap(),
    );
    assert_foreign_replay_is_reported(
        &first,
        &scope,
        Request::new(vec![HistoryTurn::assistant(foreign_adapter_turn)]),
    )
    .await
    .unwrap();

    let foreign_scope = NativeContextScope::new(
        scope.provider_id.clone(),
        scope.model_id.clone(),
        ResourceId::new("foreign.scope").unwrap(),
    )
    .unwrap();
    let mut foreign_scope_turn = completed.turn;
    foreign_scope_turn.finish.native_replay = Some(
        NativeReplayArtifact::new(
            first.descriptor().adapter_id.clone(),
            foreign_scope,
            original_payload,
        )
        .unwrap(),
    );
    assert_foreign_replay_scope_is_reported(
        &first,
        &scope,
        Request::new(vec![HistoryTurn::assistant(foreign_scope_turn)]),
    )
    .await
    .unwrap();
}

use oven_sdk::LanguageModel;
