// Conformance probe signatures return the core ModelError by value.
#![allow(clippy::result_large_err)]

pub mod common;

use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, CompactionRequest, CompletedTurn,
    Finish, FinishReason, HistoryTurn, LanguageModel, NativeReplayArtifact, Request, TextPart,
};
use oven_sdk_conformance::{
    assert_capability_honesty, assert_compaction_unsupported_before_io, assert_complete_drain,
    assert_foreign_replay_is_reported, assert_invalid_replay_reconstructs, assert_replay_artifact,
    assert_replay_round_trip, assert_stream_contract, assert_stream_lifecycle,
};
use wiremock::MockServer;

fn completed(artifact: NativeReplayArtifact) -> CompletedTurn {
    let mut finish = Finish::new(Default::default(), FinishReason::Stop);
    finish.native_replay = Some(artifact);
    CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::Text(TextPart::new("ok"))]),
        finish,
    )
}

#[tokio::test]
async fn official_chat_passes_lifecycle_complete_capability_and_replay_suites() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    assert_capability_honesty(&model).unwrap();
    assert_stream_lifecycle(&model, Request::new(Vec::new()))
        .await
        .unwrap();
    let completed = assert_complete_drain(&model, Request::new(Vec::new()))
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
    assert_replay_artifact(model.descriptor(), &scope, &completed.turn).unwrap();
    assert_replay_round_trip(
        &model,
        &scope,
        Request::new(vec![HistoryTurn::assistant(completed.turn)]),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn official_responses_passes_lifecycle_complete_capability_and_replay_suites() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = common::official_responses(&server, "gpt-5-mini");
    assert_capability_honesty(&model).unwrap();
    let response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_stream_contract(response.stream).await.unwrap();
    let completed = assert_complete_drain(&model, Request::new(Vec::new()))
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
    assert_replay_artifact(model.descriptor(), &scope, &completed.turn).unwrap();
    assert_replay_round_trip(
        &model,
        &scope,
        Request::new(vec![HistoryTurn::assistant(completed.turn)]),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn explicitly_configured_compatible_chat_passes_capability_honesty() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::compatible(&server);
    assert_capability_honesty(&model).unwrap();
    assert_complete_drain(&model, Request::new(Vec::new()))
        .await
        .unwrap();
}

#[tokio::test]
async fn unsupported_compaction_is_rejected_before_io_for_all_non_native_surfaces() {
    let server = MockServer::start().await;
    let request = CompactionRequest::new(Request::new(Vec::new()));
    assert_compaction_unsupported_before_io(
        &common::official_chat(&server, "gpt-4o-mini"),
        request.clone(),
    )
    .await
    .unwrap();
    assert_compaction_unsupported_before_io(
        &common::official_responses(&server, "gpt-5-mini"),
        request.clone(),
    )
    .await
    .unwrap();
    assert_compaction_unsupported_before_io(&common::compatible(&server), request)
        .await
        .unwrap();
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn invalid_and_foreign_replay_conformance_sequences_continue() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let scope = first
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .scope()
        .clone();
    let invalid = NativeReplayArtifact::new(
        AdapterId::new("oven.openai.chat"),
        scope.clone(),
        serde_json::Value::String("garbage".into()),
    )
    .unwrap();
    assert_invalid_replay_reconstructs(
        &model,
        &scope,
        Request::new(vec![HistoryTurn::assistant(completed(invalid))]),
    )
    .await
    .unwrap();
    let foreign = NativeReplayArtifact::new(
        AdapterId::new("foreign"),
        scope.clone(),
        serde_json::json!({}),
    )
    .unwrap();
    assert_foreign_replay_is_reported(
        &model,
        &scope,
        Request::new(vec![HistoryTurn::assistant(completed(foreign))]),
    )
    .await
    .unwrap();
}
