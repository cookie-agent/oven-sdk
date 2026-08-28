mod common;

use oven_sdk::{
    AbortSignal, CompactionRequest, HistoryTurn, InputPart, LanguageModel, Request, TextPart,
    UserMessage,
};
use oven_sdk_azure::AzureApiRoute;
use oven_sdk_conformance::{
    ToolResultFileKind, ToolResultFilePolicy, UserTurnVideoPolicy, assert_capability_honesty,
    assert_compaction_cancellation, assert_compaction_round_trip,
    assert_compaction_unsupported_before_io, assert_complete_drain, assert_native_compaction,
    assert_replay_artifact, assert_replay_round_trip, assert_stream_lifecycle,
    assert_tool_result_file_policy, assert_user_turn_video_policy,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn chat_and_responses_pass_applicable_core_04_conformance() {
    let chat_server = MockServer::start().await;
    common::mount(
        &chat_server,
        "/openai/v1/chat/completions",
        common::chat_document("ok"),
    )
    .await;
    let chat = common::provider(&chat_server, AzureApiRoute::V1)
        .chat("deployment", common::gpt4o())
        .unwrap();
    assert_capability_honesty(&chat).unwrap();
    assert_tool_result_file_policy(
        &chat,
        ToolResultFileKind::Image,
        ToolResultFilePolicy::Reject,
    )
    .unwrap();
    assert_user_turn_video_policy(&chat, UserTurnVideoPolicy::Reject)
        .await
        .unwrap();
    assert_stream_lifecycle(&chat, Request::new(Vec::new()))
        .await
        .unwrap();
    let completed = assert_complete_drain(&chat, Request::new(Vec::new()))
        .await
        .unwrap();
    let chat_scope = completed
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .scope()
        .clone();
    assert_eq!(
        completed
            .turn
            .finish
            .native_replay
            .as_ref()
            .unwrap()
            .payload()
            .get("format")
            .and_then(serde_json::Value::as_str),
        Some("oven.azure.openai.chat.assistant.v4")
    );
    assert_replay_artifact(chat.descriptor(), &chat_scope, &completed.turn).unwrap();
    assert_replay_round_trip(
        &chat,
        &chat_scope,
        Request::new(vec![HistoryTurn::assistant(completed.turn)]),
    )
    .await
    .unwrap();

    let responses_server = MockServer::start().await;
    common::mount(
        &responses_server,
        "/openai/v1/responses",
        common::responses_document("ok"),
    )
    .await;
    let responses = common::provider(&responses_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap();
    assert_capability_honesty(&responses).unwrap();
    assert_tool_result_file_policy(
        &responses,
        ToolResultFileKind::Image,
        ToolResultFilePolicy::Encode,
    )
    .unwrap();
    assert_user_turn_video_policy(&responses, UserTurnVideoPolicy::Reject)
        .await
        .unwrap();
    assert_tool_result_file_policy(
        &responses,
        ToolResultFileKind::Pdf,
        ToolResultFilePolicy::Reject,
    )
    .unwrap();
    let completed = assert_complete_drain(&responses, Request::new(Vec::new()))
        .await
        .unwrap();
    let responses_scope = completed
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .scope()
        .clone();
    assert_eq!(
        completed
            .turn
            .finish
            .native_replay
            .as_ref()
            .unwrap()
            .payload()
            .get("format")
            .and_then(serde_json::Value::as_str),
        Some("oven.azure.openai.responses.output.v4")
    );
    assert_replay_artifact(responses.descriptor(), &responses_scope, &completed.turn).unwrap();
    assert_replay_round_trip(
        &responses,
        &responses_scope,
        Request::new(vec![HistoryTurn::assistant(completed.turn)]),
    )
    .await
    .unwrap();
    let mut response = responses
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(response.stream.as_mut().next().await.is_some());
}

use futures_util::StreamExt;

#[tokio::test]
async fn responses_v1_native_compaction_passes_core_04_conformance() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses/compact"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id":"resp_compact",
            "object":"response.compaction",
            "created_at":1_764_967_971_u64,
            "output":[
                {
                    "id":"msg_user",
                    "type":"message",
                    "status":"completed",
                    "role":"user",
                    "content":[{"type":"input_text","text":"retained"}]
                },
                {
                    "id":"cmp",
                    "type":"compaction",
                    "encrypted_content":"opaque"
                }
            ],
            "usage":{
                "input_tokens":10,
                "input_tokens_details":{"cached_tokens":0,"cache_write_tokens":1},
                "output_tokens":2,
                "output_tokens_details":{"reasoning_tokens":1},
                "total_tokens":12
            }
        })))
        .mount(&server)
        .await;
    common::mount(
        &server,
        "/openai/v1/responses",
        common::responses_document("continued"),
    )
    .await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    assert_capability_honesty(&model).unwrap();
    let scope = model.native_context_scope().unwrap().clone();
    assert_native_compaction(&model, &scope, nonempty_compaction_request())
        .await
        .unwrap();
    assert_compaction_cancellation(&model, nonempty_compaction_request())
        .await
        .unwrap();
    assert_compaction_round_trip(
        &model,
        &scope,
        nonempty_compaction_request(),
        Request::new(Vec::new()),
    )
    .await
    .unwrap();

    let unsupported = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap();
    assert_compaction_unsupported_before_io(
        &unsupported,
        CompactionRequest::new(Request::new(Vec::new())),
    )
    .await
    .unwrap();
}

fn nonempty_compaction_request() -> CompactionRequest {
    CompactionRequest::new(Request::new(vec![HistoryTurn::user(UserMessage::new(
        vec![InputPart::Text(TextPart::new("compact local context"))],
    ))]))
}
