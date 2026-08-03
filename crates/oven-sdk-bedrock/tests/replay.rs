mod support;

use oven_sdk::{
    AbortSignal, AdapterId, AssistantPart, HistoryTurn, LanguageModel, NativeReplayArtifact,
    ReplayDisposition, Request, ToolContent, ToolResultPart,
};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path_regex},
};

#[tokio::test]
async fn signed_reasoning_replays_exactly_and_model_switch_discards() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex("/converse$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output":{"message":{"role":"assistant","content":[
                {"reasoningContent":{"reasoningText":{"text":"think","signature":"signed"}}},
                {"text":"answer"}
            ]}},
            "stopReason":"end_turn",
            "usage":{"inputTokens":1,"outputTokens":2,"totalTokens":3}
        })))
        .mount(&server)
        .await;
    let first_model = support::model(
        &server.uri(),
        "anthropic.claude-sonnet-4-6",
        support::FixtureKind::SignedReasoning,
    );
    let turn = first_model
        .converse(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    assert_eq!(
        turn.finish
            .native_replay
            .as_ref()
            .and_then(|artifact| artifact.payload().get("format")),
        Some(&json!("oven.bedrock.converse.assistant.v2"))
    );

    Mock::given(method("POST"))
        .and(path_regex("/converse-stream$"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("next")))
        .expect(2)
        .mount(&server)
        .await;
    let second_model = support::model(
        &server.uri(),
        "anthropic.claude-sonnet-4-6",
        support::FixtureKind::SignedReasoning,
    );
    let report = second_model
        .stream(
            Request::new(vec![HistoryTurn::assistant(turn.clone())]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        report.request.replay.decisions.as_slice(),
        [decision] if decision.disposition == ReplayDisposition::Replayed
    ));
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        body.pointer("/messages/0/content/0/reasoningContent/reasoningText/signature"),
        Some(&json!("signed"))
    );

    let switched = support::model(
        &server.uri(),
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        support::FixtureKind::SignedReasoning,
    );
    let error_or_response = switched
        .stream(
            Request::new(vec![HistoryTurn::assistant(turn)]),
            AbortSignal::default(),
        )
        .await;
    assert!(error_or_response.is_ok());
    let response = error_or_response.unwrap();
    assert!(matches!(
        response.request.replay.decisions.as_slice(),
        [first, second]
            if matches!(first.disposition, ReplayDisposition::DiscardedForeignScope { .. })
                && second.disposition == ReplayDisposition::ReconstructedNormalized
    ));
}

#[tokio::test]
async fn malformed_same_model_replay_is_discarded_and_reconstructed_safely() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex("/converse$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output":{"message":{"role":"assistant","content":[
                {"reasoningContent":{"reasoningText":{"text":"think","signature":"signed"}}},
                {"text":"answer"}
            ]}},
            "stopReason":"end_turn"
        })))
        .mount(&server)
        .await;
    let model = support::model(
        &server.uri(),
        "anthropic.claude-sonnet-4-6",
        support::FixtureKind::SignedReasoning,
    );
    let mut turn = model
        .converse(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    turn.finish.native_replay = Some(
        NativeReplayArtifact::new(
            AdapterId::new(oven_sdk_bedrock::BEDROCK_CONVERSE_ADAPTER_ID),
            model.native_context_scope().clone(),
            json!({
                "format":"oven.bedrock.converse.assistant.v2",
                "assistant_content":[{
                    "reasoningContent":{
                        "reasoningText":{"text":"think","signature":"signed"},
                        "redactedContent":"ambiguous"
                    }
                },{"text":"answer"}]
            }),
        )
        .unwrap(),
    );

    Mock::given(method("POST"))
        .and(path_regex("/converse-stream$"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("next")))
        .mount(&server)
        .await;
    let response = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        response.request.replay.decisions.as_slice(),
        [first, second]
            if matches!(first.disposition, ReplayDisposition::DiscardedInvalidPayload { .. })
                && second.disposition == ReplayDisposition::ReconstructedNormalized
    ));
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests.last().unwrap().body).unwrap();
    assert_eq!(
        body.pointer("/messages/0/content"),
        Some(&json!([{"text":"answer"}]))
    );
}

#[tokio::test]
async fn successful_native_replay_preserves_inline_tool_results() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex("/converse$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output":{"message":{"role":"assistant","content":[{
                "toolUse":{"toolUseId":"call-1","name":"lookup","input":{"key":"oven"}}
            }]}},
            "stopReason":"tool_use"
        })))
        .mount(&server)
        .await;
    let model = support::model(
        &server.uri(),
        "opaque-tool-model",
        support::FixtureKind::SignedReasoning,
    );
    let mut turn = model
        .converse(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    turn.message
        .content
        .push(AssistantPart::ToolResult(ToolResultPart::new(
            "call-1",
            ToolContent::Text("result".into()),
        )));

    Mock::given(method("POST"))
        .and(path_regex("/converse-stream$"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("next")))
        .mount(&server)
        .await;
    let response = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        response.request.replay.decisions.as_slice(),
        [decision] if decision.disposition == ReplayDisposition::Replayed
    ));
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&requests.last().expect("stream request").body).unwrap();
    assert_eq!(
        body.pointer("/messages"),
        Some(&json!([
            {"role":"assistant","content":[{
                "toolUse":{"toolUseId":"call-1","name":"lookup","input":{"key":"oven"}}
            }]},
            {"role":"user","content":[{
                "toolResult":{
                    "toolUseId":"call-1",
                    "content":[{"text":"result"}],
                    "status":"success"
                }
            }]}
        ]))
    );
}
