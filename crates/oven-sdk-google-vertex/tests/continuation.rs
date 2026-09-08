mod support;

use oven_sdk::{
    AbortSignal, AssistantPart, CompletedTurn, HistoryTurn, LanguageModel, ModelErrorKind,
    NativeReplayArtifact, ReplayCapability, Request, ToolContent, ToolMessage, ToolResultPart,
    UserMessage,
};
use oven_sdk_google_vertex::{GoogleVertexModel, GoogleVertexResource};
use serde_json::json;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn history(turn: CompletedTurn) -> Request {
    let call = turn
        .message
        .content
        .iter()
        .find_map(|part| match part {
            AssistantPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .unwrap();
    let result = ToolResultPart::new(call.id.clone(), ToolContent::Text("result".into()));
    Request::new(vec![
        HistoryTurn::assistant(turn),
        HistoryTurn::tool(ToolMessage::new(vec![result])),
    ])
}

#[tokio::test]
async fn required_signature_witness_survives_full_and_partial_calls_and_blocks_missing_or_corrupt_state()
 {
    for partial in [false, true] {
        let server = MockServer::start().await;
        let call = if partial {
            json!({"id":"provider-call","name":"lookup","partialArgs":[{"jsonPath":"$.city","stringValue":"Paris","willContinue":false}],"willContinue":false})
        } else {
            json!({"id":"provider-call","name":"lookup","args":{"city":"Paris"}})
        };
        let body = format!(
            "data: {}\n\n",
            json!({"candidates":[{"content":{"parts":[{"functionCall":call,"thoughtSignature":"opaque-signature"}]},"finishReason":"STOP"}]})
        );
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;
        let mut config = support::full_config(
            &server.uri(),
            "local-label",
            GoogleVertexResource::PublisherModel {
                publisher: "google".into(),
                model: "gemini3-wire".into(),
            },
            partial,
        );
        config.model.capabilities.replay.capability = ReplayCapability::Required;
        let model = GoogleVertexModel::new(config).unwrap();
        let first = model
            .complete(Request::new(vec![]), AbortSignal::default())
            .await
            .unwrap()
            .turn;
        assert!(oven_sdk::replay::has_required_vertex_signature(
            &first.message.content
        ));
        let serialized = serde_json::to_string(&first).unwrap();
        let first: CompletedTurn = serde_json::from_str(&serialized).unwrap();
        let control = model
            .complete(history(first.clone()), AbortSignal::default())
            .await
            .unwrap();
        assert_eq!(
            control.request.replay.decisions[0].disposition,
            oven_sdk::ReplayDisposition::Replayed
        );
        assert_eq!(
            first
                .finish
                .native_replay
                .as_ref()
                .unwrap()
                .source_wire_model_id()
                .unwrap()
                .as_str(),
            "gemini3-wire"
        );
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
        assert_eq!(
            body["contents"][0]["parts"][0]["thoughtSignature"],
            "opaque-signature"
        );
        for damage in ["missing", "changed", "removed"] {
            let mut broken = first.clone();
            let artifact = broken.finish.native_replay.take().unwrap();
            if damage != "missing" {
                let mut payload = artifact.payload().clone();
                if damage == "removed" {
                    payload["content"]["parts"][0]
                        .as_object_mut()
                        .unwrap()
                        .remove("thoughtSignature");
                } else {
                    payload["content"]["parts"][0]["thoughtSignature"] = "changed-signature".into();
                }
                broken.finish.native_replay = Some(
                    NativeReplayArtifact::new(
                        artifact.adapter_id().clone(),
                        artifact.scope().clone(),
                        payload,
                    )
                    .unwrap()
                    .with_source_wire_model_id(artifact.source_wire_model_id().unwrap().clone())
                    .unwrap(),
                );
            }
            let error = model
                .stream(history(broken), AbortSignal::default())
                .await
                .unwrap_err();
            assert_eq!(error.kind, ModelErrorKind::Replay, "{damage}");
            assert!(!error.to_string().contains("opaque-signature"));
            assert_eq!(server.received_requests().await.unwrap().len(), 2);
        }
        let mut past = first;
        past.finish.native_replay = None;
        let mut request = history(past);
        request
            .history
            .push(HistoryTurn::user(UserMessage::new(vec![
                oven_sdk::InputPart::Text(oven_sdk::TextPart::new("next turn")),
            ])));
        assert_eq!(
            model
                .complete(request, AbortSignal::default())
                .await
                .unwrap_err()
                .kind,
            ModelErrorKind::Replay
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
        model
            .complete(
                Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
                    oven_sdk::InputPart::Text(oven_sdk::TextPart::new("new history")),
                ]))]),
                AbortSignal::default(),
            )
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn ordinary_unsigned_calls_do_not_acquire_a_required_continuation_witness() {
    let server = MockServer::start().await;
    let body = format!(
        "data: {}\n\n",
        json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":"call","name":"lookup","args":{}}}]},"finishReason":"STOP"}]})
    );
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;
    let mut config = support::full_config(
        &server.uri(),
        "label",
        GoogleVertexResource::PublisherModel {
            publisher: "google".into(),
            model: "gemini3-wire".into(),
        },
        false,
    );
    config.model.capabilities.replay.capability = ReplayCapability::Required;
    let model = GoogleVertexModel::new(config).unwrap();
    let mut turn = model
        .complete(Request::new(vec![]), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    assert!(!oven_sdk::replay::has_required_vertex_signature(
        &turn.message.content
    ));
    turn.finish.native_replay = None;
    model
        .complete(history(turn), AbortSignal::default())
        .await
        .unwrap();
}

#[tokio::test]
async fn non_streaming_capture_preserves_the_required_signature_witness() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":"call","name":"lookup","args":{}},"thoughtSignature":"opaque"}]},"finishReason":"STOP"}]}))).mount(&server).await;
    let mut config = support::full_config(
        &server.uri(),
        "label",
        GoogleVertexResource::PublisherModel {
            publisher: "google".into(),
            model: "gemini3-wire".into(),
        },
        false,
    );
    config.model.capabilities.replay.capability = ReplayCapability::Required;
    let model = GoogleVertexModel::new(config).unwrap();
    let mut turn = model
        .generate_content(Request::new(vec![]), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    assert!(oven_sdk::replay::has_required_vertex_signature(
        &turn.message.content
    ));
    turn.finish.native_replay = None;
    assert_eq!(
        model
            .generate_content(history(turn), AbortSignal::default())
            .await
            .unwrap_err()
            .kind,
        ModelErrorKind::Replay
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}
