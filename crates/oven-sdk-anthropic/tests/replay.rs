use std::{collections::BTreeMap, ops::Deref, time::Duration};

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, CompletedTurn, ErrorStage, Finish,
    FinishReason, HistoryTurn, LanguageModel, ModelErrorKind, NativeContextScope,
    NativeReplayArtifact, ReasoningPart, ReplayDisposition, ReplayPolicy, Request, ResourceId,
    StreamPart, TextPart, UserMessage,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

const HEADERS: &str =
    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";

fn event(name: &str, data: &str) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

struct ScriptedModel {
    model: oven_sdk_anthropic::AnthropicModel,
    scope: NativeContextScope,
}

impl Deref for ScriptedModel {
    type Target = oven_sdk_anthropic::AnthropicModel;

    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

async fn scripted_model(body: String) -> ScriptedModel {
    scripted_model_with_policy(body, ReplayPolicy::IfValid).await
}

async fn scripted_model_with_policy(body: String, replay_policy: ReplayPolicy) -> ScriptedModel {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let model = Anthropic::builder()
        .base_url(&endpoint)
        .replay_policy(replay_policy)
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 4096];
        if socket.read(&mut request).await.unwrap() == 0 {
            return;
        }
        socket.write_all(HEADERS.as_bytes()).await.unwrap();
        for chunk in body.as_bytes().chunks(32 * 1024) {
            socket
                .write_all(format!("{:X}\r\n", chunk.len()).as_bytes())
                .await
                .unwrap();
            socket.write_all(chunk).await.unwrap();
            socket.write_all(b"\r\n").await.unwrap();
        }
        socket.write_all(b"0\r\n\r\n").await.unwrap();
        tokio::time::sleep(Duration::from_millis(250)).await;
    });
    ScriptedModel {
        model,
        scope: common::expected_native_context_scope(
            "anthropic",
            "oven.anthropic.messages",
            &endpoint,
            "claude-sonnet-4-5",
            None,
            None,
        ),
    }
}

fn terminal_response() -> String {
    [
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        event(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{}}"#,
        ),
        event("message_stop", r#"{"type":"message_stop"}"#),
    ]
    .concat()
}

fn assistant(text: &str, replay: Option<NativeReplayArtifact>) -> HistoryTurn {
    let mut finish = Finish::new(Default::default(), FinishReason::Stop);
    finish.native_replay = replay;
    HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::Text(TextPart::new(text))]),
        finish,
    ))
}

fn artifact(text: &str, scope: &NativeContextScope) -> NativeReplayArtifact {
    NativeReplayArtifact::new(
        AdapterId::new("oven.anthropic.messages"),
        scope.clone(),
        serde_json::json!({
            "format": "oven.anthropic.messages.assistant.v3",
            "message": {"role": "assistant", "content": [{"type":"text", "text":text}]},
            "stop_reason": "end_turn",
            "stop_sequence": null
        }),
    )
    .unwrap()
}

fn reasoning_assistant(replay: Option<NativeReplayArtifact>) -> HistoryTurn {
    let mut finish = Finish::new(Default::default(), FinishReason::Stop);
    finish.native_replay = replay;
    HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(vec![
            AssistantPart::Reasoning(ReasoningPart::new("reason")),
            AssistantPart::Text(TextPart::new("answer")),
        ]),
        finish,
    ))
}

#[test]
fn replay_artifacts_are_bounded_and_redacted() {
    let scope = NativeContextScope::new(
        oven_sdk::ProviderId::new("anthropic"),
        oven_sdk::ModelId::new("model"),
        ResourceId::new("resource").unwrap(),
    )
    .unwrap();
    let artifact = NativeReplayArtifact::new(
        AdapterId::new("oven.anthropic.messages"),
        scope,
        serde_json::json!({"format":"oven.anthropic.messages.assistant.v3"}),
    )
    .unwrap();
    assert!(format!("{artifact:?}").contains("<redacted>"));
}

#[tokio::test]
async fn modified_assistant_content_discards_semantically_mismatched_replay() {
    let model = scripted_model(terminal_response()).await;
    let request = Request::new(vec![
        HistoryTurn::user(UserMessage::new(Vec::new())),
        assistant("modified", Some(artifact("original", &model.scope))),
    ]);
    let response = model.stream(request, AbortSignal::default()).await.unwrap();
    assert_eq!(response.request.replay.decisions.len(), 2);
    assert!(matches!(
        response.request.replay.decisions[0].disposition,
        ReplayDisposition::DiscardedInvalidPayload { .. }
    ));
    assert_eq!(
        response.request.replay.decisions[1].disposition,
        ReplayDisposition::ReconstructedNormalized
    );
}

#[tokio::test]
async fn replay_comparison_preserves_trailing_newlines_and_ignores_only_signature() {
    let model = scripted_model(terminal_response()).await;
    let replay = NativeReplayArtifact::new(
        AdapterId::new("oven.anthropic.messages"),
        model.scope.clone(),
        serde_json::json!({
            "format": "oven.anthropic.messages.assistant.v3",
            "message": {"role": "assistant", "content": [
                {"type":"thinking", "thinking":"reason\n", "signature":"signed-value"},
                {"type":"text", "text":"answer\n"}
            ]},
            "stop_reason": "end_turn",
            "stop_sequence": null
        }),
    )
    .unwrap();
    let mut finish = Finish::new(Default::default(), FinishReason::Stop);
    finish.native_replay = Some(replay);
    let turn = HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(vec![
            AssistantPart::Reasoning(ReasoningPart::new("reason\n")),
            AssistantPart::Text(TextPart::new("answer\n")),
        ]),
        finish,
    ));
    let response = model
        .stream(
            Request::new(vec![turn, HistoryTurn::user(UserMessage::new(Vec::new()))]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_eq!(response.request.replay.decisions.len(), 1);
    assert_eq!(
        response.request.replay.decisions[0].disposition,
        ReplayDisposition::Replayed
    );
}

#[tokio::test]
async fn oversized_replay_capture_fails_before_finish() {
    let text = "x".repeat(NativeReplayArtifact::MAX_PAYLOAD_BYTES + 1024);
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#),
            event("content_block_delta", &format!(r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"{text}"}}}}"#)),
            event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
            event("message_stop", r#"{"type":"message_stop"}"#),
        ]
        .concat(),
    )
    .await;
    let mut response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut finishes = 0;
    let mut error = None;
    while let Some(item) = response.stream.next().await {
        match item {
            Ok(StreamPart::Finish { .. }) => finishes += 1,
            Ok(_) => {}
            Err(value) => error = Some(value),
        }
    }
    assert_eq!(finishes, 0);
    let error = error.expect("oversized replay capture should fail the stream");
    assert_eq!(error.kind, ModelErrorKind::Replay);
    assert_eq!(error.diagnostics.stage, ErrorStage::ReplayEncode);
}

#[tokio::test]
async fn replay_decision_log_records_every_assistant_history_turn() {
    let model = scripted_model(terminal_response()).await;
    let request = Request::new(vec![
        HistoryTurn::user(UserMessage::new(Vec::new())),
        assistant("missing", None),
        HistoryTurn::user(UserMessage::new(Vec::new())),
        assistant("replayed", Some(artifact("replayed", &model.scope))),
    ]);
    let response = model.stream(request, AbortSignal::default()).await.unwrap();
    let decisions = response.request.replay.decisions;
    assert_eq!(decisions.len(), 3);
    assert_eq!(decisions[0].history_index, 1);
    assert_eq!(decisions[0].disposition, ReplayDisposition::NoArtifact);
    assert_eq!(decisions[1].history_index, 1);
    assert_eq!(
        decisions[1].disposition,
        ReplayDisposition::ReconstructedNormalized
    );
    assert_eq!(decisions[2].history_index, 3);
    assert_eq!(decisions[2].disposition, ReplayDisposition::Replayed);
}

#[tokio::test]
async fn never_replay_reports_only_reconstructed_without_inspecting_artifacts() {
    let model = scripted_model_with_policy(terminal_response(), ReplayPolicy::Never).await;
    let foreign = NativeReplayArtifact::new(
        AdapterId::new("foreign.adapter"),
        model.scope.clone(),
        serde_json::json!({"foreign": true}),
    )
    .unwrap();
    let invalid = NativeReplayArtifact::new(
        AdapterId::new("oven.anthropic.messages"),
        model.scope.clone(),
        serde_json::json!("garbage"),
    )
    .unwrap();
    let request = Request::new(vec![
        assistant("valid", Some(artifact("valid", &model.scope))),
        HistoryTurn::user(UserMessage::new(Vec::new())),
        assistant("foreign", Some(foreign)),
        HistoryTurn::user(UserMessage::new(Vec::new())),
        assistant("invalid", Some(invalid)),
        HistoryTurn::user(UserMessage::new(Vec::new())),
        assistant("missing", None),
    ]);
    let response = model.stream(request, AbortSignal::default()).await.unwrap();
    assert_eq!(response.request.replay.decisions.len(), 4);
    assert_eq!(
        response
            .request
            .replay
            .decisions
            .iter()
            .map(|decision| (decision.history_index, &decision.disposition))
            .collect::<Vec<_>>(),
        vec![
            (0, &ReplayDisposition::ReconstructedNormalized),
            (2, &ReplayDisposition::ReconstructedNormalized),
            (4, &ReplayDisposition::ReconstructedNormalized),
            (6, &ReplayDisposition::ReconstructedNormalized),
        ]
    );
}

#[tokio::test]
async fn unsigned_reasoning_replays_byte_faithfully() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(terminal_response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let scope = common::expected_native_context_scope(
        "anthropic",
        "oven.anthropic.messages",
        &server.uri(),
        "claude-sonnet-4-5",
        None,
        None,
    );
    let unsigned = NativeReplayArtifact::new(
        AdapterId::new("oven.anthropic.messages"),
        scope,
        serde_json::json!({
            "format":"oven.anthropic.messages.assistant.v3",
            "message":{"role":"assistant","content":[
                {"type":"thinking","thinking":"reason","signature":""},
                {"type":"text","text":"answer"}
            ]},
            "stop_reason":"end_turn",
            "stop_sequence":null
        }),
    )
    .unwrap();

    let response = model
        .stream(
            Request::new(vec![
                reasoning_assistant(Some(unsigned)),
                HistoryTurn::user(UserMessage::new(Vec::new())),
            ]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.request.replay.decisions[0].disposition,
        ReplayDisposition::Replayed
    );

    let request = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(
        body["messages"][0]["content"],
        serde_json::json!([
            {"type":"thinking","thinking":"reason","signature":""},
            {"type":"text","text":"answer"}
        ])
    );
}

#[tokio::test]
async fn replay_model_switch_discards_native_continuity_and_reconstructs_safely() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(terminal_response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-4-5");
    let foreign_scope = common::expected_native_context_scope(
        "anthropic",
        "oven.anthropic.messages",
        &server.uri(),
        "claude-sonnet-4-5",
        None,
        None,
    );
    let response = model
        .stream(
            Request::new(vec![
                assistant("answer", Some(artifact("answer", &foreign_scope))),
                HistoryTurn::user(UserMessage::new(Vec::new())),
            ]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        response.request.replay.decisions[0].disposition,
        ReplayDisposition::DiscardedForeignScope { .. }
    ));
    assert_eq!(
        response.request.replay.decisions[1].disposition,
        ReplayDisposition::ReconstructedNormalized
    );
    let request = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["messages"][0]["content"][0]["text"], "answer");
}

#[tokio::test]
async fn native_context_resource_switch_is_reported_as_foreign_scope() {
    let model = scripted_model(terminal_response()).await;
    let mut foreign_scope = model.scope.clone();
    foreign_scope.resource_id = ResourceId::new("different-endpoint-resource").unwrap();
    let foreign = NativeReplayArtifact::new(
        AdapterId::new("oven.anthropic.messages"),
        foreign_scope,
        serde_json::json!({
            "format": "oven.anthropic.messages.assistant.v3",
            "message": {"role": "assistant", "content": [{"type":"text", "text":"answer"}]},
            "stop_reason": "end_turn",
            "stop_sequence": null
        }),
    )
    .unwrap();
    let response = model
        .stream(
            Request::new(vec![
                assistant("answer", Some(foreign)),
                HistoryTurn::user(UserMessage::new(Vec::new())),
            ]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        response.request.replay.decisions[0].disposition,
        ReplayDisposition::DiscardedForeignScope { .. }
    ));
}

#[tokio::test]
async fn native_context_resource_is_versioned_hashed_and_endpoint_bound() {
    let first_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(terminal_response()))
        .mount(&first_server)
        .await;
    let second_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(terminal_response()))
        .mount(&second_server)
        .await;

    async fn capture(endpoint: String, discriminator: &str) -> NativeContextScope {
        Anthropic::builder()
            .base_url(endpoint)
            .native_context_discriminator(discriminator)
            .build()
            .unwrap()
            .model("same-model")
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap()
            .turn
            .finish
            .native_replay
            .unwrap()
            .scope()
            .clone()
    }

    let canonical = capture(first_server.uri(), "tenant-a").await;
    let trailing_slash = capture(format!("{}/", first_server.uri()), "tenant-a").await;
    let other_endpoint = capture(second_server.uri(), "tenant-a").await;
    let other_discriminator = capture(first_server.uri(), "tenant-b").await;

    assert_eq!(canonical, trailing_slash);
    assert_ne!(canonical.resource_id, other_endpoint.resource_id);
    assert_ne!(canonical.resource_id, other_discriminator.resource_id);
    assert!(
        canonical
            .resource_id
            .as_str()
            .starts_with("anthropic-context-v1-")
    );
    assert!(!canonical.resource_id.as_str().contains("tenant-a"));
    assert!(!canonical.resource_id.as_str().contains("127.0.0.1"));
}

#[tokio::test]
async fn exact_same_model_redacted_reasoning_replays_authoritative_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(terminal_response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let scope = common::expected_native_context_scope(
        "anthropic",
        "oven.anthropic.messages",
        &server.uri(),
        "claude-sonnet-4-5",
        None,
        None,
    );
    let replay = NativeReplayArtifact::new(
        AdapterId::new("oven.anthropic.messages"),
        scope,
        serde_json::json!({
            "format":"oven.anthropic.messages.assistant.v3",
            "message":{"role":"assistant","content":[
                {"type":"redacted_thinking","data":"opaque"},
                {"type":"text","text":"answer"}
            ]},
            "stop_reason":"end_turn",
            "stop_sequence":null
        }),
    )
    .unwrap();
    let reasoning = ReasoningPart {
        text: String::new(),
        metadata: Some(BTreeMap::from([(
            "anthropic.redacted".into(),
            serde_json::json!("opaque"),
        )])),
    };
    let mut finish = Finish::new(Default::default(), FinishReason::Stop);
    finish.native_replay = Some(replay);
    let turn = HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(vec![
            AssistantPart::Reasoning(reasoning),
            AssistantPart::Text(TextPart::new("answer")),
        ]),
        finish,
    ));
    let response = model
        .stream(
            Request::new(vec![turn, HistoryTurn::user(UserMessage::new(Vec::new()))]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.request.replay.decisions[0].disposition,
        ReplayDisposition::Replayed
    );
    let request = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][0],
        serde_json::json!({"type":"redacted_thinking","data":"opaque"})
    );
}
mod common;

use common::Anthropic;
