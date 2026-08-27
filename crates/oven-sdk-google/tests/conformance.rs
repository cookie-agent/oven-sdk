use oven_sdk::{AbortSignal, CompactionRequest, LanguageModel, Modality, Request, StreamPart};
use oven_sdk_conformance::{
    ToolResultFileKind, ToolResultFilePolicy, assert_capability_honesty,
    assert_compaction_unsupported_before_io, assert_complete_drain, assert_declaration_honesty,
    assert_replay_artifact, assert_replay_round_trip, assert_stream_lifecycle,
    assert_tool_result_file_policy,
};
use serde_json::json;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

mod common;
use common::model;

#[tokio::test]
async fn lifecycle_complete_and_capabilities_conform() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!("data: {}\n\n", json!({
                    "candidates":[{"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}],
                    "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}
                }))),
        )
        .mount(&server)
        .await;
    let model = model(server.uri(), "gemini-2.5-flash");
    let report = assert_stream_lifecycle(&model, Request::new(Vec::new()))
        .await
        .unwrap();
    assert!(matches!(
        report.stream.parts.last(),
        Some(StreamPart::Finish { .. })
    ));
    assert_complete_drain(&model, Request::new(Vec::new()))
        .await
        .unwrap();
    assert!(
        model
            .capabilities()
            .modalities
            .input
            .contains(&Modality::audio())
    );
    assert_capability_honesty(&model).unwrap();
    assert_declaration_honesty(&model).unwrap();
    for kind in [ToolResultFileKind::Image, ToolResultFileKind::Pdf] {
        assert_tool_result_file_policy(&model, kind, ToolResultFilePolicy::Reject).unwrap();
    }
    assert_compaction_unsupported_before_io(
        &model,
        CompactionRequest::new(Request::new(Vec::new())),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn native_replay_round_trip_is_reported() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!("data: {}\n\n", json!({
                    "candidates":[{"content":{"parts":[{"text":"remember"}]},"finishReason":"STOP"}]
                }))),
        )
        .mount(&server)
        .await;
    let model = model(server.uri(), "gemini-2.5-flash");
    let turn = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    assert_replay_artifact(model.descriptor(), model.native_context_scope(), &turn).unwrap();
    let request = Request::new(vec![oven_sdk::HistoryTurn::assistant(turn)]);
    assert_replay_round_trip(&model, model.native_context_scope(), request)
        .await
        .unwrap();
}

#[tokio::test]
async fn pre_dispatch_abort_is_structured() {
    let model = model(
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini-2.5-flash",
    );
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    let error = model
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap_err();
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::Abort);
}

#[tokio::test]
async fn mid_stream_abort_emits_one_fatal_error_then_eof() {
    use futures_util::StreamExt as _;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0_u8; 4096];
        let _ = socket.read(&mut request).await.unwrap();
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
            )
            .await
            .unwrap();
        let event = format!(
            "data: {}\n\n",
            json!({"candidates":[{"content":{"parts":[{"text":"started"}]}}]})
        );
        socket
            .write_all(format!("{:X}\r\n{}\r\n", event.len(), event).as_bytes())
            .await
            .unwrap();
        socket.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    });
    let model = model(format!("http://{address}"), "gemini-2.5-flash");
    let (signal, registration) = AbortSignal::new();
    let mut response = model
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap();
    while let Some(item) = response.stream.next().await {
        if matches!(item, Ok(StreamPart::TextEnd { .. })) {
            break;
        }
    }
    registration.abort();
    let error = response.stream.next().await.unwrap().unwrap_err();
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::Abort);
    assert!(response.stream.next().await.is_none());
    server.abort();
}
