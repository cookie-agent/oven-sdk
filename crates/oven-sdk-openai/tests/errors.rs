pub mod common;

use std::time::Duration;

use oven_sdk::{AbortSignal, ErrorStage, LanguageModel, ModelErrorKind, Request, SanitizedBody};
use oven_sdk_openai::{OpenAiChatModel, OpenAiResponsesModel, OpenAiTimeouts};
use tokio::{io::AsyncWriteExt, net::TcpListener};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn http_failure_performs_one_call_without_retry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "2")
                .insert_header("x-request-id", "req_rate")
                .set_body_string(r#"{"error":{"type":"rate_limit_error","message":"slow"}}"#),
        )
        .mount(&server)
        .await;
    let error = common::official_chat(&server, "gpt-4o-mini")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::RateLimited);
    assert_eq!(error.diagnostics.request_id.as_deref(), Some("req_rate"));
    assert_eq!(error.diagnostics.retry_after, Some(Duration::from_secs(2)));
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

fn short_timeouts() -> OpenAiTimeouts {
    OpenAiTimeouts {
        connect: Duration::from_secs(1),
        headers: Duration::from_secs(1),
        stream_idle: Duration::from_millis(50),
    }
}

#[tokio::test]
async fn stalled_non_success_body_times_out_at_response_body_stage() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket
            .write_all(
                b"HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ncontent-length: 100\r\n\r\n",
            )
            .await
            .unwrap();
        socket.flush().await.unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
    });
    let mut config =
        common::official_chat_config_at(&format!("http://{address}"), "gpt-4o-mini", "secret");
    config.settings.timeouts = short_timeouts();
    let error = OpenAiChatModel::new(config)
        .unwrap()
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Timeout);
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseBody);
    assert_eq!(error.diagnostics.bytes_received, 0);
}

#[tokio::test]
async fn non_success_body_read_honors_abort() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket
            .write_all(
                b"HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ncontent-length: 100\r\n\r\n",
            )
            .await
            .unwrap();
        socket.flush().await.unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
    });
    let mut config =
        common::official_chat_config_at(&format!("http://{address}"), "gpt-4o-mini", "secret");
    config.settings.timeouts = OpenAiTimeouts {
        stream_idle: Duration::from_secs(5),
        ..short_timeouts()
    };
    let model = OpenAiChatModel::new(config).unwrap();
    let (signal, registration) = AbortSignal::new();
    let task = tokio::spawn(async move { model.stream(Request::new(Vec::new()), signal).await });
    tokio::time::sleep(Duration::from_millis(25)).await;
    registration.abort();
    let error = task.await.unwrap().unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Abort);
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseBody);
}

#[tokio::test]
async fn non_success_body_transport_error_reports_response_body_stage() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket
            .write_all(
                b"HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ncontent-length: 100\r\n\r\nabc",
            )
            .await
            .unwrap();
        socket.flush().await.unwrap();
    });
    let mut config =
        common::official_responses_config_at(&format!("http://{address}"), "gpt-5-mini", "secret");
    config.settings.timeouts = short_timeouts();
    let error = OpenAiResponsesModel::new(config)
        .unwrap()
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Transport);
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseBody);
    assert!(error.diagnostics.bytes_received <= 3);
}

#[tokio::test]
async fn oversized_non_success_body_is_bounded_but_counts_all_bytes() {
    let server = MockServer::start().await;
    let body = "x".repeat(SanitizedBody::MAX_BYTES + 100);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string(body.clone()))
        .mount(&server)
        .await;
    let error = common::official_chat(&server, "gpt-4o-mini")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    let sanitized = error.diagnostics.sanitized_body.as_ref().unwrap();
    assert_eq!(sanitized.len_bytes(), SanitizedBody::MAX_BYTES);
    assert!(sanitized.truncated());
    assert_eq!(error.diagnostics.bytes_received, body.len() as u64);
}
