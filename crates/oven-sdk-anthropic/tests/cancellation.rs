use std::time::Duration;

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, ErrorStage, LanguageModel, ModelErrorKind, Request};
use tokio::{io::AsyncWriteExt, net::TcpListener, time::timeout};

#[tokio::test]
async fn abort_before_headers_returns_abort_without_io() {
    let model = Anthropic::builder()
        .base_url("http://127.0.0.1:9")
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    let error = model
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Abort);
}

#[tokio::test]
async fn abort_mid_stream_yields_one_error_then_terminates() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = Anthropic::builder()
        .base_url(format!("http://{}", listener.local_addr().unwrap()))
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
            .await
            .unwrap();
        let body = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{}}\n\n";
        socket
            .write_all(format!("{:X}\r\n{body}\r\n", body.len()).as_bytes())
            .await
            .unwrap();
        socket.flush().await.unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
    });

    let (signal, registration) = AbortSignal::new();
    let mut response = model
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap();
    assert!(response.stream.next().await.unwrap().is_ok());
    registration.abort();
    let error = timeout(Duration::from_millis(250), response.stream.next())
        .await
        .expect("abort should interrupt the pending stream read")
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Abort);
    assert_eq!(error.diagnostics.stage, ErrorStage::StreamRead);
    assert!(response.stream.next().await.is_none());
}
mod common;

use common::Anthropic;
