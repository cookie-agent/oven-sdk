use std::time::Duration;

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, ErrorStage, LanguageModel, ModelErrorKind, Request};
use oven_sdk_openai::OpenAiChatModel;
use tokio::{io::AsyncWriteExt, net::TcpListener, time::timeout};

#[tokio::test]
async fn abort_before_headers_returns_abort() {
    let model = OpenAiChatModel::new(common::official_chat_config_at(
        "http://127.0.0.1:9",
        "gpt-4o-mini",
        "secret",
    ))
    .unwrap();
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    let error = model
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Abort);
}

#[tokio::test]
async fn abort_mid_chat_stream_yields_one_error_then_eof() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = OpenAiChatModel::new(common::official_chat_config_at(
        &format!("http://{}", listener.local_addr().unwrap()),
        "gpt-4o-mini",
        "secret",
    ))
    .unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n").await.unwrap();
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"},\"finish_reason\":null}]}\n\n";
        socket
            .write_all(format!("{:X}\r\n{body}\r\n", body.len()).as_bytes())
            .await
            .unwrap();
        socket.flush().await.unwrap();
        tokio::time::sleep(Duration::from_secs(2)).await;
    });
    let (signal, registration) = AbortSignal::new();
    let mut response = model
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap();
    assert!(response.stream.next().await.unwrap().is_ok());
    registration.abort();
    let mut error = None;
    while error.is_none() {
        let item = timeout(Duration::from_millis(300), response.stream.next())
            .await
            .expect("abort should wake stream")
            .expect("stream item");
        if let Err(found) = item {
            error = Some(found);
        }
    }
    let error = error.unwrap();
    assert_eq!(error.kind, ModelErrorKind::Abort);
    assert_eq!(error.diagnostics.stage, ErrorStage::StreamRead);
    assert!(response.stream.next().await.is_none());
}
pub mod common;
