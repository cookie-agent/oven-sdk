use std::time::{Duration, Instant};

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, LanguageModel, Request, StreamPart};
use tokio::{io::AsyncWriteExt, net::TcpListener, time::timeout};

const HEADERS: &str =
    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";

fn event(name: &str, data: &str) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, body: &str) {
    socket
        .write_all(format!("{:X}\r\n", body.len()).as_bytes())
        .await
        .unwrap();
    socket.write_all(body.as_bytes()).await.unwrap();
    socket.write_all(b"\r\n").await.unwrap();
    socket.flush().await.unwrap();
}

fn model(address: std::net::SocketAddr) -> oven_sdk_anthropic::AnthropicModel {
    Anthropic::builder()
        .base_url(format!("http://{address}"))
        .build()
        .unwrap()
        .model("claude-sonnet-4-5")
}

fn start_text() -> String {
    [
        event(
            "message_start",
            r#"{"type":"message_start","message":{"usage":{"input_tokens":1}}}"#,
        ),
        event(
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        ),
        event(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"first"}}"#,
        ),
    ]
    .concat()
}

fn finish_text() -> String {
    [
        event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
        event(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":1}}"#,
        ),
        event("message_stop", r#"{"type":"message_stop"}"#),
    ]
    .concat()
}

#[tokio::test]
async fn stream_returns_before_full_body() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = model(listener.local_addr().unwrap());
    let first = start_text();
    let rest = finish_text();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket.write_all(HEADERS.as_bytes()).await.unwrap();
        write_chunk(&mut socket, &first).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        write_chunk(&mut socket, &rest).await;
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });

    let started = Instant::now();
    let mut response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(matches!(
        response.stream.next().await,
        Some(Ok(StreamPart::StreamStart { .. }))
    ));
    assert!(matches!(
        response.stream.next().await,
        Some(Ok(StreamPart::TextStart { .. }))
    ));
    assert!(
        matches!(response.stream.next().await, Some(Ok(StreamPart::TextDelta { delta, .. })) if delta == "first")
    );
    assert!(started.elapsed() < Duration::from_millis(200));
}

#[tokio::test]
async fn same_chunk_parts_then_error_ordering() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = model(listener.local_addr().unwrap());
    let body = format!(
        "{}{}",
        start_text(),
        event("error", "this is not valid JSON")
    );
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket.write_all(HEADERS.as_bytes()).await.unwrap();
        write_chunk(&mut socket, &body).await;
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });

    let mut response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(matches!(
        response.stream.next().await,
        Some(Ok(StreamPart::StreamStart { .. }))
    ));
    assert!(matches!(
        response.stream.next().await,
        Some(Ok(StreamPart::TextStart { .. }))
    ));
    assert!(matches!(
        response.stream.next().await,
        Some(Ok(StreamPart::TextDelta { .. }))
    ));
    assert!(response.stream.next().await.unwrap().is_err());
    assert!(response.stream.next().await.is_none());
}

#[tokio::test]
async fn stream_ends_after_message_stop_despite_open_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = model(listener.local_addr().unwrap());
    let body = format!("{}{}", start_text(), finish_text());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket.write_all(HEADERS.as_bytes()).await.unwrap();
        write_chunk(&mut socket, &body).await;
        tokio::time::sleep(Duration::from_secs(3)).await;
    });

    let response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut stream = response.stream.fuse();
    timeout(Duration::from_secs(2), async {
        let mut finishes = 0;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamPart::Finish { .. }) => finishes += 1,
                Ok(_) => {}
                Err(error) => panic!("unexpected stream error: {error}"),
            }
        }
        assert_eq!(finishes, 1);
        assert!(stream.next().await.is_none());
    })
    .await
    .expect("stream should terminate after message_stop");
}

#[tokio::test]
async fn open_tool_error_closes_text_and_reasoning_before_terminal_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = model(listener.local_addr().unwrap());
    let body = [
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#),
        event("content_block_start", r#"{"type":"content_block_start","index":1,"content_block":{"type":"thinking"}}"#),
        event("content_block_start", r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call","name":"lookup"}}"#),
        event("error", r#"{"type":"error","error":{"type":"api_error","message":"failed"}}"#),
    ].concat();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket.write_all(HEADERS.as_bytes()).await.unwrap();
        write_chunk(&mut socket, &body).await;
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });

    let mut response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut parts = Vec::new();
    let mut errors = 0;
    while let Some(item) = response.stream.next().await {
        match item {
            Ok(part) => parts.push(part),
            Err(_) => errors += 1,
        }
    }
    let text_end = parts
        .iter()
        .position(|part| matches!(part, StreamPart::TextEnd { .. }))
        .unwrap();
    let reasoning_end = parts
        .iter()
        .position(|part| matches!(part, StreamPart::ReasoningEnd { .. }))
        .unwrap();
    assert!(text_end < parts.len());
    assert!(reasoning_end < parts.len());
    assert_eq!(errors, 1);
    assert!(
        !parts
            .iter()
            .any(|part| matches!(part, StreamPart::Finish { .. }))
    );
}
mod common;

use common::Anthropic;
