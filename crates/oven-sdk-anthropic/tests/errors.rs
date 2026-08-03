use std::time::{Duration, SystemTime};

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, ErrorStage, LanguageModel, ModelErrorKind, Request, SanitizedBody};
use oven_sdk_anthropic::{AnthropicTimeouts, classify_error};
use reqwest::header::{HeaderMap, HeaderValue};
use tokio::{io::AsyncWriteExt, net::TcpListener};

fn model(address: std::net::SocketAddr) -> oven_sdk_anthropic::AnthropicModel {
    Anthropic::builder()
        .base_url(format!("http://{address}"))
        .build()
        .unwrap()
        .model("claude-sonnet-4-5")
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, body: &[u8]) {
    socket
        .write_all(format!("{:X}\r\n", body.len()).as_bytes())
        .await
        .unwrap();
    socket.write_all(body).await.unwrap();
    socket.write_all(b"\r\n").await.unwrap();
    socket.flush().await.unwrap();
}

#[tokio::test]
async fn http_error_body_read_stops_at_the_cap_without_retaining_unstructured_data() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = model(listener.local_addr().unwrap());
    let chunks = [
        vec![b'a'; 32 * 1024],
        vec![b'b'; 40 * 1024],
        vec![b'c'; 8 * 1024],
    ];
    let expected = chunks.iter().map(Vec::len).sum::<usize>() as u64;
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket
            .write_all(b"HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n")
            .await
            .unwrap();
        for chunk in &chunks {
            write_chunk(&mut socket, chunk).await;
        }
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });
    let error = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseBody);
    assert!(error.diagnostics.bytes_received >= SanitizedBody::MAX_BYTES as u64);
    assert!(error.diagnostics.bytes_received < expected);
    assert!(error.diagnostics.sanitized_body.is_none());
}

#[tokio::test]
async fn header_timeout_is_classified_as_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = Anthropic::builder()
        .base_url(format!("http://{}", listener.local_addr().unwrap()))
        .timeouts(AnthropicTimeouts {
            headers: Duration::from_millis(20),
            credentials: Duration::from_secs(1),
            stream_idle: Duration::from_secs(1),
        })
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    tokio::spawn(async move {
        let (_socket, _) = listener.accept().await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
    });
    let error = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Timeout);
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseHeaders);
}

#[tokio::test]
async fn non_success_body_idle_timeout_and_abort_preserve_response_body_stage() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = Anthropic::builder()
        .base_url(format!("http://{}", listener.local_addr().unwrap()))
        .timeouts(AnthropicTimeouts {
            headers: Duration::from_secs(1),
            credentials: Duration::from_secs(1),
            stream_idle: Duration::from_millis(20),
        })
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket
            .write_all(b"HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n")
            .await
            .unwrap();
        socket.flush().await.unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
    });
    let error = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Timeout);
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseBody);
    assert_eq!(error.diagnostics.http_status, Some(400));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = Anthropic::builder()
        .base_url(format!("http://{}", listener.local_addr().unwrap()))
        .timeouts(AnthropicTimeouts {
            headers: Duration::from_secs(1),
            credentials: Duration::from_secs(1),
            stream_idle: Duration::from_secs(1),
        })
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket
            .write_all(b"HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n")
            .await
            .unwrap();
        socket.flush().await.unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
    });
    let (signal, registration) = AbortSignal::new();
    let future = model.stream(Request::new(Vec::new()), signal);
    tokio::pin!(future);
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_millis(20)) => registration.abort(),
        _ = &mut future => panic!("error body unexpectedly completed"),
    }
    let error = future.await.unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Abort);
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseBody);
    assert_eq!(error.diagnostics.http_status, Some(400));
}

#[tokio::test]
async fn non_success_partial_body_read_failure_is_transport_not_provider_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = model(listener.local_addr().unwrap());
    let partial = br#"{"error":{"type":"api_error""#;
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket
            .write_all(b"HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n")
            .await
            .unwrap();
        write_chunk(&mut socket, partial).await;
        socket.write_all(b"not-a-chunk-size\r\n").await.unwrap();
        socket.flush().await.unwrap();
    });
    let error = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Transport);
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseBody);
    assert_eq!(error.diagnostics.http_status, Some(500));
    assert_eq!(error.diagnostics.bytes_received, partial.len() as u64);
}

#[tokio::test]
async fn nested_model_code_overrides_server_status() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = model(listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let body = br#"{"error":{"message":"missing","details":{"code":"model_does_not_exist"}}}"#;
        socket
            .write_all(b"HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ncontent-length: 73\r\n\r\n")
            .await
            .unwrap();
        socket.write_all(body).await.unwrap();
    });
    let error = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::ModelNotFound);
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseBody);
}

#[test]
fn display_redacts_api_keys_and_tokens() {
    let error = classify_error(
        400,
        br#"{"error":{"type":"invalid_request_error","message":"key sk-ant-secret-token and token=very-secret leaked"}}"#,
        None,
        ErrorStage::ResponseBody,
        0,
        &HeaderMap::new(),
    );
    let display = error.to_string();
    assert!(!display.contains("sk-ant-secret-token"));
    assert!(!display.contains("very-secret"));
    let serialized = serde_json::to_string(&error).unwrap();
    assert!(!serialized.contains("sk-ant-secret-token"));
    assert!(!serialized.contains("very-secret"));
    assert!(serialized.contains("invalid_request_error"));
}

#[test]
fn request_too_large_is_not_misclassified_as_context_length() {
    let too_large = classify_error(
        413,
        br#"{"type":"error","error":{"type":"request_too_large","message":"request exceeds 32 MB"}}"#,
        None,
        ErrorStage::ResponseBody,
        0,
        &HeaderMap::new(),
    );
    assert_eq!(too_large.kind, ModelErrorKind::InvalidRequest);

    let context = classify_error(
        400,
        br#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long for the context window"}}"#,
        None,
        ErrorStage::ResponseBody,
        0,
        &HeaderMap::new(),
    );
    assert_eq!(context.kind, ModelErrorKind::ContextLength);
}

#[test]
fn unstructured_error_bodies_are_not_retained() {
    let error = classify_error(
        500,
        b"secret=do-not-retain",
        None,
        ErrorStage::ResponseBody,
        20,
        &HeaderMap::new(),
    );
    assert!(error.diagnostics.sanitized_body.is_none());
    assert!(
        !serde_json::to_string(&error)
            .unwrap()
            .contains("do-not-retain")
    );
}

#[test]
fn http_date_retry_after_is_parsed_and_lower_priority_hints_do_not_win() {
    let retry_at = SystemTime::now() + Duration::from_secs(60);
    let mut headers = HeaderMap::new();
    headers.insert(
        "retry-after",
        HeaderValue::from_str(&httpdate::fmt_http_date(retry_at)).unwrap(),
    );
    let dated = classify_error(429, b"{}", None, ErrorStage::ResponseBody, 0, &headers);
    assert!(dated.diagnostics.retry_after.unwrap() >= Duration::from_secs(58));

    headers.insert("retry-after", HeaderValue::from_static("30"));
    headers.insert("retry-after-ms", HeaderValue::from_static("25"));
    let milliseconds = classify_error(429, b"{}", None, ErrorStage::ResponseBody, 0, &headers);
    assert_eq!(
        milliseconds.diagnostics.retry_after,
        Some(Duration::from_millis(25))
    );
}

#[tokio::test]
async fn errors_report_the_stage_of_each_reachable_failure_point() {
    let connect = Anthropic::builder()
        .base_url("http://127.0.0.1:9")
        .build()
        .unwrap()
        .model("claude-sonnet-4-5")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(connect.diagnostics.stage, ErrorStage::Connect);

    let invalid = Anthropic::builder()
        .base_url("http://127.0.0.1:9")
        .build()
        .unwrap()
        .model("claude-sonnet-4-5")
        .stream(
            Request::new(Vec::new())
                .with_response_format(oven_sdk::ResponseFormat::Json { schema: None }),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(invalid.diagnostics.stage, ErrorStage::RequestValidation);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let stream_model = model(listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n").await.unwrap();
        write_chunk(
            &mut socket,
            b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{}}\n\n",
        )
        .await;
        write_chunk(&mut socket, b"event: content_block_start\ndata: {bad}\n\n").await;
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });
    let mut response = stream_model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(response.stream.next().await.unwrap().is_ok());
    let decode = response.stream.next().await.unwrap().unwrap_err();
    assert_eq!(decode.diagnostics.stage, ErrorStage::StreamDecode);
}
mod common;

use common::Anthropic;
