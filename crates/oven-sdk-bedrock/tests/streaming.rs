mod support;

use oven_sdk::{AbortSignal, AssistantPart, FinishReason, LanguageModel, Request};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn model(server: &MockServer) -> oven_sdk_bedrock::BedrockModel {
    support::model(
        &server.uri(),
        "anthropic-looking-but-opaque",
        support::FixtureKind::Text,
    )
}

#[tokio::test]
async fn eventstream_text_usage_request_id_and_lifecycle_normalize() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/vnd.amazon.eventstream")
                .insert_header("x-amzn-requestid", "request-stream")
                .set_body_bytes(support::text_stream("hello")),
        )
        .mount(&server)
        .await;
    let result = model(&server)
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "hello");
    assert_eq!(result.turn.finish.finish_reason, FinishReason::Stop);
    assert_eq!(result.turn.finish.usage.input_tokens, Some(1));
    assert_eq!(
        result.response.request_id.as_deref(),
        Some("request-stream")
    );
    assert!(result.turn.finish.native_replay.is_some());
    assert!(matches!(
        result.turn.message.content[0],
        AssistantPart::Text(_)
    ));
}

#[tokio::test]
async fn corrupted_crc_is_a_stream_error_not_a_finish() {
    let server = MockServer::start().await;
    let mut body = support::text_stream("hello");
    let last = body.len() - 1;
    body[last] ^= 1;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
        .mount(&server)
        .await;
    let error = model(&server)
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::InvalidResponse);
}

#[tokio::test]
async fn post_terminal_frames_and_truncated_trailing_bytes_are_rejected() {
    for body in [
        {
            let mut body = support::text_stream("hello");
            body.extend(support::frame(
                "messageStart",
                serde_json::json!({"role":"assistant"}),
            ));
            body
        },
        {
            let mut body = support::text_stream("hello");
            body.extend_from_slice(&[0; 11]);
            body
        },
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
            .mount(&server)
            .await;
        assert!(
            model(&server)
                .complete(Request::new(Vec::new()), AbortSignal::default())
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn eventstream_error_headers_are_typed_and_sanitized() {
    let server = MockServer::start().await;
    let body = support::frame_with_headers(
        &[
            (":message-type", "error"),
            (":error-code", "ThrottlingException"),
            (":error-message", "SECRET_PROVIDER_DETAIL"),
        ],
        Vec::new(),
    );
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
        .mount(&server)
        .await;
    let error = model(&server)
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::RateLimited);
    assert_eq!(
        error.diagnostics.vendor_code.as_deref(),
        Some("ThrottlingException")
    );
    assert!(!error.to_string().contains("SECRET_PROVIDER_DETAIL"));
    assert!(
        !error
            .diagnostics
            .sanitized_body
            .as_ref()
            .is_some_and(|body| body.text().contains("SECRET_PROVIDER_DETAIL"))
    );
}
