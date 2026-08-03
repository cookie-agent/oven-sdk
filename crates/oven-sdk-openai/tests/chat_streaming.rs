pub mod common;

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, FinishReason, LanguageModel, ModelErrorKind, Request, StreamPart};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn usage_only_chunk_is_terminally_authoritative() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("hello")).await;
    let result = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "hello");
    assert_eq!(result.turn.finish.usage.input_tokens, Some(2));
    assert_eq!(result.turn.finish.usage.output_tokens, Some(3));
    assert_eq!(result.turn.finish.usage.output_tokens_reasoning, Some(1));
    assert_eq!(result.turn.finish.usage.output_tokens_text, Some(2));
}

#[tokio::test]
async fn fragmented_parallel_tools_finalize_only_at_done() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":\"}},{\"index\":1,\"id\":\"call_2\",\"function\":{\"name\":\"two\",\"arguments\":\"{\\\"b\\\":2}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"one\",\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    common::mount(&server, "/chat/completions", body.into()).await;
    let result = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let calls = result
        .turn
        .message
        .content
        .iter()
        .filter(|part| matches!(part, oven_sdk::AssistantPart::ToolCall(_)))
        .count();
    assert_eq!(calls, 2);
    assert_eq!(result.turn.finish.finish_reason, FinishReason::ToolCalls);
}

#[tokio::test]
async fn malformed_and_empty_final_arguments_are_invalid_response() {
    for arguments in ["", "["] {
        let server = MockServer::start().await;
        let event = serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","function":{"name":"tool","arguments":arguments}}]},"finish_reason":"tool_calls"}]});
        let body = format!("data: {event}\n\ndata: [DONE]\n\n");
        common::mount(&server, "/chat/completions", body).await;
        let error = common::official_chat(&server, "gpt-4o-mini")
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
    }
}

#[tokio::test]
async fn clean_eof_requires_finish_reason() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/chat/completions",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n".into(),
    )
    .await;
    let error = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::UnexpectedEof);
}

#[tokio::test]
async fn fragmented_refusal_becomes_one_custom_part() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"not \"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"allowed\"},\"finish_reason\":\"content_filter\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    common::mount(&server, "/chat/completions", body.into()).await;
    let result = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(result.turn.message.content.iter().any(|part| {
        matches!(part, oven_sdk::AssistantPart::Custom(custom) if custom.kind == "openai.refusal" && custom.data == "not allowed")
    }));
}

#[tokio::test]
async fn mid_stream_error_emits_error_then_finish_error() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"started\"},\"finish_reason\":null}]}\n\n",
        "data: {\"error\":{\"type\":\"server_error\",\"message\":\"failed\"}}\n\n"
    );
    common::mount(&server, "/chat/completions", body.into()).await;
    let mut response = common::official_chat(&server, "gpt-4o-mini")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut parts = Vec::new();
    while let Some(item) = response.stream.next().await {
        parts.push(item.unwrap());
    }
    assert!(matches!(parts[parts.len() - 2], StreamPart::Error { .. }));
    assert!(
        matches!(&parts[parts.len() - 1], StreamPart::Finish { finish } if finish.finish_reason == FinishReason::Error)
    );
}

#[tokio::test]
async fn in_band_error_preserves_http_response_headers() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"started\"},\"finish_reason\":null}]}\n\n",
        "data: {\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow\"}}\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "text/event-stream")
                .insert_header("x-request-id", "req_in_band")
                .insert_header("retry-after-ms", "125"),
        )
        .mount(&server)
        .await;
    let mut response = common::official_chat(&server, "gpt-4o-mini")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    while let Some(item) = response.stream.next().await {
        if let StreamPart::Error { error } = item.unwrap() {
            assert_eq!(error.diagnostics.request_id.as_deref(), Some("req_in_band"));
            assert_eq!(
                error.diagnostics.retry_after,
                Some(std::time::Duration::from_millis(125))
            );
            return;
        }
    }
    panic!("missing in-band error");
}
