mod common;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AdapterId, HistoryTurn, LanguageModel, ReplayDisposition, ReplayPolicy, Request,
    SecretString, StreamPart, ToolContent, ToolMessage, ToolResultPart,
};
use oven_sdk_anthropic::{
    ANTHROPIC_MESSAGES_ADAPTER_ID, AnthropicCompatibleAuth, AnthropicRequestExt,
    AnthropicRequestOptions, MINIMAX_MESSAGES_ADAPTER_ID,
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

use common::{anthropic_capabilities, anthropic_protocol, try_compatible_model};

fn response() -> &'static str {
    "event: message_start\ndata: {\"message\":{}}\n\nevent: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\nevent: message_stop\ndata: {}\n\n"
}

async fn server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn compatible_auth_selects_api_key_bearer_or_none_with_anthropic_headers() {
    for (auth, expected_name, expected_value) in [
        (
            AnthropicCompatibleAuth::ApiKey(SecretString::new("api-secret")),
            Some("x-api-key"),
            Some("api-secret"),
        ),
        (
            AnthropicCompatibleAuth::Bearer(SecretString::new("bearer-secret")),
            Some("authorization"),
            Some("Bearer bearer-secret"),
        ),
        (AnthropicCompatibleAuth::None, None, None),
    ] {
        let server = server().await;
        let model = try_compatible_model(
            &server.uri(),
            "future-provider",
            "future-model",
            "app.future.messages",
            auth,
            anthropic_capabilities(ReplayPolicy::IfValid),
            anthropic_protocol(),
        )
        .unwrap();
        model
            .complete(
                Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
                    betas: vec!["future-beta".into()],
                    ..Default::default()
                }),
                AbortSignal::default(),
            )
            .await
            .unwrap();
        let requests = server.received_requests().await.unwrap();
        let request = &requests[0];
        assert_eq!(request.url.path(), "/messages");
        assert_eq!(request.headers["anthropic-version"], "2023-06-01");
        assert_eq!(request.headers["anthropic-beta"], "future-beta");
        match (expected_name, expected_value) {
            (Some(name), Some(value)) => assert_eq!(request.headers[name], value),
            (None, None) => {
                assert!(request.headers.get("x-api-key").is_none());
                assert!(request.headers.get("authorization").is_none());
            }
            _ => unreachable!(),
        }
    }
}

#[tokio::test]
async fn compatible_preserves_provider_and_adapter_ids_and_joins_nested_endpoint() {
    let server = server().await;
    let model = try_compatible_model(
        &format!("{}/custom/v1/", server.uri()),
        "kimi-for-coding",
        "opaque-model",
        "app.kimi.messages",
        AnthropicCompatibleAuth::None,
        anthropic_capabilities(ReplayPolicy::IfValid),
        anthropic_protocol(),
    )
    .unwrap();
    let completed = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(
        model.descriptor().identity.provider_id.as_str(),
        "kimi-for-coding"
    );
    assert_eq!(
        model.descriptor().adapter_id,
        AdapterId::new("app.kimi.messages")
    );
    assert_eq!(
        completed
            .turn
            .finish
            .native_replay
            .unwrap()
            .adapter_id()
            .as_str(),
        "app.kimi.messages"
    );
    assert_eq!(
        server.received_requests().await.unwrap()[0].url.path(),
        "/custom/v1/messages"
    );
}

#[tokio::test]
async fn compatible_signed_thinking_tool_turn_replays_without_reconstruction_warning() {
    let server = MockServer::start().await;
    let signed_tool_turn = [
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"reason\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"78aFxXtB9TS25vt+signed\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"lookup\",\"input\":{\"query\":\"oven\"}}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(signed_tool_turn))
        .mount(&server)
        .await;
    let model = try_compatible_model(
        &server.uri(),
        "kimi-for-coding",
        "kimi-for-coding",
        "app.kimi.messages",
        AnthropicCompatibleAuth::None,
        anthropic_capabilities(ReplayPolicy::IfValid),
        anthropic_protocol(),
    )
    .unwrap();

    let completed = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let artifact = completed.turn.finish.native_replay.as_ref().unwrap();
    assert_eq!(
        artifact
            .payload()
            .pointer("/message/content/0/signature")
            .and_then(serde_json::Value::as_str),
        Some("78aFxXtB9TS25vt+signed")
    );

    let mut response = model
        .stream(
            Request::new(vec![
                HistoryTurn::assistant(completed.turn),
                HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
                    "call_1",
                    ToolContent::Text("result".into()),
                )])),
            ]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.request.replay.decisions[0].disposition,
        ReplayDisposition::Replayed
    );
    assert!(matches!(
        response.stream.next().await.unwrap().unwrap(),
        StreamPart::StreamStart { warnings } if warnings.is_empty()
    ));

    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][0],
        serde_json::json!({
            "type": "thinking",
            "thinking": "reason",
            "signature": "78aFxXtB9TS25vt+signed"
        })
    );
}

#[test]
fn compatible_rejects_queries_and_reserved_adapter_ids() {
    for adapter_id in [ANTHROPIC_MESSAGES_ADAPTER_ID, MINIMAX_MESSAGES_ADAPTER_ID] {
        assert!(
            try_compatible_model(
                "https://example.test/v1",
                "provider",
                "model",
                adapter_id,
                AnthropicCompatibleAuth::None,
                anthropic_capabilities(ReplayPolicy::IfValid),
                anthropic_protocol(),
            )
            .is_err()
        );
    }
    assert!(
        try_compatible_model(
            "https://example.test/v1?route=messages",
            "provider",
            "model",
            "app.compatible.messages",
            AnthropicCompatibleAuth::None,
            anthropic_capabilities(ReplayPolicy::IfValid),
            anthropic_protocol(),
        )
        .is_err()
    );
}
