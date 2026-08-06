mod common;

use oven_sdk::{AbortSignal, AdapterId, LanguageModel, ReplayPolicy, Request, SecretString};
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
