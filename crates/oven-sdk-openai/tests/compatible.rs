pub mod common;

use std::sync::Arc;

use oven_sdk::{
    AbortSignal, AdapterId, FilePart, FileSource, HeaderOverrides, HeaderProvider, HistoryTurn,
    InputPart, JsonSchema, LanguageModel, ModelError, Request, ResponseFormat, UserMessage,
};
use oven_sdk_openai::{
    CompatibleChatOptions, OpenAiChatOptions, OpenAiChatRequestExt, OpenAiCompatibleAuth,
    OpenAiCompatibleChatModel, ReasoningField, StructuredOutputSupport,
};
use wiremock::MockServer;

#[tokio::test]
async fn compatible_chat_encodes_inline_video_as_video_url() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::compatible(&server);
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(FilePart::video(
            "video/mp4",
            FileSource::Bytes(bytes::Bytes::from_static(b"video")),
        )),
    ]))]);

    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][0],
        serde_json::json!({
            "type":"video_url",
            "video_url":{"url":"data:video/mp4;base64,dmlkZW8="}
        })
    );
}

#[tokio::test]
async fn explicit_settings_control_usage_reasoning_and_structured_downgrade() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{}\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    common::mount(&server, "/chat/completions", body.into()).await;
    let mut config = common::compatible_config(&server, "future-compatible-id");
    config.settings.stream_usage = true;
    config.settings.reasoning_field = ReasoningField::ReasoningContent;
    config.settings.structured_output = StructuredOutputSupport::JsonObject;
    let model = OpenAiCompatibleChatModel::new(config).unwrap();
    let schema = JsonSchema::new(serde_json::json!({"type":"object"})).unwrap();
    let result = model
        .complete(
            Request::new(Vec::new()).with_response_format(ResponseFormat::structured(schema)),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(result.turn.message.content.iter().any(|part| {
        matches!(part, oven_sdk::AssistantPart::Reasoning(reasoning) if reasoning.text == "think")
    }));
    assert!(
        result
            .turn
            .warnings
            .iter()
            .any(|warning| warning.contains("downgraded"))
    );
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["response_format"]["type"], "json_object");
}

#[test]
fn compatible_constructor_rejects_reserved_adapter_ids() {
    for adapter_id in ["oven.openai.chat", "oven.openai.responses"] {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(MockServer::start());
        let mut config = common::compatible_config(&server, "model");
        config.settings.adapter_id = AdapterId::new(adapter_id);
        let error = OpenAiCompatibleChatModel::new(config)
            .err()
            .expect("reserved adapter ID must fail");
        assert_eq!(error.kind, oven_sdk::ModelErrorKind::InvalidRequest);
    }
}

struct CustomAuth;

impl HeaderProvider for CustomAuth {
    fn headers(&self, _context: &oven_sdk::HeaderContext) -> Result<HeaderOverrides, ModelError> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("authorization", "Custom token".parse().unwrap());
        Ok(HeaderOverrides::new(headers))
    }
}

#[tokio::test]
async fn query_params_and_custom_auth_are_explicit() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let mut config = common::compatible_config(&server, "model");
    config.provider.auth = OpenAiCompatibleAuth::headers(Arc::new(CustomAuth));
    config
        .settings
        .query
        .push(("api-version".into(), "1".into()));
    config.settings.routing_discriminator = Some("custom-auth-route".into());
    let model = OpenAiCompatibleChatModel::new(config).unwrap();
    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let request = &server.received_requests().await.unwrap()[0];
    assert_eq!(request.headers["authorization"], "Custom token");
    assert_eq!(request.url.query(), Some("api-version=1"));
}

#[tokio::test]
async fn caller_authorization_is_not_overwritten_by_custom_auth() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let mut config = common::compatible_config(&server, "model");
    config.provider.auth = OpenAiCompatibleAuth::headers(Arc::new(CustomAuth));
    config.settings.routing_discriminator = Some("caller-auth-route".into());
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("authorization", "Caller token".parse().unwrap());
    config.provider.headers.static_headers = HeaderOverrides::new(headers);
    let model = OpenAiCompatibleChatModel::new(config).unwrap();

    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].headers["authorization"], "Caller token");
}

#[tokio::test]
async fn compatible_future_scalar_labels_pass_through_unchanged() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::compatible(&server);
    let request = Request::new(Vec::new())
        .with_openai_chat_options(OpenAiChatOptions {
            service_tier: Some("future-priority".into()),
            verbosity: Some("future-detailed".into()),
            ..Default::default()
        })
        .with_compatible_chat_options(CompatibleChatOptions {
            extra_body: serde_json::Map::from_iter([(
                "vendor_future_feature".into(),
                serde_json::json!("enabled"),
            )]),
        });
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["service_tier"], "future-priority");
    assert_eq!(body["verbosity"], "future-detailed");
    assert_eq!(body["vendor_future_feature"], "enabled");
}

#[tokio::test]
async fn compatible_extra_body_rejects_every_reserved_structural_key_before_dispatch() {
    for key in [
        "model",
        "messages",
        "stream",
        "stream_options",
        "n",
        "modalities",
        "audio",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "reasoning_effort",
        "reasoning",
        "user",
        "service_tier",
        "verbosity",
        "parallel_tool_calls",
        "tools",
        "tool_choice",
        "response_format",
    ] {
        let server = MockServer::start().await;
        let model = common::compatible(&server);
        let request =
            Request::new(Vec::new()).with_compatible_chat_options(CompatibleChatOptions {
                extra_body: serde_json::Map::from_iter([(key.into(), serde_json::json!("forged"))]),
            });
        let error = model
            .stream(request, AbortSignal::default())
            .await
            .unwrap_err();
        assert_eq!(
            error.kind,
            oven_sdk::ModelErrorKind::InvalidRequest,
            "{key}"
        );
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "{key}"
        );
    }
}
