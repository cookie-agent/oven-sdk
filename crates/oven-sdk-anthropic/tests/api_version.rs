//! `anthropic-version` selection and the request rules of API version 2026-10-21.
//!
//! Golden tests compare bodies only, so the header assertions live here.

mod common;

use oven_sdk::{
    AbortSignal, Capability, InputPart, LanguageModel, ModelErrorKind, ReplayPolicy, Request,
    TextPart, UserMessage,
};
use oven_sdk_anthropic::{
    AnthropicApiVersion, AnthropicCacheControl, AnthropicCacheTtl, AnthropicCompatibleAuth,
    AnthropicRequestExt, AnthropicRequestOptions,
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

use common::{
    Anthropic, AnthropicAws, anthropic_capabilities, anthropic_protocol, legacy_sampling_protocol,
    try_compatible_model,
};

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

async fn sent_version(server: &MockServer, model: &dyn LanguageModel) -> String {
    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    requests.last().unwrap().headers["anthropic-version"]
        .to_str()
        .unwrap()
        .to_owned()
}

fn pinned(version: AnthropicApiVersion) -> oven_sdk_anthropic::AnthropicProtocolSettings {
    oven_sdk_anthropic::AnthropicProtocolSettings {
        api_version: Some(version),
        ..anthropic_protocol()
    }
}

#[tokio::test]
async fn first_party_defaults_to_2026_10_21_and_can_pin_2023_06_01() {
    let server = server().await;
    let direct = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-5");
    assert_eq!(sent_version(&server, &direct).await, "2026-10-21");
    let direct_pinned = Anthropic::builder()
        .base_url(server.uri())
        .protocol(pinned(AnthropicApiVersion::V2023_06_01))
        .build()
        .unwrap()
        .model("claude-opus-5");
    assert_eq!(sent_version(&server, &direct_pinned).await, "2023-06-01");

    let aws = AnthropicAws::builder("us-west-2", "wrkspc_test")
        .bearer_key("test")
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-5");
    assert_eq!(sent_version(&server, &aws).await, "2026-10-21");
    let aws_pinned = AnthropicAws::builder("us-west-2", "wrkspc_test")
        .bearer_key("test")
        .base_url(server.uri())
        .protocol(pinned(AnthropicApiVersion::V2023_06_01))
        .build()
        .unwrap()
        .model("claude-opus-5");
    assert_eq!(sent_version(&server, &aws_pinned).await, "2023-06-01");
}

#[tokio::test]
async fn compatible_endpoints_default_to_2023_06_01_and_can_opt_in() {
    let server = server().await;
    let compatible = |protocol| {
        try_compatible_model(
            &server.uri(),
            "future-provider",
            "future-model",
            "app.future.messages",
            AnthropicCompatibleAuth::None,
            anthropic_capabilities(ReplayPolicy::IfValid),
            protocol,
        )
        .unwrap()
    };
    assert_eq!(
        sent_version(&server, &compatible(anthropic_protocol())).await,
        "2023-06-01"
    );
    assert_eq!(
        sent_version(
            &server,
            &compatible(pinned(AnthropicApiVersion::V2026_10_21))
        )
        .await,
        "2026-10-21"
    );
}

#[tokio::test]
async fn sampling_under_2026_10_21_is_rejected_before_dispatch_naming_the_version() {
    let server = server().await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-5");
    let mut temperature = Request::new(Vec::new());
    temperature.inference.temperature = Some(1.0);
    let mut top_p = Request::new(Vec::new());
    top_p.inference.top_p = Some(0.95);
    for request in [temperature, top_p] {
        let error = model
            .complete(request, AbortSignal::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind, ModelErrorKind::Unsupported);
        assert!(error.message.contains("2026-10-21"), "{}", error.message);
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[test]
fn sampling_capabilities_require_2023_06_01() {
    let mut sampling = anthropic_capabilities(ReplayPolicy::IfValid);
    sampling.features |= Capability::TEMPERATURE;
    let error = common::try_anthropic_model(
        "https://api.anthropic.com/v1",
        "claude-opus-5",
        sampling.clone(),
        anthropic_protocol(),
        None,
    )
    .err()
    .unwrap();
    assert!(error.message.contains("2026-10-21"), "{}", error.message);
    assert!(
        common::try_anthropic_model(
            "https://api.anthropic.com/v1",
            "claude-sonnet-4-6",
            sampling.clone(),
            legacy_sampling_protocol(),
            None,
        )
        .is_ok()
    );
    assert!(
        common::try_compatible_model(
            "https://compatible.example/v1",
            "future-provider",
            "future-model",
            "app.future.messages",
            AnthropicCompatibleAuth::None,
            sampling.clone(),
            anthropic_protocol(),
        )
        .is_ok()
    );
    assert!(
        common::try_compatible_model(
            "https://compatible.example/v1",
            "future-provider",
            "future-model",
            "app.future.messages",
            AnthropicCompatibleAuth::None,
            sampling,
            pinned(AnthropicApiVersion::V2026_10_21),
        )
        .is_err()
    );
}

#[tokio::test]
async fn one_hour_cache_ttl_adds_no_beta_header() {
    let server = server().await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-5");
    let request = Request::new(vec![oven_sdk::HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("cached prefix")),
    ]))])
    .with_anthropic_options(AnthropicRequestOptions {
        cache_control: Some(AnthropicCacheControl {
            ttl: AnthropicCacheTtl::OneHour,
        }),
        ..Default::default()
    });
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let request = &server.received_requests().await.unwrap()[0];
    assert!(request.headers.get("anthropic-beta").is_none());
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(
        body["cache_control"],
        serde_json::json!({"type":"ephemeral","ttl":"1h"})
    );
}
