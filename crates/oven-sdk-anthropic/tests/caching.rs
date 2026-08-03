use std::collections::BTreeMap;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, CustomPart, Finish, FinishReason,
    HistoryTurn, InputPart, LanguageModel, ReasoningPart, Request, SystemMessage, SystemPart,
    TextPart, ToolCallPart, ToolContent, ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};
use oven_sdk_anthropic::{
    AnthropicCacheControl, AnthropicCacheTtl, AnthropicRequestExt, AnthropicRequestOptions,
};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn marker(ttl: AnthropicCacheTtl) -> serde_json::Value {
    serde_json::to_value(AnthropicRequestOptions {
        cache_control: Some(AnthropicCacheControl { ttl }),
        ..Default::default()
    })
    .unwrap()
}

fn response() -> &'static str {
    "event: message_start\ndata: {\"message\":{}}\n\nevent: message_stop\ndata: {}\n\n"
}

#[test]
fn caching_is_explicit_capability_gated() {
    let request = Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
        cache_control: Some(AnthropicCacheControl {
            ttl: AnthropicCacheTtl::FiveMinutes,
        }),
        ..Default::default()
    });
    let mut protocol = common::anthropic_protocol();
    protocol.thinking = oven_sdk_anthropic::AnthropicThinkingSupport::None;
    protocol.thinking_disable_allowed = false;
    protocol.effort = false;
    let without_caching = Anthropic::builder()
        .capabilities(common::conservative_capabilities())
        .protocol(protocol)
        .build()
        .unwrap()
        .model("claude-future-99");
    assert!(without_caching.validate_request(&request).is_err());
}

#[tokio::test]
async fn system_and_user_message_markers_lower_to_final_eligible_blocks() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut system = SystemMessage::new(vec![SystemPart::Text(TextPart::new("system"))]);
    system.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::OneHour))]);
    let mut user = UserMessage::new(vec![InputPart::Text(TextPart::new("user"))]);
    user.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
    oven_sdk::LanguageModel::complete(
        &model,
        Request::new(vec![HistoryTurn::system(system), HistoryTurn::user(user)]),
        AbortSignal::default(),
    )
    .await
    .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"]["ttl"],
        "5m"
    );
}

#[tokio::test]
async fn marker_without_eligible_block_is_omitted_with_stream_start_warning() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut system = SystemMessage::new(Vec::new());
    system.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
    let mut stream = model
        .stream(
            Request::new(vec![HistoryTurn::system(system)]),
            AbortSignal::default(),
        )
        .await
        .unwrap()
        .stream;
    assert!(
        matches!(stream.next().await, Some(Ok(oven_sdk::StreamPart::StreamStart { warnings })) if warnings.iter().any(|warning| warning.contains("omitted")))
    );
}

#[tokio::test]
async fn ineligible_message_markers_do_not_consume_cache_slots() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");

    let mut system_custom = CustomPart::new("test.system", serde_json::json!(null));
    system_custom.metadata = Some(BTreeMap::new());
    let mut system = SystemMessage::new(vec![SystemPart::Custom(system_custom)]);
    system.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::OneHour))]);

    let mut user_custom = CustomPart::new("test.user", serde_json::json!(null));
    user_custom.metadata = Some(BTreeMap::new());
    let mut user = UserMessage::new(vec![InputPart::Custom(user_custom)]);
    user.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::OneHour))]);

    let mut assistant = AssistantMessage::new(vec![AssistantPart::Reasoning(ReasoningPart::new(
        "thinking",
    ))]);
    assistant.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::OneHour))]);

    let schema = oven_sdk::JsonSchema::new(serde_json::json!({"type":"object"})).unwrap();
    let mut request = Request::new(vec![
        HistoryTurn::system(system),
        HistoryTurn::user(user),
        HistoryTurn::assistant(CompletedTurn::new(
            assistant,
            Finish::new(Default::default(), FinishReason::Stop),
        )),
    ]);
    for index in 0..4 {
        let mut tool = ToolDefinition::new(format!("tool{index}"), "tool", schema.clone());
        tool.provider_options =
            BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
        request.tools.push(tool);
    }

    assert!(model.validate_request(&request).is_ok());
    let mut stream = model
        .stream(request, AbortSignal::default())
        .await
        .unwrap()
        .stream;
    assert!(matches!(
        stream.next().await,
        Some(Ok(oven_sdk::StreamPart::StreamStart { warnings }))
            if warnings.iter().filter(|warning| warning.contains("omitted")).count() == 4
    ));

    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert!(body.get("system").is_none());
    assert!(body["messages"].as_array().unwrap().is_empty());
    assert_eq!(body["tools"].as_array().unwrap().len(), 4);
    assert!(
        body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .all(|tool| tool["cache_control"]["ttl"] == "5m")
    );
}

#[test]
fn marked_empty_text_is_rejected_consistently_in_all_text_roles() {
    let model = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");

    let marked_empty = || {
        let mut text = TextPart::new("");
        text.metadata = Some(BTreeMap::from([(
            "anthropic".into(),
            marker(AnthropicCacheTtl::FiveMinutes),
        )]));
        text
    };
    let requests = [
        Request::new(vec![HistoryTurn::system(SystemMessage::new(vec![
            SystemPart::Text(marked_empty()),
        ]))]),
        Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(marked_empty()),
        ]))]),
        Request::new(vec![HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![AssistantPart::Text(marked_empty())]),
            Finish::new(Default::default(), FinishReason::Stop),
        ))]),
    ];
    for request in requests {
        let error = model
            .validate_request(&request)
            .expect_err("marked empty text must be rejected");
        assert!(error.message.contains("empty text"));
    }
}

#[tokio::test]
async fn unmarked_empty_text_is_filtered_consistently_in_all_text_roles() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(TextPart::new(
            "",
        ))])),
        HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(""))])),
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![AssistantPart::Text(TextPart::new(""))]),
            Finish::new(Default::default(), FinishReason::Stop),
        )),
    ]);
    oven_sdk::LanguageModel::complete(&model, request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert!(body.get("system").is_none());
    assert!(body["messages"].as_array().unwrap().is_empty());
}

#[test]
fn ttl_ordering_is_rejected_before_network() {
    let model = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut first = TextPart::new("first");
    first.metadata = Some(BTreeMap::from([(
        "anthropic".into(),
        marker(AnthropicCacheTtl::FiveMinutes),
    )]));
    let mut second = TextPart::new("second");
    second.metadata = Some(BTreeMap::from([(
        "anthropic".into(),
        marker(AnthropicCacheTtl::OneHour),
    )]));
    assert!(
        model
            .validate_request(&Request::new(vec![HistoryTurn::user(UserMessage::new(
                vec![InputPart::Text(first), InputPart::Text(second)]
            ))]))
            .is_err()
    );
}

#[test]
fn four_cache_slots_are_accepted_and_fifth_is_rejected() {
    let model = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut request = Request::new(Vec::new());
    for index in 0..4 {
        let mut tool = ToolDefinition::new(
            format!("tool{index}"),
            "tool",
            oven_sdk::JsonSchema::new(serde_json::json!({"type":"object"})).unwrap(),
        );
        tool.provider_options =
            BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
        request.tools.push(tool);
    }
    assert!(model.validate_request(&request).is_ok());
    request.tools.push(ToolDefinition::new(
        "tool4",
        "tool",
        oven_sdk::JsonSchema::new(serde_json::json!({"type":"object"})).unwrap(),
    ));
    request.tools.last_mut().unwrap().provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
    assert!(model.validate_request(&request).is_err());
}

#[tokio::test]
async fn assistant_text_and_tool_use_part_markers_lower_to_their_wire_blocks() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut text = TextPart::new("assistant text");
    text.metadata = Some(BTreeMap::from([(
        "anthropic".into(),
        marker(AnthropicCacheTtl::OneHour),
    )]));
    let mut call = ToolCallPart::new("call", "lookup", serde_json::json!({}));
    call.metadata = Some(BTreeMap::from([(
        "anthropic".into(),
        marker(AnthropicCacheTtl::FiveMinutes),
    )]));
    let assistant = CompletedTurn::new(
        AssistantMessage::new(vec![
            AssistantPart::Text(text),
            AssistantPart::ToolCall(call),
        ]),
        Finish::new(Default::default(), FinishReason::ToolCalls),
    );
    let request = Request::new(vec![
        HistoryTurn::assistant(assistant),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call",
            ToolContent::Text("done".into()),
        )])),
    ]);
    oven_sdk::LanguageModel::complete(&model, request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"]["ttl"],
        "1h"
    );
    assert_eq!(
        body["messages"][0]["content"][1]["cache_control"]["ttl"],
        "5m"
    );
}

#[tokio::test]
async fn duplicate_part_and_message_marker_uses_part_once_and_warns() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut text = TextPart::new("user");
    text.metadata = Some(BTreeMap::from([(
        "anthropic".into(),
        marker(AnthropicCacheTtl::OneHour),
    )]));
    let mut user = UserMessage::new(vec![InputPart::Text(text)]);
    user.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
    let mut response = model
        .stream(
            Request::new(vec![HistoryTurn::user(user)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(
        matches!(response.stream.next().await, Some(Ok(oven_sdk::StreamPart::StreamStart { warnings })) if warnings.iter().any(|warning| warning.contains("precedence")))
    );
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"]["ttl"],
        "1h"
    );
}

#[tokio::test]
async fn independent_part_and_message_markers_are_preserved_and_counted() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");

    let mut history = Vec::new();
    for index in 0..2 {
        let mut marked = TextPart::new(format!("marked {index}"));
        marked.metadata = Some(BTreeMap::from([(
            "anthropic".into(),
            marker(AnthropicCacheTtl::FiveMinutes),
        )]));
        let mut user = UserMessage::new(vec![
            InputPart::Text(marked),
            InputPart::Text(TextPart::new("unmarked")),
        ]);
        user.provider_options =
            BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
        history.push(HistoryTurn::user(user));
    }
    let request = Request::new(history.clone());

    assert!(model.validate_request(&request).is_ok());
    let mut response = model.stream(request, AbortSignal::default()).await.unwrap();
    assert!(matches!(
        response.stream.next().await,
        Some(Ok(oven_sdk::StreamPart::StreamStart { warnings }))
            if !warnings.iter().any(|warning| warning.contains("precedence"))
    ));

    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    for message in messages {
        let content = message["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["cache_control"]["ttl"], "5m");
        assert_eq!(content[1]["cache_control"]["ttl"], "5m");
    }

    history.push(history[0].clone());
    assert!(model.validate_request(&Request::new(history)).is_err());
}

#[tokio::test]
async fn reversed_history_uses_system_before_messages_for_cache_order() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut system = SystemMessage::new(vec![SystemPart::Text(TextPart::new("system"))]);
    system.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::OneHour))]);
    let mut user = UserMessage::new(vec![InputPart::Text(TextPart::new("user"))]);
    user.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
    oven_sdk::LanguageModel::complete(
        &model,
        Request::new(vec![HistoryTurn::user(user), HistoryTurn::system(system)]),
        AbortSignal::default(),
    )
    .await
    .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"]["ttl"],
        "5m"
    );
}

#[test]
fn reversed_history_rejects_ttl_order_that_is_invalid_on_the_wire() {
    let model = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut user = UserMessage::new(vec![InputPart::Text(TextPart::new("user"))]);
    user.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::OneHour))]);
    let mut system = SystemMessage::new(vec![SystemPart::Text(TextPart::new("system"))]);
    system.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
    let error = model
        .validate_request(&Request::new(vec![
            HistoryTurn::user(user),
            HistoryTurn::system(system),
        ]))
        .expect_err("wire-order TTL regression must be rejected");
    assert!(error.message.contains("one-hour"));
}

#[tokio::test]
async fn assistant_message_marker_skips_thinking_and_targets_last_eligible_block() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut message = AssistantMessage::new(vec![
        AssistantPart::Reasoning(ReasoningPart::new("think")),
        AssistantPart::Text(TextPart::new("answer")),
    ]);
    message.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
    let assistant =
        CompletedTurn::new(message, Finish::new(Default::default(), FinishReason::Stop));
    oven_sdk::LanguageModel::complete(
        &model,
        Request::new(vec![HistoryTurn::assistant(assistant)]),
        AbortSignal::default(),
    )
    .await
    .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"]["ttl"],
        "5m"
    );
}

#[tokio::test]
async fn reasoning_only_assistant_marker_is_omitted_warned_and_not_counted() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut message =
        AssistantMessage::new(vec![AssistantPart::Reasoning(ReasoningPart::new("think"))]);
    message.provider_options =
        BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
    let assistant =
        CompletedTurn::new(message, Finish::new(Default::default(), FinishReason::Stop));
    let mut request = Request::new(vec![HistoryTurn::assistant(assistant)]);
    for index in 0..4 {
        let mut tool = ToolDefinition::new(
            format!("tool{index}"),
            "tool",
            oven_sdk::JsonSchema::new(serde_json::json!({"type":"object"})).unwrap(),
        );
        tool.provider_options =
            BTreeMap::from([("anthropic".into(), marker(AnthropicCacheTtl::FiveMinutes))]);
        request.tools.push(tool);
    }
    let mut stream = model
        .stream(request, AbortSignal::default())
        .await
        .unwrap()
        .stream;
    assert!(
        matches!(stream.next().await, Some(Ok(oven_sdk::StreamPart::StreamStart { warnings })) if warnings.iter().any(|warning| warning.contains("assistant cache marker omitted")))
    );
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert!(
        body["messages"][0]["content"][0]
            .get("cache_control")
            .is_none()
    );
}
mod common;

use common::Anthropic;
