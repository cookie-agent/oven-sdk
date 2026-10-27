mod common;

use std::collections::BTreeMap;

use common::Anthropic;
use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, DeferredToolsCapability, Finish,
    FinishReason, HistoryTurn, InputPart, JsonSchema, LanguageModel, ReplayPolicy, Request,
    StreamPart, TextPart, ToolCallPart, ToolChoice, ToolContent, ToolDefinition, ToolMessage,
    ToolResultPart, UserMessage,
};
use oven_sdk_anthropic::{AnthropicCacheControl, AnthropicCacheTtl, AnthropicRequestOptions};
use wiremock::{Mock, MockServer, ResponseTemplate};

const RESPONSE: &str =
    "event: message_start\ndata: {\"message\":{}}\n\nevent: message_stop\ndata: {}\n\n";

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        name,
        format!("The {name} tool."),
        JsonSchema::new(serde_json::json!({"type": "object"})).expect("schema"),
    )
}

fn marked(mut tool: ToolDefinition) -> ToolDefinition {
    tool.provider_options = BTreeMap::from([(
        "anthropic".into(),
        serde_json::to_value(AnthropicRequestOptions {
            cache_control: Some(AnthropicCacheControl {
                ttl: AnthropicCacheTtl::FiveMinutes,
            }),
            ..Default::default()
        })
        .expect("options"),
    )]);
    tool
}

fn load(id: &str, names: &[&str]) -> [HistoryTurn; 2] {
    [
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![AssistantPart::ToolCall(ToolCallPart::new(
                id,
                "tool_search",
                serde_json::json!({"query": "weather"}),
            ))]),
            Finish::new(Default::default(), FinishReason::ToolCalls),
        )),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            id,
            ToolContent::LoadTools {
                names: names.iter().map(|name| (*name).to_owned()).collect(),
                text: format!("Loaded tools: {}", names.join(", ")),
            },
        )])),
    ]
}

fn history(loads: &[(&str, &[&str])]) -> Vec<HistoryTurn> {
    let mut history = vec![HistoryTurn::user(UserMessage::new(vec![InputPart::Text(
        TextPart::new("What is the weather in Paris?"),
    )]))];
    for (id, names) in loads {
        history.extend(load(id, names));
    }
    history
}

/// Sends `request` to a mock server and returns the body and the warnings.
async fn send(
    deferred_tools: DeferredToolsCapability,
    request: Request,
) -> (serde_json::Value, Vec<String>) {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(RESPONSE))
        .mount(&server)
        .await;
    let mut capabilities = common::anthropic_capabilities(ReplayPolicy::IfValid);
    capabilities.deferred_tools = deferred_tools;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .capabilities(capabilities)
        .build()
        .expect("client")
        .model("claude-sonnet-5-5");
    let mut stream = model
        .stream(request, AbortSignal::default())
        .await
        .expect("stream")
        .stream;
    let warnings = match stream.next().await {
        Some(Ok(StreamPart::StreamStart { warnings })) => warnings,
        other => panic!("expected stream_start, got {other:?}"),
    };
    while stream.next().await.is_some() {}
    let received = &server.received_requests().await.expect("requests")[0];
    (
        serde_json::from_slice(&received.body).expect("body"),
        warnings,
    )
}

#[tokio::test]
async fn deferred_tools_are_flagged_and_loads_become_tool_references() {
    let request = Request::new(history(&[("call_1", &["get_weather", "gone"])])).with_tools(vec![
        tool("tool_search"),
        tool("get_weather").deferred(),
        tool("get_time").deferred(),
    ]);
    let (body, warnings) = send(DeferredToolsCapability::Native, request).await;
    let tools = body["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 3, "every tool is declared");
    assert!(tools[0].get("defer_loading").is_none());
    assert_eq!(tools[1]["defer_loading"], true);
    assert_eq!(tools[2]["defer_loading"], true);
    let result = &body["messages"][2]["content"][0];
    assert_eq!(result["type"], "tool_result");
    assert_eq!(
        result["content"],
        serde_json::json!([{"type": "tool_reference", "tool_name": "get_weather"}]),
        "a load result holds only references, no text"
    );
    assert!(warnings.iter().any(|warning| warning.contains("`gone`")));
}

#[tokio::test]
async fn a_tool_loaded_again_is_referenced_once() {
    let request = Request::new(history(&[
        ("call_1", &["get_weather"]),
        ("call_2", &["get_weather"]),
    ]))
    .with_tools(vec![tool("tool_search"), tool("get_weather").deferred()]);
    let (body, _) = send(DeferredToolsCapability::Native, request).await;
    assert_eq!(
        body["messages"][2]["content"][0]["content"][0]["type"],
        "tool_reference"
    );
    assert_eq!(
        body["messages"][4]["content"][0]["content"], "Loaded tools: get_weather",
        "a result with nothing new to load is text"
    );
}

#[tokio::test]
async fn a_cache_marker_moves_off_a_deferred_tool() {
    let request = Request::new(history(&[])).with_tools(vec![
        tool("read"),
        tool("tool_search"),
        marked(tool("get_weather").deferred()),
    ]);
    let (body, _) = send(DeferredToolsCapability::Native, request).await;
    let tools = body["tools"].as_array().expect("tools");
    assert!(tools[2].get("cache_control").is_none());
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert!(tools[0].get("cache_control").is_none());

    let first = Request::new(history(&[])).with_tools(vec![
        marked(tool("get_weather").deferred()),
        tool("tool_search"),
    ]);
    let (body, _) = send(DeferredToolsCapability::Native, first).await;
    assert_eq!(body["tools"][1]["cache_control"]["type"], "ephemeral");
    assert!(body["tools"][0].get("cache_control").is_none());
}

#[tokio::test]
async fn without_native_support_every_tool_is_sent_up_front() {
    let request = Request::new(history(&[("call_1", &["get_weather"])]))
        .with_tools(vec![tool("tool_search"), tool("get_weather").deferred()]);
    let (body, warnings) = send(DeferredToolsCapability::Unsupported, request).await;
    assert!(
        body["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .all(|tool| tool.get("defer_loading").is_none())
    );
    assert_eq!(
        body["messages"][2]["content"][0]["content"],
        "Loaded tools: get_weather"
    );
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("sent up front"))
    );
}

#[tokio::test]
async fn loads_are_text_when_no_tools_are_sent() {
    let request = Request::new(history(&[("call_1", &["get_weather"])]))
        .with_tools(vec![tool("tool_search"), tool("get_weather").deferred()])
        .with_tool_choice(ToolChoice::None);
    let (body, _) = send(DeferredToolsCapability::Native, request).await;
    assert!(
        body.get("tools")
            .is_none_or(|tools| tools.as_array().is_some_and(Vec::is_empty))
    );
    assert_eq!(
        body["messages"][2]["content"][0]["content"],
        "Loaded tools: get_weather"
    );
}
