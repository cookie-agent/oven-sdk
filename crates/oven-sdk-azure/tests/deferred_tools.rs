mod common;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, DeferredToolsCapability, Finish,
    FinishReason, HistoryTurn, InputPart, JsonSchema, LanguageModel, Request, StreamPart, TextPart,
    ToolCallPart, ToolContent, ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};
use oven_sdk_azure::AzureApiRoute;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

const PATH: &str = "/openai/v1/responses";

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        name,
        format!("The {name} tool."),
        JsonSchema::new(serde_json::json!({"type": "object"})).expect("schema"),
    )
}

fn setup() -> common::ModelSetup {
    let mut setup = common::gpt5();
    setup.capabilities.deferred_tools = DeferredToolsCapability::Native;
    setup
}

fn assistant(call: ToolCallPart) -> HistoryTurn {
    HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::ToolCall(call)]),
        Finish::new(Default::default(), FinishReason::ToolCalls),
    ))
}

fn result(id: &str, content: ToolContent) -> HistoryTurn {
    HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(id, content)]))
}

fn searched_and_loaded() -> Vec<HistoryTurn> {
    vec![
        HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
            "What is the weather in Paris?",
        ))])),
        assistant(ToolCallPart::new(
            "call_s",
            "tool_search",
            serde_json::json!({}),
        )),
        result(
            "call_s",
            ToolContent::LoadTools {
                names: vec!["get_weather".into()],
                text: "Loaded tools: get_weather".into(),
            },
        ),
    ]
}

async fn body_of(server: &MockServer, index: usize) -> serde_json::Value {
    serde_json::from_slice(&server.received_requests().await.expect("requests")[index].body)
        .expect("body")
}

fn calls<'a>(body: &'a serde_json::Value, name: &str) -> Vec<&'a serde_json::Value> {
    body["input"]
        .as_array()
        .expect("input")
        .iter()
        .filter(|item| item["type"] == "function_call" && item["name"] == name)
        .collect()
}

#[tokio::test]
async fn deferred_tools_load_through_an_additional_tools_item() {
    let server = MockServer::start().await;
    common::mount(&server, PATH, common::responses_document("ok")).await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", setup())
        .expect("model");
    let mut history = searched_and_loaded();
    history.push(assistant(ToolCallPart::new(
        "call_w",
        "get_weather",
        serde_json::json!({}),
    )));
    history.push(result("call_w", ToolContent::Text("Sunny".into())));
    let request =
        Request::new(history).with_tools(vec![tool("tool_search"), tool("get_weather").deferred()]);
    let mut stream = model
        .stream(request, AbortSignal::default())
        .await
        .expect("stream")
        .stream;
    while stream.next().await.is_some() {}
    let body = body_of(&server, 0).await;
    assert_eq!(body["tools"].as_array().expect("tools").len(), 1);
    let input = body["input"].as_array().expect("input");
    let item = input
        .iter()
        .find(|item| item["type"] == "additional_tools")
        .expect("additional_tools item");
    assert_eq!(item["tools"][0]["name"], "get_weather");
    assert_eq!(calls(&body, "get_weather")[0]["namespace"], "get_weather");
}

#[tokio::test]
async fn a_streamed_namespace_survives_native_replay() {
    let call = r#"{"type":"function_call","id":"fc_1","call_id":"call_w","name":"get_weather","namespace":"get_weather","arguments":"{\"city\":\"Paris\"}","status":"completed"}"#;
    let completed = format!(
        r#"{{"type":"response.completed","response":{{"id":"resp_1","model":"gpt-5-mini","status":"completed","output":[{call}],"usage":{{"input_tokens":2,"output_tokens":3}}}}}}"#
    );
    let document = format!("data: {completed}\n\n");
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path(PATH))
        .respond_with(ResponseTemplate::new(200).set_body_raw(document, "text/event-stream"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    common::mount(&server, PATH, common::responses_document("ok")).await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", setup())
        .expect("model");
    let tools = || vec![tool("tool_search"), tool("get_weather").deferred()];
    let mut history = searched_and_loaded();
    let turn = model
        .complete(
            Request::new(history.clone()).with_tools(tools()),
            AbortSignal::default(),
        )
        .await
        .expect("complete")
        .turn;
    assert!(turn.finish.native_replay.is_some());
    history.push(HistoryTurn::assistant(turn));
    history.push(result("call_w", ToolContent::Text("Sunny".into())));

    let mut stream = model
        .stream(
            Request::new(history.clone()).with_tools(tools()),
            AbortSignal::default(),
        )
        .await
        .expect("stream")
        .stream;
    while stream.next().await.is_some() {}
    assert_eq!(
        calls(&body_of(&server, 1).await, "get_weather")[0]["namespace"],
        "get_weather"
    );

    let folded = Request::new(history).with_tools(vec![tool("tool_search"), tool("get_weather")]);
    let mut stream = model
        .stream(folded, AbortSignal::default())
        .await
        .expect("stream")
        .stream;
    let warnings = match stream.next().await {
        Some(Ok(StreamPart::StreamStart { warnings })) => warnings,
        other => panic!("expected stream_start, got {other:?}"),
    };
    while stream.next().await.is_some() {}
    assert!(
        warnings
            .iter()
            .all(|warning| !warning.contains("sent up front"))
    );
    assert!(
        calls(&body_of(&server, 2).await, "get_weather")[0]
            .get("namespace")
            .is_none()
    );
}

#[test]
fn azure_chat_rejects_native_deferral() {
    let mut setup = common::gpt4o();
    setup.capabilities.deferred_tools = DeferredToolsCapability::Native;
    let server_uri = "https://example.openai.azure.com";
    let config = common::chat_config(
        server_uri,
        AzureApiRoute::V1,
        "deployment",
        setup,
        oven_sdk_azure::AzureOpenAiAuth::ApiKey(oven_sdk::SecretString::new("secret")),
    );
    assert!(oven_sdk_azure::AzureOpenAiChatModel::new(config).is_err());
}
