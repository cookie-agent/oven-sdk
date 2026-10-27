pub mod common;

use std::collections::BTreeMap;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, DeferredToolsCapability, Finish,
    FinishReason, HistoryTurn, InputPart, JsonSchema, LanguageModel, Request, StreamPart, TextPart,
    ToolCallPart, ToolContent, ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};
use oven_sdk_openai::OpenAiResponsesModel;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        name,
        format!("The {name} tool."),
        JsonSchema::new(serde_json::json!({"type": "object"})).expect("schema"),
    )
}

fn model(server: &MockServer, deferred_tools: DeferredToolsCapability) -> OpenAiResponsesModel {
    let mut config = common::official_responses_config(server, "gpt-5.5");
    config.model.capabilities.deferred_tools = deferred_tools;
    OpenAiResponsesModel::new(config).expect("model")
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

fn load(names: &[&str]) -> ToolContent {
    ToolContent::LoadTools {
        names: names.iter().map(|name| (*name).to_owned()).collect(),
        text: format!("Loaded tools: {}", names.join(", ")),
    }
}

fn user(text: &str) -> HistoryTurn {
    HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(text))]))
}

/// A history that searched, loaded `get_weather`, called it, and got a result.
fn loaded_history(weather_call: ToolCallPart) -> Vec<HistoryTurn> {
    vec![
        user("What is the weather in Paris?"),
        assistant(ToolCallPart::new(
            "call_s",
            "tool_search",
            serde_json::json!({}),
        )),
        result("call_s", load(&["get_weather"])),
        assistant(weather_call),
        result("call_w", ToolContent::Text("Sunny".into())),
    ]
}

async fn send(
    model: &OpenAiResponsesModel,
    server: &MockServer,
    request: Request,
) -> (serde_json::Value, Vec<String>) {
    let before = server.received_requests().await.expect("requests").len();
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
    let received = server.received_requests().await.expect("requests");
    (
        serde_json::from_slice(&received[before].body).expect("body"),
        warnings,
    )
}

fn items_of_type<'a>(body: &'a serde_json::Value, kind: &str) -> Vec<&'a serde_json::Value> {
    body["input"]
        .as_array()
        .expect("input")
        .iter()
        .filter(|item| item["type"] == kind)
        .collect()
}

#[tokio::test]
async fn deferred_tools_load_through_an_additional_tools_item() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = model(&server, DeferredToolsCapability::Native);
    let request = Request::new(loaded_history(ToolCallPart::new(
        "call_w",
        "get_weather",
        serde_json::json!({"city": "Paris"}),
    )))
    .with_tools(vec![
        tool("tool_search"),
        tool("get_weather").deferred(),
        tool("get_time").deferred(),
    ]);
    let (body, _) = send(&model, &server, request).await;
    let names: Vec<_> = body["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .map(|tool| tool["name"].as_str().expect("name"))
        .collect();
    assert_eq!(
        names,
        ["tool_search"],
        "deferred tools are never declared up front"
    );

    let input = body["input"].as_array().expect("input");
    let load_output = input
        .iter()
        .position(|item| item["type"] == "function_call_output" && item["call_id"] == "call_s")
        .expect("load output");
    assert_eq!(input[load_output]["output"], "Loaded tools: get_weather");
    let item = &input[load_output + 1];
    assert_eq!(item["type"], "additional_tools");
    assert_eq!(item["role"], "developer");
    assert_eq!(item["tools"][0]["name"], "get_weather");
    assert_eq!(item["tools"].as_array().expect("tools").len(), 1);

    let calls = items_of_type(&body, "function_call");
    let weather = calls
        .iter()
        .find(|call| call["name"] == "get_weather")
        .expect("call");
    assert_eq!(
        weather["namespace"], "get_weather",
        "a call recorded without a namespace gets its tool's name"
    );
    let search = calls
        .iter()
        .find(|call| call["name"] == "tool_search")
        .expect("call");
    assert!(search.get("namespace").is_none());
}

#[tokio::test]
async fn a_recorded_namespace_is_sent_only_while_its_tool_is_loaded() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = model(&server, DeferredToolsCapability::Native);
    let mut call = ToolCallPart::new("call_w", "get_weather", serde_json::json!({}));
    call.metadata = Some(BTreeMap::from([(
        "openai.namespace".to_owned(),
        serde_json::json!("weather_ns"),
    )]));

    let loaded = Request::new(loaded_history(call.clone()))
        .with_tools(vec![tool("tool_search"), tool("get_weather").deferred()]);
    let (body, _) = send(&model, &server, loaded).await;
    let calls = items_of_type(&body, "function_call");
    assert_eq!(calls[1]["namespace"], "weather_ns");

    // Folded: the tool is eager now, so it lives in the default namespace.
    let folded = Request::new(loaded_history(call))
        .with_tools(vec![tool("tool_search"), tool("get_weather")]);
    let (body, _) = send(&model, &server, folded).await;
    assert!(items_of_type(&body, "additional_tools").is_empty());
    assert!(
        items_of_type(&body, "function_call")
            .iter()
            .all(|call| call.get("namespace").is_none())
    );
}

#[tokio::test]
async fn without_native_support_every_tool_is_sent_up_front() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = model(&server, DeferredToolsCapability::Unsupported);
    let request = Request::new(loaded_history(ToolCallPart::new(
        "call_w",
        "get_weather",
        serde_json::json!({}),
    )))
    .with_tools(vec![tool("tool_search"), tool("get_weather").deferred()]);
    let (body, warnings) = send(&model, &server, request).await;
    assert_eq!(body["tools"].as_array().expect("tools").len(), 2);
    assert!(items_of_type(&body, "additional_tools").is_empty());
    assert!(
        items_of_type(&body, "function_call")
            .iter()
            .all(|call| call.get("namespace").is_none())
    );
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("sent up front"))
    );
}

const NAMESPACED_CALL: &str = r#"{"type":"function_call","id":"fc_1","call_id":"call_w","name":"get_weather","namespace":"get_weather","arguments":"{\"city\":\"Paris\"}","status":"completed"}"#;

fn namespaced_call_document() -> String {
    let added = r#"{"type":"function_call","id":"fc_1","call_id":"call_w","name":"get_weather","namespace":"get_weather","arguments":""}"#;
    format!(
        concat!(
            "event: response.created\ndata: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp_1\",\"model\":\"gpt-5.5\"}}}}\n\n",
            "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{added}}}\n\n",
            "event: response.output_item.done\ndata: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{call}}}\n\n",
            "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_1\",\"model\":\"gpt-5.5\",\"status\":\"completed\",\"output\":[{call}],\"usage\":{{\"input_tokens\":2,\"output_tokens\":3}}}}}}\n\n"
        ),
        added = added,
        call = NAMESPACED_CALL,
    )
}

#[tokio::test]
async fn a_streamed_namespace_is_recorded_and_replayed_natively() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(namespaced_call_document(), "text/event-stream"),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = model(&server, DeferredToolsCapability::Native);
    let tools = || vec![tool("tool_search"), tool("get_weather").deferred()];
    let mut history = vec![
        user("What is the weather in Paris?"),
        assistant(ToolCallPart::new(
            "call_s",
            "tool_search",
            serde_json::json!({}),
        )),
        result("call_s", load(&["get_weather"])),
    ];
    let turn = model
        .complete(
            Request::new(history.clone()).with_tools(tools()),
            AbortSignal::default(),
        )
        .await
        .expect("complete")
        .turn;
    let AssistantPart::ToolCall(call) = &turn.message.content[0] else {
        panic!("expected a tool call, got {:?}", turn.message.content);
    };
    assert_eq!(
        call.metadata
            .as_ref()
            .and_then(|metadata| metadata.get("openai.namespace")),
        Some(&serde_json::json!("get_weather"))
    );
    assert!(
        turn.finish.native_replay.is_some(),
        "the namespaced item replays natively"
    );

    history.push(HistoryTurn::assistant(turn.clone()));
    history.push(result("call_w", ToolContent::Text("Sunny".into())));
    let (body, _) = send(
        &model,
        &server,
        Request::new(history.clone()).with_tools(tools()),
    )
    .await;
    let replayed = items_of_type(&body, "function_call");
    let weather = replayed
        .iter()
        .find(|call| call["name"] == "get_weather")
        .expect("call");
    assert_eq!(weather["id"], "fc_1", "replayed from the native artifact");
    assert_eq!(weather["namespace"], "get_weather");

    let folded = Request::new(history).with_tools(vec![tool("tool_search"), tool("get_weather")]);
    let (body, _) = send(&model, &server, folded).await;
    let weather = items_of_type(&body, "function_call")
        .into_iter()
        .find(|call| call["name"] == "get_weather")
        .expect("call");
    assert_eq!(weather["id"], "fc_1");
    assert!(
        weather.get("namespace").is_none(),
        "a folded tool's call drops its namespace"
    );
}

mod kimi {
    use super::*;
    use oven_sdk_openai::{CompatibleChatDeferredTools, OpenAiCompatibleChatModel};

    fn model(
        server: &MockServer,
        deferred_tools: DeferredToolsCapability,
        dialect: CompatibleChatDeferredTools,
    ) -> Result<OpenAiCompatibleChatModel, Box<oven_sdk::ModelError>> {
        let mut config = common::compatible_config(server, "kimi-k3");
        config.model.capabilities.deferred_tools = deferred_tools;
        config.settings.deferred_tools = dialect;
        OpenAiCompatibleChatModel::new(config).map_err(Box::new)
    }

    async fn send(
        model: &OpenAiCompatibleChatModel,
        server: &MockServer,
        request: Request,
    ) -> (serde_json::Value, Vec<String>) {
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
        let received = server.received_requests().await.expect("requests");
        (
            serde_json::from_slice(&received.last().expect("request").body).expect("body"),
            warnings,
        )
    }

    #[tokio::test]
    async fn loads_append_a_system_message_carrying_only_tools() {
        let server = MockServer::start().await;
        common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
        let model = model(
            &server,
            DeferredToolsCapability::Native,
            CompatibleChatDeferredTools::SystemMessage,
        )
        .expect("model");
        let request = Request::new(loaded_history(ToolCallPart::new(
            "call_w",
            "get_weather",
            serde_json::json!({}),
        )))
        .with_tools(vec![
            tool("tool_search"),
            tool("get_weather").deferred(),
            tool("get_time").deferred(),
        ]);
        let (body, _) = send(&model, &server, request).await;
        let names: Vec<_> = body["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .map(|tool| tool["function"]["name"].as_str().expect("name"))
            .collect();
        assert_eq!(names, ["tool_search"]);
        let messages = body["messages"].as_array().expect("messages");
        let load = messages
            .iter()
            .position(|message| message["role"] == "tool" && message["tool_call_id"] == "call_s")
            .expect("load result");
        assert_eq!(messages[load]["content"], "Loaded tools: get_weather");
        let tools_message = &messages[load + 1];
        assert_eq!(tools_message["role"], "system");
        assert!(
            tools_message.get("content").is_none(),
            "Kimi rejects content beside tools"
        );
        assert_eq!(tools_message["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(tools_message["tools"].as_array().expect("tools").len(), 1);
        assert_eq!(
            messages
                .iter()
                .filter(|message| message.get("tools").is_some())
                .count(),
            1,
            "only the load adds a tools message"
        );
    }

    #[tokio::test]
    async fn native_deferral_needs_a_dialect() {
        let server = MockServer::start().await;
        assert!(
            model(
                &server,
                DeferredToolsCapability::Native,
                CompatibleChatDeferredTools::Unsupported,
            )
            .is_err()
        );
        let mut official = common::official_chat_config(&server, "gpt-4o-mini");
        official.model.capabilities.deferred_tools = DeferredToolsCapability::Native;
        assert!(oven_sdk_openai::OpenAiChatModel::new(official).is_err());
    }

    #[tokio::test]
    async fn a_dialect_without_native_declaration_sends_tools_up_front() {
        let server = MockServer::start().await;
        common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
        let model = model(
            &server,
            DeferredToolsCapability::Unsupported,
            CompatibleChatDeferredTools::SystemMessage,
        )
        .expect("model");
        let request = Request::new(loaded_history(ToolCallPart::new(
            "call_w",
            "get_weather",
            serde_json::json!({}),
        )))
        .with_tools(vec![tool("tool_search"), tool("get_weather").deferred()]);
        let (body, warnings) = send(&model, &server, request).await;
        assert_eq!(body["tools"].as_array().expect("tools").len(), 2);
        assert!(
            body["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .all(|message| message.get("tools").is_none())
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("sent up front"))
        );
    }
}
