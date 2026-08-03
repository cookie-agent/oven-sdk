mod common;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, FilePart, FileSource, HistoryTurn, InferenceOptions, InputPart, JsonSchema,
    LanguageModel, Request, ResponseFormat, StreamPart, TextPart, ToolChoice, ToolDefinition,
    UserMessage,
};
use oven_sdk_azure::{
    AzureApiRoute, AzureOpenAiChatOptions, AzureOpenAiChatRequestExt, AzureOpenAiResponsesOptions,
    AzureOpenAiResponsesRequestExt,
};
use wiremock::MockServer;

#[tokio::test]
async fn chat_encodes_tools_structured_output_media_usage_and_open_labels() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/chat/completions",
        common::chat_document("{}"),
    )
    .await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .chat("arbitrary-deployment", common::gpt4o())
        .unwrap();
    let schema = JsonSchema::new(serde_json::json!({
        "type":"object",
        "properties":{"query":{"type":"string"}},
        "required":["query"],
        "additionalProperties":false
    }))
    .unwrap();
    let mut tool = ToolDefinition::new("lookup", "find", schema.clone());
    tool.provider_options
        .insert("azure_openai".into(), serde_json::json!({"strict":true}));
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("look")),
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(bytes::Bytes::from_static(b"png")),
        )),
    ]))])
    .with_tools(vec![tool])
    .with_tool_choice(ToolChoice::Tool("lookup".into()))
    .with_response_format(ResponseFormat::structured(schema))
    .with_azure_openai_chat_options(AzureOpenAiChatOptions {
        service_tier: Some("future-tier".into()),
        verbosity: Some("future-verbosity".into()),
        parallel_tool_calls: Some(false),
        ..Default::default()
    });
    let result = model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.response.request_id.as_deref(), Some("req_azure_1"));
    assert_eq!(result.turn.finish.usage.input_tokens, Some(2));
    let request = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["model"], "arbitrary-deployment");
    assert_eq!(body["tools"][0]["function"]["strict"], true);
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["service_tier"], "future-tier");
    assert_eq!(body["verbosity"], "future-verbosity");
    assert_eq!(body["messages"][0]["content"][1]["type"], "image_url");
}

#[tokio::test]
async fn chat_emits_prompt_and_completion_filter_events() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/chat/completions",
        common::chat_document("ok"),
    )
    .await;
    let mut response = common::provider(&server, AzureApiRoute::V1)
        .chat("deployment", common::gpt4o())
        .unwrap()
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut names = Vec::new();
    while let Some(item) = response.stream.next().await {
        if let StreamPart::ProviderEvent { name, .. } = item.unwrap() {
            names.push(name);
        }
    }
    assert_eq!(
        names,
        [
            "azure.prompt_filter_results",
            "azure.content_filter_results"
        ]
    );
}

#[tokio::test]
async fn responses_encodes_reasoning_structured_output_and_emits_filter_events() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/responses",
        common::responses_document("ok"),
    )
    .await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap();
    let schema = JsonSchema::new(serde_json::json!({
        "type":"object",
        "properties":{},
        "required":[],
        "additionalProperties":false
    }))
    .unwrap();
    let mut inference = InferenceOptions::new();
    inference.reasoning_effort = Some("future-effort".into());
    let request = Request::new(Vec::new())
        .with_response_format(ResponseFormat::structured(schema))
        .with_inference(inference)
        .with_azure_openai_responses_options(AzureOpenAiResponsesOptions {
            reasoning_summary: Some("future-summary".into()),
            service_tier: Some("future-tier".into()),
            ..Default::default()
        });
    let mut response = model.stream(request, AbortSignal::default()).await.unwrap();
    let mut filter_events = 0;
    let mut finish = false;
    while let Some(item) = response.stream.next().await {
        match item.unwrap() {
            StreamPart::ProviderEvent { name, .. } if name == "azure.content_filters" => {
                filter_events += 1;
            }
            StreamPart::Finish { .. } => finish = true,
            _ => {}
        }
    }
    assert_eq!(filter_events, 2);
    assert!(finish);
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["model"], "deployment");
    assert_eq!(body["reasoning"]["effort"], "future-effort");
    assert_eq!(body["reasoning"]["summary"], "future-summary");
    assert_eq!(body["text"]["format"]["type"], "json_schema");
    assert_eq!(body["store"], false);
}

#[tokio::test]
async fn conservative_configuration_rejects_unclaimed_capabilities_before_network() {
    let server = MockServer::start().await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .chat("custom-deployment", common::conservative())
        .unwrap();
    let schema = JsonSchema::new(serde_json::json!({
        "type":"object","properties":{},"required":[],"additionalProperties":false
    }))
    .unwrap();
    let request = Request::new(Vec::new()).with_response_format(ResponseFormat::structured(schema));
    assert!(model.stream(request, AbortSignal::default()).await.is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
}
