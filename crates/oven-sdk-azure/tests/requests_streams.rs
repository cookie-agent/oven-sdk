mod common;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, ContentValue, FilePart,
    FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions, InputPart, JsonSchema,
    LanguageModel, Request, ResponseFormat, StreamPart, TextPart, ToolCallPart, ToolChoice,
    ToolContent, ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};
use oven_sdk_azure::{
    AzureApiRoute, AzureOpenAiChatOptions, AzureOpenAiChatRequestExt,
    AzureOpenAiPromptCacheBreakpointExt, AzureOpenAiPromptCacheMode, AzureOpenAiPromptCacheOptions,
    AzureOpenAiPromptCacheRetention, AzureOpenAiPromptCacheTtl, AzureOpenAiResponsesOptions,
    AzureOpenAiResponsesRequestExt,
};
use wiremock::MockServer;

fn tool_result_request(file: FilePart) -> Request {
    let assistant = CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::ToolCall(ToolCallPart::new(
            "call-1",
            "inspect",
            serde_json::json!({}),
        ))]),
        Finish::new(Default::default(), FinishReason::ToolCalls),
    );
    let result = ToolResultPart::new(
        "call-1",
        ToolContent::Mixed(vec![
            ContentValue::Text("caption".into()),
            ContentValue::File(file),
        ]),
    );
    Request::new(vec![
        HistoryTurn::assistant(assistant),
        HistoryTurn::tool(ToolMessage::new(vec![result])),
    ])
}

#[tokio::test]
async fn responses_tool_result_images_encode_as_input_items_and_other_files_reject() {
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
    model
        .complete(
            tool_result_request(FilePart::image(
                "image/png",
                FileSource::Bytes(bytes::Bytes::from_static(b"png")),
            )),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let output = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .unwrap();
    assert_eq!(output["output"][0]["type"], "input_text");
    assert_eq!(output["output"][1]["type"], "input_image");
    assert!(
        output["output"][1]["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
    assert!(!String::from_utf8_lossy(&requests[0].body).contains("[112,110,103]"));

    let error = model
        .validate_request(&tool_result_request(FilePart::document(
            "application/pdf",
            FileSource::Bytes(bytes::Bytes::from_static(b"pdf")),
        )))
        .unwrap_err();
    assert_eq!(error.kind(), oven_sdk::ModelErrorKind::Unsupported);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::RequestValidation
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn chat_rejects_tool_result_files_before_dispatch() {
    let server = MockServer::start().await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .chat("deployment", common::gpt4o())
        .unwrap();
    let error = model
        .validate_request(&tool_result_request(FilePart::image(
            "image/png",
            FileSource::Bytes(bytes::Bytes::from_static(b"png")),
        )))
        .unwrap_err();
    assert_eq!(error.kind(), oven_sdk::ModelErrorKind::Unsupported);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::RequestValidation
    );
    assert_eq!(
        error.message,
        "files in tool results are not deliverable via azure-openai-chat"
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

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
        InputPart::Text(TextPart::new("look").with_azure_openai_prompt_cache_breakpoint()),
        InputPart::File(
            FilePart::image(
                "image/png",
                FileSource::Bytes(bytes::Bytes::from_static(b"png")),
            )
            .with_azure_openai_prompt_cache_breakpoint(),
        ),
    ]))])
    .with_tools(vec![tool])
    .with_tool_choice(ToolChoice::Tool("lookup".into()))
    .with_response_format(ResponseFormat::structured(schema))
    .with_azure_openai_chat_options(AzureOpenAiChatOptions {
        service_tier: Some("future-tier".into()),
        verbosity: Some("future-verbosity".into()),
        parallel_tool_calls: Some(false),
        prompt_cache_key: Some("shared-chat-prefix".into()),
        prompt_cache_retention: Some(AzureOpenAiPromptCacheRetention::TwentyFourHours),
        prompt_cache_options: Some(AzureOpenAiPromptCacheOptions {
            mode: AzureOpenAiPromptCacheMode::Explicit,
            ttl: AzureOpenAiPromptCacheTtl::ThirtyMinutes,
        }),
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
    assert_eq!(body["prompt_cache_key"], "shared-chat-prefix");
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
    assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
    assert_eq!(
        body["messages"][0]["content"][0]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
    assert_eq!(
        body["messages"][0]["content"][1]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
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
    let request = Request::new(vec![
        HistoryTurn::system(oven_sdk::SystemMessage::new(vec![
            oven_sdk::SystemPart::Text(
                TextPart::new("system").with_azure_openai_prompt_cache_breakpoint(),
            ),
        ])),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(TextPart::new("inspect").with_azure_openai_prompt_cache_breakpoint()),
            InputPart::File(
                FilePart::image(
                    "image/png",
                    FileSource::Bytes(bytes::Bytes::from_static(b"png")),
                )
                .with_azure_openai_prompt_cache_breakpoint(),
            ),
            InputPart::File(
                FilePart::document(
                    "application/pdf",
                    FileSource::Bytes(bytes::Bytes::from_static(b"pdf")),
                )
                .with_azure_openai_prompt_cache_breakpoint(),
            ),
        ])),
    ])
    .with_response_format(ResponseFormat::structured(schema))
    .with_inference(inference)
    .with_azure_openai_responses_options(AzureOpenAiResponsesOptions {
        reasoning_summary: Some("future-summary".into()),
        service_tier: Some("future-tier".into()),
        prompt_cache_key: Some("shared-responses-prefix".into()),
        prompt_cache_retention: Some(AzureOpenAiPromptCacheRetention::InMemory),
        prompt_cache_options: Some(AzureOpenAiPromptCacheOptions {
            mode: AzureOpenAiPromptCacheMode::Implicit,
            ttl: AzureOpenAiPromptCacheTtl::ThirtyMinutes,
        }),
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
    assert_eq!(body["prompt_cache_key"], "shared-responses-prefix");
    assert_eq!(body["prompt_cache_retention"], "in_memory");
    assert_eq!(body["prompt_cache_options"]["mode"], "implicit");
    assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
    assert_eq!(
        body["input"][0]["content"][0]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
    assert_eq!(
        body["input"][1]["content"][0]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
    assert_eq!(
        body["input"][1]["content"][1]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
    assert_eq!(
        body["input"][1]["content"][2]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
}

#[tokio::test]
async fn cache_keys_over_64_characters_are_rejected_before_network() {
    let server = MockServer::start().await;
    let provider = common::provider(&server, AzureApiRoute::V1);
    let chat = provider.chat("chat", common::gpt4o()).unwrap();
    let responses = provider.responses("responses", common::gpt5()).unwrap();
    let chat_request =
        Request::new(Vec::new()).with_azure_openai_chat_options(AzureOpenAiChatOptions {
            prompt_cache_key: Some("x".repeat(65)),
            ..Default::default()
        });
    let responses_request =
        Request::new(Vec::new()).with_azure_openai_responses_options(AzureOpenAiResponsesOptions {
            prompt_cache_key: Some("x".repeat(65)),
            ..Default::default()
        });

    assert!(chat.validate_request(&chat_request).is_err());
    assert!(responses.validate_request(&responses_request).is_err());

    let too_many = Request::new(vec![HistoryTurn::user(UserMessage::new(
        (0..5)
            .map(|index| {
                InputPart::Text(
                    TextPart::new(format!("block-{index}"))
                        .with_azure_openai_prompt_cache_breakpoint(),
                )
            })
            .collect(),
    ))]);
    assert!(chat.validate_request(&too_many).is_err());

    let mut malformed = Request::new(Vec::new());
    malformed.provider_options.insert(
        "azure_openai".into(),
        serde_json::json!({"chat":{"prompt_cache_options":{"mode":"future","ttl":"forever"}}}),
    );
    assert!(chat.validate_request(&malformed).is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
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
