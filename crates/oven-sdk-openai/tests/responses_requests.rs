pub mod common;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, ContentValue, FilePart,
    FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions, InputPart, JsonSchema,
    LanguageModel, Request, ResponseFormat, SystemMessage, SystemPart, TextPart, ToolCallPart,
    ToolChoice, ToolContent, ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};
use oven_sdk_openai::{
    OpenAiPromptCacheBreakpointExt, OpenAiPromptCacheMode, OpenAiPromptCacheOptions,
    OpenAiPromptCacheRetention, OpenAiPromptCacheTtl, OpenAiResponsesOptions,
    OpenAiResponsesRequestExt,
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
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = common::official_responses(&server, "gpt-5-mini");
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
async fn responses_always_sends_store_false_and_encrypted_include() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = common::official_responses(&server, "gpt-5-mini");
    let request = Request::new(Vec::new()).with_openai_responses_options(OpenAiResponsesOptions {
        include: vec![
            "file_search_call.results".into(),
            "reasoning.encrypted_content".into(),
        ],
        prompt_cache_key: Some("shared-responses-prefix".into()),
        prompt_cache_retention: Some(OpenAiPromptCacheRetention::InMemory),
        ..Default::default()
    });
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["store"], false);
    assert_eq!(body["prompt_cache_key"], "shared-responses-prefix");
    assert_eq!(body["prompt_cache_retention"], "in_memory");
    assert_eq!(
        body["include"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|value| *value == "reasoning.encrypted_content")
            .count(),
        1
    );
}

#[tokio::test]
async fn responses_rejects_oversized_prompt_cache_keys() {
    let server = MockServer::start().await;
    let model = common::official_responses(&server, "future-responses-id");
    let request = Request::new(Vec::new()).with_openai_responses_options(OpenAiResponsesOptions {
        prompt_cache_key: Some("x".repeat(65)),
        ..Default::default()
    });
    assert!(model.validate_request(&request).is_err());
}

#[tokio::test]
async fn responses_maps_system_media_tools_and_structured_output() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("{}")).await;
    let model = common::official_responses(&server, "gpt-5-mini");
    let schema = JsonSchema::new(serde_json::json!({"type":"object"})).unwrap();
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(
            TextPart::new("system").with_openai_prompt_cache_breakpoint(),
        )])),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(TextPart::new("inspect").with_openai_prompt_cache_breakpoint()),
            InputPart::File(
                FilePart::image(
                    "image/png",
                    FileSource::Bytes(bytes::Bytes::from_static(b"png")),
                )
                .with_openai_prompt_cache_breakpoint(),
            ),
            InputPart::File(
                FilePart::document("application/pdf", FileSource::Text("pdf".into()))
                    .with_openai_prompt_cache_breakpoint(),
            ),
        ])),
    ])
    .with_tools(vec![ToolDefinition::new("lookup", "find", schema.clone())])
    .with_tool_choice(ToolChoice::Required)
    .with_response_format(ResponseFormat::structured(schema))
    .with_openai_responses_options(OpenAiResponsesOptions {
        prompt_cache_options: Some(OpenAiPromptCacheOptions {
            mode: OpenAiPromptCacheMode::Implicit,
            ttl: OpenAiPromptCacheTtl::ThirtyMinutes,
        }),
        ..Default::default()
    });
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["input"][0]["role"], "developer");
    assert_eq!(
        body["input"][0]["content"][0]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
    assert_eq!(body["input"][1]["content"][0]["type"], "input_text");
    assert_eq!(
        body["input"][1]["content"][0]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
    assert_eq!(body["input"][1]["content"][1]["type"], "input_image");
    assert_eq!(
        body["input"][1]["content"][1]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
    assert_eq!(body["input"][1]["content"][2]["type"], "input_file");
    assert_eq!(
        body["input"][1]["content"][2]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
    assert_eq!(body["prompt_cache_options"]["mode"], "implicit");
    assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tool_choice"], "required");
    assert_eq!(body["text"]["format"]["type"], "json_schema");
}

#[tokio::test]
async fn responses_rejects_unknown_media_before_dispatch() {
    let server = MockServer::start().await;
    let model = common::official_responses(&server, "gpt-5-mini");
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(FilePart::video(
            "video/mp4",
            FileSource::Bytes(bytes::Bytes::new()),
        )),
    ]))]);
    assert!(model.stream(request, AbortSignal::default()).await.is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn responses_assistant_history_media_is_rejected_instead_of_silently_dropped() {
    let server = MockServer::start().await;
    let model = common::official_responses(&server, "future-responses-id");
    let turn = CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::File(FilePart::document(
            "application/pdf",
            FileSource::Bytes(bytes::Bytes::from_static(b"pdf")),
        ))]),
        Finish::new(Default::default(), FinishReason::Stop),
    );
    let error = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::Unsupported);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::RequestEncoding
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn unknown_future_responses_reasoning_effort_is_forwarded_unchanged() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = common::official_responses(&server, "future-responses-id");
    let mut request = Request::new(Vec::new());
    request.inference.reasoning_effort = Some("future-adaptive-plus".into());
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["reasoning"]["effort"], "future-adaptive-plus");
}

#[tokio::test]
async fn responses_pdf_url_and_bytes_use_distinct_wire_fields() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(FilePart::document(
            "application/pdf",
            FileSource::Url("https://example.test/document.pdf".parse().unwrap()),
        )),
        InputPart::File(FilePart::document(
            "application/pdf",
            FileSource::Bytes(bytes::Bytes::from_static(b"pdf")),
        )),
    ]))]);
    common::official_responses(&server, "gpt-5-mini")
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    let content = body["input"][0]["content"].as_array().unwrap();
    assert_eq!(content[0]["file_url"], "https://example.test/document.pdf");
    assert!(content[0].get("file_data").is_none());
    assert!(
        content[1]["file_data"]
            .as_str()
            .unwrap()
            .starts_with("data:application/pdf;base64,")
    );
    assert!(content[1].get("file_url").is_none());
}

#[tokio::test]
async fn official_responses_provider_labels_are_forwarded_unchanged() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let request = Request::new(Vec::new()).with_openai_responses_options(OpenAiResponsesOptions {
        service_tier: Some("future-burst".into()),
        verbosity: Some("future-maximal".into()),
        reasoning_summary: Some("future-insightful".into()),
        reasoning_mode: Some("future-deliberative".into()),
        truncation: Some("future-sliding".into()),
        ..Default::default()
    });
    common::official_responses(&server, "future-responses-id")
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["service_tier"], "future-burst");
    assert_eq!(body["text"]["verbosity"], "future-maximal");
    assert_eq!(body["reasoning"]["summary"], "future-insightful");
    assert_eq!(body["reasoning"]["mode"], "future-deliberative");
    assert_eq!(body["truncation"], "future-sliding");
}

#[tokio::test]
async fn non_reasoning_responses_models_reject_all_reasoning_options_without_dispatch() {
    let server = MockServer::start().await;
    let mut config = common::official_responses_config(&server, "arbitrary-nonreasoning-id");
    config
        .model
        .capabilities
        .features
        .remove(oven_sdk::Capability::REASONING);
    config.model.capabilities.replay.reasoning = false;
    config.model.capabilities.replay.capability = oven_sdk::ReplayCapability::Optional;
    let model = oven_sdk_openai::OpenAiResponsesModel::new(config).unwrap();
    let mut effort = Request::new(Vec::new());
    effort.inference.reasoning_effort = Some("future-effort".into());
    let summary = Request::new(Vec::new()).with_openai_responses_options(OpenAiResponsesOptions {
        reasoning_summary: Some("future-summary".into()),
        ..Default::default()
    });
    let mode = Request::new(Vec::new()).with_openai_responses_options(OpenAiResponsesOptions {
        reasoning_mode: Some("future-mode".into()),
        ..Default::default()
    });
    for request in [effort, summary, mode] {
        let error = model
            .stream(request, AbortSignal::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind, oven_sdk::ModelErrorKind::Unsupported);
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn responses_wire_behavior_is_independent_of_model_name_including_future_ids() {
    for model_id in ["gpt-5-mini", "provider/future-responses-2099"] {
        let server = MockServer::start().await;
        common::mount(&server, "/responses", common::responses_document("ok")).await;
        let mut inference = InferenceOptions::new();
        inference.temperature = Some(0.4);
        inference.top_p = Some(0.8);
        let mut response = common::official_responses(&server, model_id)
            .stream(
                Request::new(Vec::new()).with_inference(inference),
                AbortSignal::default(),
            )
            .await
            .unwrap();
        let warnings = match response.stream.next().await.unwrap().unwrap() {
            oven_sdk::StreamPart::StreamStart { warnings } => warnings,
            other => panic!("unexpected first part: {other:?}"),
        };
        assert!(warnings.is_empty());
        let body: serde_json::Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        assert_eq!(body["model"], model_id);
        assert_eq!(body["temperature"], 0.4);
        assert_eq!(body["top_p"], 0.8);
    }
}

#[tokio::test]
async fn ordinary_responses_models_keep_sampling_without_warnings() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let mut inference = InferenceOptions::new();
    inference.temperature = Some(0.4);
    inference.top_p = Some(0.8);
    let mut response = common::official_responses(&server, "gpt-4o-mini")
        .stream(
            Request::new(Vec::new()).with_inference(inference),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    match response.stream.next().await.unwrap().unwrap() {
        oven_sdk::StreamPart::StreamStart { warnings } => assert!(warnings.is_empty()),
        other => panic!("unexpected first part: {other:?}"),
    }
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["temperature"], 0.4);
    assert_eq!(body["top_p"], 0.8);
}

#[tokio::test]
async fn normalized_function_call_omits_absent_provider_item_id() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let mut call = ToolCallPart::new("call_1", "lookup", serde_json::json!({"x":1}));
    call.raw_input = Some("{\"x\":1}".into());
    let turn = CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::ToolCall(call)]),
        Finish::new(Default::default(), FinishReason::ToolCalls),
    );
    let result = ToolResultPart::new("call_1", ToolContent::Text("done".into()));
    let request = Request::new(vec![
        HistoryTurn::assistant(turn),
        HistoryTurn::tool(ToolMessage::new(vec![result])),
    ]);
    common::official_responses(&server, "gpt-5-mini")
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["input"][0]["type"], "function_call");
    assert!(body["input"][0].get("id").is_none());
}
