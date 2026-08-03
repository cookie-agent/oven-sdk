pub mod common;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, FilePart, FileSource, Finish,
    FinishReason, HistoryTurn, InferenceOptions, InputPart, JsonSchema, LanguageModel, Request,
    ResponseFormat, SystemMessage, SystemPart, TextPart, ToolChoice, ToolDefinition, UserMessage,
};
use oven_sdk_openai::{
    OpenAiChatOptions, OpenAiChatRequestExt, OpenAiResponsesOptions, OpenAiResponsesRequestExt,
};
use wiremock::MockServer;

#[tokio::test]
async fn official_chat_encodes_headers_stream_usage_and_single_post() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let result = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.response.request_id.as_deref(), Some("req_1"));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.headers["authorization"], "Bearer secret");
    assert_eq!(request.headers["openai-organization"], "org");
    assert_eq!(request.headers["openai-project"], "project");
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn explicit_settings_use_developer_and_max_completion_tokens() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let mut config = common::official_chat_config(&server, "future-chat-id");
    config.settings.system_message_role = oven_sdk_openai::SystemMessageRole::Developer;
    config.settings.max_tokens_field = oven_sdk_openai::MaxTokensField::MaxCompletionTokens;
    let model = oven_sdk_openai::OpenAiChatModel::new(config).unwrap();
    let mut inference = InferenceOptions::new();
    inference.max_output_tokens = Some(200);
    inference.reasoning_effort = Some("medium".into());
    let request = Request::new(vec![HistoryTurn::system(SystemMessage::new(vec![
        SystemPart::Text(TextPart::new("system")),
    ]))])
    .with_inference(inference);
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["messages"][0]["role"], "developer");
    assert_eq!(body["max_completion_tokens"], 200);
    assert_eq!(body["reasoning_effort"], "medium");
    assert!(body.get("max_tokens").is_none());
}

#[tokio::test]
async fn chat_encodes_tools_choice_strict_and_json_schema() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("{}")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let schema = JsonSchema::new(serde_json::json!({"type":"object"})).unwrap();
    let mut tool = ToolDefinition::new("lookup", "find", schema.clone());
    tool.provider_options
        .insert("openai".into(), serde_json::json!({"strict":true}));
    let request = Request::new(Vec::new())
        .with_tools(vec![tool])
        .with_tool_choice(ToolChoice::Tool("lookup".into()))
        .with_response_format(ResponseFormat::structured(schema));
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["tools"][0]["function"]["strict"], true);
    assert_eq!(body["tool_choice"]["function"]["name"], "lookup");
    assert_eq!(body["response_format"]["type"], "json_schema");
}

#[tokio::test]
async fn chat_encodes_image_and_pdf_inputs() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("look")),
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(bytes::Bytes::from_static(b"png")),
        )),
        InputPart::File(FilePart::document(
            "application/pdf",
            FileSource::Text("pdf".into()),
        )),
    ]))]);
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    let content = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content[1]["type"], "image_url");
    assert!(
        content[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
    assert_eq!(content[2]["type"], "file");
}

#[tokio::test]
async fn unsupported_audio_is_rejected_before_dispatch() {
    let server = MockServer::start().await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(FilePart::audio(
            "audio/wav",
            FileSource::Bytes(bytes::Bytes::new()),
        )),
    ]))]);
    assert!(model.stream(request, AbortSignal::default()).await.is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn assistant_history_media_is_rejected_instead_of_silently_dropped() {
    let server = MockServer::start().await;
    let model = common::official_chat(&server, "future-chat-id");
    let turn = CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(bytes::Bytes::from_static(b"png")),
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
async fn unknown_future_reasoning_effort_is_forwarded_unchanged() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "future-chat-id");
    let request = Request::new(Vec::new()).with_openai_chat_options(OpenAiChatOptions {
        reasoning_effort: Some("future-ultra-effort".into()),
        ..Default::default()
    });
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["reasoning_effort"], "future-ultra-effort");
}

#[tokio::test]
async fn official_chat_provider_labels_are_forwarded_unchanged() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "future-chat-id");
    let request = Request::new(Vec::new()).with_openai_chat_options(OpenAiChatOptions {
        service_tier: Some("future-hyperscale".into()),
        verbosity: Some("future-exhaustive".into()),
        ..Default::default()
    });
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["service_tier"], "future-hyperscale");
    assert_eq!(body["verbosity"], "future-exhaustive");
}

#[tokio::test]
async fn chat_wire_behavior_is_independent_of_model_name_including_future_ids() {
    for model_id in ["gpt-5-mini", "provider/future-chat-2099"] {
        let server = MockServer::start().await;
        common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
        let mut config = common::official_chat_config(&server, model_id);
        config.settings.system_message_role = oven_sdk_openai::SystemMessageRole::Developer;
        config.settings.max_tokens_field = oven_sdk_openai::MaxTokensField::MaxCompletionTokens;
        let model = oven_sdk_openai::OpenAiChatModel::new(config).unwrap();
        let mut inference = InferenceOptions::new();
        inference.max_output_tokens = Some(123);
        inference.temperature = Some(0.4);
        inference.top_p = Some(0.8);
        let mut response = model
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
        assert_eq!(body["max_completion_tokens"], 123);
        assert_eq!(body["temperature"], 0.4);
        assert_eq!(body["top_p"], 0.8);
    }
}

#[tokio::test]
async fn ordinary_chat_models_keep_sampling_without_warnings() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let mut inference = InferenceOptions::new();
    inference.temperature = Some(0.4);
    inference.top_p = Some(0.8);
    let mut response = common::official_chat(&server, "gpt-4o-mini")
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

#[test]
fn official_endpoint_options_share_only_the_current_openai_namespace() {
    let request = Request::new(Vec::new())
        .with_openai_chat_options(OpenAiChatOptions {
            verbosity: Some("chat-verbose".into()),
            ..Default::default()
        })
        .with_openai_responses_options(OpenAiResponsesOptions {
            reasoning_mode: Some("responses-mode".into()),
            ..Default::default()
        });
    assert_eq!(request.provider_options.len(), 1);
    assert_eq!(
        request.provider_options["openai"]["chat"]["verbosity"],
        "chat-verbose"
    );
    assert_eq!(
        request.provider_options["openai"]["responses"]["reasoning_mode"],
        "responses-mode"
    );
    assert!(!request.provider_options.contains_key("openai_chat"));
    assert!(!request.provider_options.contains_key("openai_responses"));
}
