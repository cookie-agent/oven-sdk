use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, CompactionCapability, CompletedTurn,
    ContentValue, CustomPart, FilePart, FileSource, Finish, FinishReason, HeaderOverrides,
    HeaderProvider, HistoryTurn, InferenceOptions, InputPart, JsonSchema, LanguageModel,
    ModelError, NativeReplayArtifact, ReplayDisposition, Request, ResponseFormat, SystemMessage,
    SystemPart, TextPart, ToolCallPart, ToolChoice, ToolContent, ToolDefinition, ToolMessage,
    ToolResultPart, UserMessage,
};
use oven_sdk_google::{
    GoogleModel, GoogleProviderTool, GoogleRequestExt, GoogleRequestOptions, GoogleThinkingConfig,
    GoogleToolExt, GoogleToolOptions, GoogleToolSettings,
};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::json;
use std::{sync::Arc, time::Duration};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, header, method, path, query_param},
};

mod common;
use common::{
    budget_model, config_with, default_tools, full_capabilities, level_thinking, model, model_with,
};

#[tokio::test]
async fn tool_result_files_reject_during_request_validation() {
    let server = MockServer::start().await;
    let model = common::model(format!("{}/v1beta", server.uri()), "gemini-2.5-flash");
    let assistant = CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::ToolCall(ToolCallPart::new(
            "call-1",
            "inspect",
            json!({}),
        ))]),
        Finish::new(Default::default(), FinishReason::ToolCalls),
    );
    let result = ToolResultPart::new(
        "call-1",
        ToolContent::Mixed(vec![ContentValue::File(FilePart::image(
            "image/png",
            FileSource::Bytes(bytes::Bytes::from_static(b"png")),
        ))]),
    );
    let request = Request::new(vec![
        HistoryTurn::assistant(assistant),
        HistoryTurn::tool(ToolMessage::new(vec![result])),
    ]);

    let error = model.validate_request(&request).unwrap_err();
    assert_eq!(error.kind(), oven_sdk::ModelErrorKind::Unsupported);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::RequestValidation
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

fn terminal_sse(text: &str) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "responseId":"response-1",
            "modelVersion":"gemini-2.5-flash",
            "candidates":[{"content":{"parts":[{"text":text}]},"finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2}
        })
    )
}

struct DynamicHeaders(HeaderMap);

impl HeaderProvider for DynamicHeaders {
    fn headers(&self, _context: &oven_sdk::HeaderContext) -> Result<HeaderOverrides, ModelError> {
        Ok(HeaderOverrides::new(self.0.clone()))
    }
}

#[tokio::test]
async fn dynamic_headers_cannot_override_content_type() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(terminal_sse("ok")),
        )
        .mount(&server)
        .await;
    let mut config = config_with(
        format!("{}/v1beta", server.uri()),
        "gemini-2.5-flash",
        "models/gemini-2.5-flash",
        "secret",
        full_capabilities(),
        level_thinking(),
        default_tools(),
    );
    let mut dynamic = HeaderMap::new();
    dynamic.insert(CONTENT_TYPE, HeaderValue::from_static("text/plain"));
    config.provider.headers.dynamic_headers = Some(Arc::new(DynamicHeaders(dynamic)));
    let model = GoogleModel::new(config).unwrap();

    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].headers[CONTENT_TYPE], "application/json");
}

#[tokio::test]
async fn caller_api_key_suppresses_google_auth_injection() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(terminal_sse("ok")),
        )
        .mount(&server)
        .await;
    let mut config = config_with(
        format!("{}/v1beta", server.uri()),
        "gemini-2.5-flash",
        "models/gemini-2.5-flash",
        "configured-secret",
        full_capabilities(),
        level_thinking(),
        default_tools(),
    );
    let mut headers = HeaderMap::new();
    headers.insert("x-goog-api-key", HeaderValue::from_static("caller-secret"));
    config.provider.headers.static_headers = HeaderOverrides::new(headers);
    let model = GoogleModel::new(config).unwrap();

    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].headers["x-goog-api-key"], "caller-secret");
}

#[tokio::test]
async fn streaming_request_uses_api_key_resource_and_current_response_format() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.5-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .and(header("x-goog-api-key", "secret"))
        .and(body_partial_json(json!({
            "systemInstruction":{"parts":[{"text":"system"}]},
            "generationConfig":{"responseFormat":{"text":{"mimeType":"APPLICATION_JSON","schema":{"type":"object"}}}},
            "tools":[{"googleSearch":{}}]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(terminal_sse("ok")),
        )
        .expect(1)
        .mount(&server)
        .await;

    let model = budget_model(format!("{}/v1beta", server.uri()), "gemini-2.5-flash");
    let schema = JsonSchema::new(json!({"type":"object"})).unwrap();
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(TextPart::new(
            "system",
        ))])),
        HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
            "hello",
        ))])),
    ])
    .with_response_format(ResponseFormat::structured(schema))
    .with_google_options(GoogleRequestOptions {
        provider_tools: vec![GoogleProviderTool::GoogleSearch],
        thinking_config: Some(GoogleThinkingConfig {
            thinking_budget: Some(128),
            include_thoughts: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    });
    let result = model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "ok");
}

#[tokio::test]
async fn generate_content_uses_non_streaming_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.5-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates":[{"content":{"parts":[{"text":"direct"}]},"finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let model = model(format!("{}/v1beta", server.uri()), "gemini-2.5-flash");
    let result = model
        .generate_content(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "direct");
}

#[tokio::test]
async fn unsupported_media_is_rejected_before_dispatch() {
    let server = MockServer::start().await;
    let model = model(format!("{}/v1beta", server.uri()), "gemini-2.5-flash");
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(FilePart::audio(
            "audio/unknown",
            FileSource::Bytes(vec![1, 2].into()),
        )),
    ]))]);
    assert!(model.stream(request, AbortSignal::default()).await.is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[test]
fn known_looking_and_future_names_do_not_select_behavior() {
    let tools = GoogleToolSettings {
        strict_functions: true,
        mixed_client_and_provider_tools: true,
        current_turn_signature_sentinel: true,
    };
    let known = model_with(
        "https://example.com/v1beta",
        "gemini-2.5-flash",
        "models/shared-resource",
        full_capabilities(),
        level_thinking(),
        tools,
    );
    let future = model_with(
        "https://example.com/v1beta",
        "future-name-9000",
        "models/shared-resource",
        full_capabilities(),
        level_thinking(),
        tools,
    );
    assert_eq!(known.capabilities(), future.capabilities());
    assert_ne!(known.model_id(), future.model_id());
    let request = Request::new(Vec::new()).with_google_options(GoogleRequestOptions {
        thinking_config: Some(GoogleThinkingConfig {
            thinking_level: Some("future-level".into()),
            include_thoughts: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    });
    assert!(known.supports_request(&request));
    assert!(future.supports_request(&request));
    let options = GoogleRequestOptions {
        service_tier: Some("future-tier".into()),
        ..Default::default()
    };
    assert_eq!(options.service_tier.as_deref(), Some("future-tier"));
}

#[test]
fn explicit_auth_and_resource_validation_are_eager() {
    let tools = GoogleToolSettings {
        strict_functions: true,
        mixed_client_and_provider_tools: true,
        current_turn_signature_sentinel: true,
    };
    assert!(
        GoogleModel::new(config_with(
            "https://example.com/v1beta",
            "model-id",
            "models/model-id",
            "",
            full_capabilities(),
            level_thinking(),
            tools,
        ))
        .is_err()
    );
    assert!(
        GoogleModel::new(config_with(
            "https://example.com/v1beta",
            "model-id",
            "publishers/google/models/model-id",
            "secret",
            full_capabilities(),
            level_thinking(),
            tools,
        ))
        .is_err()
    );
}

#[test]
fn native_compaction_declaration_is_rejected_eagerly() {
    let mut capabilities = full_capabilities();
    capabilities.compaction = CompactionCapability::Native;
    let error = GoogleModel::new(config_with(
        "https://example.com/v1beta",
        "model-id",
        "models/model-id",
        "secret",
        capabilities,
        level_thinking(),
        default_tools(),
    ))
    .err()
    .unwrap();
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::Unsupported);
}

#[test]
fn native_context_scope_is_canonical_cryptographic_and_redacted() {
    let canonical = model_with(
        "https://EXAMPLE.com/v1beta/",
        "model-id",
        "models/model-resource",
        full_capabilities(),
        level_thinking(),
        default_tools(),
    );
    let equivalent = model_with(
        "https://example.com/v1beta",
        "model-id",
        "models/model-resource",
        full_capabilities(),
        level_thinking(),
        default_tools(),
    );
    let different_endpoint = model_with(
        "https://example.net/v1beta",
        "model-id",
        "models/model-resource",
        full_capabilities(),
        level_thinking(),
        default_tools(),
    );
    let different_resource = model_with(
        "https://example.com/v1beta",
        "model-id",
        "models/other-resource",
        full_capabilities(),
        level_thinking(),
        default_tools(),
    );
    assert_eq!(
        canonical.native_context_scope(),
        equivalent.native_context_scope()
    );
    assert_ne!(
        canonical.native_context_scope(),
        different_endpoint.native_context_scope()
    );
    assert_ne!(
        canonical.native_context_scope(),
        different_resource.native_context_scope()
    );
    let resource = canonical.native_context_scope().resource_id.as_str();
    assert!(resource.starts_with("google-generate-content-v1-sha256:"));
    assert_eq!(
        resource.len(),
        "google-generate-content-v1-sha256:".len() + 64
    );
    assert!(
        resource["google-generate-content-v1-sha256:".len()..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    );
    for secret in [
        "EXAMPLE.com",
        "example.com",
        "v1beta",
        "model-resource",
        "models/",
    ] {
        assert!(!resource.contains(secret));
    }
    assert!(
        !format!(
            "{:?}",
            oven_sdk_google::GoogleApiKeyAuth::new("top-secret-key")
        )
        .contains("top-secret-key")
    );
    let artifact = NativeReplayArtifact::new(
        AdapterId::new("oven.google.generate-content"),
        canonical.native_context_scope().clone(),
        json!({"private":"opaque-payload"}),
    )
    .unwrap();
    let debug = format!("{artifact:?}");
    for raw in ["opaque-payload", "example.com", "v1beta", "model-resource"] {
        assert!(!debug.contains(raw));
    }
}

#[tokio::test]
async fn normalized_reasoning_effort_is_mapped_or_rejected_before_dispatch() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(terminal_sse("ok")),
        )
        .mount(&server)
        .await;

    let request = |effort: &str| {
        let mut inference = InferenceOptions::new();
        inference.reasoning_effort = Some(effort.into());
        Request::new(Vec::new()).with_inference(inference)
    };
    model(server.uri(), "level-configured-name")
        .complete(request("high"), AbortSignal::default())
        .await
        .unwrap();
    budget_model(server.uri(), "budget-configured-name")
        .complete(request("high"), AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let level: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let budget: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        level.pointer("/generationConfig/thinkingConfig/thinkingLevel"),
        Some(&json!("HIGH"))
    );
    assert_eq!(
        budget.pointer("/generationConfig/thinkingConfig/thinkingBudget"),
        Some(&json!(1_024))
    );

    let unmapped = model(server.uri(), "unmapped-name");
    assert!(
        unmapped
            .stream(request("unmapped"), AbortSignal::default())
            .await
            .is_err()
    );
    let unsupported = model_with(
        server.uri(),
        "unsupported-name",
        "models/unsupported-name",
        full_capabilities(),
        oven_sdk_google::GoogleThinkingSettings::Unsupported,
        default_tools(),
    );
    assert!(
        unsupported
            .stream(request("high"), AbortSignal::default())
            .await
            .is_err()
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[test]
fn normalized_inference_and_tools_remain_constructible() {
    let mut request = Request::new(Vec::new()).with_tools(vec![ToolDefinition::new(
        "lookup",
        "lookup",
        JsonSchema::new(json!({"type":"object"})).unwrap(),
    )]);
    request.inference = InferenceOptions::new();
    request.inference.max_output_tokens = Some(12);
    request.inference.temperature = Some(0.5);
    request.inference.top_p = Some(0.9);
    request.inference.reasoning_effort = Some("future-effort".into());
    assert_eq!(request.tools.len(), 1);
}

#[tokio::test]
async fn supported_media_sources_use_distinct_current_wire_shapes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(terminal_sse("ok")),
        )
        .mount(&server)
        .await;
    let model = model(server.uri(), "gemini-2.5-flash");
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(vec![1, 2].into()),
        )),
        InputPart::File(FilePart::document(
            "text/plain",
            FileSource::Text("hello".into()),
        )),
        InputPart::File(FilePart::audio(
            "audio/mp3",
            FileSource::Bytes(vec![3].into()),
        )),
        InputPart::File(FilePart::video(
            "video/mp4",
            FileSource::Bytes(bytes::Bytes::from_static(b"video")),
        )),
        InputPart::File(FilePart::video(
            "video/mp4",
            FileSource::Url("https://example.com/video.mp4".parse().unwrap()),
        )),
        InputPart::File(FilePart::image(
            "image/jpeg",
            FileSource::ProviderReference {
                provider: oven_sdk::ProviderId::new("google"),
                id: "files/existing".into(),
            },
        )),
    ]))]);
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let parts = body
        .pointer("/contents/0/parts")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(parts[0]["inlineData"]["data"], "AQI=");
    assert_eq!(parts[1]["inlineData"]["data"], "aGVsbG8=");
    assert_eq!(parts[2]["inlineData"]["mimeType"], "audio/mp3");
    assert_eq!(
        parts[3],
        serde_json::json!({
            "inlineData":{"mimeType":"video/mp4","data":"dmlkZW8="}
        })
    );
    assert_eq!(
        parts[4]["fileData"]["fileUri"],
        "https://example.com/video.mp4"
    );
    assert_eq!(parts[5]["fileData"]["fileUri"], "files/existing");
}

#[tokio::test]
async fn thought_signatures_survive_native_replay_request_encoding() {
    let first = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\n",
                    json!({
                        "candidates":[{"content":{"parts":[
                            {"text":"thinking","thought":true,"thoughtSignature":"signature-1"},
                            {"text":"answer"}
                        ]},"finishReason":"STOP"}]
                    })
                )),
        )
        .mount(&first)
        .await;
    let first_model = model(first.uri(), "gemini-2.5-flash");
    let turn = first_model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    let artifact = turn.finish.native_replay.as_ref().unwrap();
    assert_eq!(artifact.scope(), first_model.native_context_scope());
    let current_format = serde_json::to_vec(artifact).unwrap();
    let decoded: NativeReplayArtifact = serde_json::from_slice(&current_format).unwrap();
    assert_eq!(&decoded, artifact);
    assert_eq!(
        artifact
            .payload()
            .pointer("/parts/0/thoughtSignature")
            .and_then(serde_json::Value::as_str),
        Some("signature-1")
    );
    assert!(artifact.payload().get("format").is_none());
    assert!(artifact.payload().get("source_model").is_none());

    first_model
        .complete(
            Request::new(vec![HistoryTurn::assistant(turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let requests = first.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        body.pointer("/contents/0/parts/0/thoughtSignature")
            .and_then(serde_json::Value::as_str),
        Some("signature-1")
    );
}

#[tokio::test]
async fn endpoint_and_model_resource_changes_discard_foreign_native_context_scope() {
    let source = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\n",
                    json!({
                        "candidates":[{"content":{"parts":[{
                            "text":"signed context",
                            "thoughtSignature":"private-signature"
                        } ]},"finishReason":"STOP"}]
                    })
                )),
        )
        .mount(&source)
        .await;
    let source_model = model(source.uri(), "scope-model");
    let turn = source_model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;

    let foreign_endpoint = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(terminal_sse("endpoint changed")),
        )
        .mount(&foreign_endpoint)
        .await;
    let endpoint_result = model(foreign_endpoint.uri(), "scope-model")
        .complete(
            Request::new(vec![HistoryTurn::assistant(turn.clone())]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        endpoint_result.request.replay.decisions[0].disposition,
        ReplayDisposition::DiscardedForeignScope { .. }
    ));
    let endpoint_requests = foreign_endpoint.received_requests().await.unwrap();
    let endpoint_body: serde_json::Value =
        serde_json::from_slice(&endpoint_requests[0].body).unwrap();
    assert!(!endpoint_body.to_string().contains("private-signature"));

    let resource_model = model_with(
        source.uri(),
        "scope-model",
        "models/other-resource",
        full_capabilities(),
        level_thinking(),
        default_tools(),
    );
    let resource_result = resource_model
        .complete(
            Request::new(vec![HistoryTurn::assistant(turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        resource_result.request.replay.decisions[0].disposition,
        ReplayDisposition::DiscardedForeignScope { .. }
    ));
    let source_requests = source.received_requests().await.unwrap();
    let resource_body: serde_json::Value =
        serde_json::from_slice(&source_requests[1].body).unwrap();
    assert!(!resource_body.to_string().contains("private-signature"));
}

#[tokio::test]
async fn abort_while_waiting_for_headers_is_structured() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(5))
                .set_body_string(terminal_sse("late")),
        )
        .mount(&server)
        .await;
    let model = model(server.uri(), "gemini-2.5-flash");
    let (signal, registration) = AbortSignal::new();
    let task = tokio::spawn(async move { model.stream(Request::new(Vec::new()), signal).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    registration.abort();
    let error = task.await.unwrap().unwrap_err();
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::Abort);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::ResponseHeaders
    );
}

#[tokio::test]
async fn strict_named_function_tools_use_validated_current_wire_config() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(terminal_sse("ok")),
        )
        .mount(&server)
        .await;
    let model = model(server.uri(), "gemini-3.5-flash");
    let tool = ToolDefinition::new(
        "lookup",
        "Lookup a value",
        JsonSchema::new(json!({"type":"object","properties":{"q":{"type":"string"}}})).unwrap(),
    )
    .with_google_tool_options(GoogleToolOptions { strict: true });
    let mut request = Request::new(Vec::new()).with_tools(vec![tool]);
    request.tool_choice = ToolChoice::Tool("lookup".into());
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        body["tools"][0]["functionDeclarations"][0]["name"],
        "lookup"
    );
    assert_eq!(
        body.pointer("/toolConfig/functionCallingConfig/mode")
            .and_then(serde_json::Value::as_str),
        Some("VALIDATED")
    );
    assert_eq!(
        body.pointer("/toolConfig/functionCallingConfig/allowedFunctionNames/0")
            .and_then(serde_json::Value::as_str),
        Some("lookup")
    );
}

#[tokio::test]
async fn schema_less_json_uses_current_response_format_text_shape() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "generationConfig":{"responseFormat":{"text":{"mimeType":"APPLICATION_JSON"}}}
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(terminal_sse("{}")),
        )
        .expect(1)
        .mount(&server)
        .await;
    let model = model(server.uri(), "gemini-3.5-flash");
    model
        .complete(
            Request::new(Vec::new()).with_response_format(ResponseFormat::Json { schema: None }),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(
        body.pointer("/generationConfig/responseFormat/text/schema")
            .is_none()
    );
}

#[tokio::test]
async fn mixed_tools_enable_server_invocation_circulation_only_when_mixed() {
    for (provider_tools, expected) in [(true, Some(true)), (false, None)] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(terminal_sse("ok")),
            )
            .mount(&server)
            .await;
        let model = model(server.uri(), "gemini-3.5-flash");
        let mut request = Request::new(Vec::new()).with_tools(vec![ToolDefinition::new(
            "lookup",
            "Lookup",
            JsonSchema::new(json!({"type":"object"})).unwrap(),
        )]);
        if provider_tools {
            request = request.with_google_options(GoogleRequestOptions {
                provider_tools: vec![GoogleProviderTool::GoogleSearch],
                ..Default::default()
            });
        }
        model
            .complete(request, AbortSignal::default())
            .await
            .unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            body.pointer("/toolConfig/includeServerSideToolInvocations")
                .and_then(serde_json::Value::as_bool),
            expected
        );
        assert_eq!(
            body.pointer("/toolConfig/functionCallingConfig/mode")
                .and_then(serde_json::Value::as_str),
            Some(if provider_tools { "VALIDATED" } else { "AUTO" })
        );
    }

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(terminal_sse("ok")),
        )
        .mount(&server)
        .await;
    let model = model(server.uri(), "gemini-3.5-flash");
    model
        .complete(
            Request::new(Vec::new()).with_google_options(GoogleRequestOptions {
                provider_tools: vec![GoogleProviderTool::GoogleSearch],
                ..Default::default()
            }),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(body.get("toolConfig").is_none());
}

fn normalized_tool_turn(artifact: Option<NativeReplayArtifact>, calls: usize) -> CompletedTurn {
    let content = (0..calls)
        .map(|index| {
            AssistantPart::ToolCall(ToolCallPart::new(
                format!("call-{index}"),
                "lookup",
                json!({"index":index}),
            ))
        })
        .collect();
    let mut turn = CompletedTurn::new(
        AssistantMessage::new(content),
        Finish::new(Default::default(), FinishReason::ToolCalls),
    );
    turn.finish.native_replay = artifact;
    turn
}

fn tool_results(calls: usize) -> HistoryTurn {
    HistoryTurn::tool(ToolMessage::new(
        (0..calls)
            .map(|index| {
                ToolResultPart::new(
                    format!("call-{index}"),
                    ToolContent::Json(json!({"ok":true})),
                )
            })
            .collect(),
    ))
}

#[tokio::test]
async fn configured_reconstruction_uses_current_turn_parallel_sentinel_rules() {
    for artifact_case in 0..4 {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(terminal_sse("continued")),
            )
            .mount(&server)
            .await;
        let model = model(server.uri(), "gemini-3.5-flash");
        let artifact = match artifact_case {
            0 => None,
            1 => Some(
                NativeReplayArtifact::new(
                    AdapterId::new("foreign.adapter"),
                    model.native_context_scope().clone(),
                    json!({"opaque":true}),
                )
                .unwrap(),
            ),
            2 => Some(
                NativeReplayArtifact::new(
                    AdapterId::new("oven.google.generate-content"),
                    model.native_context_scope().clone(),
                    json!("invalid"),
                )
                .unwrap(),
            ),
            _ => Some(
                NativeReplayArtifact::new(
                    AdapterId::new("oven.google.generate-content"),
                    model_with(
                        server.uri(),
                        "gemini-3.5-flash",
                        "models/other-resource",
                        full_capabilities(),
                        level_thinking(),
                        default_tools(),
                    )
                    .native_context_scope()
                    .clone(),
                    json!({"role":"model","parts":[]}),
                )
                .unwrap(),
            ),
        };
        let request = Request::new(vec![
            HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
                "start",
            ))])),
            HistoryTurn::assistant(normalized_tool_turn(artifact, 2)),
            tool_results(2),
        ]);
        let result = model
            .complete(request, AbortSignal::default())
            .await
            .unwrap();
        assert!(
            result
                .turn
                .warnings
                .iter()
                .any(|warning| { warning.contains("skip_thought_signature_validator") })
        );
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            body.pointer("/contents/1/parts/0/thoughtSignature")
                .and_then(serde_json::Value::as_str),
            Some("skip_thought_signature_validator")
        );
        assert!(
            body.pointer("/contents/1/parts/1/thoughtSignature")
                .is_none()
        );
    }
}

#[tokio::test]
async fn configured_sentinel_is_not_added_before_the_current_turn() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(terminal_sse("continued")),
        )
        .mount(&server)
        .await;
    let model = model(server.uri(), "gemini-3.5-flash");
    model
        .complete(
            Request::new(vec![
                HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
                    "first",
                ))])),
                HistoryTurn::assistant(normalized_tool_turn(None, 1)),
                tool_results(1),
                HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
                    "next",
                ))])),
            ]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(
        body.pointer("/contents/1/parts/0/thoughtSignature")
            .is_none()
    );
}

#[tokio::test]
async fn normalized_server_tool_context_is_omitted_for_all_reconstruction_paths() {
    let server_tool_turn = |artifact| {
        let mut turn = CompletedTurn::new(
            AssistantMessage::new(vec![
                AssistantPart::Custom(CustomPart::new(
                    "google.server_tool_call",
                    json!({"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","toolName":"google_search","args":{"q":"rust"},"id":"server-1"}}),
                )),
                AssistantPart::Custom(CustomPart::new(
                    "google.server_tool_call",
                    json!({"executableCode":{"language":"PYTHON","code":"print(1)","id":"code-1"}}),
                )),
                AssistantPart::Custom(CustomPart::new(
                    "google.server_tool_result",
                    json!({"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1","id":"code-1"}}),
                )),
                AssistantPart::Custom(CustomPart::new(
                    "google.server_tool_result",
                    json!({"toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"items":[]},"id":"server-1"}}),
                )),
                AssistantPart::ToolCall(ToolCallPart::new(
                    "client-1",
                    "lookup",
                    json!({"q":"rust"}),
                )),
            ]),
            Finish::new(Default::default(), FinishReason::ToolCalls),
        );
        turn.finish.native_replay = artifact;
        turn
    };
    for artifact_case in 0..4 {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(terminal_sse("continued")),
            )
            .mount(&server)
            .await;
        let model = model(server.uri(), "gemini-3.5-flash");
        let artifact = match artifact_case {
            0 => None,
            1 => Some(
                NativeReplayArtifact::new(
                    AdapterId::new("foreign.adapter"),
                    model.native_context_scope().clone(),
                    json!({"opaque":true}),
                )
                .unwrap(),
            ),
            2 => Some(
                NativeReplayArtifact::new(
                    AdapterId::new("oven.google.generate-content"),
                    model.native_context_scope().clone(),
                    json!("invalid"),
                )
                .unwrap(),
            ),
            _ => Some(
                NativeReplayArtifact::new(
                    AdapterId::new("oven.google.generate-content"),
                    model_with(
                        server.uri(),
                        "gemini-3.5-flash",
                        "models/other-resource",
                        full_capabilities(),
                        level_thinking(),
                        default_tools(),
                    )
                    .native_context_scope()
                    .clone(),
                    json!({"role":"model","parts":[]}),
                )
                .unwrap(),
            ),
        };
        let request = Request::new(vec![
            HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
                "search",
            ))])),
            HistoryTurn::assistant(server_tool_turn(artifact)),
            HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
                "client-1",
                ToolContent::Json(json!({"answer":42})),
            )])),
        ])
        .with_tools(vec![ToolDefinition::new(
            "lookup",
            "Lookup",
            JsonSchema::new(json!({"type":"object"})).unwrap(),
        )])
        .with_google_options(GoogleRequestOptions {
            provider_tools: vec![GoogleProviderTool::GoogleSearch],
            ..Default::default()
        });
        let result = model
            .complete(request, AbortSignal::default())
            .await
            .unwrap();
        for field in [
            "toolCall",
            "toolResponse",
            "executableCode",
            "codeExecutionResult",
        ] {
            assert!(result.turn.warnings.iter().any(|warning| {
                warning.contains(field) && warning.contains("opaque thought signature")
            }));
        }
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        let parts = body
            .pointer("/contents/1/parts")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["functionCall"]["id"], "client-1");
        assert_eq!(
            parts[0]["thoughtSignature"],
            "skip_thought_signature_validator"
        );
    }
}

#[tokio::test]
async fn cached_content_request_and_usage_semantics_are_enforced() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"cachedContent":"cachedContents/cache-1"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates":[{"content":{"parts":[{"text":"cached"}]},"finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":4,"candidatesTokenCount":2}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let model = model(server.uri(), "gemini-3.1-flash-lite");
    assert!(
        model
            .capabilities()
            .features
            .contains(oven_sdk::Capability::PROMPT_CACHING)
    );
    let result = model
        .generate_content(
            Request::new(Vec::new()).with_google_options(GoogleRequestOptions {
                cached_content: Some("cachedContents/cache-1".into()),
                ..Default::default()
            }),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_eq!(result.turn.finish.usage.input_tokens, Some(10));
    assert_eq!(result.turn.finish.usage.input_tokens_no_cache, Some(6));
    assert_eq!(result.turn.finish.usage.input_tokens_cache_read, Some(4));
    assert_eq!(result.turn.finish.usage.input_tokens_cache_write, None);

    let mut no_cache = full_capabilities();
    no_cache
        .features
        .remove(oven_sdk::Capability::PROMPT_CACHING);
    let unknown = model_with(
        server.uri(),
        "gemini-3.1-flash-lite",
        "models/gemini-3.1-flash-lite",
        no_cache,
        level_thinking(),
        GoogleToolSettings {
            strict_functions: true,
            mixed_client_and_provider_tools: true,
            current_turn_signature_sentinel: true,
        },
    );
    assert!(
        unknown
            .stream(
                Request::new(Vec::new()).with_google_options(GoogleRequestOptions {
                    cached_content: Some("cachedContents/cache-1".into()),
                    ..Default::default()
                }),
                AbortSignal::default(),
            )
            .await
            .is_err()
    );
}

#[tokio::test]
async fn oversized_ai_studio_request_is_rejected_before_dispatch() {
    let server = MockServer::start().await;
    let model = model(server.uri(), "gemini-3.5-flash-lite");
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("x".repeat(20 * 1024 * 1024))),
    ]))]);
    let error = model
        .stream(request, AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::InvalidRequest);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::RequestEncoding
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn aggregate_media_count_limits_are_enforced_before_dispatch() {
    let server = MockServer::start().await;
    let model = model(server.uri(), "gemini-3.5-flash-lite");
    let media_request = |mime: &'static str, count| {
        Request::new(vec![HistoryTurn::user(UserMessage::new(
            (0..count)
                .map(|_| InputPart::File(FilePart::new(mime, FileSource::Bytes(Vec::new().into()))))
                .collect(),
        ))])
    };
    model
        .validate_request(&media_request("image/png", 3_600))
        .unwrap();
    model
        .validate_request(&media_request("video/mp4", 10))
        .unwrap();
    for request in [
        media_request("image/png", 3_601),
        media_request("video/mp4", 11),
    ] {
        let error = model
            .stream(request, AbortSignal::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind, oven_sdk::ModelErrorKind::InvalidRequest);
        assert_eq!(
            error.diagnostics.stage,
            oven_sdk::ErrorStage::RequestEncoding
        );
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}
