mod support;

use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, Capability, CompactionCapability,
    CompletedTurn, ContentValue, FilePart, FileSource, Finish, FinishReason, HeaderConfig,
    HeaderOverrides, HeaderProvider, HistoryTurn, InputPart, JsonSchema, LanguageModel,
    ModelConfig, ModelError, NativeContextScope, NativeReplayArtifact, ReplayDisposition, Request,
    ResourceId, SecretString, ToolCallPart, ToolContent, ToolDefinition, ToolMessage,
    ToolResultPart, UserMessage,
};
use oven_sdk_google_vertex::{
    GoogleVertexModel, GoogleVertexProviderTool, GoogleVertexRequestExt,
    GoogleVertexRequestOptions, GoogleVertexResource, GoogleVertexThinkingConfig,
    GoogleVertexToolExt, GoogleVertexToolOptions, VertexAuth,
};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::json;
use std::sync::Arc;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, method, path, query_param},
};

fn publisher_resource() -> GoogleVertexResource {
    GoogleVertexResource::PublisherModel {
        publisher: "google".into(),
        model: "resource-model-v1".into(),
    }
}

#[tokio::test]
async fn tool_result_files_reject_during_request_validation() {
    let server = MockServer::start().await;
    let model = support::full_model(&server.uri(), "gemini-future", publisher_resource(), true);
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

struct DynamicHeaders(HeaderMap);

impl HeaderProvider for DynamicHeaders {
    fn headers(&self) -> Result<HeaderOverrides, ModelError> {
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
                .set_body_string(format!(
                    "data: {}\n\n",
                    json!({"candidates":[{"finishReason":"STOP"}]})
                )),
        )
        .mount(&server)
        .await;
    let mut config =
        support::full_config(&server.uri(), "gemini-future", publisher_resource(), true);
    let mut dynamic = HeaderMap::new();
    dynamic.insert(CONTENT_TYPE, HeaderValue::from_static("text/plain"));
    config.provider.headers.dynamic_headers = Some(Arc::new(DynamicHeaders(dynamic)));
    let model = GoogleVertexModel::new(config).unwrap();

    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].headers[CONTENT_TYPE], "application/json");
}

#[tokio::test]
async fn future_model_id_and_typed_resource_are_behaviorally_independent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/v1beta1/projects/project/locations/global/publishers/google/models/resource-model-v1:streamGenerateContent",
        ))
        .and(query_param("alt", "sse"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\n",
                    json!({"candidates":[{"finishReason":"STOP"}]})
                )),
        )
        .expect(1)
        .mount(&server)
        .await;
    let model = support::full_model(
        &server.uri(),
        "gemini-future-2099-experimental",
        publisher_resource(),
        true,
    );
    assert_eq!(model.model_id().as_str(), "gemini-future-2099-experimental");
    assert!(
        model
            .capabilities()
            .features
            .contains(Capability::TOOL_INPUT_DELTAS)
    );
    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();

    let mut conservative = support::full_config(
        &server.uri(),
        "gemini-future-2099-experimental",
        publisher_resource(),
        false,
    );
    conservative.model.capabilities.features.remove(
        Capability::TOOL_CALLING
            | Capability::PARALLEL_TOOLS
            | Capability::PROVIDER_TOOLS
            | Capability::REASONING,
    );
    conservative.model.capabilities.replay.reasoning = false;
    conservative.settings.tools.provider_tools = false;
    conservative.settings.tools.mixed_client_and_provider_tools = false;
    conservative.settings.tools.strict_functions = false;
    conservative.settings.thinking = oven_sdk_google_vertex::GoogleVertexThinkingMode::Unsupported;
    let conservative = oven_sdk_google_vertex::GoogleVertexModel::new(conservative).unwrap();
    let tool_request = Request::new(Vec::new()).with_tools(vec![ToolDefinition::new(
        "lookup",
        "lookup",
        JsonSchema::new(json!({"type":"object"})).unwrap(),
    )]);
    assert!(!conservative.supports_request(&tool_request));
}

#[tokio::test]
async fn endpoint_resource_and_api_origin_are_never_inferred_from_model_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/v1beta1/projects/project/locations/global/endpoints/deployment-7:generateContent",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates":[{"content":{"parts":[{"text":"endpoint"}]},"finishReason":"STOP"}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let model = support::full_model(
        &server.uri(),
        "publishers/not-a-resource/models/not-used",
        GoogleVertexResource::Endpoint {
            endpoint: "deployment-7".into(),
        },
        false,
    );
    let result = model
        .generate_content(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "endpoint");
}

#[tokio::test]
async fn explicit_declarations_drive_current_schema_cache_thinking_tools_and_partial_args_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "cachedContent":"projects/project/locations/global/cachedContents/cache-1",
            "generationConfig":{"thinkingConfig":{"thinkingLevel":"HIGH"}},
            "toolConfig":{"functionCallingConfig":{"streamFunctionCallArguments":true}}
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\n",
                    json!({"candidates":[{"finishReason":"STOP"}]})
                )),
        )
        .expect(1)
        .mount(&server)
        .await;
    let model = support::full_model(&server.uri(), "future-tools", publisher_resource(), true);
    let tool = ToolDefinition::new(
        "lookup",
        "lookup",
        JsonSchema::new(json!({
            "type":"object",
            "properties":{"q":{"$ref":"#/$defs/query"}},
            "$defs":{"query":{"type":"string"}}
        }))
        .unwrap(),
    )
    .with_google_vertex_tool_options(GoogleVertexToolOptions { strict: true });
    let request = Request::new(Vec::new())
        .with_tools(vec![tool])
        .with_google_vertex_options(GoogleVertexRequestOptions {
            cached_content: Some("projects/project/locations/global/cachedContents/cache-1".into()),
            thinking_config: Some(GoogleVertexThinkingConfig {
                thinking_level: Some("HIGH".into()),
                ..Default::default()
            }),
            provider_tools: vec![GoogleVertexProviderTool::GoogleSearch],
            ..Default::default()
        });
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let declaration = &body["tools"][1]["functionDeclarations"][0];
    assert!(declaration.get("parameters").is_none());
    assert_eq!(declaration["parametersJsonSchema"]["type"], "object");
    assert_eq!(
        body.pointer("/toolConfig/functionCallingConfig/mode")
            .and_then(serde_json::Value::as_str),
        Some("VALIDATED")
    );
}

#[tokio::test]
async fn invalid_tool_roots_and_undeclared_media_fail_without_network() {
    let server = MockServer::start().await;
    let model = support::full_model(
        &server.uri(),
        "future-validation",
        publisher_resource(),
        false,
    );
    for schema in [
        json!(true),
        json!({"type":"string"}),
        json!({"type":["object"]}),
        json!({"anyOf":[{"type":"object"}]}),
        json!({"type":"object","properties":{"q":{"$ref":"#/$defs/missing"}}}),
    ] {
        let request = Request::new(Vec::new()).with_tools(vec![ToolDefinition::new(
            "lookup",
            "lookup",
            JsonSchema::new(schema).unwrap(),
        )]);
        assert!(model.stream(request, AbortSignal::default()).await.is_err());
    }
    let media = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(FilePart::image(
            "image/gif",
            FileSource::Bytes(Vec::new().into()),
        )),
    ]))]);
    assert!(model.stream(media, AbortSignal::default()).await.is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn reviewed_media_sources_use_inline_https_and_gcs_shapes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\n",
                    json!({"candidates":[{"finishReason":"STOP"}]})
                )),
        )
        .mount(&server)
        .await;
    let model = support::full_model(&server.uri(), "future-media", publisher_resource(), false);
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
            "audio/mpeg",
            FileSource::Url("https://example.com/audio.mp3".parse().unwrap()),
        )),
        InputPart::File(FilePart::video(
            "video/mp4",
            FileSource::Url("gs://bucket/video.mp4".parse().unwrap()),
        )),
    ]))]);
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(
        body.pointer("/contents/0/parts/0/inlineData/data")
            .is_some()
    );
    assert_eq!(
        body.pointer("/contents/0/parts/2/fileData/fileUri")
            .and_then(serde_json::Value::as_str),
        Some("https://example.com/audio.mp3")
    );
    assert_eq!(
        body.pointer("/contents/0/parts/3/fileData/fileUri")
            .and_then(serde_json::Value::as_str),
        Some("gs://bucket/video.mp4")
    );
}

#[tokio::test]
async fn partial_arguments_compose_for_any_explicitly_enabled_model_id() {
    let server = MockServer::start().await;
    let events = [
        json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":"call-1","name":"lookup","partialArgs":[{"jsonPath":"$.message","stringValue":"hel","willContinue":true}],"willContinue":true}}]}}]}),
        json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":"call-1","partialArgs":[{"jsonPath":"$.message","stringValue":"lo"}]}}]}}]}),
        json!({"candidates":[{"finishReason":"STOP"}]}),
    ];
    let body = events
        .into_iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>();
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;
    let model = support::full_model(
        &server.uri(),
        "brand-new-future-id",
        publisher_resource(),
        true,
    );
    let result = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let call = result
        .turn
        .message
        .content
        .iter()
        .find_map(|part| match part {
            AssistantPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .unwrap();
    assert_eq!(call.input, json!({"message":"hello"}));
}

#[tokio::test]
async fn native_context_scope_and_provider_ids_are_authoritative() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\n",
                    json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":"provider-call","name":"lookup","args":{}}}]},"finishReason":"STOP"}]})
                )),
        )
        .expect(3)
        .mount(&server)
        .await;
    let model = support::full_model(&server.uri(), "future-replay", publisher_resource(), false);
    let turn = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    let local_id = turn
        .message
        .content
        .iter()
        .find_map(|part| match part {
            AssistantPart::ToolCall(call) => Some(call.id.clone()),
            _ => None,
        })
        .unwrap();
    let history = |turn| {
        Request::new(vec![
            HistoryTurn::assistant(turn),
            HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
                local_id.clone(),
                ToolContent::Json(json!({"ok":true})),
            )])),
        ])
    };
    let replayed = model
        .complete(history(turn.clone()), AbortSignal::default())
        .await
        .unwrap();
    assert!(
        replayed
            .request
            .replay
            .decisions
            .iter()
            .any(|decision| { decision.disposition == ReplayDisposition::Replayed })
    );

    let mut foreign = turn;
    let artifact = foreign.finish.native_replay.as_ref().unwrap();
    let foreign_scope = NativeContextScope::new(
        artifact.scope().provider_id.clone(),
        artifact.scope().model_id.clone(),
        ResourceId::new("projects/other/locations/global/endpoints/other").unwrap(),
    )
    .unwrap();
    foreign.finish.native_replay = Some(
        NativeReplayArtifact::new(
            AdapterId::new("oven.google.vertex.generate-content"),
            foreign_scope,
            artifact.payload().clone(),
        )
        .unwrap(),
    );
    let reconstructed = model
        .complete(history(foreign), AbortSignal::default())
        .await
        .unwrap();
    assert!(
        reconstructed
            .request
            .replay
            .decisions
            .iter()
            .any(|decision| {
                matches!(
                    decision.disposition,
                    ReplayDisposition::DiscardedForeignScope { .. }
                )
            })
    );
    let requests = server.received_requests().await.unwrap();
    for request in &requests[1..] {
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(
            body.pointer("/contents/0/parts/0/functionCall/id")
                .and_then(serde_json::Value::as_str),
            Some("provider-call")
        );
        assert_eq!(
            body.pointer("/contents/1/parts/0/functionResponse/id")
                .and_then(serde_json::Value::as_str),
            Some("provider-call")
        );
    }
}

#[test]
fn native_context_scope_canonicalizes_equivalent_urls_and_distinguishes_gateways() {
    let equivalent_a = support::full_config(
        "http://LOCALHOST:80/gateway/",
        "future-scope",
        publisher_resource(),
        false,
    );
    let equivalent_b = support::full_config(
        "http://localhost/gateway",
        "future-scope",
        publisher_resource(),
        false,
    );
    let foreign = support::full_config(
        "http://localhost/other-gateway",
        "future-scope",
        publisher_resource(),
        false,
    );

    assert_eq!(
        equivalent_a.settings.native_context_scope,
        equivalent_b.settings.native_context_scope
    );
    assert_ne!(
        equivalent_a.settings.native_context_scope,
        foreign.settings.native_context_scope
    );
}

#[tokio::test]
async fn replay_from_a_foreign_endpoint_is_reconstructed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\n",
                    json!({"candidates":[{"content":{"parts":[{"text":"ok"}]} ,"finishReason":"STOP"}]})
                )),
        )
        .expect(2)
        .mount(&server)
        .await;
    let first = support::full_model(
        &format!("{}/gateway-a", server.uri()),
        "future-foreign-endpoint",
        publisher_resource(),
        false,
    )
    .complete(Request::new(Vec::new()), AbortSignal::default())
    .await
    .unwrap();
    let second = support::full_model(
        &format!("{}/gateway-b", server.uri()),
        "future-foreign-endpoint",
        publisher_resource(),
        false,
    )
    .complete(
        Request::new(vec![HistoryTurn::assistant(first.turn)]),
        AbortSignal::default(),
    )
    .await
    .unwrap();

    assert!(matches!(
        second.request.replay.decisions.first(),
        Some(oven_sdk::ReplayDecision {
            disposition: ReplayDisposition::DiscardedForeignScope { .. },
            ..
        })
    ));
}

#[tokio::test]
async fn replay_artifact_omits_raw_endpoint_headers_and_auth_secrets() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\n",
                    json!({"candidates":[{"finishReason":"STOP"}]})
                )),
        )
        .mount(&server)
        .await;
    let mut config = support::full_config(
        &format!("{}/private-gateway-marker", server.uri()),
        "future-redaction",
        publisher_resource(),
        false,
    );
    config.provider.auth = VertexAuth::AccessToken(SecretString::new("oauth-secret-marker"));
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-private-routing-marker",
        HeaderValue::from_static("header-secret-marker"),
    );
    config.provider.headers = HeaderConfig {
        static_headers: HeaderOverrides::new(headers),
        dynamic_headers: None,
    };
    let result = oven_sdk_google_vertex::GoogleVertexModel::new(config)
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let artifact = result.turn.finish.native_replay.as_ref().unwrap();
    let serialized = serde_json::to_string(artifact).unwrap();
    let debug = format!("{artifact:?}");

    for private in [
        server.uri().as_str(),
        "private-gateway-marker",
        "x-private-routing-marker",
        "header-secret-marker",
        "oauth-secret-marker",
    ] {
        assert!(!serialized.contains(private), "serialized {private}");
        assert!(!debug.contains(private), "debug {private}");
    }
    let resource_id = artifact.scope().resource_id.as_str();
    assert!(
        resource_id.starts_with("google.vertex.generate-content.native-context-scope.v1.sha256.")
    );
    assert_eq!(resource_id.rsplit('.').next().unwrap().len(), 43);
}

#[test]
fn constructor_rejects_native_compaction_scope_and_partial_capability_mismatches() {
    let mut compaction =
        support::full_config("https://example.com", "future", publisher_resource(), false);
    compaction.model.capabilities.compaction = CompactionCapability::Native;
    let error = oven_sdk_google_vertex::GoogleVertexModel::new(compaction)
        .err()
        .expect("native compaction must be rejected");
    assert_eq!(error.kind(), oven_sdk::ModelErrorKind::Unsupported);

    let mut scope =
        support::full_config("https://example.com", "future", publisher_resource(), false);
    scope.settings.native_context_scope.resource_id = ResourceId::new("wrong-resource").unwrap();
    assert!(oven_sdk_google_vertex::GoogleVertexModel::new(scope).is_err());

    let mut partial =
        support::full_config("https://example.com", "future", publisher_resource(), true);
    partial
        .model
        .capabilities
        .features
        .remove(Capability::TOOL_INPUT_DELTAS);
    assert!(
        oven_sdk_google_vertex::GoogleVertexModel::new(ModelConfig::new(
            partial.provider,
            partial.model,
            partial.settings,
        ))
        .is_err()
    );
}
