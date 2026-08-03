mod common;

use oven_sdk::{
    AbortSignal, CompactionCapability, CompactionRequest, ErrorStage, HistoryTurn, InputPart,
    LanguageModel, ModelErrorKind, NativeContextWindow, Request, SecretString, TextPart,
    UserMessage,
};
use oven_sdk_azure::{
    AzureApiRoute, AzureApiVersion, AzureOpenAiAuth, AzureOpenAiCompactionOptions,
    AzureOpenAiCompactionRequestExt, AzureOpenAiResponsesCompaction, AzureOpenAiResponsesModel,
    AzureOpenAiResponsesSettings,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn compact_document() -> serde_json::Value {
    serde_json::json!({
        "id":"resp_compact_1",
        "object":"response.compaction",
        "created_at":1_764_967_971_u64,
        "output":[
            {
                "id":"msg_user_1",
                "type":"message",
                "status":"completed",
                "role":"user",
                "content":[{"type":"input_text","text":"old context"}]
            },
            {
                "id":"cmp_1",
                "type":"compaction",
                "encrypted_content":"opaque-encrypted-context",
                "created_by":"azure-openai"
            }
        ],
        "usage":{
            "input_tokens":139,
            "input_tokens_details":{"cached_tokens":10,"cache_write_tokens":4},
            "output_tokens":438,
            "output_tokens_details":{"reasoning_tokens":64},
            "total_tokens":577
        }
    })
}

async fn mount_compaction(server: &MockServer, value: serde_json::Value) {
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses/compact"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(value)
                .insert_header("apim-request-id", "req_compact_1"),
        )
        .mount(server)
        .await;
}

fn user_request(text: &str) -> Request {
    Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new(text)),
    ]))])
}

#[tokio::test]
async fn responses_v1_compaction_returns_and_reuses_the_canonical_window() {
    let server = MockServer::start().await;
    mount_compaction(&server, compact_document()).await;
    common::mount(
        &server,
        "/openai/v1/responses",
        common::responses_document("continued"),
    )
    .await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    let request = CompactionRequest::new(user_request("compact this"))
        .with_azure_openai_compaction_options(AzureOpenAiCompactionOptions {
            instructions: Some("preserve the coding task".into()),
            prompt_cache_key: Some("cache-key".into()),
            prompt_cache_retention: Some("24h".into()),
            service_tier: Some("future-azure-tier-2027".into()),
        });
    let compacted = model
        .compact(request, AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(
        compacted.native_context.scope(),
        model.native_context_scope().unwrap()
    );
    assert_eq!(compacted.usage.input_tokens, Some(139));
    assert_eq!(compacted.usage.input_tokens_no_cache, Some(125));
    assert_eq!(compacted.usage.input_tokens_cache_read, Some(10));
    assert_eq!(compacted.usage.input_tokens_cache_write, Some(4));
    assert_eq!(compacted.usage.output_tokens_reasoning, Some(64));
    assert_eq!(
        compacted.response.request_id.as_deref(),
        Some("req_compact_1")
    );
    assert_eq!(
        compacted
            .native_context
            .payload()
            .pointer("/output/1/type")
            .and_then(serde_json::Value::as_str),
        Some("compaction")
    );

    let serialized = serde_json::to_string(&compacted.native_context).unwrap();
    assert!(!serialized.contains("test-v1-route"));
    assert!(!serialized.contains("secret"));
    assert!(!serialized.contains(&server.uri()));

    model
        .complete(
            user_request("continue").with_native_context(compacted.native_context),
            AbortSignal::default(),
        )
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let compact_body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(compact_body["model"], "deployment");
    assert_eq!(compact_body["instructions"], "preserve the coding task");
    assert_eq!(compact_body["prompt_cache_key"], "cache-key");
    assert_eq!(compact_body["prompt_cache_retention"], "24h");
    assert_eq!(compact_body["service_tier"], "future-azure-tier-2027");
    assert!(compact_body.get("previous_response_id").is_none());
    assert!(compact_body.get("prompt_cache_options").is_none());
    assert!(compact_body.get("stream").is_none());
    assert!(compact_body.get("store").is_none());

    let continuation: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    let input = continuation["input"].as_array().unwrap();
    assert_eq!(input[0]["id"], "msg_user_1");
    assert_eq!(input[1]["type"], "compaction");
    assert_eq!(input[2]["role"], "user");
}

#[tokio::test]
async fn future_service_tier_label_is_passed_through_unchanged() {
    let server = MockServer::start().await;
    mount_compaction(&server, compact_document()).await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    let request = CompactionRequest::new(user_request("compact future tier"))
        .with_azure_openai_compaction_options(AzureOpenAiCompactionOptions {
            service_tier: Some("azure-future-tier-2028".into()),
            ..Default::default()
        });
    model
        .compact(request, AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["service_tier"], "azure-future-tier-2028");
}

#[test]
fn native_compaction_requires_responses_v1_and_matching_explicit_settings() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::start());

    let mut native_chat = common::gpt5_chat();
    native_chat.capabilities.compaction = CompactionCapability::Native;
    assert!(
        common::provider(&server, AzureApiRoute::V1)
            .chat("deployment", native_chat)
            .is_err()
    );

    let mut missing = common::gpt5_compaction();
    missing.compaction = AzureOpenAiResponsesCompaction::Unsupported;
    assert!(
        common::provider(&server, AzureApiRoute::V1)
            .responses("deployment", missing)
            .is_err()
    );

    let mut unsupported = common::gpt5();
    unsupported.compaction = AzureOpenAiResponsesCompaction::V1 {
        routing_discriminator: "route".into(),
    };
    assert!(
        common::provider(&server, AzureApiRoute::V1)
            .responses("deployment", unsupported)
            .is_err()
    );

    for route in [
        AzureApiRoute::V1Preview,
        AzureApiRoute::Dated(AzureApiVersion::new("2026-01-01-preview").unwrap()),
    ] {
        assert!(
            common::provider(&server, route)
                .responses("deployment", common::gpt5_compaction())
                .is_err()
        );
    }

    for discriminator in ["", "   ", "bad\nroute"] {
        let mut setup = common::gpt5_compaction();
        setup.compaction = AzureOpenAiResponsesCompaction::V1 {
            routing_discriminator: discriminator.into(),
        };
        assert!(
            common::provider(&server, AzureApiRoute::V1)
                .responses("deployment", setup)
                .is_err()
        );
    }
}

#[tokio::test]
async fn compaction_is_bounded_fail_closed_and_cancellable() {
    let malformed_server = MockServer::start().await;
    let mut malformed = compact_document();
    malformed["output"] = serde_json::json!([{
        "id":"msg",
        "type":"message",
        "status":"completed",
        "role":"user",
        "content":[{"type":"input_text","text":"missing compaction"}]
    }]);
    mount_compaction(&malformed_server, malformed).await;
    let model = common::provider(&malformed_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    let error = model
        .compact(
            CompactionRequest::new(user_request("malformed response")),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::InvalidResponse);
    assert_eq!(error.diagnostics.stage, ErrorStage::NativeContextDecode);

    let oversized_server = MockServer::start().await;
    let mut oversized = compact_document();
    oversized["output"] = serde_json::Value::Array(
        (0..129)
            .map(|index| {
                serde_json::json!({
                    "id":format!("msg_{index}"),
                    "type":"message",
                    "status":"completed",
                    "role":"user",
                    "content":[{"type":"input_text","text":"x"}]
                })
            })
            .collect(),
    );
    mount_compaction(&oversized_server, oversized).await;
    let model = common::provider(&oversized_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    assert!(
        model
            .compact(
                CompactionRequest::new(user_request("oversized response")),
                AbortSignal::default()
            )
            .await
            .is_err()
    );

    let cache_server = MockServer::start().await;
    let mut inconsistent_cache = compact_document();
    inconsistent_cache["usage"]["input_tokens_details"] =
        serde_json::json!({"cached_tokens":100,"cache_write_tokens":50});
    mount_compaction(&cache_server, inconsistent_cache).await;
    let model = common::provider(&cache_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    let error = model
        .compact(
            CompactionRequest::new(user_request("invalid cache usage")),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::InvalidResponse);
    assert_eq!(error.diagnostics.stage, ErrorStage::NativeContextDecode);

    let cancelled_server = MockServer::start().await;
    let model = common::provider(&cancelled_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    let error = model
        .compact(CompactionRequest::new(user_request("cancel")), signal)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::Abort);
    assert!(
        cancelled_server
            .received_requests()
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn forged_native_context_fingerprint_is_rejected_before_generation_io() {
    let server = MockServer::start().await;
    mount_compaction(&server, compact_document()).await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    let result = model
        .compact(
            CompactionRequest::new(user_request("forge")),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let mut payload = result.native_context.payload().clone();
    payload["fingerprint"] = "forged".into();
    let forged = NativeContextWindow::new(
        result.native_context.adapter_id().clone(),
        result.native_context.scope().clone(),
        payload,
    )
    .unwrap();
    let error = model
        .stream(
            user_request("continue").with_native_context(forged),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::NativeContext);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn compaction_options_and_http_errors_fail_before_unsafe_continuation() {
    let invalid_server = MockServer::start().await;
    let model = common::provider(&invalid_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    let mut invalid_request = user_request("invalid options");
    invalid_request.provider_options.insert(
        "azure_openai".into(),
        serde_json::json!({"compaction":{"prompt_cache_options":{"mode":"explicit","ttl":"30m"}}}),
    );
    let request = CompactionRequest::new(invalid_request);
    let error = model
        .compact(request, AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::InvalidRequest);
    assert!(invalid_server.received_requests().await.unwrap().is_empty());

    let mut stored_request = user_request("stored response");
    stored_request.provider_options.insert(
        "azure_openai".into(),
        serde_json::json!({"compaction":{"previous_response_id":"resp_stored"}}),
    );
    let error = model
        .compact(
            CompactionRequest::new(stored_request),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::InvalidRequest);
    assert!(invalid_server.received_requests().await.unwrap().is_empty());

    let error_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses/compact"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("apim-request-id", "req_compact_quota")
                .insert_header("x-ms-retry-after-ms", "50")
                .set_body_json(serde_json::json!({
                    "error":{"code":"insufficient_quota","message":"quota exceeded"}
                })),
        )
        .mount(&error_server)
        .await;
    let model = common::provider(&error_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    let error = model
        .compact(
            CompactionRequest::new(user_request("quota")),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::Quota);
    assert_eq!(
        error.diagnostics.request_id.as_deref(),
        Some("req_compact_quota")
    );
    assert_eq!(
        error.diagnostics.retry_after,
        Some(std::time::Duration::from_millis(50))
    );
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseBody);
}

#[tokio::test]
async fn zero_encoded_local_input_is_rejected_without_io() {
    let server = MockServer::start().await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    let request = CompactionRequest::new(Request::new(Vec::new()))
        .with_azure_openai_compaction_options(AzureOpenAiCompactionOptions {
            instructions: Some("instructions alone are not local context".into()),
            service_tier: Some("future-tier".into()),
            ..Default::default()
        });
    let error = model
        .compact(request, AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::InvalidRequest);
    assert_eq!(error.diagnostics.stage, ErrorStage::NativeContextEncode);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[test]
fn settings_type_requires_explicit_compaction_selection() {
    let settings = AzureOpenAiResponsesSettings::default();
    assert_eq!(
        settings.compaction,
        AzureOpenAiResponsesCompaction::Unsupported
    );
}

#[test]
fn native_scope_uses_the_non_secret_routing_discriminator_not_credentials() {
    let configure = |secret: &str, discriminator: &str| {
        let mut setup = common::gpt5_compaction();
        setup.compaction = AzureOpenAiResponsesCompaction::V1 {
            routing_discriminator: discriminator.into(),
        };
        AzureOpenAiResponsesModel::new(common::responses_config(
            "https://example.test",
            AzureApiRoute::V1,
            "deployment",
            setup,
            AzureOpenAiAuth::ApiKey(SecretString::new(secret)),
        ))
        .unwrap()
    };
    let first = configure("first-secret", "route-a");
    let rotated = configure("rotated-secret", "route-a");
    let rerouted = configure("first-secret", "route-b");
    assert_eq!(first.native_context_scope(), rotated.native_context_scope());
    assert_ne!(
        first.native_context_scope(),
        rerouted.native_context_scope()
    );
    let serialized = serde_json::to_string(first.native_context_scope().unwrap()).unwrap();
    assert!(!serialized.contains("route-a"));
    assert!(!serialized.contains("first-secret"));
}
