pub mod common;

use oven_sdk::{
    AbortSignal, CompactionCapability, CompactionRequest, HistoryTurn, InputPart, LanguageModel,
    ModelErrorKind, NativeContextWindow, Request, TextPart, UserMessage,
};
use oven_sdk_conformance::{
    assert_compaction_cancellation, assert_compaction_round_trip, assert_native_compaction,
    assert_native_context_window,
};
use oven_sdk_openai::{
    OpenAiChatModel, OpenAiCompatibleChatModel, OpenAiPromptCacheOptions,
    OpenAiResponsesCompaction, OpenAiResponsesCompactionOptions,
    OpenAiResponsesCompactionRequestExt, OpenAiResponsesModel,
};
use wiremock::MockServer;
use wiremock::{
    Mock, ResponseTemplate,
    matchers::{method, path},
};

fn user(text: &str) -> HistoryTurn {
    HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(text))]))
}

fn large_compact_document() -> serde_json::Value {
    let mut output = (0..130)
        .map(|message_index| {
            let content = if message_index == 0 {
                (0..130)
                    .map(|content_index| {
                        serde_json::json!({
                            "type": "input_text",
                            "text": format!("retained-{content_index}"),
                            "prompt_cache_breakpoint": {"mode": "explicit"}
                        })
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![serde_json::json!({
                    "type": "input_text",
                    "text": format!("message-{message_index}")
                })]
            };
            serde_json::json!({
                "type": "message",
                "id": format!("msg_large_{message_index}"),
                "role": "user",
                "status": "completed",
                "content": content
            })
        })
        .collect::<Vec<_>>();
    output.push(serde_json::json!({
        "type": "compaction",
        "id": "cmp_large",
        "encrypted_content": "opaque-large-window"
    }));
    serde_json::json!({
        "id": "cmp_large_response",
        "created_at": 1_754_000_001_u64,
        "object": "response.compaction",
        "output": output,
        "usage": {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 1010
        }
    })
}

#[test]
fn chat_and_compatible_chat_reject_native_compaction_declarations() {
    let mut official =
        common::official_chat_config_at("https://example.test/v1", "gpt-4o-mini", "secret");
    official.model.capabilities.compaction = CompactionCapability::Native;
    assert!(matches!(
        OpenAiChatModel::new(official),
        Err(error) if error.kind() == ModelErrorKind::InvalidRequest
    ));

    let mut compatible =
        common::compatible_config_at("https://example.test/v1", "fixture-model", "secret");
    compatible.model.capabilities.compaction = CompactionCapability::Native;
    assert!(matches!(
        OpenAiCompatibleChatModel::new(compatible),
        Err(error) if error.kind() == ModelErrorKind::InvalidRequest
    ));
}

#[test]
fn responses_requires_matching_explicit_compaction_setting_and_capability() {
    let mut native_without_setting =
        common::official_responses_config_at("https://example.test/v1", "gpt-5-mini", "secret");
    native_without_setting.model.capabilities.compaction = CompactionCapability::Native;
    assert!(matches!(
        OpenAiResponsesModel::new(native_without_setting),
        Err(error) if error.kind() == ModelErrorKind::InvalidRequest
    ));

    let mut setting_without_native =
        common::official_responses_config_at("https://example.test/v1", "gpt-5-mini", "secret");
    setting_without_native.settings.compaction = OpenAiResponsesCompaction::V1;
    assert!(matches!(
        OpenAiResponsesModel::new(setting_without_native),
        Err(error) if error.kind() == ModelErrorKind::InvalidRequest
    ));
}

#[tokio::test]
async fn native_compaction_preserves_canonical_output_options_usage_and_metadata() {
    let server = MockServer::start().await;
    common::mount_compaction(&server).await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let request = CompactionRequest::new(Request::new(vec![user("compact this")]))
        .with_openai_responses_compaction_options(OpenAiResponsesCompactionOptions {
            instructions: Some("preserve the working context\nnext\tcolumn\rline".into()),
            prompt_cache_key: Some("cache-route".into()),
            prompt_cache_options: Some(OpenAiPromptCacheOptions {
                mode: "explicit".into(),
                ttl: "30m".into(),
            }),
            prompt_cache_retention: Some("future-retention".into()),
            service_tier: Some("future-burst".into()),
        });
    let result = model
        .compact(request, AbortSignal::default())
        .await
        .unwrap();

    assert_eq!(result.usage.input_tokens, Some(120));
    assert_eq!(result.usage.input_tokens_cache_read, Some(20));
    assert_eq!(result.usage.output_tokens, Some(8));
    assert_eq!(result.usage.output_tokens_text, Some(5));
    assert_eq!(result.usage.output_tokens_reasoning, Some(3));
    assert_eq!(result.response.http_status, Some(200));
    assert_eq!(result.response.request_id.as_deref(), Some("req_compact_1"));
    assert_eq!(
        result.response.response_metadata["openai.compaction_id"],
        "cmp_1"
    );
    assert_eq!(
        result.native_context.payload()["output"],
        common::compact_document()["output"]
    );

    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(requests[0].url.path(), "/responses/compact");
    assert_eq!(body["model"], "gpt-5-mini");
    assert_eq!(body["input"][0]["role"], "user");
    assert_eq!(
        body["instructions"],
        "preserve the working context\nnext\tcolumn\rline"
    );
    assert_eq!(body["prompt_cache_key"], "cache-route");
    assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
    assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
    assert_eq!(body["prompt_cache_retention"], "future-retention");
    assert_eq!(body["service_tier"], "future-burst");
}

#[tokio::test]
async fn native_context_continuation_prepends_the_exact_compacted_window() {
    let server = MockServer::start().await;
    common::mount_compaction(&server).await;
    common::mount(
        &server,
        "/responses",
        common::responses_document("continued"),
    )
    .await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let compacted = model
        .compact(
            CompactionRequest::new(Request::new(vec![user("old context")])),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    model
        .complete(
            Request::new(vec![user("new turn")]).with_native_context(compacted.native_context),
            AbortSignal::default(),
        )
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let continuation = requests
        .iter()
        .find(|request| request.url.path() == "/responses")
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&continuation.body).unwrap();
    let canonical = common::compact_document()["output"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        &body["input"].as_array().unwrap()[..canonical.len()],
        canonical
    );
    assert_eq!(body["input"][canonical.len()]["role"], "user");
    assert_eq!(body["input"][0]["id"], "msg_retained_1");
    assert_eq!(body["input"][0]["status"], "completed");
    assert_eq!(body["input"][1]["created_by"], "openai");
    for content_index in 0..3 {
        assert_eq!(
            body["input"][0]["content"][content_index]["prompt_cache_breakpoint"],
            serde_json::json!({"mode":"explicit"})
        );
    }
}

#[tokio::test]
async fn large_retained_window_and_content_continue_without_arbitrary_item_caps() {
    let server = MockServer::start().await;
    let compact_document = large_compact_document();
    Mock::given(method("POST"))
        .and(path("/responses/compact"))
        .respond_with(ResponseTemplate::new(200).set_body_json(compact_document.clone()))
        .mount(&server)
        .await;
    common::mount(
        &server,
        "/responses",
        common::responses_document("continued"),
    )
    .await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let compacted = model
        .compact(
            CompactionRequest::new(Request::new(vec![user("old context")])),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    model
        .complete(
            Request::new(vec![user("new turn")]).with_native_context(compacted.native_context),
            AbortSignal::default(),
        )
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let continuation = requests
        .iter()
        .find(|request| request.url.path() == "/responses")
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&continuation.body).unwrap();
    let canonical = compact_document["output"].as_array().unwrap();
    assert!(canonical.len() > 128);
    assert!(canonical[0]["content"].as_array().unwrap().len() > 128);
    assert_eq!(
        &body["input"].as_array().unwrap()[..canonical.len()],
        canonical
    );
}

#[tokio::test]
async fn native_compaction_passes_core_conformance_round_trip_and_cancellation() {
    let server = MockServer::start().await;
    common::mount_compaction(&server).await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let seed = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let scope = seed
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .scope()
        .clone();
    let compaction = CompactionRequest::new(Request::new(vec![user("compact")]));
    let result = assert_native_compaction(&model, &scope, compaction.clone())
        .await
        .unwrap();
    assert_native_context_window(model.descriptor(), &scope, &result.native_context).unwrap();
    assert_compaction_cancellation(&model, compaction.clone())
        .await
        .unwrap();
    assert_compaction_round_trip(
        &model,
        &scope,
        compaction,
        Request::new(vec![user("continue")]),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn foreign_and_noncurrent_native_context_are_rejected_before_io() {
    let server = MockServer::start().await;
    common::mount_compaction(&server).await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let result = model
        .compact(
            CompactionRequest::new(Request::new(vec![user("compact")])),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let scope = result.native_context.scope().clone();
    let invalid = NativeContextWindow::new(
        result.native_context.adapter_id().clone(),
        scope,
        serde_json::json!({"output": common::compact_document()["output"]}),
    )
    .unwrap();
    let error = model
        .stream(
            Request::new(vec![user("continue")]).with_native_context(invalid),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::NativeContext);
    assert_eq!(
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .filter(|request| request.url.path() == "/responses")
            .count(),
        0
    );
}

#[tokio::test]
async fn tampered_native_context_fingerprint_is_rejected_before_io() {
    let server = MockServer::start().await;
    common::mount_compaction(&server).await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let result = model
        .compact(
            CompactionRequest::new(Request::new(vec![user("compact")])),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let mut payload = result.native_context.payload().clone();
    payload["output"][0]["content"][0]["text"] = "tampered".into();
    let tampered = NativeContextWindow::new(
        result.native_context.adapter_id().clone(),
        result.native_context.scope().clone(),
        payload,
    )
    .unwrap();
    let error = model
        .stream(
            Request::new(vec![user("continue")]).with_native_context(tampered),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::NativeContext);
}

#[tokio::test]
async fn invalid_compaction_options_are_rejected_before_io() {
    let server = MockServer::start().await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let request = CompactionRequest::new(Request::new(vec![user("compact")]))
        .with_openai_responses_compaction_options(OpenAiResponsesCompactionOptions {
            prompt_cache_options: Some(OpenAiPromptCacheOptions {
                mode: "future-mode".into(),
                ttl: "forever".into(),
            }),
            ..Default::default()
        });
    let error = model
        .compact(request, AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::InvalidRequest);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn instructions_reject_nul_and_disallowed_controls_before_io() {
    let server = MockServer::start().await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    for instructions in ["unsafe\0text", "unsafe\u{7}text"] {
        let request = CompactionRequest::new(Request::new(vec![user("compact")]))
            .with_openai_responses_compaction_options(OpenAiResponsesCompactionOptions {
                instructions: Some(instructions.into()),
                ..Default::default()
            });
        let error = model
            .compact(request, AbortSignal::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ModelErrorKind::InvalidRequest);
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn compaction_http_errors_keep_structured_diagnostics() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses/compact"))
        .respond_with(
            ResponseTemplate::new(429)
                .set_body_json(serde_json::json!({
                    "error": {"type": "rate_limit_error", "code": "rate_limit", "message": "slow"}
                }))
                .insert_header("x-request-id", "req_compact_error")
                .insert_header("retry-after-ms", "25"),
        )
        .mount(&server)
        .await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let error = model
        .compact(
            CompactionRequest::new(Request::new(vec![user("compact")])),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::RateLimited);
    assert_eq!(error.diagnostics.http_status, Some(429));
    assert_eq!(
        error.diagnostics.request_id.as_deref(),
        Some("req_compact_error")
    );
    assert_eq!(error.diagnostics.stage, oven_sdk::ErrorStage::ResponseBody);
}

#[tokio::test]
async fn noncanonical_compaction_response_is_rejected_without_a_window() {
    let server = MockServer::start().await;
    let mut body = common::compact_document();
    body["legacy"] = true.into();
    Mock::given(method("POST"))
        .and(path("/responses/compact"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let error = model
        .compact(
            CompactionRequest::new(Request::new(vec![user("compact")])),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::InvalidResponse);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::NativeContextDecode
    );
}

#[tokio::test]
async fn compaction_cancellation_while_waiting_for_headers_returns_abort() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses/compact"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(std::time::Duration::from_secs(2))
                .set_body_json(common::compact_document()),
        )
        .mount(&server)
        .await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let (signal, registration) = AbortSignal::new();
    let operation = model.compact(
        CompactionRequest::new(Request::new(vec![user("compact")])),
        signal,
    );
    tokio::pin!(operation);
    tokio::select! {
        result = &mut operation => panic!("compaction completed before cancellation: {result:?}"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(25)) => registration.abort(),
    }
    let error = operation.await.unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::Abort);
}
