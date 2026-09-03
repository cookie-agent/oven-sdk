use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use oven_sdk::{
    AbortSignal, Capability, ErrorStage, FilePart, FileSource, HistoryTurn, InputPart,
    LanguageModel, ModelErrorKind, ProviderId, ReplayPolicy, Request, UserMessage,
};
use oven_sdk_anthropic::{
    ANTHROPIC_AWS_MESSAGES_ADAPTER_ID, ANTHROPIC_AWS_PROVIDER_ID, AnthropicAwsCredentials,
    AnthropicAwsRequestExt, AnthropicAwsRequestOptions, AnthropicTimeouts,
    MINIMAX_MESSAGES_ADAPTER_ID, MINIMAX_PROVIDER_ID, MiniMaxMediaExt, MiniMaxMediaOptions,
    MiniMaxRequestExt, MiniMaxRequestOptions, MiniMaxThinking,
};
use reqwest::header::{HeaderMap, HeaderValue};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn response() -> &'static str {
    concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"test\",\"usage\":{\"input_tokens\":1}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    )
}

async fn success_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    server
}

#[test]
fn aws_secret_credentials_use_redacted_secret_strings() {
    let credentials = AnthropicAwsCredentials {
        access_key_id: "AKID".into(),
        secret_access_key: oven_sdk::SecretString::new("secret-value"),
        session_token: Some(oven_sdk::SecretString::new("session-value")),
    };
    let debug = format!("{credentials:?}");
    assert!(!debug.contains("secret-value"));
    assert!(!debug.contains("session-value"));
    assert_eq!(
        credentials.secret_access_key.expose_secret(),
        "secret-value"
    );
    assert_eq!(
        credentials.session_token.as_ref().unwrap().expose_secret(),
        "session-value"
    );
}

#[tokio::test]
async fn minimax_uses_bearer_auth_default_wire_options_and_distinct_identity() {
    let server = success_server().await;
    let model = MiniMax::builder()
        .api_key("mini-secret")
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("MiniMax-M3");
    let mut request = Request::new(Vec::new()).with_minimax_options(MiniMaxRequestOptions {
        thinking: Some(MiniMaxThinking::Adaptive),
        service_tier: Some("future-priority".into()),
        user_id: Some("user-1".into()),
    });
    request.inference.temperature = Some(2.0);
    request.inference.top_p = Some(0.95);
    request.inference.max_output_tokens = Some(123);
    request.provider_options.insert(
        "anthropic".into(),
        serde_json::json!({"thinking":{"enabled":{"budget_tokens":18446744073709551615_u64}}}),
    );
    let completed = model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();

    let descriptor = model.descriptor();
    assert_eq!(
        descriptor.identity.provider_id.as_str(),
        MINIMAX_PROVIDER_ID
    );
    assert_eq!(descriptor.adapter_id.as_str(), MINIMAX_MESSAGES_ADAPTER_ID);
    assert_eq!(
        completed
            .turn
            .finish
            .native_replay
            .as_ref()
            .unwrap()
            .adapter_id()
            .as_str(),
        MINIMAX_MESSAGES_ADAPTER_ID
    );
    assert_eq!(
        completed
            .turn
            .finish
            .native_replay
            .as_ref()
            .unwrap()
            .payload()["format"],
        "oven.minimax.messages.assistant.v3"
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].headers["authorization"], "Bearer mini-secret");
    assert!(requests[0].headers.get("anthropic-version").is_none());
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["service_tier"], "future-priority");
    assert_eq!(body["temperature"], 2.0);
    assert_eq!(body["top_p"], 0.95);
    assert_eq!(body["max_tokens"], 123);
    assert_eq!(
        completed
            .turn
            .finish
            .native_replay
            .as_ref()
            .unwrap()
            .scope()
            .model_id
            .as_str(),
        "MiniMax-M3"
    );
}

#[tokio::test]
async fn minimax_m3_media_options_and_file_references_are_encoded() {
    let server = success_server().await;
    let model = MiniMax::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("MiniMax-M3");
    let image = FilePart::image(
        "image/webp",
        FileSource::Bytes(bytes::Bytes::from_static(b"image")),
    )
    .with_minimax_media_options(MiniMaxMediaOptions {
        detail: Some("future-detail".into()),
        ..Default::default()
    });
    let video = FilePart::video(
        "video/mp4",
        FileSource::ProviderReference {
            provider: ProviderId::new("minimax"),
            id: "file_123".into(),
        },
    )
    .with_minimax_media_options(MiniMaxMediaOptions {
        detail: Some("high".into()),
        fps: Some(2.5),
        max_long_side_pixel: Some(1080),
    });
    model
        .complete(
            Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
                InputPart::File(image),
                InputPart::File(video),
            ]))]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let content = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "image");
    assert_eq!(content[0]["source"]["detail"], "future-detail");
    assert_eq!(content[1]["type"], "video");
    assert_eq!(content[1]["source"]["url"], "mm_file://file_123");
    assert_eq!(content[1]["source"]["fps"], 2.5);
    assert_eq!(content[1]["source"]["max_long_side_pixel"], 1080);
}

#[tokio::test]
async fn minimax_m2_rejects_media_before_dispatch() {
    let server = MockServer::start().await;
    let model = MiniMax::builder()
        .base_url(server.uri())
        .capabilities(minimax_capabilities(ReplayPolicy::IfValid, false))
        .build()
        .unwrap()
        .model("MiniMax-M2.7");
    let error = model
        .stream(
            Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
                InputPart::File(FilePart::image(
                    "image/png",
                    FileSource::Bytes(bytes::Bytes::new()),
                )),
            ]))]),
            AbortSignal::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Unsupported);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn anthropic_aws_bearer_key_sets_workspace_endpoint_options_and_identity() {
    let server = success_server().await;
    let model = AnthropicAws::builder("us-west-2", "wrkspc_test")
        .bearer_key("aws-platform-key")
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-6");
    let completed = model
        .complete(
            Request::new(Vec::new()).with_anthropic_aws_options(AnthropicAwsRequestOptions {
                inference_geo: Some("future-geo".into()),
            }),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let descriptor = model.descriptor();
    assert_eq!(
        descriptor.identity.provider_id.as_str(),
        ANTHROPIC_AWS_PROVIDER_ID
    );
    assert_eq!(
        descriptor.adapter_id.as_str(),
        ANTHROPIC_AWS_MESSAGES_ADAPTER_ID
    );
    assert_eq!(
        completed.turn.finish.native_replay.unwrap().payload()["format"],
        "oven.anthropic.aws.messages.assistant.v3"
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].headers["x-api-key"], "aws-platform-key");
    assert_eq!(requests[0].headers["anthropic-workspace-id"], "wrkspc_test");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["inference_geo"], "future-geo");
}

#[tokio::test]
async fn aws_native_context_scope_binds_endpoint_region_workspace_without_secrets() {
    let server = success_server().await;

    async fn capture(
        endpoint: &str,
        region: &str,
        workspace: &str,
        key: &str,
    ) -> oven_sdk::NativeReplayArtifact {
        AnthropicAws::builder(region, workspace)
            .bearer_key(key)
            .base_url(endpoint)
            .build()
            .unwrap()
            .model("same-model")
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap()
            .turn
            .finish
            .native_replay
            .unwrap()
    }

    let baseline = capture(&server.uri(), "us-west-2", "workspace-a", "secret-key").await;
    let other_region = capture(&server.uri(), "us-east-1", "workspace-a", "secret-key").await;
    let other_workspace = capture(&server.uri(), "us-west-2", "workspace-b", "secret-key").await;
    assert_ne!(
        baseline.scope().resource_id,
        other_region.scope().resource_id
    );
    assert_ne!(
        baseline.scope().resource_id,
        other_workspace.scope().resource_id
    );
    let serialized = serde_json::to_string(&baseline).unwrap();
    assert!(!serialized.contains("workspace-a"));
    assert!(!serialized.contains("secret-key"));
    assert!(!serialized.contains(&server.uri()));
}

#[tokio::test]
async fn anthropic_aws_sigv4_invokes_provider_and_signs_exact_request_body() {
    let server = success_server().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let provider_calls = Arc::clone(&calls);
    let model = AnthropicAws::builder("us-west-2", "wrkspc_test")
        .credential_provider(move || {
            provider_calls.fetch_add(1, Ordering::SeqCst);
            async {
                Ok(AnthropicAwsCredentials {
                    access_key_id: "AKIDEXAMPLE".into(),
                    secret_access_key: oven_sdk::SecretString::new("secret"),
                    session_token: Some(oven_sdk::SecretString::new("session-token")),
                })
            }
        })
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-4-8");
    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let requests = server.received_requests().await.unwrap();
    let authorization = requests[0].headers["authorization"].to_str().unwrap();
    assert!(authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"));
    assert!(authorization.contains("/us-west-2/aws-external-anthropic/aws4_request"));
    assert_eq!(requests[0].headers["x-amz-security-token"], "session-token");
    assert!(requests[0].headers.contains_key("x-amz-date"));
    assert_eq!(requests[0].headers["anthropic-workspace-id"], "wrkspc_test");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "claude-opus-4-8");
}

#[test]
fn anthropic_aws_rejects_static_signing_headers() {
    for (name, value) in [
        ("x-amz-date", "20260801T000000Z"),
        ("x-amz-security-token", "caller-token"),
        ("x-amz-content-sha256", "caller-hash"),
    ] {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_str(value).unwrap(),
        );
        assert!(
            AnthropicAws::builder("us-west-2", "workspace")
                .bearer_key("configured")
                .default_headers(headers.clone())
                .build()
                .is_err()
        );
        assert!(
            AnthropicAws::builder("us-west-2", "workspace")
                .static_credentials(AnthropicAwsCredentials {
                    access_key_id: "id".into(),
                    secret_access_key: oven_sdk::SecretString::new("secret"),
                    session_token: None,
                })
                .default_headers(headers)
                .build()
                .is_err()
        );
    }
}

#[tokio::test]
async fn anthropic_aws_caller_auth_suppresses_configured_auth() {
    let server = success_server().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let provider_calls = Arc::clone(&calls);
    let model = AnthropicAws::builder("us-west-2", "workspace")
        .credential_provider(move || {
            provider_calls.fetch_add(1, Ordering::SeqCst);
            async {
                Ok(AnthropicAwsCredentials {
                    access_key_id: "id".into(),
                    secret_access_key: oven_sdk::SecretString::new("secret"),
                    session_token: None,
                })
            }
        })
        .header_provider(Arc::new(|| {
            HeaderMap::from_iter([(
                reqwest::header::HeaderName::from_static("cookie"),
                HeaderValue::from_static("caller-session=value"),
            )])
        }))
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-4-8");
    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].headers["cookie"], "caller-session=value");
    assert!(requests[0].headers.get("authorization").is_none());

    let bearer = AnthropicAws::builder("us-west-2", "workspace")
        .bearer_key("configured")
        .header_provider(Arc::new(|| {
            HeaderMap::from_iter([(
                reqwest::header::HeaderName::from_static("cookie"),
                HeaderValue::from_static("caller-session=value"),
            )])
        }))
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-4-8");
    bearer
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[1].headers["cookie"], "caller-session=value");
    assert!(requests[1].headers.get("x-api-key").is_none());
}

#[tokio::test]
async fn anthropic_aws_credential_provider_timeout_and_abort_are_local() {
    let server = MockServer::start().await;
    let stalled = AnthropicAws::builder("us-west-2", "workspace")
        .credential_provider(std::future::pending)
        .timeouts(AnthropicTimeouts {
            headers: Duration::from_secs(1),
            credentials: Duration::from_millis(20),
            stream_idle: Duration::from_secs(1),
        })
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-4-8");
    let timeout = stalled
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(timeout.kind, ModelErrorKind::Timeout);
    assert_eq!(timeout.diagnostics.stage, ErrorStage::RequestEncoding);

    let aborting = AnthropicAws::builder("us-west-2", "workspace")
        .credential_provider(std::future::pending)
        .timeouts(AnthropicTimeouts {
            headers: Duration::from_secs(1),
            credentials: Duration::from_secs(1),
            stream_idle: Duration::from_secs(1),
        })
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-4-8");
    let (signal, registration) = AbortSignal::new();
    let future = aborting.stream(Request::new(Vec::new()), signal);
    tokio::pin!(future);
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_millis(20)) => registration.abort(),
        _ = &mut future => panic!("credential provider unexpectedly completed"),
    }
    let aborted = future.await.unwrap_err();
    assert_eq!(aborted.kind, ModelErrorKind::Abort);
    assert_eq!(aborted.diagnostics.stage, ErrorStage::RequestEncoding);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[test]
fn explicit_provider_declarations_are_distinct() {
    let minimax = MiniMax::builder().build().unwrap().model("MiniMax-M3");
    assert!(
        minimax
            .capabilities()
            .modalities
            .input
            .contains(&oven_sdk::Modality::video())
    );
    let aws = AnthropicAws::builder("us-east-1", "wrkspc_test")
        .static_credentials(AnthropicAwsCredentials {
            access_key_id: "id".into(),
            secret_access_key: oven_sdk::SecretString::new("secret"),
            session_token: None,
        })
        .build()
        .unwrap()
        .model("claude-opus-4-8");
    assert!(
        aws.capabilities()
            .modalities
            .input
            .contains(&oven_sdk::Modality::pdf())
    );
}

#[test]
fn minimax_model_id_does_not_select_protocol_behavior() {
    let provider = MiniMax::builder().build().unwrap();
    let request = Request::new(Vec::new()).with_minimax_options(MiniMaxRequestOptions {
        thinking: Some(MiniMaxThinking::Adaptive),
        ..Default::default()
    });
    for id in [
        "MiniMax-M3",
        "future-minimax-deployment",
        "arbitrary-resource-id",
    ] {
        let model = provider.model(id);
        assert!(
            model
                .capabilities()
                .features
                .contains(Capability::REASONING)
        );
        assert!(model.validate_request(&request).is_ok(), "{id}");
    }
}

#[tokio::test]
async fn new_provider_abort_before_dispatch_performs_no_io() {
    let server = MockServer::start().await;
    let minimax = MiniMax::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("MiniMax-M3");
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    let error = minimax
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Abort);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn minimax_error_mapping_uses_messages_envelope() {
    for (status, body, expected) in [
        (
            404,
            r#"{"type":"error","request_id":"req_1","error":{"type":"not_found_error","message":"Model does not exist."}}"#,
            ModelErrorKind::ModelNotFound,
        ),
        (
            413,
            r#"{"type":"error","error":{"type":"request_too_large","message":"too large"}}"#,
            ModelErrorKind::InvalidRequest,
        ),
        (
            529,
            r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#,
            ModelErrorKind::Overload,
        ),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_string(body))
            .mount(&server)
            .await;
        let model = MiniMax::builder()
            .base_url(server.uri())
            .build()
            .unwrap()
            .model("MiniMax-M3");
        let error = model
            .stream(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind, expected);
        if status == 404 {
            assert_eq!(error.diagnostics.request_id.as_deref(), Some("req_1"));
        }
    }
}

#[tokio::test]
async fn anthropic_aws_request_too_large_is_invalid_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(413).set_body_string(
            r#"{"type":"error","error":{"type":"request_too_large","message":"request exceeds 32 MB"}}"#,
        ))
        .mount(&server)
        .await;
    let model = AnthropicAws::builder("us-west-2", "workspace")
        .bearer_key("key")
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-4-8");
    let error = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::InvalidRequest);
}
mod common;

use common::{AnthropicAws, MiniMax, minimax_capabilities};
