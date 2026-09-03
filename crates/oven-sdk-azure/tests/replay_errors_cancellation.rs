mod common;

use std::{sync::Arc, time::Duration};

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, ErrorStage, HeaderConfig, HeaderOverrides, HeaderProvider, HistoryTurn,
    LanguageModel, ModelError, ModelErrorKind, ReplayDisposition, Request, SecretString,
};
use oven_sdk_azure::{AzureApiRoute, AzureOpenAiAuth, AzureOpenAiChatModel, AzureOpenAiTimeouts};
use reqwest::header::{HeaderMap, HeaderValue};
use tokio::{io::AsyncWriteExt, net::TcpListener, time::timeout};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

struct DynamicHeaders(HeaderMap);

impl HeaderProvider for DynamicHeaders {
    fn headers(&self, _context: &oven_sdk::HeaderContext) -> Result<HeaderOverrides, ModelError> {
        Ok(HeaderOverrides::new(self.0.clone()))
    }
}

#[tokio::test]
async fn replay_is_bound_to_api_model_id_and_headers() {
    let first_server = MockServer::start().await;
    common::mount(
        &first_server,
        "/openai/v1/chat/completions",
        common::chat_document("ok"),
    )
    .await;
    let first_model = common::provider(&first_server, AzureApiRoute::V1)
        .with_header("x-azure-test", "one")
        .chat("deployment-a", common::gpt4o())
        .unwrap();
    let first = first_model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(
        first
            .turn
            .finish
            .native_replay
            .as_ref()
            .unwrap()
            .adapter_id()
            .as_str(),
        "oven.azure.openai.chat"
    );

    let header_model = common::provider(&first_server, AzureApiRoute::V1)
        .with_header("x-azure-test", "two")
        .chat("deployment-a", common::gpt4o())
        .unwrap();
    let header_result = header_model
        .complete(
            Request::new(vec![HistoryTurn::assistant(first.turn.clone())]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        header_result.request.replay.decisions.first(),
        Some(oven_sdk::ReplayDecision {
            disposition: ReplayDisposition::DiscardedForeignScope { .. },
            ..
        })
    ));

    let second_server = MockServer::start().await;
    common::mount(
        &second_server,
        "/openai/v1/chat/completions",
        common::chat_document("ok"),
    )
    .await;
    let second_model = common::provider(&second_server, AzureApiRoute::V1)
        .with_header("x-azure-test", "one")
        .chat("deployment-a", common::gpt4o())
        .unwrap();
    let second = second_model
        .complete(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        second.request.replay.decisions.as_slice(),
        [
            oven_sdk::ReplayDecision {
                disposition: ReplayDisposition::DiscardedForeignScope { .. },
                ..
            },
            oven_sdk::ReplayDecision {
                disposition: ReplayDisposition::ReconstructedNormalized,
                ..
            }
        ]
    ));
}

#[tokio::test]
async fn replay_serialization_redacts_endpoint_headers_auth_and_tenant_data() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/chat/completions",
        common::chat_document("ok"),
    )
    .await;

    let mut dynamic = HeaderMap::new();
    dynamic.append(
        "x-private-tenant-id",
        HeaderValue::from_static("tenant-secret-value"),
    );
    dynamic.append("x-repeated-secret", HeaderValue::from_static("first"));
    dynamic.append("x-repeated-secret", HeaderValue::from_static("second"));
    let mut config = common::chat_config(
        server.uri(),
        AzureApiRoute::V1,
        "deployment",
        common::gpt4o(),
        AzureOpenAiAuth::ApiKey(SecretString::new("api-key-secret-value")),
    );
    config.provider.headers = HeaderConfig {
        static_headers: HeaderOverrides::new({
            let mut headers = HeaderMap::new();
            headers.insert(
                "x-static-secret-name",
                HeaderValue::from_static("static-secret-value"),
            );
            headers
        }),
        dynamic_headers: Some(Arc::new(DynamicHeaders(dynamic))),
    };
    let first = AzureOpenAiChatModel::new(config)
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let artifact = first.turn.finish.native_replay.as_ref().unwrap();
    let serialized = serde_json::to_string(artifact).unwrap();
    let debug = format!("{artifact:?}");
    for private in [
        server.uri().as_str(),
        "x-private-tenant-id",
        "tenant-secret-value",
        "x-repeated-secret",
        "first",
        "second",
        "x-static-secret-name",
        "static-secret-value",
        "api-key",
        "api-key-secret-value",
    ] {
        assert!(!serialized.contains(private), "serialized {private}");
        assert!(!debug.contains(private), "debug {private}");
    }
    assert_eq!(
        artifact
            .payload()
            .pointer("/binding/version")
            .and_then(serde_json::Value::as_str),
        Some("azure.openai.native_context_scope.v1")
    );
    let digest = artifact
        .payload()
        .pointer("/binding/sha256")
        .and_then(serde_json::Value::as_str)
        .unwrap();
    assert_eq!(digest.len(), 43);

    let mut changed = common::chat_config(
        server.uri(),
        AzureApiRoute::V1,
        "deployment",
        common::gpt4o(),
        AzureOpenAiAuth::ApiKey(SecretString::new("different-api-key")),
    );
    changed.provider.headers = HeaderConfig::empty();
    let result = AzureOpenAiChatModel::new(changed)
        .unwrap()
        .complete(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        result.request.replay.decisions.first(),
        Some(oven_sdk::ReplayDecision {
            disposition: ReplayDisposition::DiscardedForeignScope { .. },
            ..
        })
    ));
}

#[tokio::test]
async fn equivalent_endpoint_and_header_configurations_share_replay_scope() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/chat/completions",
        common::chat_document("ok"),
    )
    .await;

    let configure = |api: String, reversed: bool| {
        let mut config = common::chat_config(
            api,
            AzureApiRoute::V1,
            "deployment",
            common::gpt4o(),
            AzureOpenAiAuth::ApiKey(SecretString::new("same-secret")),
        );
        let mut headers = HeaderMap::new();
        if reversed {
            headers.insert("x-second", HeaderValue::from_static("two"));
            headers.insert("x-first", HeaderValue::from_static("one"));
        } else {
            headers.insert("x-first", HeaderValue::from_static("one"));
            headers.insert("x-second", HeaderValue::from_static("two"));
        }
        config.provider.headers.static_headers = HeaderOverrides::new(headers);
        config
    };

    let first = AzureOpenAiChatModel::new(configure(server.uri(), false))
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let first_scope = first
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .scope()
        .clone();
    let second = AzureOpenAiChatModel::new(configure(format!("{}/", server.uri()), true))
        .unwrap()
        .complete(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        second.turn.finish.native_replay.as_ref().unwrap().scope(),
        &first_scope
    );
    assert!(matches!(
        second.request.replay.decisions.first(),
        Some(oven_sdk::ReplayDecision {
            disposition: ReplayDisposition::Replayed,
            ..
        })
    ));
}

#[tokio::test]
async fn azure_error_envelopes_preserve_request_id_retry_quota_filter_and_body_stage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("apim-request-id", "req_quota")
                .insert_header("x-ms-retry-after-ms", "25")
                .set_body_json(serde_json::json!({
                    "error": {"code":"insufficient_quota","message":"quota exceeded"}
                })),
        )
        .mount(&server)
        .await;
    let error = common::provider(&server, AzureApiRoute::V1)
        .chat("deployment", common::gpt4o())
        .unwrap()
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Quota);
    assert_eq!(error.diagnostics.request_id.as_deref(), Some("req_quota"));
    assert_eq!(
        error.diagnostics.retry_after,
        Some(Duration::from_millis(25))
    );
    assert_eq!(error.diagnostics.stage, ErrorStage::ResponseBody);
    assert!(error.diagnostics.bytes_received > 0);

    let filter_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": {"code":"content_filter","inner_error":{"code":"ResponsibleAIPolicyViolation"}}
        })))
        .mount(&filter_server)
        .await;
    let error = common::provider(&filter_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap()
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::ContentFilter);
}

#[tokio::test]
async fn cancellation_covers_credentials_headers_and_midstream_locally() {
    let credential_server = MockServer::start().await;
    let mut config = common::chat_config(
        credential_server.uri(),
        AzureApiRoute::V1,
        "deployment",
        common::gpt4o(),
        AzureOpenAiAuth::Entra(Arc::new(|| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok("token".into())
            })
        })),
    );
    config.settings.timeouts = AzureOpenAiTimeouts {
        credentials: Duration::from_secs(60),
        ..Default::default()
    };
    let provider = AzureOpenAiChatModel::new(config).unwrap();
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    let error = provider
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Abort);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
            .await
            .unwrap();
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"start\"},\"finish_reason\":null}]}\n\n";
        socket
            .write_all(format!("{:X}\r\n{body}\r\n", body.len()).as_bytes())
            .await
            .unwrap();
        socket.flush().await.unwrap();
        tokio::time::sleep(Duration::from_secs(2)).await;
    });
    let (signal, registration) = AbortSignal::new();
    let config = common::chat_config(
        api,
        AzureApiRoute::V1,
        "deployment",
        common::gpt4o(),
        AzureOpenAiAuth::ApiKey(SecretString::new("secret")),
    );
    let mut response = AzureOpenAiChatModel::new(config)
        .unwrap()
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap();
    assert!(response.stream.next().await.is_some());
    registration.abort();
    let mut error = None;
    while error.is_none() {
        let item = timeout(Duration::from_millis(300), response.stream.next())
            .await
            .expect("abort should wake stream")
            .expect("stream item");
        if let Err(found) = item {
            error = Some(found);
        }
    }
    let error = error.unwrap();
    assert_eq!(error.kind, ModelErrorKind::Abort);
    assert_eq!(error.diagnostics.stage, ErrorStage::StreamRead);
    assert!(response.stream.next().await.is_none());
}
