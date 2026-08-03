mod common;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use oven_sdk::{AbortSignal, HeaderOverrides, LanguageModel, Request, SecretString};
use oven_sdk_azure::{AzureApiRoute, AzureApiVersion, AzureOpenAiAuth, AzureOpenAiChatModel};
use reqwest::header::{HeaderMap, HeaderValue};
use wiremock::MockServer;

#[tokio::test]
async fn v1_preview_and_dated_routes_are_exact() {
    let v1 = MockServer::start().await;
    common::mount(
        &v1,
        "/openai/v1/chat/completions",
        common::chat_document("ok"),
    )
    .await;
    common::provider(&v1, AzureApiRoute::V1)
        .chat("deployment one", common::gpt4o())
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let request = &v1.received_requests().await.unwrap()[0];
    assert!(request.url.query().is_none());
    assert_eq!(request.headers["api-key"], "secret");
    assert!(request.headers.get("authorization").is_none());

    let preview = MockServer::start().await;
    common::mount(
        &preview,
        "/openai/v1/responses",
        common::responses_document("ok"),
    )
    .await;
    common::provider(&preview, AzureApiRoute::V1Preview)
        .responses("deployment", common::gpt5())
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(
        preview.received_requests().await.unwrap()[0]
            .url
            .query_pairs()
            .collect::<Vec<_>>(),
        [("api-version".into(), "preview".into())]
    );

    let dated = MockServer::start().await;
    common::mount(
        &dated,
        "/openai/deployments/deployment%20one/chat/completions",
        common::chat_document("ok"),
    )
    .await;
    common::provider(
        &dated,
        AzureApiRoute::Dated(AzureApiVersion::new("2025-04-01-preview").unwrap()),
    )
    .chat("deployment one", common::gpt4o())
    .unwrap()
    .complete(Request::new(Vec::new()), AbortSignal::default())
    .await
    .unwrap();
    assert_eq!(
        dated.received_requests().await.unwrap()[0]
            .url
            .query_pairs()
            .collect::<Vec<_>>(),
        [("api-version".into(), "2025-04-01-preview".into())]
    );
}

#[tokio::test]
async fn entra_provider_is_async_per_request() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/chat/completions",
        common::chat_document("ok"),
    )
    .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let provider_calls = Arc::clone(&calls);
    let config = common::chat_config(
        server.uri(),
        AzureApiRoute::V1,
        "deployment",
        common::gpt4o(),
        AzureOpenAiAuth::Entra(Arc::new(move || {
            let number = provider_calls.fetch_add(1, Ordering::SeqCst) + 1;
            Box::pin(async move { Ok(format!("token-{number}")) })
        })),
    );
    let model = AzureOpenAiChatModel::new(config).unwrap();
    for _ in 0..2 {
        model
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap();
    }
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].headers["authorization"], "Bearer token-1");
    assert_eq!(requests[1].headers["authorization"], "Bearer token-2");
    assert!(requests[0].headers.get("api-key").is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn api_versions_endpoints_and_protected_headers_validate() {
    assert!(AzureApiVersion::new("2025-04-01-preview").is_ok());
    assert!(AzureApiVersion::new("preview").is_err());
    let invalid_endpoint = common::chat_config(
        "https://example.test/path",
        AzureApiRoute::V1,
        "deployment",
        common::conservative(),
        AzureOpenAiAuth::ApiKey(SecretString::new("x")),
    );
    assert!(AzureOpenAiChatModel::new(invalid_endpoint).is_err());

    let mut headers = HeaderMap::new();
    headers.insert("authorization", HeaderValue::from_static("secret"));
    let mut protected = common::chat_config(
        "https://example.test",
        AzureApiRoute::V1,
        "deployment",
        common::conservative(),
        AzureOpenAiAuth::ApiKey(SecretString::new("x")),
    );
    protected.provider.headers.static_headers = HeaderOverrides::new(headers);
    assert!(AzureOpenAiChatModel::new(protected).is_err());
}
