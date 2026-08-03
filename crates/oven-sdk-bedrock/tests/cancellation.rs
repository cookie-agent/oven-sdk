mod support;

use std::{sync::Arc, time::Duration};

use oven_sdk::{AbortSignal, LanguageModel, ModelErrorKind, Request};
use oven_sdk_bedrock::{AwsCredentials, BedrockAuth};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

#[tokio::test]
async fn cancellation_before_headers_drops_the_request_future() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(5))
                .set_body_bytes(Vec::new()),
        )
        .mount(&server)
        .await;
    let model = support::model(&server.uri(), "unknown", support::FixtureKind::Text);
    let (signal, registration) = AbortSignal::new();
    let call = model.stream(Request::new(Vec::new()), signal);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(25)).await;
        registration.abort();
    });
    let error = call.await.unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Abort);
}

#[tokio::test]
async fn cancellation_interrupts_credential_resolution() {
    let provider = Arc::new(|| async {
        std::future::pending::<Result<AwsCredentials, oven_sdk::ModelError>>().await
    });
    let model = support::model_with_auth(
        "https://bedrock-runtime.us-east-1.amazonaws.com",
        "unknown",
        support::FixtureKind::Text,
        BedrockAuth::Provider(provider),
    );
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    let error = model
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Abort);
}
