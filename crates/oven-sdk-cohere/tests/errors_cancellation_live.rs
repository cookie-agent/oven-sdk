mod common;

use oven_sdk::{AbortSignal, LanguageModel, ModelErrorKind, Request};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn errors_and_predispatch_cancellation_are_typed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/chat"))
        .respond_with(
            ResponseTemplate::new(429).set_body_json(serde_json::json!({"message":"slow"})),
        )
        .mount(&server)
        .await;
    let model = common::model(&server, "opaque");
    let error = match model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("expected rate-limit error"),
    };
    assert_eq!(error.kind, ModelErrorKind::RateLimited);

    let (signal, registration) = AbortSignal::new();
    registration.abort();
    let error = match model.stream(Request::new(Vec::new()), signal).await {
        Err(error) => error,
        Ok(_) => panic!("expected abort error"),
    };
    assert_eq!(error.kind, ModelErrorKind::Abort);
}

#[tokio::test]
#[ignore = "requires COHERE_API_KEY, COHERE_MODEL, and COHERE_ENDPOINT"]
async fn live_gate() {
    let token = std::env::var("COHERE_API_KEY").unwrap();
    let model_id = std::env::var("COHERE_MODEL").unwrap();
    let endpoint = std::env::var("COHERE_ENDPOINT").unwrap();
    let provider = oven_sdk::ProviderConfig::new(
        oven_sdk::ProviderId::new("cohere"),
        oven_sdk::ApiEndpoint::parse(endpoint).unwrap(),
        oven_sdk_cohere::CohereAuth::bearer(oven_sdk::SecretString::new(token)),
        oven_sdk::HeaderConfig::empty(),
    )
    .unwrap();
    let declaration =
        oven_sdk::ModelDeclaration::new(oven_sdk::ModelId::new(model_id), common::capabilities())
            .unwrap();
    let model = oven_sdk_cohere::CohereModel::new(oven_sdk::ModelConfig::new(
        provider,
        declaration,
        oven_sdk_cohere::CohereSettings::default(),
    ))
    .unwrap();
    oven_sdk_conformance::assert_stream_lifecycle(&model, Request::new(Vec::new()))
        .await
        .unwrap();
}
