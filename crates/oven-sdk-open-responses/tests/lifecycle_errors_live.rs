mod common;

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, LanguageModel, ModelErrorKind, Request};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn rejects_sequence_gaps_and_missing_done() {
    let server = MockServer::start().await;
    common::mount(&server, common::bad_sequence_stream()).await;
    let model = common::generic_model(&server, "opaque");
    let mut response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(response.stream.next().await.unwrap().is_ok());
    let error = loop {
        match response.stream.next().await {
            Some(Err(error)) => break error,
            Some(Ok(_)) => continue,
            None => panic!("expected lifecycle error"),
        }
    };
    assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
}

#[tokio::test]
async fn http_errors_and_predispatch_abort_are_typed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
            "error":{"type":"too_many_requests","code":"rate_limit","message":"slow"}
        })))
        .mount(&server)
        .await;
    let model = common::generic_model(&server, "opaque");
    let error = match model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("expected rate-limit error"),
    };
    assert_eq!(error.kind, ModelErrorKind::RateLimited);
    assert_eq!(error.diagnostics.http_status, Some(429));
    assert!(error.diagnostics.sanitized_body.is_some());
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    let error = match model.stream(Request::new(Vec::new()), signal).await {
        Err(error) => error,
        Ok(_) => panic!("expected abort error"),
    };
    assert_eq!(error.kind, ModelErrorKind::Abort);
}

#[tokio::test]
async fn streaming_errors_are_classified_without_fabricated_http_status() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"model\":\"opaque\"}}\n\n",
            "event: error\ndata: {\"type\":\"error\",\"sequence_number\":1,\"error\":{\"type\":\"not_found\",\"code\":\"model_not_found\",\"message\":\"missing\"}}\n\n",
            "event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":2,\"response\":{\"id\":\"resp_1\",\"status\":\"failed\",\"model\":\"opaque\",\"error\":{\"code\":\"model_not_found\",\"message\":\"missing\"},\"usage\":null}}\n\n",
            "data: [DONE]\n\n"
        )
        .into(),
    )
    .await;
    let mut response = common::generic_model(&server, "opaque")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut streamed_error = None;
    while let Some(item) = response.stream.next().await {
        if let oven_sdk::StreamPart::Error { error } = item.unwrap() {
            streamed_error = Some(error);
        }
    }
    let error = streamed_error.unwrap();
    assert_eq!(error.kind, ModelErrorKind::ModelNotFound);
    assert_eq!(error.diagnostics.http_status, None);
    assert!(error.diagnostics.sanitized_body.is_some());
}

#[tokio::test]
#[ignore = "requires OPEN_RESPONSES_TOKEN, OPEN_RESPONSES_MODEL, OPEN_RESPONSES_ENDPOINT, and OPEN_RESPONSES_PROFILE"]
async fn live_gate() {
    let token = std::env::var("OPEN_RESPONSES_TOKEN").unwrap();
    let model_id = std::env::var("OPEN_RESPONSES_MODEL").unwrap();
    let endpoint = std::env::var("OPEN_RESPONSES_ENDPOINT").unwrap();
    let profile = std::env::var("OPEN_RESPONSES_PROFILE").unwrap();
    let provider = oven_sdk::ProviderConfig::new(
        oven_sdk::ProviderId::new(profile.clone()),
        oven_sdk::ApiEndpoint::parse(endpoint).unwrap(),
        oven_sdk_open_responses::OpenResponsesAuth::bearer(oven_sdk::SecretString::new(token)),
        oven_sdk::HeaderConfig::empty(),
    )
    .unwrap();
    let declaration = oven_sdk::ModelDeclaration::new(
        oven_sdk::ModelId::new(model_id),
        common::capabilities(false),
    )
    .unwrap();
    let model = oven_sdk_open_responses::OpenResponsesModel::new(oven_sdk::ModelConfig::new(
        provider,
        declaration,
        oven_sdk_open_responses::OpenResponsesSettings {
            transport: oven_sdk_open_responses::OpenResponsesTransport::Generic { profile },
            timeouts: Default::default(),
            strict_json_schema: true,
            strict_tools: true,
            parallel_tool_calls: true,
            store: false,
            include: Vec::new(),
            reasoning_summary: None,
        },
    ))
    .unwrap();
    oven_sdk_conformance::assert_stream_lifecycle(&model, Request::new(Vec::new()))
        .await
        .unwrap();
}
