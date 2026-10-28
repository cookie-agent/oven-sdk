pub mod common;

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AdapterId, BoxFuture, InferenceOptions, LanguageModel, ModelConfig, ModelError,
    ModelErrorKind, ProviderConfig, ProviderId, Request, SecretString, StreamPart,
};
use oven_sdk_openai::{
    OpenAiAuth, OpenAiCompatibleAuth, OpenAiResponsesModel, OpenAiResponsesOptions,
    OpenAiResponsesRequestExt, OpenAiResponsesSettings, OpenAiResponsesWireProfile,
    OpenAiTokenSource,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

/// Hands out `tokens` in order, moving on only when the current one is
/// rejected.
struct Rotating {
    tokens: Vec<&'static str>,
    current: AtomicUsize,
    rejected: Mutex<Vec<String>>,
}

impl Rotating {
    fn new(tokens: Vec<&'static str>) -> Arc<Self> {
        Arc::new(Self {
            tokens,
            current: AtomicUsize::new(0),
            rejected: Mutex::new(Vec::new()),
        })
    }

    fn rejected(&self) -> Vec<String> {
        self.rejected.lock().unwrap().clone()
    }
}

impl OpenAiTokenSource for Rotating {
    fn token(&self) -> BoxFuture<'_, Result<SecretString, ModelError>> {
        let index = self.current.load(Ordering::SeqCst);
        let token = self.tokens[index.min(self.tokens.len() - 1)];
        Box::pin(async move { Ok(SecretString::new(token)) })
    }

    fn rejected(&self, token: &SecretString) {
        self.rejected
            .lock()
            .unwrap()
            .push(token.expose_secret().to_owned());
        self.current.fetch_add(1, Ordering::SeqCst);
    }
}

fn config(
    server: &MockServer,
    auth: OpenAiCompatibleAuth,
    discriminator: Option<&str>,
) -> ModelConfig<OpenAiCompatibleAuth, OpenAiResponsesSettings> {
    let base = common::official_responses_config(server, "gpt-6-luna");
    let mut settings = OpenAiResponsesSettings::new();
    settings.routing_discriminator = discriminator.map(str::to_owned);
    ModelConfig::new(
        ProviderConfig::new(
            ProviderId::new("chatgpt"),
            base.provider.api,
            auth,
            base.provider.headers,
        )
        .unwrap(),
        base.model,
        settings,
    )
}

fn model(server: &MockServer, source: Arc<Rotating>) -> OpenAiResponsesModel {
    OpenAiResponsesModel::new_compatible(
        config(
            server,
            OpenAiCompatibleAuth::token_source(source),
            Some("account:one"),
        ),
        AdapterId::new("test.chatgpt.responses"),
    )
    .unwrap()
}

async fn mount_for(server: &MockServer, bearer: &str, status: u16, body: &str) {
    let template = if status == 200 {
        ResponseTemplate::new(200).set_body_raw(body.to_owned(), "text/event-stream")
    } else {
        ResponseTemplate::new(status).set_body_raw(body.to_owned(), "application/json")
    };
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", format!("Bearer {bearer}").as_str()))
        .respond_with(template)
        .mount(server)
        .await;
}

#[tokio::test]
async fn a_refused_token_is_reported_and_retried_once_with_a_fresh_one() {
    let server = MockServer::start().await;
    mount_for(
        &server,
        "stale",
        401,
        r#"{"error":{"message":"Invalid token","type":"invalid_request_error","code":"invalid_api_key"}}"#,
    )
    .await;
    mount_for(&server, "fresh", 200, &common::responses_document("ok")).await;
    let source = Rotating::new(vec!["stale", "fresh"]);
    let turn = model(&server, Arc::clone(&source))
        .complete(Request::new(vec![]), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    assert!(matches!(
        turn.message.content.as_slice(),
        [oven_sdk::AssistantPart::Text(text)] if text.text == "ok"
    ));
    assert_eq!(source.rejected(), vec!["stale".to_owned()]);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn a_second_refusal_is_the_error_without_another_retry() {
    let server = MockServer::start().await;
    for bearer in ["one", "two"] {
        mount_for(
            &server,
            bearer,
            401,
            r#"{"error":{"message":"Invalid token","code":"invalid_api_key"}}"#,
        )
        .await;
    }
    let source = Rotating::new(vec!["one", "two", "three"]);
    let error = model(&server, Arc::clone(&source))
        .complete(Request::new(vec![]), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Auth);
    assert_eq!(source.rejected(), vec!["one".to_owned()]);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn a_401_for_a_refused_operation_is_not_a_token_problem() {
    let server = MockServer::start().await;
    mount_for(
        &server,
        "token",
        401,
        r#"{"error":{"message":"This ChatPass credential is not authorized for the requested operation.","type":"rejected_by_hardened_oauth_boundary","code":"hardened_oauth_rule_missing"}}"#,
    )
    .await;
    let source = Rotating::new(vec!["token"]);
    let error = model(&server, Arc::clone(&source))
        .complete(Request::new(vec![]), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Unsupported);
    assert!(source.rejected().is_empty());
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn replay_is_bound_to_the_account_not_to_the_token() {
    let server = MockServer::start().await;
    for bearer in ["one", "two"] {
        mount_for(&server, bearer, 200, &common::responses_document("ok")).await;
    }
    let source = Rotating::new(vec!["one", "two"]);
    let model = model(&server, Arc::clone(&source));
    let first = model
        .complete(Request::new(vec![]), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    OpenAiTokenSource::rejected(source.as_ref(), &SecretString::new("one"));
    let second = model
        .complete(Request::new(vec![]), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    let scope = |turn: &oven_sdk::CompletedTurn| {
        turn.finish.native_replay.as_ref().unwrap().scope().clone()
    };
    assert_eq!(scope(&first), scope(&second));
    assert!(!format!("{:?}", scope(&first)).contains("one"));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].headers["authorization"], "Bearer one");
    assert_eq!(requests[1].headers["authorization"], "Bearer two");
}

#[test]
fn a_token_source_needs_an_account_and_no_other_auth() {
    let server_uri = "http://127.0.0.1:9";
    let base = common::official_responses_config_at(server_uri, "gpt-6-luna", "secret");
    let build = |auth: OpenAiCompatibleAuth, discriminator: Option<&str>| {
        let mut settings = OpenAiResponsesSettings::new();
        settings.routing_discriminator = discriminator.map(str::to_owned);
        OpenAiResponsesModel::new_compatible(
            ModelConfig::new(
                ProviderConfig::new(
                    ProviderId::new("chatgpt"),
                    base.provider.api.clone(),
                    auth,
                    base.provider.headers.clone(),
                )
                .unwrap(),
                base.model.clone(),
                settings,
            ),
            AdapterId::new("test.chatgpt.responses"),
        )
        .is_ok()
    };
    let source = || OpenAiCompatibleAuth::token_source(Rotating::new(vec!["token"]));
    assert!(!build(source(), None));
    assert!(!build(source(), Some(" ")));
    let mut combined = source();
    combined.bearer = Some(SecretString::new("static"));
    assert!(!build(combined, Some("account:one")));
    assert!(build(source(), Some("account:one")));
}

#[tokio::test]
async fn the_token_sharing_profile_drops_refused_fields() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let base = common::official_responses_config(&server, "gpt-6-luna");
    let mut settings = OpenAiResponsesSettings::new();
    settings.wire_profile = OpenAiResponsesWireProfile::ChatGptTokenSharing;
    let model = OpenAiResponsesModel::new(ModelConfig::new(
        ProviderConfig::new(
            ProviderId::new("openai"),
            base.provider.api,
            OpenAiAuth::new(SecretString::new("secret")),
            base.provider.headers,
        )
        .unwrap(),
        base.model,
        settings,
    ))
    .unwrap();
    let mut inference = InferenceOptions::new();
    inference.max_output_tokens = Some(4096);
    inference.temperature = Some(0.5);
    inference.top_p = Some(0.9);
    let request = Request::new(vec![])
        .with_inference(inference)
        .with_openai_responses_options(OpenAiResponsesOptions {
            service_tier: Some("priority".into()),
            user: Some("someone".into()),
            truncation: Some("auto".into()),
            verbosity: Some("low".into()),
            prompt_cache_key: Some("session".into()),
            ..OpenAiResponsesOptions::default()
        });
    let mut response = model.stream(request, AbortSignal::default()).await.unwrap();
    let warnings = match response.stream.next().await {
        Some(Ok(StreamPart::StreamStart { warnings })) => warnings,
        other => panic!("expected stream start, got {other:?}"),
    };
    while response.stream.next().await.is_some() {}
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    for field in [
        "max_output_tokens",
        "temperature",
        "top_p",
        "service_tier",
        "user",
        "truncation",
    ] {
        assert!(body.get(field).is_none(), "{field} was sent: {body}");
    }
    assert_eq!(body["prompt_cache_key"], "session");
    assert_eq!(body["text"]["verbosity"], "low");
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    for field in ["temperature", "top_p", "service_tier", "user", "truncation"] {
        assert!(
            warnings.iter().any(|warning| warning.contains(field)),
            "no warning for {field}: {warnings:?}"
        );
    }
    assert!(
        !warnings
            .iter()
            .any(|warning| warning.contains("max_output_tokens"))
    );
}
