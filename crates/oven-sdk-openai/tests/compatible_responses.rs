pub mod common;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use oven_sdk::{
    AbortSignal, AdapterId, HistoryTurn, LanguageModel, ModelConfig, ProviderConfig, ProviderId,
    ReplayDisposition, Request, SecretString,
};
use oven_sdk_openai::{OpenAiCompatibleAuth, OpenAiResponsesModel};
use wiremock::MockServer;

struct HeaderAuth(Arc<AtomicUsize>);

impl oven_sdk::HeaderProvider for HeaderAuth {
    fn headers(
        &self,
        context: &oven_sdk::HeaderContext,
    ) -> Result<oven_sdk::HeaderOverrides, oven_sdk::ModelError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-api-key", context.session_id.parse().unwrap());
        Ok(oven_sdk::HeaderOverrides::new(headers))
    }
}

#[tokio::test]
async fn header_provider_auth_resolves_once_without_gating_standard_replay() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let config = common::official_responses_config(&server, "model");
    let mut config = ModelConfig::new(
        ProviderConfig::new(
            ProviderId::new("gateway"),
            config.provider.api,
            OpenAiCompatibleAuth::headers(Arc::new(HeaderAuth(Arc::clone(&calls)))),
            config.provider.headers,
        )
        .unwrap(),
        config.model,
        config.settings,
    );
    assert!(
        OpenAiResponsesModel::new_compatible(
            config.clone(),
            AdapterId::new("test.header.responses")
        )
        .is_ok()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    config.settings.routing_discriminator = Some("header:x-api-key".into());
    let model =
        OpenAiResponsesModel::new_compatible(config, AdapterId::new("test.header.responses"))
            .unwrap();
    let request = |history, secret| {
        Request::new(history).with_header_context(oven_sdk::HeaderContext::new(secret))
    };
    let first = model
        .complete(request(vec![], "secret-one"), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        !format!("{:?}", first.finish.native_replay.as_ref().unwrap().scope())
            .contains("secret-one")
    );
    for secret in ["secret-one", "secret-two"] {
        let response = model
            .stream(
                request(vec![HistoryTurn::assistant(first.clone())], secret),
                AbortSignal::default(),
            )
            .await
            .unwrap();
        assert!(matches!(
            response.request.replay.decisions[0].disposition,
            ReplayDisposition::Replayed
        ));
        for secret in ["secret-one", "secret-two"] {
            assert!(!format!("{:?}", response.request.replay).contains(secret));
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    let requests = server.received_requests().await.unwrap();
    for (request, secret) in requests
        .iter()
        .zip(["secret-one", "secret-one", "secret-two"])
    {
        assert_eq!(request.headers["x-api-key"], secret);
        assert!(!request.headers.contains_key("authorization"));
        assert_eq!(request.url.path(), "/responses");
    }
}

fn model(server: &MockServer, provider: &str, auth: OpenAiCompatibleAuth) -> OpenAiResponsesModel {
    let config = common::official_responses_config(server, "model");
    OpenAiResponsesModel::new_compatible(
        ModelConfig::new(
            ProviderConfig::new(
                ProviderId::new(provider),
                config.provider.api,
                auth,
                config.provider.headers,
            )
            .unwrap(),
            config.model,
            config.settings,
        ),
        AdapterId::new("test.compatible.responses"),
    )
    .unwrap()
}

#[tokio::test]
async fn compatible_auth_routing_and_native_replay_use_responses_codecs() {
    for bearer in [false, true] {
        let server = MockServer::start().await;
        common::mount(&server, "/responses", common::responses_document("ok")).await;
        let auth = if bearer {
            OpenAiCompatibleAuth::bearer(SecretString::new("test-token"))
        } else {
            OpenAiCompatibleAuth::none()
        };
        let model = model(&server, "gateway", auth);
        let first = model
            .complete(Request::new(vec![]), AbortSignal::default())
            .await
            .unwrap();
        let replay = model
            .stream(
                Request::new(vec![HistoryTurn::assistant(first.turn)]),
                AbortSignal::default(),
            )
            .await
            .unwrap();
        assert!(matches!(
            replay.request.replay.decisions[0].disposition,
            ReplayDisposition::Replayed
        ));
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests[0].headers.contains_key("authorization"), bearer);
        let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
        assert_eq!(body["input"][0]["content"][0]["text"], "ok");
        assert_eq!(body["store"], false);
    }
}

#[tokio::test]
async fn foreign_scope_replays_text_but_missing_encrypted_state_fails_closed() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let first = model(&server, "one", OpenAiCompatibleAuth::none())
        .complete(Request::new(vec![]), AbortSignal::default())
        .await
        .unwrap();
    let other = model(&server, "two", OpenAiCompatibleAuth::none());
    let replay = other
        .stream(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        replay.request.replay.decisions[0].disposition,
        ReplayDisposition::Replayed
    ));
    let mut turn = other
        .complete(Request::new(vec![]), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    turn.finish.native_replay = None;
    turn.finish.finish_reason = oven_sdk::FinishReason::ToolCalls;
    turn.message.content.push(oven_sdk::AssistantPart::ToolCall(
        oven_sdk::ToolCallPart::new("required-call", "inspect", serde_json::json!({})),
    ));
    turn.message
        .content
        .push(oven_sdk::AssistantPart::Custom(oven_sdk::CustomPart::new(
            "openai.responses.reasoning_continuation",
            serde_json::json!({"item_id":"rs","encrypted_sha256":"missing"}),
        )));
    let count = server.received_requests().await.unwrap().len();
    let error = other
        .stream(
            Request::new(vec![
                HistoryTurn::assistant(turn),
                HistoryTurn::tool(oven_sdk::ToolMessage::new(vec![
                    oven_sdk::ToolResultPart::new(
                        "required-call",
                        oven_sdk::ToolContent::Text("result".into()),
                    ),
                ])),
            ]),
            AbortSignal::default(),
        )
        .await
        .expect_err("required continuation must fail");
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::Replay);
    assert_eq!(server.received_requests().await.unwrap().len(), count);
}

#[tokio::test]
async fn encrypted_reasoning_requires_equal_known_wire_id_and_preserves_standard_siblings() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"portable\"}]},{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[],\"encrypted_content\":\"opaque\"}]}}\n\n".into()).await;
    let first = model(&server, "source-provider", OpenAiCompatibleAuth::none())
        .complete(Request::new(vec![]), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    for (target_id, known_source, enabled, encrypted) in [
        ("model", true, true, true),
        ("different", true, true, false),
        ("model", false, true, false),
        ("model", true, false, false),
    ] {
        let mut config = common::official_responses_config(&server, target_id);
        if !enabled {
            config
                .model
                .capabilities
                .features
                .remove(oven_sdk::Capability::REASONING);
            config.model.capabilities.replay = oven_sdk::ReplayDeclaration {
                policy: oven_sdk::ReplayPolicy::Never,
                capability: oven_sdk::ReplayCapability::Unsupported,
                reasoning: false,
            };
        }
        let target = OpenAiResponsesModel::new_compatible(
            ModelConfig::new(
                ProviderConfig::new(
                    ProviderId::new("different-provider"),
                    config.provider.api,
                    OpenAiCompatibleAuth::bearer(SecretString::new("different-key")),
                    config.provider.headers,
                )
                .unwrap(),
                config.model,
                config.settings,
            ),
            AdapterId::new("different-adapter.responses"),
        )
        .unwrap();
        let mut source = first.clone();
        if !known_source {
            let artifact = source.finish.native_replay.take().unwrap();
            source.finish.native_replay = Some(
                oven_sdk::NativeReplayArtifact::new(
                    artifact.adapter_id().clone(),
                    artifact.scope().clone(),
                    artifact.payload().clone(),
                )
                .unwrap(),
            );
        }
        let result = target
            .complete(
                Request::new(vec![HistoryTurn::assistant(source)]),
                AbortSignal::default(),
            )
            .await
            .unwrap();
        assert_eq!(
            result.request.replay.decisions[0].disposition,
            if enabled {
                ReplayDisposition::Replayed
            } else {
                ReplayDisposition::ReconstructedNormalized
            }
        );
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&requests.last().unwrap().body).unwrap();
        assert_eq!(body["model"], target_id);
        assert_eq!(body["input"][0]["content"][0]["text"], "portable");
        assert_eq!(body.to_string().contains("opaque"), encrypted);
        if enabled {
            assert_eq!(body["input"][0]["id"], "msg");
        }
    }
}

#[tokio::test]
async fn azure_responses_envelope_replays_on_compatible_responses_after_integrity_validation() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let (items, fingerprint) = oven_sdk::replay::azure_responses_capture(&[
        serde_json::json!({"type":"message","id":"azure-msg","role":"assistant","content":[{"type":"output_text","text":"portable"}]}),
        serde_json::json!({"type":"reasoning","id":"rs","summary":[],"encrypted_content":"opaque"}),
    ]).unwrap();
    let mut finish = oven_sdk::Finish::new(Default::default(), oven_sdk::FinishReason::Stop);
    finish.native_replay = Some(oven_sdk::NativeReplayArtifact::capture(
        AdapterId::new("oven.azure.openai.responses"),
        oven_sdk::NativeContextScope::new(ProviderId::new("azure.openai"), oven_sdk::ModelId::new("model"), oven_sdk::ResourceId::new("different-deployment-metadata").unwrap()).unwrap(),
        serde_json::json!({"format":"oven.azure.openai.responses.output.v4","binding":{"version":"source-version","sha256":"source-provenance"},"items":items,"fingerprint":fingerprint}),
    ).unwrap());
    let turn = oven_sdk::CompletedTurn::new(
        oven_sdk::AssistantMessage::new(vec![
            oven_sdk::AssistantPart::Text(oven_sdk::TextPart::new("portable")),
            oven_sdk::AssistantPart::Custom(oven_sdk::CustomPart::new(
                oven_sdk::replay::AZURE_RESPONSES_FINGERPRINT_KIND,
                serde_json::json!(fingerprint),
            )),
        ]),
        finish,
    );
    let result = model(&server, "other-provider", OpenAiCompatibleAuth::none())
        .complete(
            Request::new(vec![HistoryTurn::assistant(turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.request.replay.decisions[0].disposition,
        ReplayDisposition::Replayed
    );
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["input"][0]["id"], "azure-msg");
    assert_eq!(body["input"][1]["encrypted_content"], "opaque");
    assert!(!body.to_string().contains("source-provenance"));
}
