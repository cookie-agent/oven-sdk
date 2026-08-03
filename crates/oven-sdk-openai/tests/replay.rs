pub mod common;

use std::sync::Arc;

use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, CompletedTurn, Finish, FinishReason,
    HeaderOverrides, HeaderProvider, HistoryTurn, LanguageModel, ModelError, ModelErrorKind,
    ModelId, NativeContextScope, NativeReplayArtifact, ProviderId, ReasoningPart,
    ReplayDisposition, ReplayPolicy, Request, ResourceId, TextPart,
};
use oven_sdk_openai::{OpenAiCompatibleChatModel, ReasoningField};
use wiremock::MockServer;

fn completed(artifact: NativeReplayArtifact) -> CompletedTurn {
    let mut finish = Finish::new(Default::default(), FinishReason::Stop);
    finish.native_replay = Some(artifact);
    CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::Text(TextPart::new("ok"))]),
        finish,
    )
}

fn with_payload(turn: &CompletedTurn, payload: serde_json::Value) -> CompletedTurn {
    let original = turn.finish.native_replay.as_ref().unwrap();
    let artifact = NativeReplayArtifact::new(
        original.adapter_id().clone(),
        original.scope().clone(),
        payload,
    )
    .unwrap();
    let mut forged = turn.clone();
    forged.finish.native_replay = Some(artifact);
    forged
}

async fn assert_invalid_replay(model: &impl LanguageModel, turn: CompletedTurn) {
    let response = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        response.request.replay.decisions.as_slice(),
        [
            oven_sdk::ReplayDecision {
                disposition: ReplayDisposition::DiscardedInvalidPayload { .. },
                ..
            },
            oven_sdk::ReplayDecision {
                disposition: ReplayDisposition::ReconstructedNormalized,
                ..
            }
        ]
    ));
}

struct RouteHeaders;

impl HeaderProvider for RouteHeaders {
    fn headers(&self) -> Result<HeaderOverrides, ModelError> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-route", "dynamic".parse().unwrap());
        Ok(HeaderOverrides::new(headers))
    }
}

#[tokio::test]
async fn chat_current_format_round_trips_and_foreign_reconstructs() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let scope = first
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .scope()
        .clone();
    let replayed = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        replayed.request.replay.decisions.as_slice(),
        [oven_sdk::ReplayDecision {
            disposition: ReplayDisposition::Replayed,
            ..
        }]
    ));
    let foreign =
        NativeReplayArtifact::new(AdapterId::new("foreign"), scope, serde_json::json!({})).unwrap();
    let reconstructed = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(completed(foreign))]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        reconstructed.request.replay.decisions.as_slice(),
        [
            _,
            oven_sdk::ReplayDecision {
                disposition: ReplayDisposition::ReconstructedNormalized,
                ..
            }
        ]
    ));
}

#[tokio::test]
async fn invalid_same_adapter_payload_discards_and_reconstructs() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let valid = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let garbage = NativeReplayArtifact::new(
        AdapterId::new("oven.openai.chat"),
        valid.turn.finish.native_replay.unwrap().scope().clone(),
        serde_json::Value::String("garbage".into()),
    )
    .unwrap();
    let response = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(completed(garbage))]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        response.request.replay.decisions.as_slice(),
        [
            oven_sdk::ReplayDecision {
                disposition: ReplayDisposition::DiscardedInvalidPayload { .. },
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
async fn same_adapter_foreign_scope_discards_and_reconstructs() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let first_model = common::official_chat(&server, "model-one");
    let first = first_model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let second_model = common::official_chat(&server, "model-two");
    let response = second_model
        .stream(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        response.request.replay.decisions.as_slice(),
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
async fn routing_discriminator_changes_versioned_cryptographic_replay_scope() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let mut first_config = common::official_chat_config(&server, "same-model");
    first_config.provider.headers.dynamic_headers = Some(Arc::new(RouteHeaders));
    first_config.settings.routing_discriminator = Some("route-a".into());
    let first_model = oven_sdk_openai::OpenAiChatModel::new(first_config).unwrap();
    let first = first_model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let resource_id = first
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .scope()
        .resource_id
        .as_str();
    assert!(resource_id.starts_with("openai-scope-v1-sha256-"));

    let mut second_config = common::official_chat_config(&server, "same-model");
    second_config.provider.headers.dynamic_headers = Some(Arc::new(RouteHeaders));
    second_config.settings.routing_discriminator = Some("route-b".into());
    let second_model = oven_sdk_openai::OpenAiChatModel::new(second_config).unwrap();
    let response = second_model
        .stream(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        response.request.replay.decisions.as_slice(),
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
async fn dynamic_routing_headers_require_explicit_discriminator() {
    let server = MockServer::start().await;
    let mut config = common::official_responses_config(&server, "same-model");
    config.provider.headers.dynamic_headers = Some(Arc::new(RouteHeaders));
    let error = oven_sdk_openai::OpenAiResponsesModel::new(config)
        .err()
        .expect("dynamic routing without discriminator must fail");
    assert_eq!(error.kind, ModelErrorKind::InvalidRequest);
}

#[tokio::test]
async fn organization_and_project_are_bound_into_replay_scope() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let first_model = common::official_chat(&server, "same-model");
    let first = first_model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut second_config = common::official_chat_config(&server, "same-model");
    second_config.provider.auth.organization = Some("other-org".into());
    second_config.provider.auth.project = Some("other-project".into());
    let second_model = oven_sdk_openai::OpenAiChatModel::new(second_config).unwrap();
    let second = second_model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_ne!(
        first.turn.finish.native_replay.unwrap().scope().resource_id,
        second
            .turn
            .finish
            .native_replay
            .unwrap()
            .scope()
            .resource_id
    );
}

#[tokio::test]
async fn responses_encrypted_items_round_trip() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = common::official_responses(&server, "gpt-5-mini");
    let first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let response = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_eq!(response.request.replay.decisions.len(), 1);
    assert!(matches!(
        response.request.replay.decisions[0].disposition,
        ReplayDisposition::Replayed
    ));
}

#[tokio::test]
async fn responses_summary_and_raw_reasoning_round_trip_with_same_adapter() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":0,\"delta\":\"summary thought\"}\n\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"raw thought\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"reasoning\",\"id\":\"reasoning-0\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"summary thought\"}],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"raw thought\"}],\"encrypted_content\":\"opaque\"}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let model = common::official_responses(&server, "gpt-5-mini");
    let first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let second = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        second.request.replay.decisions.as_slice(),
        [oven_sdk::ReplayDecision {
            disposition: ReplayDisposition::Replayed,
            ..
        }]
    ));
}

#[tokio::test]
async fn responses_replay_rejects_reorder_merge_split_stripping_and_unknown_extras() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"a\"},{\"type\":\"output_text\",\"text\":\"b\"}]},{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"thought\"}],\"encrypted_content\":\"opaque\"}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let model = common::official_responses(&server, "future-responses-id");
    let first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let original = first
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .payload()
        .clone();

    let mut reordered = original.clone();
    reordered["items"].as_array_mut().unwrap().swap(0, 1);

    let mut merged = original.clone();
    merged["items"][0]["content"] = serde_json::json!([{"type":"output_text","text":"ab"}]);

    let mut split = original.clone();
    split["items"][0]["content"] = serde_json::json!([
        {"type":"output_text","text":"a"},
        {"type":"output_text","text":""},
        {"type":"output_text","text":"b"}
    ]);

    let mut stripped = original.clone();
    stripped["items"][1]
        .as_object_mut()
        .unwrap()
        .remove("encrypted_content");

    let mut replaced_encrypted = original.clone();
    replaced_encrypted["items"][1]["encrypted_content"] = "forged".into();

    let mut unknown_field = original.clone();
    unknown_field["items"][0]["unexpected"] = true.into();

    let mut unknown_item = original.clone();
    unknown_item["items"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"type":"unknown","id":"forged"}));

    for payload in [
        reordered,
        merged,
        split,
        stripped,
        replaced_encrypted,
        unknown_field,
        unknown_item,
    ] {
        assert_invalid_replay(&model, with_payload(&first.turn, payload)).await;
    }
}

#[tokio::test]
async fn responses_raw_reasoning_mismatch_discards_replay() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = common::official_responses(&server, "gpt-5-mini");
    let valid = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let artifact = NativeReplayArtifact::new(
        AdapterId::new("oven.openai.responses"),
        valid.turn.finish.native_replay.unwrap().scope().clone(),
        serde_json::json!({
            "format":"oven.openai.responses.output.v1",
            "items":[{
                "type":"reasoning",
                "id":"rs",
                "summary":[],
                "content":[{"type":"reasoning_text","text":"artifact thought"}],
                "encrypted_content":"opaque"
            }],
            "store":false,
            "status":"completed",
            "incomplete_details":null
        }),
    )
    .unwrap();
    let mut finish = Finish::new(Default::default(), FinishReason::Stop);
    finish.native_replay = Some(artifact);
    let turn = CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::Reasoning(ReasoningPart::new(
            "different thought",
        ))]),
        finish,
    );
    let response = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        response.request.replay.decisions.as_slice(),
        [
            oven_sdk::ReplayDecision {
                disposition: ReplayDisposition::DiscardedInvalidPayload { .. },
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
async fn replay_policy_never_records_only_reconstruction_and_captures_nothing() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let mut config = common::official_chat_config(&server, "gpt-4o-mini");
    config.model.capabilities.replay.policy = ReplayPolicy::Never;
    config.model.capabilities.replay.capability = oven_sdk::ReplayCapability::Unsupported;
    let model = oven_sdk_openai::OpenAiChatModel::new(config).unwrap();
    let artifact = NativeReplayArtifact::new(
        AdapterId::new("oven.openai.chat"),
        NativeContextScope::new(
            ProviderId::new("openai"),
            ModelId::new("gpt-4o-mini"),
            ResourceId::new("arbitrary").unwrap(),
        )
        .unwrap(),
        serde_json::json!({"format":"anything"}),
    )
    .unwrap();
    let result = model
        .complete(
            Request::new(vec![HistoryTurn::assistant(completed(artifact))]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        result.request.replay.decisions.as_slice(),
        [oven_sdk::ReplayDecision {
            disposition: ReplayDisposition::ReconstructedNormalized,
            ..
        }]
    ));
    assert!(result.turn.finish.native_replay.is_none());
}

#[tokio::test]
async fn oversized_new_chat_replay_fails_closed() {
    let server = MockServer::start().await;
    let text = "x".repeat(NativeReplayArtifact::MAX_PAYLOAD_BYTES + 1);
    common::mount(&server, "/chat/completions", common::chat_document(&text)).await;
    let error = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Replay);
}

#[tokio::test]
async fn compatible_chat_replay_round_trips_configured_reasoning_wire_field() {
    for (field, wire_field, other_field) in [
        (
            ReasoningField::ReasoningContent,
            "reasoning_content",
            "reasoning",
        ),
        (ReasoningField::Reasoning, "reasoning", "reasoning_content"),
    ] {
        let server = MockServer::start().await;
        let body = format!(
            concat!(
                "data: {{\"choices\":[{{\"index\":0,\"delta\":{{{wire_field:?}:\"thought\"}},\"finish_reason\":null}}]}}\n\n",
                "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"ok\"}},\"finish_reason\":\"stop\"}}]}}\n\n",
                "data: [DONE]\n\n"
            ),
            wire_field = wire_field
        );
        common::mount(&server, "/chat/completions", body).await;
        let mut config = common::compatible_config(&server, "model");
        config.settings.adapter_id = AdapterId::new(format!("fixture.{wire_field}.chat"));
        config.settings.reasoning_field = field;
        let model = OpenAiCompatibleChatModel::new(config).unwrap();
        let first = model
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap();
        let second = model
            .stream(
                Request::new(vec![HistoryTurn::assistant(first.turn)]),
                AbortSignal::default(),
            )
            .await
            .unwrap();
        assert!(matches!(
            second.request.replay.decisions.as_slice(),
            [oven_sdk::ReplayDecision {
                disposition: ReplayDisposition::Replayed,
                ..
            }]
        ));
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
        assert_eq!(body["messages"][0][wire_field], "thought");
        assert!(body["messages"][0].get(other_field).is_none());
    }
}
