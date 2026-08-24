mod common;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use oven_sdk::{
    AbortSignal, AdapterId, ErrorStage, FilePart, FileSource, HeaderOverrides, HistoryTurn,
    InputPart, JsonSchema, LanguageModel, ModelError, ModelErrorKind, ModelLimits,
    NativeReplayArtifact, ReplayCapability, ReplayDisposition, Request, ResponseFormat,
    UserMessage,
};
use oven_sdk_azure::{
    AZURE_OPENAI_CHAT_ADAPTER_ID, AZURE_OPENAI_RESPONSES_ADAPTER_ID, AzureApiRoute,
    AzureApiVersion, AzureOpenAiAuth, AzureOpenAiChatModel,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use sha2::{Digest, Sha256};
use wiremock::MockServer;

fn media_request(files: Vec<FilePart>) -> Request {
    Request::new(vec![HistoryTurn::user(UserMessage::new(
        files.into_iter().map(InputPart::File).collect(),
    ))])
}

fn strict_schema(value: serde_json::Value) -> Request {
    Request::new(Vec::new())
        .with_response_format(ResponseFormat::structured(JsonSchema::new(value).unwrap()))
}

fn assert_discarded(result: &oven_sdk::CompleteResult) {
    assert!(matches!(
        result.request.replay.decisions.as_slice(),
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

async fn responses_error(document: String) -> ModelError {
    let server = MockServer::start().await;
    common::mount(&server, "/openai/v1/responses", document).await;
    common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err()
}

fn assert_typed_invalid(error: &ModelError, stage: ErrorStage) {
    assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
    assert_eq!(error.diagnostics.stage, stage);
    assert!(error.diagnostics.bytes_received > 0);
}

#[test]
fn model_names_never_infer_capabilities_limits_or_replay() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::start());
    let provider = common::provider(&server, AzureApiRoute::V1);
    let conservative = provider
        .chat("gpt-4.1-2025-04-14", common::conservative())
        .unwrap();
    let capabilities = &conservative.descriptor().capabilities;
    assert_eq!(
        conservative.descriptor().identity.provider_id.as_str(),
        "azure.openai"
    );
    assert!(capabilities.features.is_empty());
    assert_eq!(capabilities.limits.context, None);
    assert_eq!(capabilities.limits.output, None);
    assert_eq!(
        capabilities.replay.capability,
        ReplayCapability::Unsupported
    );

    let mut explicit = common::gpt4o();
    explicit.capabilities.limits = ModelLimits::new(Some(300_000), None, Some(32_768));
    let configured = provider
        .chat("arbitrary-deployment", explicit.clone())
        .unwrap();
    assert_eq!(configured.capabilities().limits.context, Some(300_000));
    assert_eq!(configured.capabilities().limits.output, Some(32_768));

    explicit.revision = None;
    assert!(provider.chat("arbitrary-deployment", explicit).is_err());

    let mut opaque_without_declaration = common::gpt5();
    opaque_without_declaration.capabilities.replay.reasoning = false;
    assert!(
        provider
            .responses("deployment", opaque_without_declaration)
            .is_err()
    );

    let mut remote_cancellation = common::conservative();
    remote_cancellation.capabilities.cancellation =
        oven_sdk::CancellationCapability::RemoteBestEffort;
    assert!(provider.chat("deployment", remote_cancellation).is_err());
}

#[tokio::test]
async fn conservative_configuration_does_not_request_or_capture_encrypted_replay() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/responses",
        common::responses_document("ok"),
    )
    .await;
    let result = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::conservative())
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(result.turn.finish.native_replay.is_none());
    let request: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert!(
        !request["include"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "reasoning.encrypted_content")
    );
}

#[tokio::test]
async fn forged_chat_replay_with_extra_fields_is_rejected() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/chat/completions",
        common::chat_document("ok"),
    )
    .await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .chat("deployment", common::gpt4o())
        .unwrap();
    let mut first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    let artifact = first.finish.native_replay.as_ref().unwrap();
    let mut payload = artifact.payload().clone();
    payload["message"]["forged"] = true.into();
    first.finish.native_replay = Some(
        NativeReplayArtifact::new(
            AdapterId::new(AZURE_OPENAI_CHAT_ADAPTER_ID),
            artifact.scope().clone(),
            payload,
        )
        .unwrap(),
    );
    let result = model
        .complete(
            Request::new(vec![HistoryTurn::assistant(first)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_discarded(&result);
}

fn encrypted_document(secret: &str) -> String {
    format!(
        concat!(
            "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp\"}}}}\n\n",
            "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[],\"encrypted_content\":{secret:?}}}}}\n\n",
            "data: {{\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":0,\"delta\":\"summary\"}}\n\n",
            "data: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[{{\"type\":\"summary_text\",\"text\":\"summary\"}}],\"encrypted_content\":{secret:?}}}}}\n\n",
            "data: {{\"type\":\"response.completed\",\"response\":{{\"status\":\"completed\",\"output\":[{{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[{{\"type\":\"summary_text\",\"text\":\"summary\"}}],\"encrypted_content\":{secret:?}}}]}}}}\n\n"
        ),
        secret = secret
    )
}

#[tokio::test]
async fn forged_responses_hidden_continuation_with_recomputed_payload_hash_is_rejected() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/responses",
        encrypted_document("original-encrypted"),
    )
    .await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap();
    let mut first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    let artifact = first.finish.native_replay.as_ref().unwrap();
    let mut payload = artifact.payload().clone();
    payload["items"][0]["encrypted_content"] = "forged-encrypted".into();
    let encoded = serde_json::to_vec(payload["items"].as_array().unwrap()).unwrap();
    payload["fingerprint"] = URL_SAFE_NO_PAD.encode(Sha256::digest(encoded)).into();
    first.finish.native_replay = Some(
        NativeReplayArtifact::new(
            AdapterId::new(AZURE_OPENAI_RESPONSES_ADAPTER_ID),
            artifact.scope().clone(),
            payload,
        )
        .unwrap(),
    );
    let result = model
        .complete(
            Request::new(vec![HistoryTurn::assistant(first)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_discarded(&result);
    let requests = server.received_requests().await.unwrap();
    assert!(!String::from_utf8_lossy(&requests[1].body).contains("forged-encrypted"));
}

#[tokio::test]
async fn responses_replay_capture_strips_provider_extras_and_rejects_unknown_items() {
    let safe_server = MockServer::start().await;
    let safe = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp\"}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\",\"annotations\":[{\"secret\":\"x\"}]}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\",\"annotations\":[{\"secret\":\"x\"}]}]}]}}\n\n"
    );
    common::mount(&safe_server, "/openai/v1/responses", safe.into()).await;
    let result = common::provider(&safe_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let replay = result.turn.finish.native_replay.unwrap();
    assert!(replay.payload().pointer("/items/0/status").is_none());
    assert!(
        replay
            .payload()
            .pointer("/items/0/content/0/annotations")
            .is_none()
    );

    let unknown_server = MockServer::start().await;
    let unknown = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp\"}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"future_item\",\"id\":\"future\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"future_item\",\"id\":\"future\"}]}}\n\n"
    );
    common::mount(&unknown_server, "/openai/v1/responses", unknown.into()).await;
    let error = common::provider(&unknown_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
}

#[tokio::test]
async fn responses_provider_indices_are_checked_before_vector_extension() {
    for (event_type, index_field) in [
        ("response.output_text.delta", "content_index"),
        ("response.refusal.delta", "content_index"),
        ("response.reasoning_summary_text.delta", "summary_index"),
        ("response.reasoning_text.delta", "content_index"),
    ] {
        for index in [u64::MAX, 128, 127, 16] {
            let event = serde_json::json!({
                "type": event_type,
                "output_index": 0,
                (index_field): index,
                "delta": "x"
            });
            let document = format!(
                "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp\"}}}}\n\ndata: {event}\n\n"
            );
            let error = responses_error(document).await;
            assert_typed_invalid(&error, ErrorStage::StreamEvent);
        }
    }

    for output_index in [u64::MAX, 128, 127] {
        let event = serde_json::json!({
            "type":"response.output_text.delta",
            "output_index":output_index,
            "content_index":0,
            "delta":"x"
        });
        let error = responses_error(format!("data: {event}\n\n")).await;
        assert_typed_invalid(&error, ErrorStage::StreamEvent);
    }

    let output_gap = concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":2,\"content_index\":0,\"delta\":\"x\"}\n\n"
    );
    let error = responses_error(output_gap.into()).await;
    assert_typed_invalid(&error, ErrorStage::StreamEvent);

    for (item_type, field, part_type) in [
        ("message", "content", "output_text"),
        ("reasoning", "content", "reasoning_text"),
        ("reasoning", "summary", "summary_text"),
    ] {
        let slots = vec![serde_json::json!({"type":part_type,"text":"x"}); 129];
        let item = serde_json::json!({
            "type":item_type,
            "id":"item",
            "role":"assistant",
            (field):slots
        });
        let event = serde_json::json!({
            "type":"response.output_item.added",
            "output_index":0,
            "item":item
        });
        let error = responses_error(format!("data: {event}\n\n")).await;
        assert_typed_invalid(&error, ErrorStage::StreamEvent);
    }
}

#[tokio::test]
async fn responses_terminal_payloads_are_fail_closed() {
    let valid_item = serde_json::json!({
        "type":"message",
        "id":"msg",
        "role":"assistant",
        "content":[{"type":"output_text","text":"ok"}]
    });
    let invalid_events = [
        serde_json::json!({"type":"response.completed"}),
        serde_json::json!({"type":"response.completed","response":{"output":[valid_item.clone()]}}),
        serde_json::json!({"type":"response.completed","response":{"status":"incomplete","output":[valid_item.clone()]}}),
        serde_json::json!({"type":"response.completed","response":{"status":"completed"}}),
        serde_json::json!({"type":"response.completed","response":{"status":"completed","output":null}}),
        serde_json::json!({"type":"response.completed","response":{"status":"completed","output":[]}}),
        serde_json::json!({"type":"response.completed","response":{"status":"completed","output":[{"type":"hosted_tool_call","id":"hosted"}]}}),
        serde_json::json!({"type":"response.completed","response":{"status":"completed","output":[valid_item.clone()],"incomplete_details":{"reason":"max_output_tokens"}}}),
        serde_json::json!({"type":"response.incomplete","response":{"status":"incomplete","output":[valid_item.clone()]}}),
        serde_json::json!({"type":"response.incomplete","response":{"status":"incomplete","output":[valid_item.clone()],"incomplete_details":null}}),
        serde_json::json!({"type":"response.incomplete","response":{"status":"incomplete","output":[valid_item.clone()],"incomplete_details":{"reason":"future_reason"}}}),
        serde_json::json!({"type":"response.incomplete","response":{"status":"completed","output":[valid_item.clone()],"incomplete_details":{"reason":"max_output_tokens"}}}),
    ];
    for event in invalid_events {
        let error = responses_error(format!("data: {event}\n\n")).await;
        assert_typed_invalid(&error, ErrorStage::StreamFinalize);
    }

    for (reason, expected) in [
        ("max_output_tokens", oven_sdk::FinishReason::Length),
        ("content_filter", oven_sdk::FinishReason::ContentFilter),
    ] {
        let server = MockServer::start().await;
        let event = serde_json::json!({
            "type":"response.incomplete",
            "response":{
                "status":"incomplete",
                "output":[valid_item.clone()],
                "incomplete_details":{"reason":reason}
            }
        });
        common::mount(
            &server,
            "/openai/v1/responses",
            format!("data: {event}\n\n"),
        )
        .await;
        let result = common::provider(&server, AzureApiRoute::V1)
            .responses("deployment", common::gpt5())
            .unwrap()
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap();
        assert_eq!(result.turn.finish.finish_reason, expected);
    }
}

#[test]
fn injected_clients_are_not_a_public_configuration_surface_and_protected_headers_are_rejected() {
    for name in [
        "authorization",
        "api-key",
        "host",
        "content-type",
        "content-length",
    ] {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_static("malicious"),
        );
        let mut config = common::chat_config(
            "https://example.test",
            AzureApiRoute::V1,
            "deployment",
            common::conservative(),
            AzureOpenAiAuth::ApiKey(oven_sdk::SecretString::new("configured")),
        );
        config.provider.headers.static_headers = HeaderOverrides::new(headers);
        assert!(AzureOpenAiChatModel::new(config).is_err(), "{name}");
    }
}

#[test]
fn chat_and_responses_media_limits_are_surface_specific_and_checked() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::start());
    let provider = common::provider(&server, AzureApiRoute::V1);
    let chat = provider.chat("deployment", common::gpt4o()).unwrap();
    let responses = provider.responses("deployment", common::gpt5()).unwrap();

    let images = |count| {
        (0..count)
            .map(|_| FilePart::image("image/png", FileSource::Bytes(Bytes::from_static(b"x"))))
            .collect::<Vec<_>>()
    };
    assert!(chat.validate_request(&media_request(images(10))).is_ok());
    assert!(chat.validate_request(&media_request(images(11))).is_err());
    assert!(
        responses
            .validate_request(&media_request(images(50)))
            .is_ok()
    );
    assert!(
        responses
            .validate_request(&media_request(images(51)))
            .is_err()
    );

    let combined_images = media_request(
        [16, 16, 16, 2]
            .into_iter()
            .map(|mib| {
                FilePart::image(
                    "image/png",
                    FileSource::Bytes(Bytes::from(vec![0; mib * 1024 * 1024])),
                )
            })
            .collect(),
    );
    assert!(responses.validate_request(&combined_images).is_err());

    let pdf = |mib: usize| {
        FilePart::document(
            "application/pdf",
            FileSource::Bytes(Bytes::from(vec![0; mib * 1024 * 1024])),
        )
    };
    assert!(chat.validate_request(&media_request(vec![pdf(1)])).is_err());
    assert!(
        responses
            .validate_request(&media_request(vec![pdf(49)]))
            .is_ok()
    );
    assert!(
        responses
            .validate_request(&media_request(vec![pdf(50)]))
            .is_err()
    );
    assert!(
        responses
            .validate_request(&media_request(vec![pdf(25), pdf(25)]))
            .is_err()
    );
    assert!(
        responses
            .validate_request(&media_request(vec![FilePart::document(
                "application/pdf",
                FileSource::Text("not-pdf-bytes".into()),
            )]))
            .is_err()
    );
}

#[test]
fn strict_schema_references_enforce_depth_cycles_and_local_definition_boundaries() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::start());
    let model = common::provider(&server, AzureApiRoute::V1)
        .chat("deployment", common::gpt4o())
        .unwrap();
    let object = |name: &str, value: serde_json::Value| {
        let mut properties = serde_json::Map::new();
        properties.insert(name.to_owned(), value);
        serde_json::json!({
            "type":"object",
            "properties":properties,
            "required":[name],
            "additionalProperties":false
        })
    };
    let valid = serde_json::json!({
        "type":"object",
        "properties":{"root":{"$ref":"#/$defs/A"}},
        "required":["root"],
        "additionalProperties":false,
        "$defs":{
            "A":object("a", serde_json::json!({"$ref":"#/$defs/B"})),
            "B":object("b", serde_json::json!({"$ref":"#/$defs/C"})),
            "C":object("c", serde_json::json!({"$ref":"#/$defs/D"})),
            "D":object("value", serde_json::json!({"type":"string"}))
        }
    });
    assert!(model.validate_request(&strict_schema(valid)).is_ok());

    let too_deep = serde_json::json!({
        "type":"object",
        "properties":{"root":{"$ref":"#/$defs/A"}},
        "required":["root"],
        "additionalProperties":false,
        "$defs":{
            "A":object("a", serde_json::json!({"$ref":"#/$defs/B"})),
            "B":object("b", serde_json::json!({"$ref":"#/$defs/C"})),
            "C":object("c", serde_json::json!({"$ref":"#/$defs/D"})),
            "D":object("d", serde_json::json!({"$ref":"#/$defs/E"})),
            "E":object("value", serde_json::json!({"type":"string"}))
        }
    });
    assert!(model.validate_request(&strict_schema(too_deep)).is_err());

    for invalid in [
        serde_json::json!({
            "type":"object","properties":{"x":{"$ref":"#/$defs/Missing"}},
            "required":["x"],"additionalProperties":false,"$defs":{}
        }),
        serde_json::json!({
            "type":"object","properties":{"x":{"$ref":"#/properties/x"}},
            "required":["x"],"additionalProperties":false
        }),
        serde_json::json!({
            "type":"object","properties":{"x":{"$ref":"#/$defs/A/properties/x"}},
            "required":["x"],"additionalProperties":false,
            "$defs":{"A":object("x", serde_json::json!({"type":"string"}))}
        }),
        serde_json::json!({
            "type":"object","properties":{"x":{"$ref":"#/$defs/A"}},
            "required":["x"],"additionalProperties":false,
            "$defs":{
                "A":{"$ref":"#/$defs/B"},
                "B":{"$ref":"#/$defs/A"}
            }
        }),
    ] {
        assert!(model.validate_request(&strict_schema(invalid)).is_err());
    }
}

#[tokio::test]
async fn chat_requires_a_choice_and_finish_reason_for_success() {
    for document in [
        "data: [DONE]\n\n".to_owned(),
        "data: {\"prompt_filter_results\":[],\"choices\":[]}\n\ndata: [DONE]\n\n".to_owned(),
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1}}\n\ndata: [DONE]\n\n".to_owned(),
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n".to_owned(),
    ] {
        let server = MockServer::start().await;
        common::mount(&server, "/openai/v1/chat/completions", document).await;
        let error = common::provider(&server, AzureApiRoute::V1)
            .chat("deployment", common::gpt4o())
            .unwrap()
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap_err();
        assert!(matches!(
            error.kind,
            ModelErrorKind::UnexpectedEof | ModelErrorKind::InvalidResponse
        ));
    }
}

#[test]
fn api_version_rejects_year_zero() {
    assert!(AzureApiVersion::new("0000-01-01").is_err());
    assert!(AzureApiVersion::new("0000-02-29-preview").is_err());
}
