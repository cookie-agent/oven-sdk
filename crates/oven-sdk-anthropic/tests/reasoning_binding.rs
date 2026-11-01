//! Reasoning bound to its request prefix: the refusal's error kind,
//! `thinking.block_binding`, and `input_transformations`.

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, ErrorStage, LanguageModel, ModelErrorKind, ReasoningBinding,
    ReasoningTransformation, ReasoningTransformationKind, ReplayPolicy, Request, StreamPart,
};
use oven_sdk_anthropic::{
    AnthropicRequestExt, AnthropicRequestOptions, AnthropicThinking, classify_error,
};
use reqwest::header::HeaderMap;
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;
use common::Anthropic;

const BETA: &str = "thinking-binding-controls-2026-08-01";

fn empty_stream() -> &'static str {
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
}

fn classify(message: &str) -> oven_sdk::ModelError {
    let body = serde_json::json!({
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    });
    classify_error(
        400,
        body.to_string().as_bytes(),
        None,
        ErrorStage::ResponseBody,
        0,
        &HeaderMap::new(),
    )
}

#[test]
fn a_refused_stale_block_is_a_reasoning_prefix_mismatch() {
    let error = classify(
        "messages.5.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\". The `tools` list differs from the one this block was created with.",
    );
    assert_eq!(error.kind, ModelErrorKind::ReasoningPrefixMismatch);
    assert!(!error.retryable);
    // A gateway that masks the indices still says why.
    let masked = classify(
        "***.***.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation.",
    );
    assert_eq!(masked.kind, ModelErrorKind::ReasoningPrefixMismatch);
}

#[test]
fn a_signature_that_failed_on_its_own_stays_an_invalid_request() {
    let error = classify("messages.1.content.0: Invalid `signature` in `thinking` block");
    assert_eq!(error.kind, ModelErrorKind::InvalidRequest);
}

async fn sent_body(
    binding: ReasoningBinding,
    thinking: Option<AnthropicThinking>,
    default_adaptive: bool,
) -> (serde_json::Value, Option<String>) {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(empty_stream()))
        .mount(&server)
        .await;
    let mut capabilities = common::anthropic_capabilities(ReplayPolicy::IfValid);
    capabilities.replay.reasoning_binding = binding;
    let mut protocol = common::anthropic_protocol();
    protocol.thinking_default_active = default_adaptive;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .capabilities(capabilities)
        .protocol(protocol)
        .build()
        .unwrap()
        .model("claude-test");
    let request = Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
        thinking,
        ..Default::default()
    });
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body = serde_json::from_slice(&requests[0].body).unwrap();
    let beta = requests[0]
        .headers
        .get("anthropic-beta")
        .map(|value| value.to_str().unwrap().to_owned());
    (body, beta)
}

fn sends_drop_block(body: &serde_json::Value) -> bool {
    body["thinking"]["block_binding"]["prefix_mismatch_behavior"] == "drop_block"
}

#[tokio::test]
async fn a_binding_model_asks_for_drop_block_with_adaptive_thinking() {
    let (body, beta) = sent_body(
        ReasoningBinding::Prefix,
        Some(AnthropicThinking::Adaptive { display: None }),
        false,
    )
    .await;
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert!(sends_drop_block(&body));
    assert_eq!(beta.as_deref(), Some(BETA));
}

#[tokio::test]
async fn drop_block_is_never_sent_without_adaptive_thinking() {
    let (body, beta) = sent_body(
        ReasoningBinding::Prefix,
        Some(AnthropicThinking::Enabled {
            budget_tokens: 1024,
            display: None,
        }),
        false,
    )
    .await;
    assert!(body["thinking"].get("block_binding").is_none());
    assert!(beta.is_none());
    let (body, beta) = sent_body(ReasoningBinding::Prefix, None, false).await;
    assert!(body.get("thinking").is_none());
    assert!(beta.is_none());
}

#[tokio::test]
async fn a_default_adaptive_model_carries_drop_block_on_its_default() {
    let (body, beta) = sent_body(ReasoningBinding::Prefix, None, true).await;
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert!(sends_drop_block(&body));
    assert_eq!(beta.as_deref(), Some(BETA));
}

#[tokio::test]
async fn a_model_that_does_not_bind_sends_no_binding_controls() {
    let (body, beta) = sent_body(
        ReasoningBinding::None,
        Some(AnthropicThinking::Adaptive { display: None }),
        true,
    )
    .await;
    assert!(body["thinking"].get("block_binding").is_none());
    assert!(beta.is_none());
    let (body, _) = sent_body(ReasoningBinding::None, None, true).await;
    assert!(body.get("thinking").is_none());
}

async fn finish_of(stream: &'static str) -> oven_sdk::Finish {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(stream))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-test");
    let response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let parts = response.stream.collect::<Vec<_>>().await;
    parts
        .into_iter()
        .find_map(|part| match part.unwrap() {
            StreamPart::Finish { finish } => Some(finish),
            _ => None,
        })
        .unwrap()
}

fn transformation(
    kind: ReasoningTransformationKind,
    path: &str,
    reason: &str,
) -> ReasoningTransformation {
    ReasoningTransformation {
        kind,
        path: path.into(),
        reason: reason.into(),
    }
}

#[tokio::test]
async fn transformations_on_message_start_reach_the_finish() {
    let finish = finish_of(concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"input_transformations\":[",
        "{\"type\":\"thinking_dropped\",\"path\":\"messages.3.content.0\",\"reason\":\"prefix_binding_mismatch\"},",
        "{\"type\":\"thinking_mismatch_allowed\",\"path\":\"messages.5.content.0\",\"reason\":\"prefix_binding_mismatch\"},",
        "{\"type\":\"a_future_type\",\"path\":\"messages.7.content.0\",\"reason\":\"x\"},",
        "{\"type\":\"thinking_dropped\",\"reason\":\"prefix_binding_mismatch\"}",
        "]}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ))
    .await;
    assert_eq!(
        finish.reasoning_transformations,
        vec![
            transformation(
                ReasoningTransformationKind::Dropped,
                "messages.3.content.0",
                "prefix_binding_mismatch",
            ),
            transformation(
                ReasoningTransformationKind::MismatchAllowed,
                "messages.5.content.0",
                "prefix_binding_mismatch",
            ),
        ]
    );
}

#[tokio::test]
async fn a_final_delta_replaces_the_transformations_after_a_fallback() {
    let finish = finish_of(concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"input_transformations\":[",
        "{\"type\":\"thinking_dropped\",\"path\":\"messages.3.content.0\",\"reason\":\"prefix_binding_mismatch\"}",
        "]}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{},\"input_transformations\":[",
        "{\"type\":\"thinking_dropped\",\"path\":\"messages.1.content.0\",\"reason\":\"model_binding_mismatch\"}",
        "]}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ))
    .await;
    assert_eq!(
        finish.reasoning_transformations,
        vec![transformation(
            ReasoningTransformationKind::Dropped,
            "messages.1.content.0",
            "model_binding_mismatch",
        )]
    );
}

#[tokio::test]
async fn a_response_without_transformations_reports_none() {
    let finish = finish_of(empty_stream()).await;
    assert!(finish.reasoning_transformations.is_empty());
}
