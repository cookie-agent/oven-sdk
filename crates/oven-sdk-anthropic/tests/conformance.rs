// The conformance helper's required probe signature returns core ModelError directly.
#![allow(clippy::result_large_err)]

mod common;

use common::{Anthropic, AnthropicAws, MiniMax};

use std::time::Duration;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, Capability, CompactionCapability,
    CompactionRequest, CompletedTurn, Finish, FinishReason, JsonSchema, LanguageModel, ModelError,
    ModelErrorKind, NativeReplayArtifact, ReasoningPart, Request, StreamPart, TextPart, ToolChoice,
    ToolDefinition,
};
use oven_sdk_anthropic::{
    AnthropicCacheControl, AnthropicCacheTtl, AnthropicRequestExt, AnthropicRequestOptions,
    AnthropicThinking,
};
use oven_sdk_conformance::{
    CapabilityProbe, assert_capability_honesty, assert_capability_honesty_with,
    assert_compaction_unsupported_before_io, assert_complete_drain, assert_declaration_honesty,
    assert_error_taxonomy, assert_foreign_replay_is_reported, assert_history_round_trip,
    assert_invalid_replay_reconstructs, assert_malformed_payload_returns_error,
    assert_replay_artifact, assert_replay_round_trip, assert_stream_contract,
    assert_stream_lifecycle, assert_validate_for_consistency,
    sse::{ChunkPattern, chunk_bytes},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SSE_HEADERS: &str =
    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";

fn response() -> String {
    concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    )
    .into()
}

fn event(name: &str, data: &str) -> String {
    format!("event: {name}\r\ndata: {data}\r\n\r\n")
}

fn model(address: std::net::SocketAddr) -> oven_sdk_anthropic::AnthropicModel {
    Anthropic::builder()
        .api_key("test")
        .base_url(format!("http://{address}"))
        .build()
        .unwrap()
        .model("claude-sonnet-4-5")
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, body: &[u8]) {
    socket
        .write_all(format!("{:X}\r\n", body.len()).as_bytes())
        .await
        .unwrap();
    socket.write_all(body).await.unwrap();
    socket.write_all(b"\r\n").await.unwrap();
    socket.flush().await.unwrap();
}

async fn scripted_sse_model(chunks: Vec<Vec<u8>>) -> oven_sdk_anthropic::AnthropicModel {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = model(listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 4096];
        if socket.read(&mut request).await.unwrap() == 0 {
            return;
        }
        socket.write_all(SSE_HEADERS.as_bytes()).await.unwrap();
        for chunk in chunks {
            write_chunk(&mut socket, &chunk).await;
        }
        socket.write_all(b"0\r\n\r\n").await.unwrap();
        tokio::time::sleep(Duration::from_millis(25)).await;
    });
    model
}

async fn scripted_http_error(status: u16, body: &'static str) -> ModelError {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = model(listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 4096];
        if socket.read(&mut request).await.unwrap() == 0 {
            return;
        }
        socket
            .write_all(
                format!(
                    "HTTP/1.1 {status} Error\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
    });
    model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err()
}

async fn decoder_error(body: String) -> ModelError {
    let model = scripted_sse_model(chunk_bytes(body.as_bytes(), &ChunkPattern::OneByte)).await;
    match model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
    {
        Err(error) => error,
        Ok(mut response) => loop {
            match response.stream.next().await {
                Some(Err(error)) => return error,
                Some(Ok(_)) => {}
                None => panic!("malformed Anthropic stream unexpectedly ended cleanly"),
            }
        },
    }
}

fn pathological_response() -> String {
    [
        ": keepalive\r\n".into(),
        event("ping", r#"{"type":"ping"}"#),
        event(
            "message_start",
            r#"{"type":"message_start","message":{"usage":{"input_tokens":1}}}"#,
        ),
        event(
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        ),
        event(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hé"}}"#,
        ),
        event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
        event(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":1}}"#,
        ),
        event("message_stop", r#"{"type":"message_stop"}"#),
    ]
    .concat()
}

async fn model_with_wiremock() -> (MockServer, oven_sdk_anthropic::AnthropicModel) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .api_key("test")
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    (server, model)
}

fn completed(artifact: Option<NativeReplayArtifact>) -> CompletedTurn {
    let mut finish = Finish::new(Default::default(), FinishReason::Stop);
    finish.native_replay = artifact;
    CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::Text(TextPart::new("ok"))]),
        finish,
    )
}

#[tokio::test]
async fn adapter_backed_stream_contract_handles_pathological_sse_chunks() {
    let document = pathological_response();
    let model = scripted_sse_model(chunk_bytes(document.as_bytes(), &ChunkPattern::OneByte)).await;
    let response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_stream_contract(response.stream).await.unwrap();

    let model = scripted_sse_model(chunk_bytes(document.as_bytes(), &ChunkPattern::OneByte)).await;
    assert_stream_lifecycle(&model, Request::new(Vec::new()))
        .await
        .unwrap();
}

#[tokio::test]
async fn adapter_backed_errors_feed_error_taxonomy() {
    let mut errors = Vec::new();
    for (status, body, expected) in [
        (
            400,
            r#"{"error":{"type":"invalid_request_error","message":"bad"}}"#,
            ModelErrorKind::InvalidRequest,
        ),
        (
            401,
            r#"{"error":{"type":"authentication_error","message":"bad key"}}"#,
            ModelErrorKind::Auth,
        ),
        (
            429,
            r#"{"error":{"type":"rate_limit_error","message":"slow"}}"#,
            ModelErrorKind::RateLimited,
        ),
        (
            500,
            r#"{"error":{"type":"api_error","message":"down"}}"#,
            ModelErrorKind::Provider,
        ),
        (
            529,
            r#"{"error":{"type":"overloaded_error","message":"busy"}}"#,
            ModelErrorKind::Overload,
        ),
    ] {
        let error = scripted_http_error(status, body).await;
        assert_eq!(error.kind, expected);
        errors.push(error);
    }

    let mid_stream = scripted_sse_model(vec![
        event("message_start", r#"{"type":"message_start","message":{}}"#).into_bytes(),
        event(
            "error",
            r#"{"type":"error","error":{"type":"api_error","message":"failed"}}"#,
        )
        .into_bytes(),
    ])
    .await;
    let mut response = mid_stream
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    while let Some(item) = response.stream.next().await {
        if let Ok(StreamPart::Error { error }) = item {
            errors.push(error);
            break;
        }
    }

    let malformed = decoder_error(format!(
        "{}event: content_block_delta\r\ndata: {{\"type\":\r\n\r\n",
        event("message_start", r#"{"type":"message_start","message":{}}"#)
    ))
    .await;
    assert_eq!(malformed.kind, ModelErrorKind::InvalidResponse);
    errors.push(malformed);
    assert_eq!(errors.len(), 7);
    assert_error_taxonomy(&errors).unwrap();
}

#[tokio::test]
async fn adapter_backed_malformed_sse_returns_decoder_errors() {
    let truncated_json = decoder_error(format!(
        "{}event: content_block_delta\r\ndata: {{\"type\":\r\n\r\n",
        event("message_start", r#"{"type":"message_start","message":{}}"#)
    ))
    .await;
    assert_malformed_payload_returns_error(|| Err(truncated_json)).unwrap();

    let unknown_event = decoder_error(format!(
        "{}{}",
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        event("unknown_event", r#"{"type":"unknown_event"}"#),
    ))
    .await;
    assert_eq!(unknown_event.kind, ModelErrorKind::InvalidResponse);
    assert_malformed_payload_returns_error(|| Err(unknown_event)).unwrap();

    let invalid_tool_args = decoder_error(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call","name":"lookup"}}"#,
            ),
            event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"["}}"#,
            ),
            event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
        ]
        .concat(),
    )
    .await;
    assert_eq!(invalid_tool_args.kind, ModelErrorKind::InvalidResponse);
    assert_malformed_payload_returns_error(|| Err(invalid_tool_args)).unwrap();
}

#[tokio::test]
async fn full_anthropic_conformance_suite() {
    let (_server, model) = model_with_wiremock().await;
    let baseline = Request::new(Vec::new());
    let lifecycle = assert_stream_lifecycle(&model, baseline.clone())
        .await
        .unwrap();
    let completed_result = assert_complete_drain(&model, baseline.clone())
        .await
        .unwrap();
    assert_capability_honesty(&model).unwrap();
    assert_declaration_honesty(&model).unwrap();
    let cache = Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
        cache_control: Some(AnthropicCacheControl {
            ttl: AnthropicCacheTtl::FiveMinutes,
        }),
        ..Default::default()
    });
    assert_capability_honesty_with(
        &model,
        [CapabilityProbe {
            capability: Capability::PROMPT_CACHING,
            name: "prompt_caching",
            request: cache,
        }],
    )
    .unwrap();
    assert_validate_for_consistency(&model, &baseline).unwrap();
    let native_context_scope = completed_result
        .turn
        .finish
        .native_replay
        .as_ref()
        .unwrap()
        .scope()
        .clone();
    assert_replay_artifact(
        &model.descriptor(),
        &native_context_scope,
        &completed_result.turn,
    )
    .unwrap();
    assert_replay_round_trip(
        &model,
        &native_context_scope,
        Request::new(vec![oven_sdk::HistoryTurn::assistant(
            completed_result.turn.clone(),
        )]),
    )
    .await
    .unwrap();
    let invalid = NativeReplayArtifact::new(
        AdapterId::new("oven.anthropic.messages"),
        native_context_scope.clone(),
        serde_json::Value::String("garbage".into()),
    )
    .unwrap();
    assert_invalid_replay_reconstructs(
        &model,
        &native_context_scope,
        Request::new(vec![oven_sdk::HistoryTurn::assistant(completed(Some(
            invalid,
        )))]),
    )
    .await
    .unwrap();
    let foreign = NativeReplayArtifact::new(
        AdapterId::new("foreign"),
        native_context_scope.clone(),
        serde_json::json!({}),
    )
    .unwrap();
    assert_foreign_replay_is_reported(
        &model,
        &native_context_scope,
        Request::new(vec![oven_sdk::HistoryTurn::assistant(completed(Some(
            foreign,
        )))]),
    )
    .await
    .unwrap();
    assert_history_round_trip(&model, completed_result.turn).unwrap();
    assert_eq!(lifecycle.stream.finish.finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn minimax_and_anthropic_aws_baseline_conformance() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let minimax = MiniMax::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("MiniMax-M3");
    assert_stream_lifecycle(&minimax, Request::new(Vec::new()))
        .await
        .unwrap();
    assert_capability_honesty(&minimax).unwrap();
    assert_declaration_honesty(&minimax).unwrap();

    let aws = AnthropicAws::builder("us-west-2", "wrkspc_test")
        .bearer_key("test")
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-6");
    assert_complete_drain(&aws, Request::new(Vec::new()))
        .await
        .unwrap();
    assert_capability_honesty(&aws).unwrap();
    assert_declaration_honesty(&aws).unwrap();
}

#[tokio::test]
async fn all_concrete_models_inherit_unsupported_compaction_without_io() {
    let server = MockServer::start().await;
    let direct = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("direct-model");
    let minimax = MiniMax::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("minimax-model");
    let aws = AnthropicAws::builder("us-west-2", "workspace")
        .bearer_key("key")
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("aws-model");

    for model in [
        &direct as &dyn LanguageModel,
        &minimax as &dyn LanguageModel,
        &aws as &dyn LanguageModel,
    ] {
        assert_eq!(
            model.capabilities().compaction,
            CompactionCapability::Unsupported
        );
        assert_compaction_unsupported_before_io(
            model,
            CompactionRequest::new(Request::new(Vec::new())),
        )
        .await
        .unwrap();
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[test]
fn direct_and_aws_enforce_thinking_request_conformance_before_dispatch() {
    let direct = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let aws = AnthropicAws::builder("us-west-2", "wrkspc_test")
        .bearer_key("test")
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");

    for model in [&direct as &dyn LanguageModel, &aws as &dyn LanguageModel] {
        let tool = ToolDefinition::new(
            "lookup",
            "lookup",
            JsonSchema::new(serde_json::json!({"type":"object"})).unwrap(),
        );
        let forced = Request::new(Vec::new())
            .with_tools(vec![tool])
            .with_tool_choice(ToolChoice::Required)
            .with_anthropic_options(AnthropicRequestOptions {
                thinking: Some(AnthropicThinking::Enabled {
                    budget_tokens: 1024,
                    display: None,
                }),
                ..Default::default()
            });
        assert!(model.validate_request(&forced).is_err());

        let prefill = Request::new(vec![oven_sdk::HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![AssistantPart::Reasoning(ReasoningPart::new(
                "prefill",
            ))]),
            Finish::new(Default::default(), FinishReason::Stop),
        ))])
        .with_anthropic_options(AnthropicRequestOptions {
            thinking: Some(AnthropicThinking::Enabled {
                budget_tokens: 1024,
                display: None,
            }),
            ..Default::default()
        });
        assert!(model.validate_request(&prefill).is_err());

        let mut zero = Request::new(Vec::new());
        zero.inference.max_output_tokens = Some(0);
        assert!(model.validate_request(&zero).is_err());
    }
}
