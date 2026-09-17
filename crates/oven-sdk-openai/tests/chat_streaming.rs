pub mod common;

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, FinishReason, LanguageModel, ModelErrorKind, Request, StreamPart};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn usage_only_chunk_is_terminally_authoritative() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("hello")).await;
    let result = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "hello");
    assert_eq!(result.turn.finish.usage.input_tokens, Some(2));
    assert_eq!(result.turn.finish.usage.output_tokens, Some(3));
    assert_eq!(result.turn.finish.usage.output_tokens_reasoning, Some(1));
    assert_eq!(result.turn.finish.usage.output_tokens_text, Some(2));
}

#[tokio::test]
async fn qwen_usage_separates_cached_input_and_top_level_reasoning() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/chat/completions",
        common::qwen_usage_chat_document("{\"cached_tokens\":1216}"),
    )
    .await;
    let result = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let usage = result.turn.finish.usage;
    assert_eq!(usage.input_tokens_cache_read, Some(1216));
    assert_eq!(usage.input_tokens_no_cache, Some(48));
    assert_eq!(usage.output_tokens_reasoning, Some(20));
    assert_eq!(usage.output_tokens_text, Some(0));
}

#[tokio::test]
async fn qwen_cache_miss_keeps_cache_and_no_cache_unknown() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/chat/completions",
        common::qwen_usage_chat_document("null"),
    )
    .await;
    let result = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let usage = result.turn.finish.usage;
    assert_eq!(usage.input_tokens_cache_read, None);
    assert_eq!(usage.input_tokens_no_cache, None);
    assert_eq!(usage.output_tokens_reasoning, Some(20));
    assert_eq!(usage.output_tokens_text, Some(0));
}

#[tokio::test]
async fn fragmented_parallel_tools_finalize_only_at_done() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":\"}},{\"index\":1,\"id\":\"call_2\",\"function\":{\"name\":\"two\",\"arguments\":\"{\\\"b\\\":2}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"one\",\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    common::mount(&server, "/chat/completions", body.into()).await;
    let result = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let calls = result
        .turn
        .message
        .content
        .iter()
        .filter(|part| matches!(part, oven_sdk::AssistantPart::ToolCall(_)))
        .count();
    assert_eq!(calls, 2);
    assert_eq!(result.turn.finish.finish_reason, FinishReason::ToolCalls);
}

#[tokio::test]
async fn repeated_tool_index_and_missing_identity_finalize_with_stable_ids() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"a\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"b\",\"arguments\":\"{}\"}},{\"index\":2,\"function\":{\"arguments\":\"{}\"}},{\"index\":3,\"id\":\"google-call-2\",\"function\":{\"name\":\"real\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    common::mount(&server, "/chat/completions", body.into()).await;
    let completed = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let calls = completed
        .turn
        .message
        .content
        .iter()
        .filter_map(|part| match part {
            oven_sdk::AssistantPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        calls
            .iter()
            .map(|call| call.id.as_str())
            .collect::<Vec<_>>(),
        ["call_a", "call_a-1", "google-call-2", "google-call-2-1"]
    );
    assert_eq!(calls[3].name, "");
    let replay = completed.turn.finish.native_replay.unwrap();
    assert!(
        replay
            .payload()
            .pointer("/message/tool_calls/3/function/name")
            .is_none()
    );
}

#[tokio::test]
async fn empty_final_arguments_become_empty_object() {
    let server = MockServer::start().await;
    let event = serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","function":{"name":"tool","arguments":""}}]},"finish_reason":"tool_calls"}]});
    let body = format!("data: {event}\n\ndata: [DONE]\n\n");
    common::mount(&server, "/chat/completions", body).await;
    let completed = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let call = completed
        .turn
        .message
        .content
        .iter()
        .find_map(|part| match part {
            oven_sdk::AssistantPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("finalized tool call");
    assert_eq!(call.input, serde_json::json!({}));
    assert!(call.raw_input.is_none());
    assert!(completed.turn.warnings.is_empty());
}

#[tokio::test]
async fn invalid_final_arguments_are_marked_invalid() {
    let server = MockServer::start().await;
    let event = serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","function":{"name":"tool","arguments":"["}}]},"finish_reason":"tool_calls"}]});
    let body = format!("data: {event}\n\ndata: [DONE]\n\n");
    common::mount(&server, "/chat/completions", body).await;
    let completed = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let call = completed
        .turn
        .message
        .content
        .iter()
        .find_map(|part| match part {
            oven_sdk::AssistantPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("finalized tool call");
    assert!(call.input.is_null());
    assert_eq!(call.raw_input.as_deref(), Some("["));
    assert_eq!(
        completed.turn.warnings,
        vec![
            "tool call `call` finalized with arguments that are not a valid JSON object; input surfaced as null"
        ]
    );
}

#[tokio::test]
async fn non_object_final_arguments_are_marked_invalid() {
    for arguments in ["[]", "123", "\"x\"", "null"] {
        let server = MockServer::start().await;
        let event = serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","function":{"name":"tool","arguments":arguments}}]},"finish_reason":"tool_calls"}]});
        let body = format!("data: {event}\n\ndata: [DONE]\n\n");
        common::mount(&server, "/chat/completions", body).await;
        let completed = common::official_chat(&server, "gpt-4o-mini")
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap();
        let call = completed
            .turn
            .message
            .content
            .iter()
            .find_map(|part| match part {
                oven_sdk::AssistantPart::ToolCall(call) => Some(call),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{arguments} must finalize a tool call"));
        assert!(call.input.is_null(), "{arguments} must surface null input");
        assert_eq!(call.raw_input.as_deref(), Some(arguments));
        assert_eq!(
            completed.turn.warnings,
            vec![
                "tool call `call` finalized with arguments that are not a valid JSON object; input surfaced as null"
            ],
            "{arguments} must carry the marked-invalid warning"
        );
    }
}

#[tokio::test]
async fn zero_argument_tool_replays_canonical_empty_object() {
    let server = MockServer::start().await;
    let event = serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","function":{"name":"noop","arguments":""}}]},"finish_reason":"tool_calls"}]});
    let body = format!("data: {event}\n\ndata: [DONE]\n\n");
    common::mount(&server, "/chat/completions", body).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(first.turn.finish.finish_reason, FinishReason::ToolCalls);
    let artifact = first
        .turn
        .finish
        .native_replay
        .as_ref()
        .expect("replay artifact");
    assert_eq!(
        artifact
            .payload()
            .pointer("/message/tool_calls/0/function/arguments"),
        Some(&serde_json::Value::String("{}".into())),
        "canonical zero-argument calls replay as an empty object"
    );
    let replayed = model
        .stream(
            Request::new(vec![
                oven_sdk::HistoryTurn::assistant(first.turn),
                oven_sdk::HistoryTurn::tool(oven_sdk::ToolMessage::new(vec![
                    oven_sdk::ToolResultPart::new(
                        "call",
                        oven_sdk::ToolContent::Text("done".into()),
                    ),
                ])),
            ]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        replayed.request.replay.decisions.as_slice(),
        [oven_sdk::ReplayDecision {
            disposition: oven_sdk::ReplayDisposition::Replayed,
            ..
        }]
    ));
}

#[tokio::test]
async fn truncated_tool_arguments_replay_the_recorded_raw_string() {
    let server = MockServer::start().await;
    let truncated = "{\"query\":\"par";
    let event = serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","function":{"name":"search","arguments":truncated}}]},"finish_reason":"length"}]});
    let body = format!("data: {event}\n\ndata: [DONE]\n\n");
    common::mount(&server, "/chat/completions", body).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(first.turn.finish.finish_reason, FinishReason::Length);
    let call = first
        .turn
        .message
        .content
        .iter()
        .find_map(|part| match part {
            oven_sdk::AssistantPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("finalized tool call");
    assert!(call.input.is_null());
    assert_eq!(call.raw_input.as_deref(), Some(truncated));

    let second = model
        .stream(
            Request::new(vec![
                oven_sdk::HistoryTurn::assistant(first.turn),
                oven_sdk::HistoryTurn::tool(oven_sdk::ToolMessage::new(vec![
                    oven_sdk::ToolResultPart::new(
                        "call",
                        oven_sdk::ToolContent::Text("failed".into()),
                    ),
                ])),
            ]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        second.request.replay.decisions.as_slice(),
        [
            oven_sdk::ReplayDecision {
                disposition: oven_sdk::ReplayDisposition::DiscardedInvalidPayload { .. },
                ..
            },
            oven_sdk::ReplayDecision {
                disposition: oven_sdk::ReplayDisposition::ReconstructedNormalized,
                ..
            }
        ]
    ));
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        body["messages"][0]["tool_calls"][0]["function"]["arguments"], truncated,
        "reconstruction must keep the verbatim recorded argument bytes"
    );
}

#[tokio::test]
async fn clean_eof_without_finish_reason_is_unexpected_eof() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/chat/completions",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n".into(),
    )
    .await;
    let error = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::UnexpectedEof);
    assert!(error.message.contains("ended before the [DONE] marker"));
}

#[tokio::test]
async fn clean_eof_after_finish_reason_without_done_completes() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/chat/completions",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".into(),
    )
    .await;
    let completed = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(completed.turn.finish.finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn fragmented_refusal_becomes_one_custom_part() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"not \"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"allowed\"},\"finish_reason\":\"content_filter\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    common::mount(&server, "/chat/completions", body.into()).await;
    let result = common::official_chat(&server, "gpt-4o-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(result.turn.message.content.iter().any(|part| {
        matches!(part, oven_sdk::AssistantPart::Custom(custom) if custom.kind == "openai.refusal" && custom.data == "not allowed")
    }));
}

#[tokio::test]
async fn mid_stream_error_emits_error_then_finish_error() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"started\"},\"finish_reason\":null}]}\n\n",
        "data: {\"error\":{\"type\":\"server_error\",\"message\":\"failed\"}}\n\n"
    );
    common::mount(&server, "/chat/completions", body.into()).await;
    let mut response = common::official_chat(&server, "gpt-4o-mini")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut parts = Vec::new();
    while let Some(item) = response.stream.next().await {
        parts.push(item.unwrap());
    }
    assert!(matches!(parts[parts.len() - 2], StreamPart::Error { .. }));
    assert!(
        matches!(&parts[parts.len() - 1], StreamPart::Finish { finish } if finish.finish_reason == FinishReason::Error)
    );
}

#[tokio::test]
async fn in_band_error_preserves_http_response_headers() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"started\"},\"finish_reason\":null}]}\n\n",
        "data: {\"error\":{\"type\":\"rate_limit_error\",\"message\":\"daily request limit reached; Bearer stream-secret\"}}\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "text/event-stream")
                .insert_header("x-request-id", "req_in_band")
                .insert_header("retry-after-ms", "125"),
        )
        .mount(&server)
        .await;
    let mut response = common::official_chat(&server, "gpt-4o-mini")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    while let Some(item) = response.stream.next().await {
        if let StreamPart::Error { error } = item.unwrap() {
            assert_eq!(error.diagnostics.request_id.as_deref(), Some("req_in_band"));
            let body = error.diagnostics.sanitized_body.as_ref().unwrap();
            assert!(body.text().contains("daily request limit reached"));
            assert!(!body.truncated());
            assert!(
                !serde_json::to_string(&error)
                    .unwrap()
                    .contains("stream-secret")
            );
            assert_eq!(
                error.diagnostics.retry_after,
                Some(std::time::Duration::from_millis(125))
            );
            return;
        }
    }
    panic!("missing in-band error");
}
