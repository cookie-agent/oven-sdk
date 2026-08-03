pub mod common;

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, FinishReason, LanguageModel, ModelErrorKind, Request, StreamPart};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn full_item_only_message_synthesizes_complete_text_lifecycle() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_full\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"full text\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg_full\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"full text\"}]}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let mut response = common::official_responses(&server, "gpt-5-mini")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut parts = Vec::new();
    while let Some(item) = response.stream.next().await {
        parts.push(item.unwrap());
    }
    assert!(matches!(parts[1], StreamPart::TextStart { ref id, .. } if id == "msg_full"));
    assert!(
        matches!(parts[2], StreamPart::TextDelta { ref id, ref delta, .. } if id == "msg_full" && delta == "full text")
    );
    assert!(matches!(parts[3], StreamPart::TextEnd { ref id, .. } if id == "msg_full"));
}

#[tokio::test]
async fn full_item_only_reasoning_synthesizes_complete_reasoning_lifecycle() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_full\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"full thought\"}],\"encrypted_content\":\"opaque\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"reasoning\",\"id\":\"rs_full\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"full thought\"}],\"encrypted_content\":\"opaque\"}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let mut response = common::official_responses(&server, "gpt-5-mini")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut parts = Vec::new();
    while let Some(item) = response.stream.next().await {
        parts.push(item.unwrap());
    }
    assert!(
        matches!(parts[1], StreamPart::ReasoningStart { ref id, .. } if id == "rs_full:summary:0")
    );
    assert!(
        matches!(parts[2], StreamPart::ReasoningDelta { ref id, ref delta, .. } if id == "rs_full:summary:0" && delta == "full thought")
    );
    assert!(
        matches!(parts[3], StreamPart::ReasoningEnd { ref id, .. } if id == "rs_full:summary:0")
    );
}

#[tokio::test]
async fn terminal_only_message_synthesizes_once() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg_terminal\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"terminal text\"}]}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let result = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "terminal text");
    assert_eq!(
        result
            .turn
            .message
            .content
            .iter()
            .filter(|part| matches!(part, oven_sdk::AssistantPart::Text(_)))
            .count(),
        1
    );
}

#[tokio::test]
async fn terminal_only_reasoning_synthesizes_once() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"reasoning\",\"id\":\"rs_terminal\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"terminal thought\"}],\"encrypted_content\":\"opaque\"}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let result = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let reasoning = result
        .turn
        .message
        .content
        .iter()
        .filter_map(|part| match part {
            oven_sdk::AssistantPart::Reasoning(reasoning) => Some(reasoning.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(reasoning, ["terminal thought"]);
}

#[tokio::test]
async fn terminal_only_function_synthesizes_once() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"id\":\"fc_terminal\",\"call_id\":\"call_terminal\",\"name\":\"lookup\",\"arguments\":\"{\\\"x\\\":1}\"}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let result = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let calls = result
        .turn
        .message
        .content
        .iter()
        .filter_map(|part| match part {
            oven_sdk::AssistantPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_terminal");
    assert_eq!(calls[0].provider_item_id.as_deref(), Some("fc_terminal"));
}

#[tokio::test]
async fn differing_terminal_message_is_invalid_response() {
    assert_authoritative_mismatch(concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"done text\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"different text\"}]}]}}\n\n"
    ))
    .await;
}

#[tokio::test]
async fn differing_terminal_reasoning_is_invalid_response() {
    assert_authoritative_mismatch(concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"done thought\"}],\"encrypted_content\":\"opaque\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"different thought\"}],\"encrypted_content\":\"opaque\"}]}}\n\n"
    ))
    .await;
}

#[tokio::test]
async fn differing_terminal_function_is_invalid_response() {
    assert_authoritative_mismatch(concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc\",\"call_id\":\"call\",\"name\":\"lookup\",\"arguments\":\"{\\\"x\\\":1}\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"id\":\"fc\",\"call_id\":\"call\",\"name\":\"lookup\",\"arguments\":\"{\\\"x\\\":2}\"}]}}\n\n"
    ))
    .await;
}

async fn assert_authoritative_mismatch(body: &str) {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", body.into()).await;
    let error = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::StreamFinalize
    );
}

#[tokio::test]
async fn text_deltas_are_written_into_replay_items() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("hello")).await;
    let result = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "hello");
    let artifact = result.turn.finish.native_replay.unwrap();
    assert_eq!(
        artifact
            .payload()
            .pointer("/items/0/content/0/text")
            .and_then(serde_json::Value::as_str),
        Some("hello")
    );
}

#[tokio::test]
async fn function_call_uses_call_id_and_provider_item_id() {
    let server = MockServer::start().await;
    let body = concat!(
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp\"}}\n\n",
        "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n",
        "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"x\\\":1}\"}\n\n",
        "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"x\\\":1}\"}}\n\n",
        "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"x\\\":1}\"}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let result = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let call = result
        .turn
        .message
        .content
        .iter()
        .find_map(|part| match part {
            oven_sdk::AssistantPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .unwrap();
    assert_eq!(call.id, "call_1");
    assert_eq!(call.provider_item_id.as_deref(), Some("fc_1"));
    assert_eq!(result.turn.finish.finish_reason, FinishReason::ToolCalls);
}

#[tokio::test]
async fn reasoning_summary_indices_have_stable_lifecycle() {
    let server = MockServer::start().await;
    let body = concat!(
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[]}}\n\n",
        "event: response.reasoning_summary_text.delta\ndata: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":1,\"delta\":\"thought\"}\n\n",
        "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"\"},{\"type\":\"summary_text\",\"text\":\"thought\"}],\"encrypted_content\":\"opaque\"}}\n\n",
        "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"\"},{\"type\":\"summary_text\",\"text\":\"thought\"}],\"encrypted_content\":\"opaque\"}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let result = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let reasoning = result
        .turn
        .message
        .content
        .iter()
        .filter_map(|part| match part {
            oven_sdk::AssistantPart::Reasoning(reasoning) => Some(reasoning.text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(reasoning, "thought");
    assert_eq!(
        result
            .turn
            .finish
            .native_replay
            .unwrap()
            .payload()
            .pointer("/items/0/encrypted_content")
            .and_then(serde_json::Value::as_str),
        Some("opaque")
    );
}

#[tokio::test]
async fn incomplete_reasons_map_to_finish_reasons() {
    for (wire, expected) in [
        ("max_output_tokens", FinishReason::Length),
        ("content_filter", FinishReason::ContentFilter),
    ] {
        let server = MockServer::start().await;
        let body = format!(
            "event: response.created\ndata: {{\"type\":\"response.created\",\"response\":{{}}}}\n\nevent: response.incomplete\ndata: {{\"type\":\"response.incomplete\",\"response\":{{\"status\":\"incomplete\",\"output\":[],\"incomplete_details\":{{\"reason\":{wire:?}}}}}}}\n\n"
        );
        common::mount(&server, "/responses", body).await;
        let result = common::official_responses(&server, "gpt-5-mini")
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap();
        assert_eq!(result.turn.finish.finish_reason, expected);
    }
}

#[tokio::test]
async fn eof_before_terminal_event_is_unexpected_eof() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/responses",
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{}}\n\n".into(),
    )
    .await;
    let error = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::UnexpectedEof);
}

#[tokio::test]
async fn malformed_done_arguments_are_invalid_response() {
    let server = MockServer::start().await;
    let body = concat!(
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc\",\"call_id\":\"call\",\"name\":\"tool\"}}\n\n",
        "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc\",\"call_id\":\"call\",\"name\":\"tool\",\"arguments\":\"[\"}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let error = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
}

#[tokio::test]
async fn function_done_arguments_never_substitute_for_missing_item_arguments() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc\",\"call_id\":\"call\",\"name\":\"tool\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"arguments\":\"{}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc\",\"call_id\":\"call\",\"name\":\"tool\"}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let error = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_finalize_error(error);
}

#[tokio::test]
async fn malformed_function_required_fields_return_typed_errors_without_panicking() {
    for (field, value) in [
        ("id", None),
        ("id", Some("")),
        ("call_id", None),
        ("call_id", Some("")),
        ("name", None),
        ("name", Some("")),
        ("arguments", None),
        ("arguments", Some("")),
    ] {
        let server = MockServer::start().await;
        let mut item = serde_json::json!({
            "type":"function_call",
            "id":"fc",
            "call_id":"call",
            "name":"tool",
            "arguments":"{}"
        });
        match value {
            Some(value) => item[field] = value.into(),
            None => {
                item.as_object_mut().unwrap().remove(field);
            }
        }
        let body = format!(
            "data: {{\"type\":\"response.created\",\"response\":{{}}}}\n\ndata: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{item}}}\n\n"
        );
        common::mount(&server, "/responses", body).await;
        let mut response = common::official_responses(&server, "gpt-5-mini")
            .stream(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap();
        let mut failure = None;
        while let Some(item) = response.stream.next().await {
            match item {
                Ok(part) => assert!(!matches!(
                    part,
                    StreamPart::ToolCallEnd { .. }
                        | StreamPart::ToolCall { .. }
                        | StreamPart::Finish { .. }
                )),
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        assert_finalize_error(failure.expect("malformed function item must fail"));
    }
}

fn assert_finalize_error(error: oven_sdk::ModelError) {
    assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::StreamFinalize
    );
    assert!(error.diagnostics.bytes_received > 0);
}

#[tokio::test]
async fn later_reasoning_summary_index_synthesizes_prior_empty_slots_immediately() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":2,\"delta\":\"third\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"\"},{\"type\":\"summary_text\",\"text\":\"\"},{\"type\":\"summary_text\",\"text\":\"third\"}] }]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let mut response = common::official_responses(&server, "gpt-5-mini")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut parts = Vec::new();
    while let Some(part) = response.stream.next().await {
        parts.push(part.unwrap());
    }
    assert!(matches!(&parts[1], StreamPart::ReasoningStart { id, .. } if id == "rs:summary:0"));
    assert!(matches!(&parts[2], StreamPart::ReasoningEnd { id, .. } if id == "rs:summary:0"));
    assert!(matches!(&parts[3], StreamPart::ReasoningStart { id, .. } if id == "rs:summary:1"));
    assert!(matches!(&parts[4], StreamPart::ReasoningEnd { id, .. } if id == "rs:summary:1"));
    assert!(matches!(&parts[5], StreamPart::ReasoningStart { id, .. } if id == "rs:summary:2"));
    assert!(
        matches!(&parts[6], StreamPart::ReasoningDelta { id, delta, .. } if id == "rs:summary:2" && delta == "third")
    );
}

#[tokio::test]
async fn interleaved_reasoning_summary_indices_preserve_index_order() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":0,\"delta\":\"fir\"}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":2,\"delta\":\"third\"}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":0,\"delta\":\"st\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"first\"},{\"type\":\"summary_text\",\"text\":\"\"},{\"type\":\"summary_text\",\"text\":\"third\"}]}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let result = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let reasoning = result
        .turn
        .message
        .content
        .iter()
        .filter_map(|part| match part {
            oven_sdk::AssistantPart::Reasoning(part) => Some(part.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(reasoning, ["first", "", "third"]);
}

#[tokio::test]
async fn terminal_reasoning_must_preserve_synthesized_earlier_empty_slot() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":1,\"delta\":\"later\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"reasoning\",\"id\":\"rs\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"unexpected earlier\"},{\"type\":\"summary_text\",\"text\":\"later\"}]}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let error = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_finalize_error(error);
}

#[tokio::test]
async fn hosted_items_are_captured_without_becoming_client_tool_calls() {
    let server = MockServer::start().await;
    let body = concat!(
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"completed\"}}\n\n",
        "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"completed\"}}\n\n",
        "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"completed\"}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let result = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.finish.finish_reason, FinishReason::Stop);
    assert!(
        !result
            .turn
            .message
            .content
            .iter()
            .any(|part| matches!(part, oven_sdk::AssistantPart::ToolCall(_)))
    );
    assert_eq!(
        result
            .turn
            .finish
            .native_replay
            .unwrap()
            .payload()
            .pointer("/items/0/type")
            .and_then(serde_json::Value::as_str),
        Some("web_search_call")
    );
}

#[tokio::test]
async fn response_failed_after_output_emits_error_then_finish_error() {
    let server = MockServer::start().await;
    let body = concat!(
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg\",\"content\":[]}}\n\n",
        "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"partial\"}\n\n",
        "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"type\":\"server_error\",\"message\":\"failed\"}}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let mut response = common::official_responses(&server, "gpt-5-mini")
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
async fn dispatch_uses_payload_type_when_event_names_are_missing_or_wrong() {
    let server = MockServer::start().await;
    let body = concat!(
        "event: wrong.created\ndata: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"payload routed\"}]}}\n\n",
        "event: not-a-terminal\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"payload routed\"}]}]}}\n\n"
    );
    common::mount(&server, "/responses", body.into()).await;
    let result = common::official_responses(&server, "gpt-5-mini")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "payload routed");
}

#[tokio::test]
async fn in_band_error_preserves_http_response_headers() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow\"}}}\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "text/event-stream")
                .insert_header("x-request-id", "req_in_band")
                .insert_header("retry-after", "3"),
        )
        .mount(&server)
        .await;
    let mut response = common::official_responses(&server, "gpt-5-mini")
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    while let Some(item) = response.stream.next().await {
        if let StreamPart::Error { error } = item.unwrap() {
            assert_eq!(error.diagnostics.request_id.as_deref(), Some("req_in_band"));
            assert_eq!(
                error.diagnostics.retry_after,
                Some(std::time::Duration::from_secs(3))
            );
            return;
        }
    }
    panic!("missing in-band error");
}
