mod common;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AssistantPart, FinishReason, HistoryTurn, LanguageModel, ModelErrorKind,
    ReplayCapability, ReplayPolicy, Request,
};
use oven_sdk_open_responses::OpenResponsesModel;
use wiremock::MockServer;

fn sse(events: Vec<(&str, serde_json::Value)>, done: bool) -> String {
    let mut output = String::new();
    for (name, value) in events {
        output.push_str(&format!("event: {name}\ndata: {value}\n\n"));
    }
    if done {
        output.push_str("data: [DONE]\n\n");
    }
    output
}

fn response_prefix() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "response.created",
            serde_json::json!({"type":"response.created","sequence_number":0,"response":{"id":"resp_1","status":"in_progress","model":"opaque"}}),
        ),
        (
            "response.in_progress",
            serde_json::json!({"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_1","status":"in_progress","model":"opaque"}}),
        ),
    ]
}

fn official_reasoning_stream() -> String {
    let item = serde_json::json!({
        "type":"reasoning","id":"rs_1","status":"completed",
        "summary":[{"type":"summary_text","text":"summary"}],
        "content":[{"type":"reasoning_text","text":"reasoning"}],
        "encrypted_content":null
    });
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", serde_json::json!({"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"reasoning","id":"rs_1","status":"in_progress","summary":[],"content":[],"encrypted_content":null}})),
        ("response.reasoning_summary_part.added", serde_json::json!({"type":"response.reasoning_summary_part.added","sequence_number":3,"item_id":"rs_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}})),
        ("response.reasoning_summary_text.delta", serde_json::json!({"type":"response.reasoning_summary_text.delta","sequence_number":4,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"summary"})),
        ("response.reasoning_summary_text.done", serde_json::json!({"type":"response.reasoning_summary_text.done","sequence_number":5,"item_id":"rs_1","output_index":0,"summary_index":0,"text":"summary"})),
        ("response.reasoning_summary_part.done", serde_json::json!({"type":"response.reasoning_summary_part.done","sequence_number":6,"item_id":"rs_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":"summary"}})),
        ("response.content_part.added", serde_json::json!({"type":"response.content_part.added","sequence_number":7,"item_id":"rs_1","output_index":0,"content_index":0,"part":{"type":"reasoning_text","text":""}})),
        ("response.reasoning.delta", serde_json::json!({"type":"response.reasoning.delta","sequence_number":8,"item_id":"rs_1","output_index":0,"content_index":0,"delta":"reasoning"})),
        ("response.reasoning.done", serde_json::json!({"type":"response.reasoning.done","sequence_number":9,"item_id":"rs_1","output_index":0,"content_index":0,"text":"reasoning"})),
        ("response.content_part.done", serde_json::json!({"type":"response.content_part.done","sequence_number":10,"item_id":"rs_1","output_index":0,"content_index":0,"part":{"type":"reasoning_text","text":"reasoning"}})),
        ("response.output_item.done", serde_json::json!({"type":"response.output_item.done","sequence_number":11,"output_index":0,"item":item.clone()})),
        ("response.completed", serde_json::json!({"type":"response.completed","sequence_number":12,"response":{"id":"resp_1","status":"completed","model":"opaque","output":[item],"usage":null}})),
    ]);
    sse(events, true)
}

fn hugging_face_reasoning_stream() -> String {
    let item = serde_json::json!({
        "type":"reasoning","id":"rs_hf","status":"completed","summary":[],
        "content":[{"type":"reasoning_text","text":"legacy"}]
    });
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", serde_json::json!({"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"reasoning","id":"rs_hf","status":"in_progress","summary":[],"content":[]}})),
        ("response.reasoning_text.delta", serde_json::json!({"type":"response.reasoning_text.delta","sequence_number":3,"item_id":"rs_hf","output_index":0,"content_index":0,"delta":"legacy"})),
        ("response.reasoning_text.done", serde_json::json!({"type":"response.reasoning_text.done","sequence_number":4,"item_id":"rs_hf","output_index":0,"content_index":0,"text":"legacy"})),
        ("response.output_item.done", serde_json::json!({"type":"response.output_item.done","sequence_number":5,"output_index":0,"item":item.clone()})),
        ("response.completed", serde_json::json!({"type":"response.completed","sequence_number":6,"response":{"id":"resp_1","status":"completed","model":"opaque","output":[item],"usage":null}})),
    ]);
    sse(events, true)
}

fn function_stream(include_arguments_done: bool) -> String {
    let item = serde_json::json!({"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"});
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", serde_json::json!({"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}})),
        ("response.function_call_arguments.delta", serde_json::json!({"type":"response.function_call_arguments.delta","sequence_number":3,"item_id":"fc_1","output_index":0,"delta":"{}"})),
    ]);
    let mut sequence = 4;
    if include_arguments_done {
        events.push(("response.function_call_arguments.done", serde_json::json!({"type":"response.function_call_arguments.done","sequence_number":sequence,"item_id":"fc_1","output_index":0,"arguments":"{}"})));
        sequence += 1;
    }
    events.extend([
        ("response.output_item.done", serde_json::json!({"type":"response.output_item.done","sequence_number":sequence,"output_index":0,"item":item.clone()})),
        ("response.completed", serde_json::json!({"type":"response.completed","sequence_number":sequence + 1,"response":{"id":"resp_1","status":"completed","model":"opaque","output":[item],"usage":null}})),
    ]);
    sse(events, true)
}

fn incomplete_stream(response_status: &str, include_reason: bool) -> String {
    let item = serde_json::json!({"type":"message","id":"msg_incomplete","status":"incomplete","role":"assistant","content":[]});
    let mut response = serde_json::json!({
        "id":"resp_1","status":response_status,"model":"opaque","output":[item.clone()],"usage":null
    });
    if include_reason {
        response["incomplete_details"] = serde_json::json!({"reason":"max_output_tokens"});
    }
    let terminal = if response_status == "incomplete" {
        "response.incomplete"
    } else {
        "response.completed"
    };
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", serde_json::json!({"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"message","id":"msg_incomplete","status":"in_progress","role":"assistant","content":[]}})),
        ("response.output_item.done", serde_json::json!({"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":item})),
        (terminal, serde_json::json!({"type":terminal,"sequence_number":4,"response":response})),
    ]);
    sse(events, true)
}

fn phase_text_stream() -> String {
    let item = serde_json::json!({
        "type":"message","id":"msg_phase","status":"completed","role":"assistant",
        "phase":"commentary","content":[{"type":"output_text","text":"phase","annotations":[]}]
    });
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", serde_json::json!({"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"message","id":"msg_phase","status":"in_progress","role":"assistant","phase":"commentary","content":[]}})),
        ("response.content_part.added", serde_json::json!({"type":"response.content_part.added","sequence_number":3,"item_id":"msg_phase","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}})),
        ("response.output_text.delta", serde_json::json!({"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_phase","output_index":0,"content_index":0,"delta":"phase"})),
        ("response.output_text.done", serde_json::json!({"type":"response.output_text.done","sequence_number":5,"item_id":"msg_phase","output_index":0,"content_index":0,"text":"phase"})),
        ("response.content_part.done", serde_json::json!({"type":"response.content_part.done","sequence_number":6,"item_id":"msg_phase","output_index":0,"content_index":0,"part":{"type":"output_text","text":"phase","annotations":[]}})),
        ("response.output_item.done", serde_json::json!({"type":"response.output_item.done","sequence_number":7,"output_index":0,"item":item.clone()})),
        ("response.completed", serde_json::json!({"type":"response.completed","sequence_number":8,"response":{"id":"resp_1","status":"completed","model":"opaque","output":[item],"usage":null}})),
    ]);
    sse(events, true)
}

fn mixed_refusal_stream(refusal_first: bool) -> String {
    let ordered = if refusal_first {
        [("refusal", "blocked"), ("output_text", "after")]
    } else {
        [("output_text", "before"), ("refusal", "blocked")]
    };
    let content = ordered
        .iter()
        .map(|(kind, text)| {
            if *kind == "refusal" {
                serde_json::json!({"type":"refusal","refusal":text})
            } else {
                serde_json::json!({"type":"output_text","text":text,"annotations":[]})
            }
        })
        .collect::<Vec<_>>();
    let item = serde_json::json!({
        "type":"message","id":"msg_mixed","status":"completed","role":"assistant",
        "content":content
    });
    let mut events = response_prefix();
    events.push(("response.output_item.added", serde_json::json!({"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"message","id":"msg_mixed","status":"in_progress","role":"assistant","content":[]}})));
    let mut sequence = 3_u64;
    for (index, (kind, text)) in ordered.iter().enumerate() {
        let (delta_event, done_event, part) = if *kind == "refusal" {
            (
                "response.refusal.delta",
                "response.refusal.done",
                serde_json::json!({"type":"refusal","refusal":""}),
            )
        } else {
            (
                "response.output_text.delta",
                "response.output_text.done",
                serde_json::json!({"type":"output_text","text":"","annotations":[]}),
            )
        };
        events.push(("response.content_part.added", serde_json::json!({"type":"response.content_part.added","sequence_number":sequence,"item_id":"msg_mixed","output_index":0,"content_index":index,"part":part})));
        sequence += 1;
        events.push((delta_event, serde_json::json!({"type":delta_event,"sequence_number":sequence,"item_id":"msg_mixed","output_index":0,"content_index":index,"delta":text})));
        sequence += 1;
        let done = if *kind == "refusal" {
            serde_json::json!({"type":done_event,"sequence_number":sequence,"item_id":"msg_mixed","output_index":0,"content_index":index,"refusal":text})
        } else {
            serde_json::json!({"type":done_event,"sequence_number":sequence,"item_id":"msg_mixed","output_index":0,"content_index":index,"text":text})
        };
        events.push((done_event, done));
        sequence += 1;
        let final_part = if *kind == "refusal" {
            serde_json::json!({"type":"refusal","refusal":text})
        } else {
            serde_json::json!({"type":"output_text","text":text,"annotations":[]})
        };
        events.push(("response.content_part.done", serde_json::json!({"type":"response.content_part.done","sequence_number":sequence,"item_id":"msg_mixed","output_index":0,"content_index":index,"part":final_part})));
        sequence += 1;
    }
    if refusal_first {
        let lifecycle = events.split_off(3);
        events.push(lifecycle[0].clone());
        events.extend(lifecycle[4..].iter().cloned());
        events.extend(lifecycle[1..4].iter().cloned());
    }
    for (sequence_number, (_, value)) in events.iter_mut().enumerate() {
        value["sequence_number"] = (sequence_number as u64).into();
    }
    sequence = events.len() as u64;
    events.extend([
        ("response.output_item.done", serde_json::json!({"type":"response.output_item.done","sequence_number":sequence,"output_index":0,"item":item.clone()})),
        ("response.completed", serde_json::json!({"type":"response.completed","sequence_number":sequence + 1,"response":{"id":"resp_1","status":"completed","model":"opaque","output":[item],"usage":null}})),
    ]);
    sse(events, true)
}

fn function_output_stream(output: serde_json::Value) -> String {
    let item = serde_json::json!({
        "type":"function_call_output","id":"fco_1","call_id":"call_1",
        "status":"completed","output":output
    });
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", serde_json::json!({"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"function_call_output","id":"fco_1","call_id":"call_1","status":"in_progress","output":""}})),
        ("response.output_item.done", serde_json::json!({"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":item.clone()})),
        ("response.completed", serde_json::json!({"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","status":"completed","model":"opaque","output":[item],"usage":null}})),
    ]);
    sse(events, true)
}

async fn stream_error(model: &OpenResponsesModel) -> oven_sdk::ModelError {
    let mut response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    while let Some(item) = response.stream.next().await {
        if let Err(error) = item {
            return error;
        }
    }
    panic!("expected stream error")
}

#[tokio::test]
async fn official_reasoning_and_summary_lifecycles_are_normalized_in_order() {
    let server = MockServer::start().await;
    common::mount(&server, official_reasoning_stream()).await;
    let completed = common::generic_model(&server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let reasoning = completed
        .turn
        .message
        .content
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(reasoning.len(), 2);
    assert_eq!(reasoning[0].text, "summary");
    assert_eq!(
        reasoning[0].metadata.as_ref().unwrap()["open_responses.kind"],
        "summary_text"
    );
    assert_eq!(reasoning[1].text, "reasoning");
    assert_eq!(
        reasoning[1].metadata.as_ref().unwrap()["open_responses.kind"],
        "reasoning_text"
    );
}

#[tokio::test]
async fn assistant_phase_is_preserved_in_normalized_text_metadata() {
    let server = MockServer::start().await;
    common::mount(&server, phase_text_stream()).await;
    let completed = common::generic_model(&server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(matches!(
        &completed.turn.message.content[0],
        AssistantPart::Text(text)
            if text.metadata.as_ref().unwrap()["open_responses.phase"] == "commentary"
    ));
}

#[tokio::test]
async fn refusal_content_preserves_before_and_after_text_order_with_replay() {
    for refusal_first in [true, false] {
        let server = MockServer::start().await;
        common::mount(&server, mixed_refusal_stream(refusal_first)).await;
        let model = common::generic_model(&server, "opaque");
        let completed = model
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap();
        let order = completed
            .turn
            .message
            .content
            .iter()
            .filter_map(|part| match part {
                AssistantPart::Text(_) => Some("output_text"),
                AssistantPart::Custom(custom) if custom.kind == "open_responses.refusal" => {
                    Some("refusal")
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let expected = if refusal_first {
            vec!["refusal", "output_text"]
        } else {
            vec!["output_text", "refusal"]
        };
        assert_eq!(order, expected);

        model
            .complete(
                Request::new(vec![HistoryTurn::assistant(completed.turn)]),
                AbortSignal::default(),
            )
            .await
            .unwrap();
        let requests = server.received_requests().await.unwrap();
        let replay_body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
        let replay_order = replay_body["input"][0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|part| part["type"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(replay_order, expected);
    }
}

#[tokio::test]
async fn function_output_media_is_validated_before_normalization() {
    for output in [
        serde_json::json!([{"type":"input_image","image_url":"data:text/plain;base64,AAAA"}]),
        serde_json::json!([{"type":"input_file","file_url":"ftp://example.com/file.pdf"}]),
    ] {
        let server = MockServer::start().await;
        common::mount(&server, function_output_stream(output)).await;
        let error = stream_error(&common::generic_model(&server, "opaque")).await;
        assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
    }
}

#[tokio::test]
async fn lifecycle_markers_status_and_sequence_are_optional() {
    let item = serde_json::json!({"type":"message","id":"msg_1","role":"assistant","content":[]});
    let body = sse(
        vec![
            (
                "wrong-name",
                serde_json::json!({"type":"response.output_item.added","output_index":0,"item":item.clone()}),
            ),
            (
                "also-wrong",
                serde_json::json!({"type":"response.output_item.done","sequence_number":9,"output_index":0,"item":item.clone()}),
            ),
            (
                "terminal-name",
                serde_json::json!({"type":"response.completed","sequence_number":42,"response":{"output":[{"type":"message","id":"different","role":"assistant","content":[]}]}}),
            ),
        ],
        true,
    );
    let server = MockServer::start().await;
    common::mount(&server, body).await;
    let completed = common::generic_model(&server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(completed.turn.finish.native_replay.is_some());
}

#[tokio::test]
async fn hugging_face_legacy_reasoning_events_require_the_explicit_profile() {
    let hf_server = MockServer::start().await;
    common::mount(&hf_server, hugging_face_reasoning_stream()).await;
    let completed = common::hugging_face_model(&hf_server, "org/model:provider")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(matches!(
        &completed.turn.message.content[0],
        AssistantPart::Reasoning(reasoning) if reasoning.text == "legacy"
    ));

    let generic_server = MockServer::start().await;
    common::mount(&generic_server, hugging_face_reasoning_stream()).await;
    let error = stream_error(&common::generic_model(&generic_server, "opaque")).await;
    assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
}

#[tokio::test]
async fn item_identity_stays_bound_but_function_arguments_done_is_optional() {
    let identity_server = MockServer::start().await;
    common::mount(
        &identity_server,
        common::text_stream("ok").replacen("\"item_id\":\"msg_1\"", "\"item_id\":\"other\"", 1),
    )
    .await;
    let error = stream_error(&common::generic_model(&identity_server, "opaque")).await;
    assert_eq!(error.kind, ModelErrorKind::InvalidResponse);

    let missing_server = MockServer::start().await;
    common::mount(&missing_server, function_stream(false)).await;
    let completed = common::generic_model(&missing_server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(completed.turn.finish.finish_reason, FinishReason::ToolCalls);

    let valid_server = MockServer::start().await;
    common::mount(&valid_server, function_stream(true)).await;
    let completed = common::generic_model(&valid_server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(completed.turn.finish.finish_reason, FinishReason::ToolCalls);
}

#[tokio::test]
async fn eof_after_queued_parts_is_still_unexpected_eof() {
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", serde_json::json!({"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"message","id":"msg_1","status":"in_progress","role":"assistant","content":[]}})),
        ("response.content_part.added", serde_json::json!({"type":"response.content_part.added","sequence_number":3,"item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}})),
        ("response.output_text.delta", serde_json::json!({"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"queued"})),
    ]);
    let server = MockServer::start().await;
    common::mount(&server, sse(events, false)).await;
    let error = stream_error(&common::generic_model(&server, "opaque")).await;
    assert_eq!(error.kind, ModelErrorKind::UnexpectedEof);
}

#[tokio::test]
async fn incomplete_items_accept_missing_or_crossed_status_and_reason() {
    let valid_server = MockServer::start().await;
    common::mount(&valid_server, incomplete_stream("incomplete", true)).await;
    let completed = common::generic_model(&valid_server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(completed.turn.finish.finish_reason, FinishReason::Length);

    for (status, reason) in [("completed", false), ("incomplete", false)] {
        let server = MockServer::start().await;
        common::mount(&server, incomplete_stream(status, reason)).await;
        common::generic_model(&server, "opaque")
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn optional_replay_omits_oversize_artifact_but_required_replay_fails() {
    let text = "x".repeat(2 * 1024 * 1024);
    let optional_server = MockServer::start().await;
    common::mount(&optional_server, common::text_stream(&text)).await;
    let completed = common::generic_model(&optional_server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(completed.turn.finish.native_replay.is_none());

    let required_server = MockServer::start().await;
    common::mount(&required_server, common::text_stream(&text)).await;
    let mut config = common::generic_config(&required_server, "opaque");
    config.model.capabilities.replay.policy = ReplayPolicy::IfValid;
    config.model.capabilities.replay.capability = ReplayCapability::Required;
    let error = stream_error(&OpenResponsesModel::new(config).unwrap()).await;
    assert_eq!(error.kind, ModelErrorKind::Replay);
}

#[tokio::test]
async fn sparse_content_index_reconciles_once_and_captures_valid_replay() {
    let item = serde_json::json!({
        "type":"message","id":"msg_sparse","role":"assistant",
        "content":[{"type":"output_text","text":"once","annotations":[]}]
    });
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_sparse","role":"assistant","content":[]}})),
        ("response.content_part.added", serde_json::json!({"type":"response.content_part.added","output_index":0,"content_index":2,"item_id":"msg_sparse","part":{"type":"output_text","text":"","annotations":[]}})),
        ("response.output_text.delta", serde_json::json!({"type":"response.output_text.delta","output_index":0,"content_index":2,"item_id":"msg_sparse","delta":"once"})),
        ("response.output_item.done", serde_json::json!({"type":"response.output_item.done","output_index":0,"item":item.clone()})),
        ("response.completed", serde_json::json!({"type":"response.completed","response":{"output":[item]}})),
    ]);
    let server = MockServer::start().await;
    common::mount(&server, sse(events, true)).await;
    let model = common::generic_model(&server, "opaque");
    let completed = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(completed.turn.text(), "once");
    assert_eq!(
        completed
            .turn
            .message
            .content
            .iter()
            .filter(|part| matches!(part, AssistantPart::Text(_)))
            .count(),
        1
    );
    assert!(completed.turn.finish.native_replay.is_some());
    model
        .complete(
            Request::new(vec![HistoryTurn::assistant(completed.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn open_refusal_is_closed_without_losing_or_duplicating_content() {
    let item = serde_json::json!({
        "type":"message","id":"msg_refusal","role":"assistant",
        "content":[{"type":"refusal","refusal":"blocked"}]
    });
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_refusal","role":"assistant","content":[]}})),
        ("response.content_part.added", serde_json::json!({"type":"response.content_part.added","output_index":0,"content_index":4,"item_id":"msg_refusal","part":{"type":"refusal","refusal":""}})),
        ("response.refusal.delta", serde_json::json!({"type":"response.refusal.delta","output_index":0,"content_index":4,"item_id":"msg_refusal","delta":"blocked"})),
        ("response.output_item.done", serde_json::json!({"type":"response.output_item.done","output_index":0,"item":item.clone()})),
        ("response.completed", serde_json::json!({"type":"response.completed","response":{"output":[item]}})),
    ]);
    let server = MockServer::start().await;
    common::mount(&server, sse(events, true)).await;
    let completed = common::generic_model(&server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let refusals = completed
        .turn
        .message
        .content
        .iter()
        .filter(|part| matches!(part, AssistantPart::Custom(custom) if custom.data == "blocked"))
        .count();
    assert_eq!(refusals, 1);
    assert!(completed.turn.finish.native_replay.is_some());
}

#[tokio::test]
async fn identical_sparse_parts_consume_distinct_terminal_annotations() {
    let annotation = |title: &str| {
        serde_json::json!({
            "type":"url_citation","start_index":0,"end_index":4,
            "url":format!("https://example.com/{title}"),"title":title
        })
    };
    let item = serde_json::json!({
        "type":"message","id":"msg_identical","role":"assistant","content":[
            {"type":"output_text","text":"same","annotations":[annotation("first")]},
            {"type":"output_text","text":"same","annotations":[annotation("second")]}
        ]
    });
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_identical","role":"assistant","content":[]}})),
        ("response.content_part.added", serde_json::json!({"type":"response.content_part.added","output_index":0,"content_index":1,"item_id":"msg_identical","part":{"type":"output_text","text":"","annotations":[]}})),
        ("response.output_text.delta", serde_json::json!({"type":"response.output_text.delta","output_index":0,"content_index":1,"item_id":"msg_identical","delta":"same"})),
        ("response.content_part.added", serde_json::json!({"type":"response.content_part.added","output_index":0,"content_index":3,"item_id":"msg_identical","part":{"type":"output_text","text":"","annotations":[]}})),
        ("response.output_text.delta", serde_json::json!({"type":"response.output_text.delta","output_index":0,"content_index":3,"item_id":"msg_identical","delta":"same"})),
        ("response.output_item.done", serde_json::json!({"type":"response.output_item.done","output_index":0,"item":item.clone()})),
        ("response.completed", serde_json::json!({"type":"response.completed","response":{"output":[item]}})),
    ]);
    let server = MockServer::start().await;
    common::mount(&server, sse(events, true)).await;
    let completed = common::generic_model(&server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(
        completed
            .turn
            .message
            .content
            .iter()
            .filter(|part| matches!(part, AssistantPart::Text(text) if text.text == "same"))
            .count(),
        2
    );
    let titles = completed
        .turn
        .message
        .content
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Source(source) => source.title.as_deref(),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(titles, ["first", "second"]);
    let replay = completed.turn.finish.native_replay.unwrap();
    assert_eq!(
        replay.payload()["items"][0]["content"][0]["annotations"][0]["title"],
        "first"
    );
    assert_eq!(
        replay.payload()["items"][0]["content"][1]["annotations"][0]["title"],
        "second"
    );
}

#[tokio::test]
async fn duplicate_function_ids_are_disambiguated_before_collection() {
    let added = |index: usize, id: &str, name: &str| {
        serde_json::json!({
            "type":"response.output_item.added","output_index":index,
            "item":{"type":"function_call","id":id,"call_id":"same","name":name,"arguments":"{}"}
        })
    };
    let done = |index: usize, id: &str, name: &str| {
        serde_json::json!({
            "type":"response.output_item.done","output_index":index,
            "item":{"type":"function_call","id":id,"call_id":"same","name":name,"arguments":"{}"}
        })
    };
    let terminal = serde_json::json!([
        {"type":"function_call","id":"fc_a","call_id":"same","name":"a","arguments":"{}"},
        {"type":"function_call","id":"fc_b","call_id":"same","name":"b","arguments":"{}"}
    ]);
    let mut events = response_prefix();
    events.extend([
        ("response.output_item.added", added(0, "fc_a", "a")),
        ("response.output_item.done", done(0, "fc_a", "a")),
        ("response.output_item.added", added(1, "fc_b", "b")),
        ("response.output_item.done", done(1, "fc_b", "b")),
        (
            "response.completed",
            serde_json::json!({"type":"response.completed","response":{"output":terminal}}),
        ),
    ]);
    let server = MockServer::start().await;
    common::mount(&server, sse(events, true)).await;
    let completed = common::generic_model(&server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let ids = completed
        .turn
        .message
        .content
        .iter()
        .filter_map(|part| match part {
            AssistantPart::ToolCall(call) => Some(call.id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(ids, ["same", "same-1"]);
    assert!(completed.turn.finish.native_replay.is_some());
}
