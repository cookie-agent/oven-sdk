use std::time::Duration;

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AssistantPart, ErrorStage, FinishReason, LanguageModel, ModelErrorKind,
    ReplayCapability, Request, StreamPart,
};
use oven_sdk_anthropic::AnthropicTimeouts;
use oven_sdk_conformance::sse::{ChunkPattern, LineEnding, SseEvent, chunk_bytes, encode_sse};
use tokio::{io::AsyncWriteExt, net::TcpListener};

const HEADERS: &str = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nrequest-id: req_header\r\ntransfer-encoding: chunked\r\n\r\n";

fn event(name: &str, data: &str) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, body: &str) {
    socket
        .write_all(format!("{:X}\r\n", body.len()).as_bytes())
        .await
        .unwrap();
    socket.write_all(body.as_bytes()).await.unwrap();
    socket.write_all(b"\r\n").await.unwrap();
    socket.flush().await.unwrap();
}

fn model(address: std::net::SocketAddr) -> oven_sdk_anthropic::AnthropicModel {
    Anthropic::builder()
        .base_url(format!("http://{address}"))
        .build()
        .unwrap()
        .model("claude-sonnet-4-5")
}

async fn scripted_model(body: String) -> oven_sdk_anthropic::AnthropicModel {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = model(listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket.write_all(HEADERS.as_bytes()).await.unwrap();
        write_chunk(&mut socket, &body).await;
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });
    model
}

async fn scripted_aws_model(body: String) -> oven_sdk_anthropic::AnthropicAwsModel {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = AnthropicAws::builder("us-west-2", "workspace")
        .bearer_key("key")
        .base_url(format!("http://{}", listener.local_addr().unwrap()))
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    spawn_scripted_response(listener, body);
    model
}

async fn scripted_minimax_model(body: String) -> oven_sdk_anthropic::MiniMaxModel {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model = MiniMax::builder()
        .base_url(format!("http://{}", listener.local_addr().unwrap()))
        .build()
        .unwrap()
        .model("MiniMax-M3");
    spawn_scripted_response(listener, body);
    model
}

fn spawn_scripted_response(listener: TcpListener, body: String) {
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket.write_all(HEADERS.as_bytes()).await.unwrap();
        write_chunk(&mut socket, &body).await;
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });
}

async fn collect(model: &dyn LanguageModel) -> Vec<Result<StreamPart, oven_sdk::ModelError>> {
    let mut response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut parts = Vec::new();
    while let Some(item) = response.stream.next().await {
        parts.push(item);
    }
    parts
}

async fn assert_reasoning_accepted(model: &dyn LanguageModel) {
    assert_eq!(
        model.capabilities().replay.capability,
        ReplayCapability::Required
    );
    let parts = collect(model).await;
    assert!(!parts.iter().any(Result::is_err));
    assert!(parts.iter().any(
        |part| matches!(part, Ok(StreamPart::Finish { finish }) if finish.native_replay.is_some())
    ));
}

async fn assert_index_failure(model: &dyn LanguageModel) {
    let parts = collect(model).await;
    assert!(
        !parts
            .iter()
            .any(|part| matches!(part, Ok(StreamPart::Finish { .. })))
    );
    let error = parts.into_iter().find_map(Result::err).unwrap();
    assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
    assert!(matches!(
        error.diagnostics.stage,
        ErrorStage::StreamEvent | ErrorStage::StreamFinalize
    ));
    assert!(error.diagnostics.bytes_received > 0);
}

async fn assert_all_protocols_accept_indices(body: String) {
    let direct = scripted_model(body.clone()).await;
    let aws = scripted_aws_model(body.clone()).await;
    let minimax = scripted_minimax_model(body).await;
    for model in [
        &direct as &dyn LanguageModel,
        &aws as &dyn LanguageModel,
        &minimax as &dyn LanguageModel,
    ] {
        let parts = collect(model).await;
        assert!(!parts.iter().any(Result::is_err));
        assert!(
            parts
                .iter()
                .any(|part| matches!(part, Ok(StreamPart::Finish { .. })))
        );
    }
}

fn terminal() -> String {
    [
        event("message_delta", r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":1}}"#),
        event("message_stop", r#"{"type":"message_stop"}"#),
    ]
    .concat()
}

#[test]
fn bom_at_stream_start_is_ignored() {
    let mut parser = oven_sdk_anthropic::sse::Parser::default();
    let mut events = Vec::new();
    for chunk in chunk_bytes(
        b"\xEF\xBB\xBFevent: ping\ndata: {}\n\n",
        &ChunkPattern::OneByte,
    ) {
        events.extend(parser.feed(&chunk).unwrap());
    }
    events.extend(parser.finish().unwrap());
    assert_eq!(events[0].name, "ping");
}

#[test]
fn sse_parser_handles_one_byte_multiline_comments_and_bare_cr() {
    let bytes = encode_sse(
        &[
            SseEvent::comment("activity"),
            SseEvent::named("message_delta", "first").with_data_line("second"),
        ],
        LineEnding::Lf,
    );
    let bare_cr = String::from_utf8(bytes)
        .unwrap()
        .replace('\n', "\r")
        .into_bytes();
    let mut parser = oven_sdk_anthropic::sse::Parser::default();
    let mut events = Vec::new();
    for chunk in chunk_bytes(&bare_cr, &ChunkPattern::OneByte) {
        events.extend(parser.feed(&chunk).unwrap());
    }
    events.extend(parser.finish().unwrap());
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, "message_delta");
    assert_eq!(events[0].data, "first\nsecond");
}

#[tokio::test]
async fn ping_event_resets_stream_idle_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let timeouts = AnthropicTimeouts {
        headers: Duration::from_secs(1),
        credentials: Duration::from_secs(1),
        stream_idle: Duration::from_millis(200),
    };
    let model = Anthropic::builder()
        .base_url(format!("http://{}", listener.local_addr().unwrap()))
        .timeouts(timeouts)
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        socket.write_all(HEADERS.as_bytes()).await.unwrap();
        write_chunk(
            &mut socket,
            &event("message_start", r#"{"type":"message_start","message":{}}"#),
        )
        .await;
        tokio::time::sleep(Duration::from_millis(125)).await;
        write_chunk(&mut socket, &event("ping", r#"{"type":"ping"}"#)).await;
        tokio::time::sleep(Duration::from_millis(125)).await;
        write_chunk(&mut socket, &terminal()).await;
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });
    let parts = collect(&model).await;
    assert!(
        parts
            .iter()
            .any(|part| matches!(part, Ok(StreamPart::Finish { .. })))
    );
    assert!(!parts.iter().any(Result::is_err));
}

#[tokio::test]
async fn interleaved_content_block_starts_preserve_provider_order() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
            ),
            event(
                "content_block_start",
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"thinking"}}"#,
            ),
            event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"sig"}}"#,
            ),
            event(
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            event(
                "content_block_stop",
                r#"{"type":"content_block_stop","index":1}"#,
            ),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let starts = collect(&model)
        .await
        .into_iter()
        .filter_map(Result::ok)
        .filter_map(|part| match part {
            StreamPart::TextStart { id, .. } | StreamPart::ReasoningStart { id, .. } => Some(id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(starts, ["text:0", "thinking:1"]);
}

#[tokio::test]
async fn parallel_tool_blocks_are_tracked_by_index() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"first","name":"one"}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"second","name":"two"}}"#),
            event("content_block_delta", r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"b\":2}"}}"#),
            event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":1}"#),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let calls = collect(&model)
        .await
        .into_iter()
        .filter_map(Result::ok)
        .filter_map(|part| match part {
            StreamPart::ToolCall { tool_call } => Some((tool_call.id, tool_call.input)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        calls,
        [
            ("first".into(), serde_json::json!({"a": 1})),
            ("second".into(), serde_json::json!({"b": 2}))
        ]
    );
}

#[tokio::test]
async fn tool_input_finalizes_only_at_content_block_stop() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call","name":"lookup"}}"#),
            event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"ready\":true}"}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let parts = collect(&model).await;
    let end = parts
        .iter()
        .position(|part| matches!(part, Ok(StreamPart::ToolCallEnd { id, .. }) if id == "call"))
        .unwrap();
    let call = parts
        .iter()
        .position(
            |part| matches!(part, Ok(StreamPart::ToolCall { tool_call }) if tool_call.id == "call"),
        )
        .unwrap();
    assert_eq!(call, end + 1);

    let missing_stop = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"missing","name":"lookup"}}"#),
            event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"ready\":true}"}}"#),
            event("message_stop", r#"{"type":"message_stop"}"#),
        ]
        .concat(),
    )
    .await;
    let parts = collect(&missing_stop).await;
    assert!(!parts.iter().any(Result::is_err));
    assert!(parts.iter().any(
        |part| matches!(part, Ok(StreamPart::ToolCall { tool_call }) if tool_call.id == "missing")
    ));
}

#[tokio::test]
async fn message_stop_after_closed_blocks_emits_finish() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
            ),
            event(
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            terminal(),
        ]
        .concat(),
    )
    .await;
    assert!(
        collect(&model)
            .await
            .into_iter()
            .any(|part| matches!(part, Ok(StreamPart::Finish { .. })))
    );
}

#[tokio::test]
async fn successful_finish_uses_header_request_id() {
    let model = scripted_model(
        [
            event(
                "message_start",
                r#"{"type":"message_start","message":{"request_id":"req_body"}}"#,
            ),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let mut response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(response.response.request_id.as_deref(), Some("req_header"));
    let mut finish = None;
    while let Some(part) = response.stream.next().await {
        if let StreamPart::Finish { finish: value } = part.unwrap() {
            finish = Some(value);
        }
    }
    assert_eq!(
        finish.unwrap().response_metadata["anthropic.request_id"],
        "req_header"
    );
}

#[tokio::test]
async fn message_start_full_tool_block_has_complete_lifecycle() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{"content":[{"type":"tool_use","id":"initial","name":"lookup","input":{"q":"x"}}]}}"#),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let parts = collect(&model).await;
    assert!(
        parts.iter().any(
            |part| matches!(part, Ok(StreamPart::ToolCallStart { id, .. }) if id == "initial")
        )
    );
    assert!(
        parts
            .iter()
            .any(|part| matches!(part, Ok(StreamPart::ToolCallEnd { id, .. }) if id == "initial"))
    );
    assert!(parts.iter().any(|part| matches!(part, Ok(StreamPart::ToolCall { tool_call }) if tool_call.id == "initial" && tool_call.input == serde_json::json!({"q":"x"}))));
}

#[tokio::test]
async fn empty_tool_input_finalizes_to_empty_object() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"empty","name":"lookup"}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
            terminal(),
        ]
        .concat(),
    )
    .await;
    assert!(collect(&model).await.into_iter().any(|part| matches!(part, Ok(StreamPart::ToolCall { tool_call }) if tool_call.input == serde_json::json!({}) && tool_call.raw_input.as_deref() == Some("{}"))));
}

#[tokio::test]
async fn signature_and_redacted_thinking_are_captured_in_replay() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}"#),
            event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reason"}}"#),
            event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque"}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":1}"#),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let parts = collect(&model).await;
    assert!(parts.iter().any(|part| matches!(part, Ok(StreamPart::ReasoningStart { metadata: Some(metadata), .. }) if metadata["anthropic.redacted"] == "opaque")));
    let replay = parts
        .into_iter()
        .find_map(|part| match part.unwrap() {
            StreamPart::Finish { finish } => finish.native_replay,
            _ => None,
        })
        .unwrap();
    assert_eq!(
        replay
            .payload()
            .pointer("/message/content/0/signature")
            .and_then(serde_json::Value::as_str),
        Some("sig")
    );
    assert_eq!(
        replay
            .payload()
            .pointer("/message/content/1/data")
            .and_then(serde_json::Value::as_str),
        Some("opaque")
    );
    assert_eq!(
        replay.payload()["format"],
        "oven.anthropic.messages.assistant.v3"
    );
    assert_eq!(replay.scope().model_id.as_str(), "claude-sonnet-4-5");
}

#[tokio::test]
async fn required_replay_declarations_accept_only_valid_authoritative_reasoning() {
    let signed = [
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}"#),
        event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}"#),
        event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
        terminal(),
    ]
    .concat();
    let redacted = [
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}"#),
        event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
        terminal(),
    ]
    .concat();
    let direct = scripted_model(signed.clone()).await;
    let aws = scripted_aws_model(redacted).await;
    let minimax = scripted_minimax_model(signed).await;
    for model in [
        &direct as &dyn LanguageModel,
        &aws as &dyn LanguageModel,
        &minimax as &dyn LanguageModel,
    ] {
        assert_eq!(
            model.capabilities().replay.capability,
            ReplayCapability::Required
        );
        let parts = collect(model).await;
        assert!(!parts.iter().any(Result::is_err));
        assert!(parts.iter().any(|part| matches!(part, Ok(StreamPart::Finish { finish }) if finish.native_replay.is_some())));
    }
}

#[tokio::test]
async fn unsigned_thinking_and_empty_redacted_data_are_captured_faithfully() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}"#,
            ),
            event(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reason"}}"#,
            ),
            event(
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            event(
                "content_block_start",
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":""}}"#,
            ),
            event(
                "content_block_stop",
                r#"{"type":"content_block_stop","index":1}"#,
            ),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let completed = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();

    assert_eq!(
        completed.turn.finish.native_replay.unwrap().payload(),
        &serde_json::json!({
            "format": "oven.anthropic.messages.assistant.v3",
            "message": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reason", "signature": ""},
                {"type": "redacted_thinking", "data": ""}
            ]},
            "stop_reason": "end_turn",
            "stop_sequence": null
        })
    );
}

#[tokio::test]
async fn non_string_redacted_reasoning_and_minimax_redaction_are_preserved() {
    let redacted = |data: &str| {
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event(
                "content_block_start",
                &format!(r#"{{"type":"content_block_start","index":0,"content_block":{{"type":"redacted_thinking","data":{data}}}}}"#),
            ),
            event(
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            terminal(),
        ]
        .concat()
    };

    assert_reasoning_accepted(&scripted_model(redacted("null")).await).await;
    assert_reasoning_accepted(&scripted_minimax_model(redacted("\"opaque\"")).await).await;
}

#[tokio::test]
async fn noncontiguous_duplicate_and_reused_indices_finish_for_all_protocols() {
    let signed_thinking_without_zero = [
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        event("content_block_start", r#"{"type":"content_block_start","index":1,"content_block":{"type":"thinking"}}"#),
        event("content_block_delta", r#"{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"signed"}}"#),
        event("content_block_stop", r#"{"type":"content_block_stop","index":1}"#),
        event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#),
        terminal(),
    ]
    .concat();
    assert_all_protocols_accept_indices(signed_thinking_without_zero).await;

    let reused_after_stop = [
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        event(
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        ),
        event(
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        event(
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        ),
        terminal(),
    ]
    .concat();
    assert_all_protocols_accept_indices(reused_after_stop).await;

    let duplicate_active_start = [
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        event(
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        ),
        event(
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        ),
        terminal(),
    ]
    .concat();
    assert_all_protocols_accept_indices(duplicate_active_start).await;

    let gap = [
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        event(
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        ),
        event(
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        event(
            "content_block_start",
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"text"}}"#,
        ),
        event(
            "content_block_stop",
            r#"{"type":"content_block_stop","index":2}"#,
        ),
        event(
            "content_block_start",
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text"}}"#,
        ),
        terminal(),
    ]
    .concat();
    assert_all_protocols_accept_indices(gap).await;

    let start_after_terminal = [
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        terminal(),
        event(
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
        ),
    ]
    .concat();
    let direct = scripted_model(start_after_terminal).await;
    assert_index_failure(&direct).await;
}

#[tokio::test]
async fn contiguous_multi_block_native_content_finishes_for_all_protocols() {
    let body = [
        event("message_start", r#"{"type":"message_start","message":{}}"#),
        event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}"#),
        event("content_block_start", r#"{"type":"content_block_start","index":1,"content_block":{"type":"text"}}"#),
        event("content_block_start", r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call","name":"lookup","input":{}}}"#),
        event("content_block_delta", r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}"#),
        event("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}"#),
        event("content_block_stop", r#"{"type":"content_block_stop","index":2}"#),
        event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
        event("content_block_stop", r#"{"type":"content_block_stop","index":1}"#),
        terminal(),
    ]
    .concat();
    let direct = scripted_model(body.clone()).await;
    let aws = scripted_aws_model(body.clone()).await;
    let minimax = scripted_minimax_model(body).await;
    for model in [
        &direct as &dyn LanguageModel,
        &aws as &dyn LanguageModel,
        &minimax as &dyn LanguageModel,
    ] {
        let completed = model
            .complete(Request::new(Vec::new()), AbortSignal::default())
            .await
            .unwrap();
        let normalized = completed
            .turn
            .message
            .content
            .iter()
            .map(|part| match part {
                AssistantPart::Reasoning(_) => "thinking",
                AssistantPart::Text(_) => "text",
                AssistantPart::ToolCall(_) => "tool_use",
                other => panic!("unexpected normalized part: {other:?}"),
            })
            .collect::<Vec<_>>();
        let replay = completed.turn.finish.native_replay.unwrap();
        let content = replay
            .payload()
            .pointer("/message/content")
            .and_then(serde_json::Value::as_array)
            .unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[2]["type"], "tool_use");
        assert!(content.iter().all(|block| !block.is_null()));
        assert_eq!(normalized, ["thinking", "text", "tool_use"]);
        assert_eq!(
            content
                .iter()
                .map(|block| block["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            normalized
        );
    }
}

#[tokio::test]
async fn sparse_out_of_order_indices_keep_start_order_in_normalized_and_replay_content() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":5,"content_block":{"type":"text"}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":1,"content_block":{"type":"text"}}"#),
            event("content_block_delta", r#"{"type":"content_block_delta","index":5,"delta":{"type":"text_delta","text":"first"}}"#),
            event("content_block_delta", r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"second"}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":1}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":5}"#),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let completed = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let texts = completed
        .turn
        .message
        .content
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(texts, ["first", "second"]);
    let replay = completed.turn.finish.native_replay.unwrap();
    let content = replay.payload().pointer("/message/content").unwrap();
    assert_eq!(content[0]["text"], "first");
    assert_eq!(content[1]["text"], "second");
}

#[tokio::test]
async fn unknown_content_block_lifecycle_is_forwarded_without_failing() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":7,"content_block":{"type":"future_block","value":"start"}}"#),
            event("content_block_delta", r#"{"type":"content_block_delta","index":7,"delta":{"type":"future_delta","value":"delta"}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":7}"#),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let parts = collect(&model).await;
    assert!(!parts.iter().any(Result::is_err));
    assert_eq!(
        parts
            .iter()
            .filter(|part| matches!(part, Ok(StreamPart::ProviderEvent { .. })))
            .count(),
        3
    );
    assert!(parts.iter().any(
        |part| matches!(part, Ok(StreamPart::Finish { finish }) if finish.native_replay.is_some())
    ));
}

#[tokio::test]
async fn duplicate_and_synthesized_tool_ids_are_disambiguated_in_replay() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"same","name":"a","input":{}}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"same","name":"b","input":{}}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":1}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","name":"c","input":{}}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":2}"#),
            event("content_block_start", r#"{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"google-call-2","name":"d","input":{}}}"#),
            event("content_block_stop", r#"{"type":"content_block_stop","index":3}"#),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let completed = model
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
    assert_eq!(ids, ["same", "same-1", "google-call-2", "google-call-2-1"]);
    let replay = completed.turn.finish.native_replay.unwrap();
    let replay_ids = replay.payload()["message"]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(replay_ids, ids);
}

#[tokio::test]
async fn stream_emits_exactly_one_finish() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            terminal(),
        ]
        .concat(),
    )
    .await;
    let finishes = collect(&model)
        .await
        .into_iter()
        .filter(|part| matches!(part, Ok(StreamPart::Finish { .. })))
        .count();
    assert_eq!(finishes, 1);
}

#[tokio::test]
async fn usage_includes_cache_read_and_creation_tokens() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{"usage":{"input_tokens":2}}}"#),
            event("message_delta", r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":2,"cache_read_input_tokens":3,"cache_creation_input_tokens":5,"output_tokens":7}}"#),
            event("message_stop", r#"{"type":"message_stop"}"#),
        ]
        .concat(),
    )
    .await;
    let finish = collect(&model)
        .await
        .into_iter()
        .find_map(|part| match part.unwrap() {
            StreamPart::Finish { finish } => Some(finish),
            _ => None,
        })
        .unwrap();
    assert_eq!(finish.usage.input_tokens, Some(10));
    assert_eq!(finish.usage.input_tokens_no_cache, Some(2));
    assert_eq!(finish.usage.input_tokens_cache_read, Some(3));
    assert_eq!(finish.usage.input_tokens_cache_write, Some(5));
}

#[tokio::test]
async fn output_only_usage_delta_preserves_message_start_input_and_cache_fields() {
    let model = scripted_model(
        [
            event(
                "message_start",
                r#"{"type":"message_start","message":{"usage":{"input_tokens":2,"cache_read_input_tokens":3,"cache_creation_input_tokens":5}}}"#,
            ),
            event(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#,
            ),
            event("message_stop", r#"{"type":"message_stop"}"#),
        ]
        .concat(),
    )
    .await;
    let finish = collect(&model)
        .await
        .into_iter()
        .find_map(|part| match part.unwrap() {
            StreamPart::Finish { finish } => Some(finish),
            _ => None,
        })
        .unwrap();
    assert_eq!(finish.usage.input_tokens, Some(10));
    assert_eq!(finish.usage.input_tokens_no_cache, Some(2));
    assert_eq!(finish.usage.input_tokens_cache_read, Some(3));
    assert_eq!(finish.usage.input_tokens_cache_write, Some(5));
    assert_eq!(finish.usage.output_tokens, Some(7));
}

#[tokio::test]
async fn unreasonable_block_indices_fail_without_allocation() {
    let model = scripted_model(
        [
            event("message_start", r#"{"type":"message_start","message":{}}"#),
            event(
                "content_block_start",
                r#"{"type":"content_block_start","index":18446744073709551615,"content_block":{"type":"text"}}"#,
            ),
        ]
        .concat(),
    )
    .await;
    let error = collect(&model)
        .await
        .into_iter()
        .find_map(Result::err)
        .unwrap();
    assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
    assert_eq!(error.diagnostics.stage, oven_sdk::ErrorStage::StreamEvent);
    assert!(error.diagnostics.bytes_received > 0);
}

#[tokio::test]
async fn provider_usage_overflow_and_inconsistent_breakdown_are_invalid_responses() {
    for body in [
        [
            event("message_start", r#"{"type":"message_start","message":{"usage":{"input_tokens":18446744073709551615}}}"#),
            event("message_delta", r#"{"type":"message_delta","delta":{},"usage":{"cache_read_input_tokens":1}}"#),
        ]
        .concat(),
        event(
            "message_start",
            r#"{"type":"message_start","message":{"usage":{"output_tokens":1,"output_tokens_details":{"thinking_tokens":2}}}}"#,
        ),
    ] {
        let model = scripted_model(body).await;
        let error = collect(&model)
            .await
            .into_iter()
            .find_map(Result::err)
            .unwrap();
        assert_eq!(error.kind, ModelErrorKind::InvalidResponse);
        assert_eq!(error.diagnostics.stage, oven_sdk::ErrorStage::StreamEvent);
        assert!(error.diagnostics.bytes_received > 0);
    }
}

#[tokio::test]
async fn clean_eof_before_message_stop_returns_unexpected_eof() {
    let model = scripted_model(event(
        "message_start",
        r#"{"type":"message_start","message":{}}"#,
    ))
    .await;
    let parts = collect(&model).await;
    assert!(
        matches!(parts.last(), Some(Err(error)) if error.kind == ModelErrorKind::UnexpectedEof)
    );
}

#[tokio::test]
async fn ordinary_in_band_error_becomes_error_and_finish_error() {
    let model = scripted_model(
        [
            event(
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","model":"claude","request_id":"req_body"}}"#,
            ),
            event(
                "error",
                r#"{"type":"error","error":{"type":"api_error","message":"failed"}}"#,
            ),
        ]
        .concat(),
    )
    .await;
    let parts = collect(&model).await;
    assert!(
        matches!(&parts[1], Ok(StreamPart::Error { error }) if error.diagnostics.request_id.as_deref() == Some("req_header"))
    );
    assert!(
        matches!(&parts[2], Ok(StreamPart::Finish { finish }) if finish.finish_reason == FinishReason::Error)
    );
    let finish = parts
        .into_iter()
        .find_map(|part| match part.unwrap() {
            StreamPart::Finish { finish } => Some(finish),
            _ => None,
        })
        .unwrap();
    assert_eq!(finish.response_metadata["anthropic.message_id"], "msg_1");
    assert_eq!(finish.response_metadata["anthropic.model"], "claude");
    assert_eq!(
        finish.response_metadata["anthropic.request_id"],
        "req_header"
    );
}

#[tokio::test]
async fn first_overloaded_error_rejects_stream_creation() {
    let model = scripted_model(event(
        "error",
        r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#,
    ))
    .await;
    let error = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Overload);
    assert_eq!(error.diagnostics.http_status, Some(529));
    assert!(error.retryable);
}
mod common;

use common::{Anthropic, AnthropicAws, MiniMax};
