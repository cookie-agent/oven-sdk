//! Parity suite: Oven `oven-sdk-openai` Chat SSE parser vs Vercel AI SDK
//! `@ai-sdk/openai-compatible` expected behavior.
//!
//! Each test names the Vercel behavior and asserts the Oven parser's observed
//! behavior. Tests whose expectations match Vercel assert equality directly.
//! Tests that document an intentional Oven divergence are prefixed
//! `known_divergence_` and assert the *current* behavior with a comment; this
//! keeps the suite green while making the divergence visible.
//!
//! Vercel references (vercel/ai main, rev 6dcd9237):
//!   packages/openai-compatible/src/chat/openai-compatible-chat-language-model.ts
//!   packages/provider-utils/src/streaming-tool-call-tracker.ts
//!   packages/ai/src/generate-text/parse-tool-call.ts

pub mod common;

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, LanguageModel, ModelError, Request, StreamPart};
use oven_sdk_openai::{OpenAiChatModel, ReasoningField};
use wiremock::MockServer;

#[derive(Debug, PartialEq, Eq)]
enum Run {
    /// Stream completed; normalized parts in order, no error.
    Ok(Vec<String>),
    /// `stream()` itself failed before yielding parts.
    DispatchErr(String),
    /// Parts were yielded, then the stream errored.
    StreamErr { parts: Vec<String>, error: String },
}

fn norm_part(part: &StreamPart) -> String {
    match part {
        StreamPart::StreamStart { .. } => "stream_start".into(),
        StreamPart::TextStart { id, .. } => format!("text_start(id={id:?})"),
        StreamPart::TextDelta { id, delta, .. } => {
            format!("text_delta(id={id:?},delta={delta:?})")
        }
        StreamPart::TextEnd { id, .. } => format!("text_end(id={id:?})"),
        StreamPart::ReasoningStart { id, .. } => format!("reasoning_start(id={id:?})"),
        StreamPart::ReasoningDelta { id, delta, .. } => {
            format!("reasoning_delta(id={id:?},delta={delta:?})")
        }
        StreamPart::ReasoningEnd { id, .. } => format!("reasoning_end(id={id:?})"),
        StreamPart::ToolCallStart { id, name, .. } => {
            format!("tool_start(id={id:?},name={name:?})")
        }
        StreamPart::ToolCallDelta { id, delta, .. } => {
            format!("tool_delta(id={id:?},delta={delta:?})")
        }
        StreamPart::ToolCallEnd { id, .. } => format!("tool_end(id={id:?})"),
        StreamPart::ToolCall { tool_call } => format!(
            "tool_call(id={:?},name={:?},input={},raw={:?})",
            tool_call.id, tool_call.name, tool_call.input, tool_call.raw_input
        ),
        StreamPart::Custom { part } => format!("custom(kind={:?},data={})", part.kind, part.data),
        StreamPart::Error { error } => format!("error(kind={:?})", error.kind),
        StreamPart::Finish { finish } => {
            format!("finish(reason={})", finish.finish_reason)
        }
        other => format!("other({other:?})"),
    }
}

fn norm_err(error: &ModelError) -> String {
    format!("{:?}", error.kind)
}

/// SSE body from JSON event objects.
fn body(events: &[serde_json::Value], done: bool) -> String {
    let mut out = String::new();
    for event in events {
        out.push_str("data: ");
        out.push_str(&event.to_string());
        out.push_str("\n\n");
    }
    if done {
        out.push_str("data: [DONE]\n\n");
    }
    out
}

async fn run_official(body: String) -> Run {
    run_with(body, |server| common::official_chat(server, "gpt-4o-mini")).await
}

async fn run_reasoning(body: String) -> Run {
    run_with(body, |server| {
        let mut config = common::official_chat_config(server, "gpt-4o-mini");
        config.settings.reasoning_field = ReasoningField::ReasoningContent;
        OpenAiChatModel::new(config).expect("reasoning chat model")
    })
    .await
}

async fn run_with<F>(body: String, build: F) -> Run
where
    F: FnOnce(&MockServer) -> OpenAiChatModel,
{
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", body).await;
    let model = build(&server);
    let response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await;
    let mut response = match response {
        Ok(response) => response,
        Err(error) => return Run::DispatchErr(norm_err(&error)),
    };
    let mut parts = Vec::new();
    while let Some(item) = response.stream.next().await {
        match item {
            Ok(part) => parts.push(norm_part(&part)),
            Err(error) => {
                return Run::StreamErr {
                    parts,
                    error: norm_err(&error),
                };
            }
        }
    }
    Run::Ok(parts)
}

fn finish(reason: &str) -> serde_json::Value {
    serde_json::json!({"choices":[{"index":0,"delta":{},"finish_reason":reason}]})
}

fn chunk(delta: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"choices":[{"index":0,"delta":delta,"finish_reason":null}]})
}

// ---------------------------------------------------------------------------
// 1. name arrives AFTER args. Vercel buffers tool deltas by index until the
//    name arrives (openai-compatible-chat-language-model.ts:472-528), so it
//    works. Oven gates `ToolCallStart` on id+name and emits the buffered
//    arguments once the name lands. Expected: MATCH.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn parity_01_late_name_is_buffered() {
    let events = vec![
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather"}}]},"finish_reason":"tool_calls"}]}),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "tool_start(id=\"call_1\",name=\"get_weather\")".into(),
        "tool_delta(id=\"call_1\",delta=\"{\\\"city\\\":\\\"Paris\\\"}\")".into(),
        "tool_end(id=\"call_1\")".into(),
        "tool_call(id=\"call_1\",name=\"get_weather\",input={\"city\":\"Paris\"},raw=Some(\"{\\\"city\\\":\\\"Paris\\\"}\"))".into(),
        "finish(reason=tool_calls)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "DIVERGENCE CLASS: late-name buffering (expected match)"
    );
}

// ---------------------------------------------------------------------------
// 2. missing function.name entirely, valid args. DELIBERATE OVEN DIVERGENCE:
//    Vercel defers the throw to flush (`Expected 'function.name'`); Oven emits
//    name "" silently and completes the stream. Rationale: discarding an
//    otherwise usable turn on a cosmetic provider quirk is worse than letting
//    cookie-agent normalize at persist time and degrade dispatch to a
//    recoverable tool error (unknown name -> ToolFailure). This may change in a
//    future sprint; the divergence is asserted here on purpose.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn known_divergence_missing_name_emits_empty_name_and_completes() {
    let events = vec![
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":null}]}),
        finish("tool_calls"),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "tool_start(id=\"call_1\",name=\"\")".into(),
        "tool_delta(id=\"call_1\",delta=\"{\\\"city\\\":\\\"Paris\\\"}\")".into(),
        "tool_end(id=\"call_1\")".into(),
        "tool_call(id=\"call_1\",name=\"\",input={\"city\":\"Paris\"},raw=Some(\"{\\\"city\\\":\\\"Paris\\\"}\"))".into(),
        "finish(reason=tool_calls)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "DELIBERATE DIVERGENCE: missing function.name becomes \"\" and the stream completes"
    );
}

// ---------------------------------------------------------------------------
// 3. name "" explicit. Vercel accepts it (`function?.name == null` is false for
//    ""; streaming-tool-call-tracker.ts:167). Oven also accepts (pointer gives
//    Some("")). Expected: MATCH.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn parity_03_explicit_empty_name_is_accepted() {
    let events = vec![
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "tool_start(id=\"call_1\",name=\"\")".into(),
        "tool_delta(id=\"call_1\",delta=\"{}\")".into(),
        "tool_end(id=\"call_1\")".into(),
        "tool_call(id=\"call_1\",name=\"\",input={},raw=Some(\"{}\"))".into(),
        "finish(reason=tool_calls)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "DIVERGENCE CLASS: explicit empty name acceptance (expected match)"
    );
}

// ---------------------------------------------------------------------------
// 4. empty-string arguments on a zero-arg tool. Vercel maps "" to {}. Oven now
//    canonicalizes whitespace-only arguments to `{}` with no streamed delta and
//    `raw_input: None`, so it matches. Note there is intentionally no
//    `tool_delta` (the argument block stays closed at the collector).
// ---------------------------------------------------------------------------
#[tokio::test]
async fn parity_04_empty_arguments_become_empty_object() {
    let events = vec![
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"noop","arguments":""}}]},"finish_reason":"tool_calls"}]}),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "tool_start(id=\"call_1\",name=\"noop\")".into(),
        "tool_end(id=\"call_1\")".into(),
        "tool_call(id=\"call_1\",name=\"noop\",input={},raw=None)".into(),
        "finish(reason=tool_calls)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "empty tool arguments map to {{}} and the stream completes"
    );
}

// ---------------------------------------------------------------------------
// 5. invalid/truncated JSON arguments on finish_reason=length. Vercel keeps the
//    raw string and defers validation downstream. Oven now marks the call
//    invalid: it emits the tool call with `input: null` and preserves the raw
//    bytes in `raw_input` (and in the native replay payload), never fabricating
//    a parsed value. The null input fails every published object schema, so the
//    failure surfaces as a recoverable tool error rather than killing the
//    stream.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn parity_05_invalid_json_arguments_are_marked_invalid() {
    let events = vec![
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"broken","arguments":"["}}]},"finish_reason":"length"}]}),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "tool_start(id=\"call_1\",name=\"broken\")".into(),
        "tool_delta(id=\"call_1\",delta=\"[\")".into(),
        "tool_end(id=\"call_1\")".into(),
        "tool_call(id=\"call_1\",name=\"broken\",input=null,raw=Some(\"[\"))".into(),
        "finish(reason=length)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "invalid tool JSON is marked invalid with null input and preserved raw bytes"
    );
}

// ---------------------------------------------------------------------------
// 6. continuation delta with NO index across two interleaved parallel calls.
//    Vercel correlates id -> index -> latest-touched (streaming-tool-call-
//    tracker.ts:110-129), so the orphan fragment goes to the latest call
//    (call_1/beta). Oven now routes id-less/index-less continuations to the
//    last-touched call as well. Expected: MATCH.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn parity_06_indexless_continuation_routes_to_latest_touched() {
    let events = vec![
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_0","function":{"name":"alpha","arguments":"{\"a\":"}}]},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_1","function":{"name":"beta","arguments":"{\"b\":"}}]},"finish_reason":null}]}),
        // continuation with neither id nor index
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":"1}"}}]},"finish_reason":null}]}),
        finish("tool_calls"),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "tool_start(id=\"call_0\",name=\"alpha\")".into(),
        "tool_delta(id=\"call_0\",delta=\"{\\\"a\\\":\")".into(),
        "tool_start(id=\"call_1\",name=\"beta\")".into(),
        "tool_delta(id=\"call_1\",delta=\"{\\\"b\\\":\")".into(),
        "tool_delta(id=\"call_1\",delta=\"1}\")".into(),
        "tool_end(id=\"call_0\")".into(),
        "tool_call(id=\"call_0\",name=\"alpha\",input=null,raw=Some(\"{\\\"a\\\":\"))".into(),
        "tool_end(id=\"call_1\")".into(),
        "tool_call(id=\"call_1\",name=\"beta\",input={\"b\":1},raw=Some(\"{\\\"b\\\":1}\"))".into(),
        "finish(reason=tool_calls)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "index-omitted continuation uses latest-touched, not index 0"
    );
}

// ---------------------------------------------------------------------------
// 7. same id reused with different index. Vercel keys by id and merges into one
//    call (streaming-tool-call-tracker.ts:112-127). Oven now also routes by id,
//    producing a single call whose merged argument text is not a valid JSON
//    object; that becomes a marked-invalid call (null input + preserved raw
//    bytes). Expected: MATCH on call identity, intentional null-input divergence
//    on validation.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn parity_07_same_id_merges_into_one_call() {
    let events = vec![
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"a","arguments":"{\"x\":1}"}}]},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_a","function":{"name":"a","arguments":"{\"y\":2}"}}]},"finish_reason":"tool_calls"}]}),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "tool_start(id=\"call_a\",name=\"a\")".into(),
        "tool_delta(id=\"call_a\",delta=\"{\\\"x\\\":1}\")".into(),
        "tool_delta(id=\"call_a\",delta=\"{\\\"y\\\":2}\")".into(),
        "tool_end(id=\"call_a\")".into(),
        "tool_call(id=\"call_a\",name=\"a\",input=null,raw=Some(\"{\\\"x\\\":1}{\\\"y\\\":2}\"))"
            .into(),
        "finish(reason=tool_calls)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "same provider id merges into one call; merged invalid args are marked invalid"
    );
}

// ---------------------------------------------------------------------------
// 8a. stream ends without finish_reason but WITH [DONE]. Vercel: flush sets
//     finishReason error + emits an error part. Oven: clean FinishReason::Unknown.
//     KNOWN DIVERGENCE (asserts current Oven behavior).
// ---------------------------------------------------------------------------
#[tokio::test]
async fn known_divergence_done_without_finish_reason_is_unknown() {
    let events = vec![chunk(serde_json::json!({"content":"partial"}))];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "text_start(id=\"0\")".into(),
        "text_delta(id=\"0\",delta=\"partial\")".into(),
        "text_end(id=\"0\")".into(),
        "finish(reason=unknown)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "KNOWN DIVERGENCE: Oven maps a missing finish_reason with [DONE] to Unknown, not Error"
    );
}

// ---------------------------------------------------------------------------
// 8b. stream ends without finish_reason and WITHOUT [DONE]. Vercel: same error
//     path as 8a. Oven: `unexpected_eof` after the parts already yielded, with
//     no Finish part (the block is never closed). KNOWN DIVERGENCE.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn known_divergence_eof_without_finish_reason() {
    let events = vec![chunk(serde_json::json!({"content":"partial"}))];
    let observed = run_official(body(&events, false)).await;
    let expected = Run::StreamErr {
        parts: vec![
            "stream_start".into(),
            "text_start(id=\"0\")".into(),
            "text_delta(id=\"0\",delta=\"partial\")".into(),
        ],
        error: "UnexpectedEof".into(),
    };
    assert_eq!(
        observed, expected,
        "KNOWN DIVERGENCE: Oven raises unexpected_eof with no Finish part"
    );
}

// ---------------------------------------------------------------------------
// 9. content "" and whitespace-only chunks. Vercel: "" -> no text events;
//    whitespace is non-empty so it DOES emit a text delta. Oven: ignores "" and
//    emits whitespace. Expected: MATCH at the streaming layer. NOTE: Oven's
//    whitespace fix (db6a5aa, `is_semantic_text`) applies to the
//    normalized/replay turn, not to raw streaming events.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn parity_09_empty_and_whitespace_content() {
    let events = vec![
        chunk(serde_json::json!({"content":""})),
        chunk(serde_json::json!({"content":" "})),
        chunk(serde_json::json!({"content":"x"})),
        finish("stop"),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "text_start(id=\"0\")".into(),
        "text_delta(id=\"0\",delta=\" \")".into(),
        "text_delta(id=\"0\",delta=\"x\")".into(),
        "text_end(id=\"0\")".into(),
        "finish(reason=stop)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "DIVERGENCE CLASS: empty/whitespace content handling (expected match) - verify #3d94c3e/#db6a5aa"
    );
}

// ---------------------------------------------------------------------------
// 10. reasoning_content streaming + a chunk with tool_calls: [] mid-reasoning.
//     Vercel only closes reasoning when `tool_calls.length > 0`
//     (language-model.ts:686), so reasoning stays open. Oven now guards the
//     close on a non-empty array and matches. Expected: MATCH.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn parity_10_empty_tool_calls_do_not_close_reasoning() {
    let events = vec![
        chunk(serde_json::json!({"reasoning_content":"think1"})),
        chunk(serde_json::json!({"tool_calls":[]})),
        chunk(serde_json::json!({"reasoning_content":"think2"})),
        chunk(serde_json::json!({"content":"answer"})),
        finish("stop"),
    ];
    let observed = run_reasoning(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "reasoning_start(id=\"reasoning:0\")".into(),
        "reasoning_delta(id=\"reasoning:0\",delta=\"think1\")".into(),
        "reasoning_delta(id=\"reasoning:0\",delta=\"think2\")".into(),
        "reasoning_end(id=\"reasoning:0\")".into(),
        "text_start(id=\"0\")".into(),
        "text_delta(id=\"0\",delta=\"answer\")".into(),
        "text_end(id=\"0\")".into(),
        "finish(reason=stop)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "`tool_calls: []` must not close an open reasoning block"
    );
}

// ---------------------------------------------------------------------------
// 11a. choice index 1 only. Vercel reads choices[0] regardless of its index.
//      Oven hard-errors during the early peek, so `stream()` returns DispatchErr.
//      KNOWN DIVERGENCE (asserts current Oven behavior).
// ---------------------------------------------------------------------------
#[tokio::test]
async fn known_divergence_choice_index_nonzero() {
    let events = vec![
        serde_json::json!({"choices":[{"index":1,"delta":{"content":"hi"},"finish_reason":"stop"}]}),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::DispatchErr("InvalidResponse".into());
    assert_eq!(
        observed, expected,
        "KNOWN DIVERGENCE: non-zero choice index is rejected instead of tolerated"
    );
}

// ---------------------------------------------------------------------------
// 11b. two choices. Vercel uses choices[0] and silently ignores the rest. Oven
//      hard-errors on choices.len() > 1 during the early peek -> DispatchErr.
//      KNOWN DIVERGENCE.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn known_divergence_multiple_choices() {
    let events = vec![serde_json::json!({"choices":[
        {"index":0,"delta":{"content":"first"},"finish_reason":"stop"},
        {"index":1,"delta":{"content":"second"},"finish_reason":"stop"}
    ]})];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::DispatchErr("InvalidResponse".into());
    assert_eq!(
        observed, expected,
        "KNOWN DIVERGENCE: extra choices fail the stream instead of being ignored"
    );
}

// ---------------------------------------------------------------------------
// 12. content array form [{type:"text",text:...}]. Vercel extracts the text.
//     Oven only handles `content.as_str()` and silently drops arrays. KNOWN
//     DIVERGENCE (asserts current Oven behavior: the text block never opens).
// ---------------------------------------------------------------------------
#[tokio::test]
async fn known_divergence_content_array_form() {
    let events = vec![
        serde_json::json!({"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"hi"}]},"finish_reason":"stop"}]}),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec!["stream_start".into(), "finish(reason=stop)".into()]);
    assert_eq!(
        observed, expected,
        "KNOWN DIVERGENCE: content array form is dropped, not flattened to text"
    );
}

// ---------------------------------------------------------------------------
// 13. unknown extra fields in delta and top-level chunks. Both sides tolerate
//     them (Vercel uses looseObject; Oven ignores unknown keys). Expected: MATCH.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn parity_13_unknown_fields_tolerated() {
    let events = vec![
        serde_json::json!({"id":"chat_1","model":"m","created":1,"system_fingerprint":"fp","extra_top":"ignored","choices":[{"index":0,"delta":{"content":"hi","foo":123,"nested":{"a":1}},"finish_reason":null}]}),
        finish("stop"),
    ];
    let observed = run_official(body(&events, true)).await;
    let expected = Run::Ok(vec![
        "stream_start".into(),
        "text_start(id=\"0\")".into(),
        "text_delta(id=\"0\",delta=\"hi\")".into(),
        "text_end(id=\"0\")".into(),
        "finish(reason=stop)".into(),
    ]);
    assert_eq!(
        observed, expected,
        "DIVERGENCE CLASS: unknown fields tolerance (expected match)"
    );
}
