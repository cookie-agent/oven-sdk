pub mod common;

use futures_util::StreamExt;
use oven_sdk::{AbortSignal, HistoryTurn, LanguageModel, ModelConfig, Request, StreamPart};
use oven_sdk_openai::{
    OpenAiResponsesCompaction, OpenAiResponsesModel, OpenAiResponsesOptions,
    OpenAiResponsesRequestExt, OpenAiResponsesSettings,
};
use serde_json::Value;
use wiremock::MockServer;

/// A response the server compacted in the stream, as the live route sends
/// it: compaction items beside the answer, and an empty terminal `output`.
fn compacted_document() -> String {
    let events = [
        r#"{"type":"response.created","response":{"id":"resp_1","model":"gpt-6-luna"}}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"compaction","id":"cmp_1","encrypted_content":"partial"}}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"compaction","id":"cmp_1","encrypted_content":"state-one"}}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}"#,
        r#"{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"ok"}"#,
        r#"{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"ok"}]}}"#,
        r#"{"type":"response.output_item.added","output_index":2,"item":{"type":"compaction","id":"cmp_2","encrypted_content":"partial"}}"#,
        r#"{"type":"response.output_item.done","output_index":2,"item":{"type":"compaction","id":"cmp_2","encrypted_content":"state-two"}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_1","model":"gpt-6-luna","status":"completed","output":[],"usage":{"input_tokens":7276,"output_tokens":106}}}"#,
    ];
    events
        .iter()
        .map(|event| {
            let kind = serde_json::from_str::<Value>(event).unwrap()["type"]
                .as_str()
                .unwrap()
                .to_owned();
            format!("event: {kind}\ndata: {event}\n\n")
        })
        .collect()
}

fn model(
    server: &MockServer,
    model_id: &str,
    compaction: OpenAiResponsesCompaction,
) -> OpenAiResponsesModel {
    let base = common::official_responses_config(server, model_id);
    let mut settings = OpenAiResponsesSettings::new();
    settings.compaction = compaction;
    OpenAiResponsesModel::new(ModelConfig::new(base.provider, base.model, settings)).unwrap()
}

fn with_threshold(request: Request, threshold: u64) -> Request {
    request.with_openai_responses_options(OpenAiResponsesOptions {
        compact_threshold: Some(threshold),
        ..OpenAiResponsesOptions::default()
    })
}

async fn bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect()
}

#[tokio::test]
async fn an_in_stream_model_sends_its_threshold_and_others_drop_it_with_a_warning() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let in_stream = model(&server, "gpt-6-luna", OpenAiResponsesCompaction::InStream);
    in_stream
        .complete(
            with_threshold(Request::new(vec![]), 190_400),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let plain = model(
        &server,
        "gpt-6-luna",
        OpenAiResponsesCompaction::Unsupported,
    );
    let mut response = plain
        .stream(
            with_threshold(Request::new(vec![]), 190_400),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let warnings = match response.stream.next().await {
        Some(Ok(StreamPart::StreamStart { warnings })) => warnings,
        other => panic!("expected stream start, got {other:?}"),
    };
    while response.stream.next().await.is_some() {}
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("compact_threshold"))
    );
    let sent = bodies(&server).await;
    assert_eq!(
        sent[0]["context_management"],
        serde_json::json!([{"type": "compaction", "compact_threshold": 190_400}])
    );
    assert!(sent[1].get("context_management").is_none());
}

#[tokio::test]
async fn a_compacted_response_completes_and_replays_its_compaction_to_the_same_model_only() {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", compacted_document()).await;
    let luna = model(&server, "gpt-6-luna", OpenAiResponsesCompaction::InStream);
    let turn = luna
        .complete(
            with_threshold(Request::new(vec![]), 2_000),
            AbortSignal::default(),
        )
        .await
        .unwrap()
        .turn;
    assert!(matches!(
        turn.message.content.as_slice(),
        [oven_sdk::AssistantPart::Text(text)] if text.text == "ok"
    ));
    let replayed = turn.finish.native_replay.as_ref().unwrap().payload()["items"].clone();
    assert_eq!(
        replayed
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["compaction", "message", "compaction"]
    );

    // The same model continues from its compaction state.
    luna.complete(
        Request::new(vec![HistoryTurn::assistant(turn.clone())]),
        AbortSignal::default(),
    )
    .await
    .unwrap();
    // Another model of the provider continues from the turns themselves.
    let sol = model(&server, "gpt-6-sol", OpenAiResponsesCompaction::InStream);
    let mut other = turn.clone();
    other.finish.native_replay = turn.finish.native_replay.clone();
    sol.complete(
        Request::new(vec![HistoryTurn::assistant(other)]),
        AbortSignal::default(),
    )
    .await
    .unwrap();
    let sent = bodies(&server).await;
    let types = |body: &Value| {
        body["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| item["type"].as_str().map(str::to_owned))
            .collect::<Vec<_>>()
    };
    assert_eq!(types(&sent[1]), ["compaction", "message", "compaction"]);
    assert_eq!(
        sent[1]["input"][0]["encrypted_content"],
        Value::String("state-one".into())
    );
    assert_eq!(types(&sent[2]), ["message"]);
}
