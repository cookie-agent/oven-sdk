mod common;

use oven_sdk::{
    AbortSignal, FilePart, FileSource, HistoryTurn, InputPart, JsonSchema, LanguageModel, Request,
    StreamPart, TextPart, ToolDefinition, UserMessage,
};
use wiremock::MockServer;

#[tokio::test]
async fn encodes_exact_images_and_collects_citations_usage() {
    let server = MockServer::start().await;
    common::mount(&server, common::text_stream("ok")).await;
    let model = common::model(&server, "opaque");
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("inspect")),
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(bytes::Bytes::from_static(b"png")),
        )),
    ]))]);
    let result = model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.finish.usage.input_tokens, Some(2));
    assert!(
        result
            .turn
            .message
            .content
            .iter()
            .any(|part| matches!(part, oven_sdk::AssistantPart::Source(_)))
    );
    let received = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["model"], "opaque");
    assert!(
        body["messages"][0]["content"][1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
}

#[tokio::test]
async fn parallel_tool_plan_and_calls_have_strict_lifecycle() {
    let server = MockServer::start().await;
    common::mount(&server, common::tool_stream()).await;
    let model = common::model(&server, "opaque");
    let schema = JsonSchema::new(serde_json::json!({
        "type":"object","properties":{"q":{"type":"string"}},"required":["q"]
    }))
    .unwrap();
    let mut response = model
        .stream(
            Request::new(Vec::new())
                .with_tools(vec![ToolDefinition::new("lookup", "find", schema)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let mut calls = 0;
    let mut plan = false;
    use futures_util::StreamExt;
    while let Some(item) = response.stream.next().await {
        match item.unwrap() {
            StreamPart::ToolCall { .. } => calls += 1,
            StreamPart::ReasoningDelta { metadata, .. }
                if metadata
                    .as_ref()
                    .and_then(|value| value.get("cohere.kind"))
                    .and_then(serde_json::Value::as_str)
                    == Some("tool_plan") =>
            {
                plan = true
            }
            _ => {}
        }
    }
    assert_eq!(calls, 2);
    assert!(plan);
}

#[tokio::test]
async fn unsupported_image_mime_is_rejected_before_network() {
    let server = MockServer::start().await;
    let model = common::model(&server, "opaque");
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(FilePart::image(
            "image/svg+xml",
            FileSource::Bytes(bytes::Bytes::from_static(b"svg")),
        )),
    ]))]);
    assert!(model.stream(request, AbortSignal::default()).await.is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
}
