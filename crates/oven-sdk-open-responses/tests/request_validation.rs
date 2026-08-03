mod common;

use std::collections::BTreeMap;

use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, ContentValue, CustomPart,
    FilePart, FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions, InputPart,
    JsonSchema, LanguageModel, ModelErrorKind, ReasoningPart, Request, ResponseFormat, SourcePart,
    TextPart, ToolApprovalPart, ToolCallPart, ToolContent, ToolDefinition, ToolResultPart,
    UserMessage,
};
use oven_sdk_open_responses::{OpenResponsesRequestExt, OpenResponsesRequestOptions};
use wiremock::MockServer;

fn completed(parts: Vec<AssistantPart>) -> HistoryTurn {
    HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(parts),
        Finish::new(Default::default(), FinishReason::Stop),
    ))
}

#[tokio::test]
async fn normalized_reconstruction_preserves_representable_order_and_phase() {
    let server = MockServer::start().await;
    common::mount(&server, common::text_stream("ok")).await;

    let mut text = TextPart::new("alpha");
    text.metadata = Some(BTreeMap::from([(
        "open_responses.phase".into(),
        "commentary".into(),
    )]));
    let mut source = SourcePart::new();
    source.url = Some("https://example.com/source".parse().unwrap());
    source.title = Some("Source".into());
    source.metadata = Some(BTreeMap::from([
        ("type".into(), "url_citation".into()),
        ("start_index".into(), 0.into()),
        ("end_index".into(), 5.into()),
        ("url".into(), "https://example.com/source".into()),
        ("title".into(), "Source".into()),
    ]));
    let mut refusal = CustomPart::new("open_responses.refusal", "blocked".into());
    refusal.metadata = Some(BTreeMap::from([(
        "open_responses.phase".into(),
        "final_answer".into(),
    )]));
    let reasoning = ReasoningPart::new("summary");
    let mut call = ToolCallPart::new("call_1", "lookup", serde_json::json!({}));
    call.provider_item_id = Some("fc_1".into());
    call.raw_input = Some("{}".into());
    let result = ToolResultPart::new(
        "call_1",
        ToolContent::Mixed(vec![
            ContentValue::Text("done".into()),
            ContentValue::File(FilePart::document(
                "application/pdf",
                FileSource::Url("https://example.com/result.pdf".parse().unwrap()),
            )),
        ]),
    );
    let request = Request::new(vec![completed(vec![
        AssistantPart::Text(text),
        AssistantPart::Source(source),
        AssistantPart::Custom(refusal),
        AssistantPart::Reasoning(reasoning),
        AssistantPart::ToolCall(call),
        AssistantPart::ToolResult(result),
    ])]);

    common::generic_model(&server, "opaque")
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    let input = body["input"].as_array().unwrap();
    assert_eq!(
        input
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "message",
            "message",
            "reasoning",
            "function_call",
            "function_call_output"
        ]
    );
    assert_eq!(input[0]["phase"], "commentary");
    assert_eq!(input[0]["content"][0]["text"], "alpha");
    assert_eq!(
        input[0]["content"][0]["annotations"][0]["type"],
        "url_citation"
    );
    assert_eq!(input[1]["phase"], "final_answer");
    assert_eq!(input[1]["content"][0]["type"], "refusal");
    assert_eq!(input[2]["summary"][0]["type"], "summary_text");
    assert_eq!(input[3]["id"], "fc_1");
    assert_eq!(input[4]["output"][1]["type"], "input_file");
}

#[tokio::test]
async fn official_request_constraints_are_enforced_before_dispatch() {
    let server = MockServer::start().await;
    let model = common::generic_model(&server, "opaque");

    let mut too_small = InferenceOptions::new();
    too_small.max_output_tokens = Some(15);
    let requests = [
        Request::new(Vec::new()).with_inference(too_small),
        Request::new(Vec::new()).with_tools(vec![ToolDefinition::new(
            "bad name",
            "bad",
            JsonSchema::new(serde_json::json!({"type":"object"})).unwrap(),
        )]),
        Request::new(Vec::new()).with_response_format(ResponseFormat::structured(
            JsonSchema::new(serde_json::json!(true)).unwrap(),
        )),
        Request::new(Vec::new()).with_open_responses_options(OpenResponsesRequestOptions {
            safety_identifier: Some("x".repeat(65)),
            ..Default::default()
        }),
    ];
    for request in requests {
        let error = model
            .stream(request, AbortSignal::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind, ModelErrorKind::InvalidRequest);
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn unsupported_history_and_tool_media_are_rejected_without_dropping() {
    let server = MockServer::start().await;
    let model = common::generic_model(&server, "opaque");

    let mut call = ToolCallPart::new("call_1", "lookup", serde_json::json!({}));
    call.raw_input = Some("{}".into());
    let result = ToolResultPart::new(
        "call_1",
        ToolContent::Mixed(vec![ContentValue::File(FilePart::image(
            "image/png",
            FileSource::Url("ftp://example.com/image.png".parse().unwrap()),
        ))]),
    );
    let media = Request::new(vec![completed(vec![
        AssistantPart::ToolCall(call),
        AssistantPart::ToolResult(result),
    ])]);
    let file = Request::new(vec![completed(vec![AssistantPart::File(
        FilePart::document(
            "application/pdf",
            FileSource::Bytes(bytes::Bytes::from_static(b"pdf")),
        ),
    )])]);
    let source = Request::new(vec![completed(vec![AssistantPart::Source(
        SourcePart::new(),
    )])]);
    let approval = Request::new(vec![completed(vec![AssistantPart::ToolApproval(
        ToolApprovalPart::new("call_1"),
    )])]);
    let custom = Request::new(vec![completed(vec![AssistantPart::Custom(
        CustomPart::new("provider:unsupported", serde_json::json!({})),
    )])]);
    let mut message = AssistantMessage::new(vec![AssistantPart::Text(TextPart::new("text"))]);
    message
        .provider_options
        .insert("other".into(), serde_json::json!({"ignored":true}));
    let provider_options = Request::new(vec![HistoryTurn::assistant(CompletedTurn::new(
        message,
        Finish::new(Default::default(), FinishReason::Stop),
    ))]);
    let oversize = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(vec![0_u8; 16 * 1024 * 1024].into()),
        )),
    ]))]);

    for request in [media, file, source, approval, custom, provider_options] {
        assert!(!model.supports_request(&request));
        let error = model
            .stream(request, AbortSignal::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind, ModelErrorKind::Unsupported);
    }
    assert!(!model.supports_request(&oversize));
    let error = model
        .stream(oversize, AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::InvalidRequest);
    assert!(server.received_requests().await.unwrap().is_empty());
}
