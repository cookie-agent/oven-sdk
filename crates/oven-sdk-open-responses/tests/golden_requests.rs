//! Pins the exact JSON bodies sent for representative Open Responses requests.
//!
//! Expected bodies live in `tests/golden/*.json` and are compared as parsed
//! `serde_json::Value`s, so key order and whitespace are irrelevant. Set
//! `OVEN_SDK_BLESS_GOLDEN=1` to rewrite them after an intentional wire change.

mod common;

use std::{collections::BTreeMap, path::PathBuf};

use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, ContentValue, CustomPart,
    FilePart, FileSource, Finish, FinishReason, HistoryTurn, InputPart, JsonSchema, LanguageModel,
    ReasoningPart, Request, ResponseFormat, SystemMessage, SystemPart, TextPart, ToolCallPart,
    ToolChoice, ToolContent, ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};
use oven_sdk_open_responses::{OpenResponsesRequestExt, OpenResponsesRequestOptions};
use serde_json::{Value, json};
use wiremock::MockServer;

fn assert_golden(name: &str, body: &Value) {
    let file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.json"));
    if std::env::var_os("OVEN_SDK_BLESS_GOLDEN").is_some() {
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, serde_json::to_string_pretty(body).unwrap() + "\n").unwrap();
    }
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(&file).unwrap_or_else(|_| panic!("missing golden {name}")),
    )
    .unwrap();
    assert_eq!(body, &expected, "request body for {name} changed");
}

async fn last_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.unwrap();
    serde_json::from_slice(&requests.last().unwrap().body).unwrap()
}

async fn server() -> MockServer {
    let server = MockServer::start().await;
    common::mount(&server, common::text_stream("ok")).await;
    server
}

fn user_text(text: &str) -> HistoryTurn {
    HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(text))]))
}

fn finish(reason: FinishReason) -> Finish {
    Finish::new(Default::default(), reason)
}

fn tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "lookup",
            "Looks a key up",
            JsonSchema::new(json!({
                "type":"object",
                "properties":{"q":{"type":"string"}},
                "required":["q"],
                "additionalProperties":false
            }))
            .unwrap(),
        ),
        ToolDefinition::new(
            "list_files",
            "",
            JsonSchema::new(json!({"type":"object","properties":{},"additionalProperties":false}))
                .unwrap(),
        ),
    ]
}

fn metadata(key: &str, value: Value) -> Option<BTreeMap<String, Value>> {
    Some(BTreeMap::from([(key.to_owned(), value)]))
}

fn custom_item(kind: &str, id: &str) -> CustomPart {
    CustomPart::new(
        kind,
        json!({"type":kind,"id":id,"status":"completed","payload":{"n":1}}),
    )
}

#[tokio::test]
async fn golden_open_responses_plain_text() {
    let server = server().await;
    common::generic_model(&server, "opaque")
        .complete(
            Request::new(vec![user_text("hello")]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_golden("open_responses_plain_text", &last_body(&server).await);
}

#[tokio::test]
async fn golden_open_responses_options_and_tools() {
    let server = server().await;
    let mut config = common::generic_config(&server, "opaque");
    config.settings.reasoning_summary = Some("auto".into());
    config.settings.include = vec!["message.output_text.logprobs".into()];
    let model = oven_sdk_open_responses::OpenResponsesModel::new(config).unwrap();
    let mut request = Request::new(vec![user_text("look up x")]).with_open_responses_options(
        OpenResponsesRequestOptions {
            instructions: Some("Be brief.".into()),
            metadata: BTreeMap::from([("tenant".into(), "a".into()), ("run".into(), "7".into())]),
            service_tier: Some("default".into()),
            truncation: Some("auto".into()),
            text_verbosity: Some("low".into()),
            max_tool_calls: Some(3),
            safety_identifier: Some("user-1".into()),
            prompt_cache_key: Some("cache-key".into()),
        },
    );
    request.tools = tools();
    request.tool_choice = ToolChoice::Tool("lookup".into());
    request.inference.max_output_tokens = Some(512);
    request.inference.temperature = Some(0.25);
    request.inference.top_p = Some(0.5);
    request.inference.reasoning_effort = Some("high".into());
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "open_responses_options_and_tools",
        &last_body(&server).await,
    );

    let mut config = common::generic_config(&server, "opaque");
    config.settings.reasoning_summary = Some("concise".into());
    let model = oven_sdk_open_responses::OpenResponsesModel::new(config).unwrap();
    let mut request = Request::new(vec![user_text("summary only")]);
    request.tools = tools();
    request.tool_choice = ToolChoice::Required;
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("open_responses_summary_only", &last_body(&server).await);
}

fn assistant_history() -> Vec<HistoryTurn> {
    let mut phased = TextPart::new("Checking the docs.");
    phased.metadata = metadata("open_responses.phase", json!("commentary"));
    let mut source = oven_sdk::SourcePart::new();
    source.metadata = Some(BTreeMap::from([
        ("type".to_owned(), json!("url_citation")),
        ("url".to_owned(), json!("https://example.com/doc")),
        ("title".to_owned(), json!("Doc")),
        ("start_index".to_owned(), json!(0)),
        ("end_index".to_owned(), json!(5)),
    ]));
    let mut refusal = CustomPart::new("open_responses.refusal", json!("I can't."));
    refusal.metadata = metadata("open_responses.phase", json!("final_answer"));
    let mut raw = ToolCallPart::new("call_a", "lookup", json!({"q":"x"}));
    raw.raw_input = Some("{\"q\": \"x\"}".into());
    let mut with_item = ToolCallPart::new("call_b", "list_files", json!({}));
    with_item.provider_item_id = Some("fc_b".into());
    vec![
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![
                AssistantPart::Reasoning(ReasoningPart::new("plan")),
                AssistantPart::Text(phased),
                AssistantPart::Source(source),
                AssistantPart::Text(TextPart::new("Plain.")),
                AssistantPart::Custom(custom_item("acme:widget", "w_1")),
                AssistantPart::Custom(refusal),
                AssistantPart::ToolCall(raw),
                AssistantPart::ToolCall(with_item),
            ]),
            finish(FinishReason::ToolCalls),
        )),
        HistoryTurn::tool(ToolMessage::new(vec![
            ToolResultPart::new("call_a", ToolContent::Text("found \"x\"".into())),
            ToolResultPart::new(
                "call_b",
                ToolContent::Mixed(vec![
                    ContentValue::Text("files".into()),
                    ContentValue::Json(json!(["a", "b"])),
                    ContentValue::File(FilePart::image(
                        "image/png",
                        FileSource::Bytes(bytes::Bytes::from_static(b"png")),
                    )),
                ]),
            ),
        ])),
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![
                AssistantPart::ToolCall(ToolCallPart::new("call_c", "lookup", json!({"q":"y"}))),
                AssistantPart::ToolCall(ToolCallPart::new("call_d", "lookup", json!({"q":"z"}))),
                AssistantPart::ToolCall(ToolCallPart::new("call_e", "lookup", json!({"q":"w"}))),
            ]),
            finish(FinishReason::ToolCalls),
        )),
        HistoryTurn::tool(ToolMessage::new(vec![
            ToolResultPart::new("call_c", ToolContent::Json(json!({"ok":true,"s":"é"}))),
            ToolResultPart::new(
                "call_d",
                ToolContent::Denied {
                    reason: Some("user declined".into()),
                },
            ),
            ToolResultPart::new("call_e", ToolContent::Denied { reason: None }),
        ])),
    ]
}

#[tokio::test]
async fn golden_open_responses_history_and_media() {
    let server = server().await;
    let mut pdf = FilePart::document(
        "application/pdf",
        FileSource::Bytes(bytes::Bytes::from_static(b"%PDF")),
    );
    pdf.filename = Some("report.pdf".into());
    let mut history = vec![
        HistoryTurn::system(SystemMessage::new(vec![
            SystemPart::Text(TextPart::new("You are terse.")),
            SystemPart::Custom(custom_item("acme:policy", "p_1")),
            SystemPart::Text(TextPart::new("Answer in English.")),
        ])),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(TextPart::new("Describe these.")),
            InputPart::File(FilePart::image(
                "image/png",
                FileSource::Bytes(bytes::Bytes::from_static(b"png-bytes")),
            )),
            InputPart::File(FilePart::image(
                "image/jpeg",
                FileSource::Url("https://example.test/a.jpg".parse().unwrap()),
            )),
            InputPart::File(pdf),
            InputPart::File(FilePart::document(
                "application/pdf",
                FileSource::Bytes(bytes::Bytes::from_static(b"%PDF-2")),
            )),
            InputPart::File(FilePart::document(
                "application/pdf",
                FileSource::Url("https://example.test/doc.pdf".parse().unwrap()),
            )),
            InputPart::Custom(custom_item("acme:note", "n_1")),
            InputPart::Text(TextPart::new("Thanks.")),
        ])),
    ];
    history.extend(assistant_history());
    history.push(user_text("continue"));
    let mut request = Request::new(history);
    request.tools = tools();
    common::generic_model(&server, "opaque")
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "open_responses_history_and_media",
        &last_body(&server).await,
    );
}

#[tokio::test]
async fn golden_open_responses_replay() {
    let server = server().await;
    let model = common::generic_model(&server, "opaque");
    let first = model
        .complete(Request::new(vec![user_text("hi")]), AbortSignal::default())
        .await
        .unwrap();
    let request = Request::new(vec![
        user_text("hi"),
        HistoryTurn::assistant(first.turn),
        user_text("again"),
    ]);
    let streamed = model.stream(request, AbortSignal::default()).await.unwrap();
    assert!(matches!(
        streamed.request.replay.decisions[0].disposition,
        oven_sdk::ReplayDisposition::Replayed
    ));
    drop(streamed);
    assert_golden("open_responses_replay", &last_body(&server).await);
}

#[tokio::test]
async fn golden_open_responses_hugging_face_reasoning() {
    let server = server().await;
    let mut text_kind = ReasoningPart::new("raw thinking");
    text_kind.metadata = metadata("open_responses.kind", json!("reasoning_text"));
    let mut summary_kind = ReasoningPart::new("summarized");
    summary_kind.metadata = metadata("open_responses.kind", json!("reasoning_summary"));
    let request = Request::new(vec![
        user_text("think"),
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![
                AssistantPart::Reasoning(text_kind),
                AssistantPart::Reasoning(summary_kind),
                AssistantPart::Reasoning(ReasoningPart::new("default kind")),
                AssistantPart::Text(TextPart::new("Answer.")),
            ]),
            finish(FinishReason::Stop),
        )),
        user_text("more"),
    ]);
    common::hugging_face_model(&server, "org/model")
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "open_responses_hugging_face_reasoning",
        &last_body(&server).await,
    );
}

#[tokio::test]
async fn golden_open_responses_structured_output() {
    let server = server().await;
    let model = common::generic_model(&server, "opaque");
    let schema = JsonSchema::new(json!({
        "type":"object",
        "properties":{"answer":{"type":"string"}},
        "required":["answer"],
        "additionalProperties":false
    }))
    .unwrap();
    for (name, format, verbosity) in [
        (
            "open_responses_json_schema_verbosity",
            ResponseFormat::Json {
                schema: Some(schema.clone()),
            },
            Some("high"),
        ),
        (
            "open_responses_json_schema",
            ResponseFormat::Json {
                schema: Some(schema),
            },
            None,
        ),
        (
            "open_responses_json_object_verbosity",
            ResponseFormat::Json { schema: None },
            Some("low"),
        ),
        (
            "open_responses_json_object",
            ResponseFormat::Json { schema: None },
            None,
        ),
        (
            "open_responses_text_verbosity",
            ResponseFormat::Text,
            Some("medium"),
        ),
    ] {
        let mut request = Request::new(vec![user_text("format")]).with_open_responses_options(
            OpenResponsesRequestOptions {
                text_verbosity: verbosity.map(str::to_owned),
                ..OpenResponsesRequestOptions::default()
            },
        );
        request.response_format = format;
        model
            .complete(request, AbortSignal::default())
            .await
            .unwrap();
        assert_golden(name, &last_body(&server).await);
    }
}
