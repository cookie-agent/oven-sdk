//! Pins the exact JSON bodies sent for representative requests.
//!
//! Expected bodies live in `tests/golden/*.json` and are compared as parsed
//! `serde_json::Value`s, so key order and whitespace are irrelevant. Set
//! `OVEN_SDK_BLESS_GOLDEN=1` to rewrite them after an intentional wire change.

pub mod common;

use std::path::PathBuf;

use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompactionRequest, CompletedTurn, ContentValue,
    CustomPart, FilePart, FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions,
    InputPart, JsonSchema, LanguageModel, ReasoningPart, Request, ResponseFormat, SystemMessage,
    SystemPart, TextPart, ToolCallPart, ToolChoice, ToolContent, ToolDefinition, ToolMessage,
    ToolResultPart, UserMessage,
};
use oven_sdk_openai::{
    CompatibleChatOptions, MaxTokensField, OpenAiChatOptions, OpenAiChatRequestExt,
    OpenAiCompatibleChatModel, OpenAiPromptCacheBreakpointExt, OpenAiPromptCacheMode,
    OpenAiPromptCacheOptions, OpenAiPromptCacheTtl, OpenAiResponsesCompactionOptions,
    OpenAiResponsesCompactionRequestExt, OpenAiResponsesOptions, OpenAiResponsesRequestExt,
    ReasoningField, StructuredOutputSupport, SystemMessageRole,
};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

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

/// Serves `first` once, then the plain document for every later request.
async fn mount_once(server: &MockServer, endpoint: &str, first: String) {
    Mock::given(method("POST"))
        .and(path(endpoint))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(first, "text/event-stream")
                .insert_header("x-request-id", "req_first"),
        )
        .up_to_n_times(1)
        .with_priority(1)
        .mount(server)
        .await;
}

fn user_text(text: &str) -> HistoryTurn {
    HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(text))]))
}

fn finish(reason: FinishReason) -> Finish {
    Finish::new(Default::default(), reason)
}

fn tool_call(id: &str, name: &str, input: Value, raw: Option<&str>) -> ToolCallPart {
    let mut call = ToolCallPart::new(id, name, input);
    call.raw_input = raw.map(str::to_owned);
    call
}

fn tools() -> Vec<ToolDefinition> {
    let mut strict = ToolDefinition::new(
        "lookup",
        "Looks a key up",
        JsonSchema::new(json!({
            "type":"object",
            "properties":{"q":{"type":"string"}},
            "required":["q"],
            "additionalProperties":false
        }))
        .unwrap(),
    );
    strict
        .provider_options
        .insert("openai".into(), json!({"strict":true}));
    vec![
        strict,
        ToolDefinition::new(
            "list_files",
            "",
            JsonSchema::new(json!({"type":"object","properties":{}})).unwrap(),
        ),
    ]
}

fn inference() -> InferenceOptions {
    let mut inference = InferenceOptions::new();
    inference.max_output_tokens = Some(2048);
    inference.temperature = Some(0.25);
    inference.top_p = Some(0.5);
    inference
}

fn output_schema() -> JsonSchema {
    JsonSchema::new(json!({
        "type":"object",
        "properties":{"answer":{"type":"string"},"score":{"type":"number"}},
        "required":["answer","score"],
        "additionalProperties":false
    }))
    .unwrap()
}

fn png(bytes: &'static [u8]) -> FilePart {
    FilePart::image(
        "image/png",
        FileSource::Bytes(bytes::Bytes::from_static(bytes)),
    )
}

fn pdf(bytes: &'static [u8], filename: Option<&str>) -> FilePart {
    let mut file = FilePart::document(
        "application/pdf",
        FileSource::Bytes(bytes::Bytes::from_static(bytes)),
    );
    file.filename = filename.map(str::to_owned);
    file
}

fn url(value: &str) -> FileSource {
    FileSource::Url(value.parse().unwrap())
}

fn system_two_parts() -> HistoryTurn {
    HistoryTurn::system(SystemMessage::new(vec![
        SystemPart::Text(TextPart::new("You are terse.")),
        SystemPart::Custom(CustomPart::new("ignored.custom", json!({"x":1}))),
        SystemPart::Text(TextPart::new("Answer in English.")),
    ]))
}

fn system_with_breakpoint() -> HistoryTurn {
    HistoryTurn::system(SystemMessage::new(vec![
        SystemPart::Text(TextPart::new("Stable prefix.").with_openai_prompt_cache_breakpoint()),
        SystemPart::Text(TextPart::new("")),
        SystemPart::Text(TextPart::new("Second block.")),
    ]))
}

/// An assistant tool-call turn followed by results every OpenAI surface can
/// encode as text.
fn text_tool_results() -> [HistoryTurn; 2] {
    let ids = ["call_a", "call_b", "call_c", "call_d", "call_e", "call_f"];
    let calls = HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(
            ids.iter()
                .map(|id| AssistantPart::ToolCall(tool_call(id, "lookup", json!({"q":id}), None)))
                .collect(),
        ),
        finish(FinishReason::ToolCalls),
    ));
    let mut failed = ToolResultPart::new("call_b", ToolContent::Text("boom: not found".into()));
    failed.is_error = true;
    let results = HistoryTurn::tool(ToolMessage::new(vec![
        ToolResultPart::new(
            "call_a",
            ToolContent::Text("plain \"quoted\"\nresult".into()),
        ),
        failed,
        ToolResultPart::new(
            "call_c",
            ToolContent::Json(json!({"rows":[1,2,3],"nested":{"ok":true,"s":"é"}})),
        ),
        ToolResultPart::new(
            "call_d",
            ToolContent::Denied {
                reason: Some("user declined".into()),
            },
        ),
        ToolResultPart::new("call_e", ToolContent::Denied { reason: None }),
        ToolResultPart::new(
            "call_f",
            ToolContent::Mixed(vec![
                ContentValue::Text("caption".into()),
                ContentValue::Json(json!({"answer":42})),
            ]),
        ),
    ]));
    [calls, results]
}

fn paired_results() -> HistoryTurn {
    HistoryTurn::tool(ToolMessage::new(vec![
        ToolResultPart::new("call_a", ToolContent::Text("found".into())),
        ToolResultPart::new("call_b", ToolContent::Json(json!(["a.txt", "b.txt"]))),
    ]))
}

fn normalized_assistant_with_calls() -> HistoryTurn {
    let mut with_item = tool_call("call_b", "list_files", json!({"dir":"/tmp"}), None);
    with_item.provider_item_id = Some("fc_item_b".into());
    HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(vec![
            AssistantPart::Reasoning(ReasoningPart::new("visible thought")),
            AssistantPart::Text(TextPart::new("Let me ")),
            AssistantPart::Text(TextPart::new("check.")),
            AssistantPart::ToolCall(tool_call(
                "call_a",
                "lookup",
                json!({"q":"x"}),
                Some("{\"q\": \"x\"}"),
            )),
            AssistantPart::ToolCall(with_item),
            AssistantPart::Text(TextPart::new("Trailing text.")),
        ]),
        finish(FinishReason::ToolCalls),
    ))
}

fn assistant_text(text: &str) -> HistoryTurn {
    HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::Text(TextPart::new(text))]),
        finish(FinishReason::Stop),
    ))
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

fn responses_options() -> OpenAiResponsesOptions {
    OpenAiResponsesOptions {
        include: vec!["message.output_text.logprobs".into()],
        user: Some("user-123".into()),
        service_tier: Some("flex".into()),
        verbosity: Some("low".into()),
        reasoning_summary: Some("auto".into()),
        reasoning_mode: None,
        truncation: Some("auto".into()),
        parallel_tool_calls: Some(false),
        prompt_cache_key: Some("cache-key".into()),
        prompt_cache_retention: None,
        prompt_cache_options: Some(OpenAiPromptCacheOptions {
            mode: Some(OpenAiPromptCacheMode::Explicit),
            ttl: Some(OpenAiPromptCacheTtl::TwelveHours),
            prewarm: None,
        }),
        compact_threshold: None,
    }
}

fn responses_rich_document() -> String {
    let output = json!([
        {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"plan the lookup"}],"encrypted_content":"enc-state-1"},
        {"type":"message","id":"msg_1","status":"completed","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"Checking.","annotations":[]}]},
        {"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}
    ]);
    format!(
        "event: response.completed\ndata: {}\n\n",
        json!({"type":"response.completed","response":{
            "id":"resp_rich","model":"gpt-5-mini","status":"completed","output":output,
            "usage":{"input_tokens":5,"output_tokens":7,"output_tokens_details":{"reasoning_tokens":3}}
        }})
    )
}

async fn responses_server() -> MockServer {
    let server = MockServer::start().await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    server
}

#[tokio::test]
async fn golden_responses_plain_text() {
    let server = responses_server().await;
    let model = common::official_responses(&server, "gpt-5-mini");
    model
        .complete(
            Request::new(vec![user_text("hello")]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_golden("responses_plain_text", &last_body(&server).await);
}

#[tokio::test]
async fn golden_responses_options_and_tools() {
    let server = responses_server().await;
    let model = common::official_responses(&server, "gpt-5-mini");
    let mut request = Request::new(vec![system_two_parts(), user_text("look up x")])
        .with_openai_responses_options(responses_options());
    request.tools = tools();
    request.tool_choice = ToolChoice::Tool("lookup".into());
    request.inference = inference();
    request.inference.reasoning_effort = Some("high".into());
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("responses_options_and_tools", &last_body(&server).await);
}

#[tokio::test]
async fn golden_responses_history_media_and_tool_results() {
    let server = responses_server().await;
    let model = common::official_responses(&server, "gpt-5-mini");
    let mut request = Request::new(vec![
        system_with_breakpoint(),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(TextPart::new("Describe these.").with_openai_prompt_cache_breakpoint()),
            InputPart::File(
                png(b"png-bytes")
                    .with_openai_prompt_cache_breakpoint_ttl(OpenAiPromptCacheTtl::TwelveHours),
            ),
            InputPart::File(FilePart::image(
                "image/jpeg",
                url("https://example.test/a.jpg"),
            )),
            InputPart::File(pdf(b"%PDF-1.7", Some("report.pdf"))),
            InputPart::File(pdf(b"%PDF-1.4", None)),
            InputPart::File(FilePart::document(
                "application/pdf",
                url("https://example.test/doc.pdf"),
            )),
            InputPart::File(FilePart::document(
                "application/pdf",
                FileSource::Text("text-source".into()),
            )),
            InputPart::Custom(CustomPart::new("ignored", json!(null))),
        ])),
        normalized_assistant_with_calls(),
        HistoryTurn::tool(ToolMessage::new(vec![
            ToolResultPart::new("call_a", ToolContent::Text("found".into())),
            ToolResultPart::new(
                "call_b",
                ToolContent::Mixed(vec![
                    ContentValue::Text("screenshot".into()),
                    ContentValue::Json(json!({"w":10})),
                    ContentValue::File(png(b"tool-png")),
                    ContentValue::File(pdf(b"tool-pdf", Some("tool.pdf"))),
                ]),
            ),
        ])),
        assistant_text("Done with files."),
        text_tool_results()[0].clone(),
        text_tool_results()[1].clone(),
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![
                AssistantPart::Text(TextPart::new("Inline result follows.")),
                AssistantPart::ToolCall(tool_call("call_inline", "lookup", json!({}), None)),
                AssistantPart::ToolResult(ToolResultPart::new(
                    "call_inline",
                    ToolContent::Text("inline output".into()),
                )),
                AssistantPart::Text(TextPart::new("After inline.")),
            ]),
            finish(FinishReason::Stop),
        )),
        user_text("thanks"),
    ]);
    request.tools = tools();
    request.tool_choice = ToolChoice::Required;
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "responses_history_media_and_tool_results",
        &last_body(&server).await,
    );
}

#[tokio::test]
async fn golden_responses_reasoning_replay() {
    let server = MockServer::start().await;
    mount_once(&server, "/responses", responses_rich_document()).await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let model = common::official_responses(&server, "gpt-5-mini");
    let first = model
        .complete(
            Request::new(vec![user_text("look up x")]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(first.turn.finish.native_replay.is_some());
    let mut request = Request::new(vec![
        user_text("look up x"),
        HistoryTurn::assistant(first.turn),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call_1",
            ToolContent::Text("value-of-x".into()),
        )])),
        assistant_text("x is value-of-x"),
        user_text("and y?"),
    ]);
    request.tools = tools();
    request.inference.reasoning_effort = Some("medium".into());
    let streamed = model.stream(request, AbortSignal::default()).await.unwrap();
    assert!(matches!(
        streamed.request.replay.decisions[0].disposition,
        oven_sdk::ReplayDisposition::Replayed
    ));
    drop(streamed);
    assert_golden("responses_reasoning_replay", &last_body(&server).await);
}

#[tokio::test]
async fn golden_responses_structured_output() {
    let server = responses_server().await;
    let model = common::official_responses(&server, "gpt-5-mini");
    let mut request = Request::new(vec![user_text("score it")]).with_openai_responses_options(
        OpenAiResponsesOptions {
            verbosity: Some("high".into()),
            ..OpenAiResponsesOptions::default()
        },
    );
    request.response_format = ResponseFormat::Json {
        schema: Some(output_schema()),
    };
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "responses_structured_json_schema",
        &last_body(&server).await,
    );

    let mut request = Request::new(vec![user_text("any json")]);
    request.response_format = ResponseFormat::Json { schema: None };
    request.tools = tools();
    request.tool_choice = ToolChoice::None;
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("responses_json_object", &last_body(&server).await);

    let request = Request::new(vec![user_text("just verbose")]).with_openai_responses_options(
        OpenAiResponsesOptions {
            verbosity: Some("medium".into()),
            ..OpenAiResponsesOptions::default()
        },
    );
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("responses_verbosity_only", &last_body(&server).await);
}

#[tokio::test]
async fn golden_responses_compaction_and_native_context() {
    let server = responses_server().await;
    common::mount_compaction(&server).await;
    let model = common::official_responses_native(&server, "gpt-5-mini");
    let request = Request::new(vec![
        system_two_parts(),
        user_text("long history"),
        assistant_text("noted"),
    ]);
    let request = CompactionRequest::new(request).with_openai_responses_compaction_options(
        OpenAiResponsesCompactionOptions {
            instructions: Some("Keep decisions.".into()),
            prompt_cache_key: Some("compact-key".into()),
            prompt_cache_options: Some(OpenAiPromptCacheOptions {
                mode: Some(OpenAiPromptCacheMode::Implicit),
                ttl: Some(OpenAiPromptCacheTtl::ThirtyMinutes),
                prewarm: None,
            }),
            prompt_cache_retention: None,
            service_tier: Some("default".into()),
        },
    );
    let compacted = model
        .compact(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("responses_compaction_request", &last_body(&server).await);

    let mut request = Request::new(vec![user_text("continue")]);
    request.native_context = Some(compacted.native_context);
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("responses_native_context", &last_body(&server).await);
}

// ---------------------------------------------------------------------------
// Chat Completions
// ---------------------------------------------------------------------------

fn chat_options() -> OpenAiChatOptions {
    OpenAiChatOptions {
        user: Some("user-123".into()),
        reasoning_effort: Some("low".into()),
        service_tier: Some("priority".into()),
        verbosity: Some("medium".into()),
        parallel_tool_calls: Some(true),
        prompt_cache_key: Some("chat-key".into()),
        prompt_cache_retention: None,
        prompt_cache_options: Some(OpenAiPromptCacheOptions {
            mode: Some(OpenAiPromptCacheMode::Implicit),
            ttl: Some(OpenAiPromptCacheTtl::ThirtyMinutes),
            prewarm: None,
        }),
    }
}

fn chat_tool_document() -> String {
    concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Checking.\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"x\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    )
    .to_owned()
}

fn chat_reasoning_document() -> String {
    concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking hard\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Calling.\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    )
    .to_owned()
}

fn chat_history_with_media() -> Vec<HistoryTurn> {
    vec![
        system_with_breakpoint(),
        system_two_parts(),
        user_text("plain user"),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(TextPart::new("joined ")),
            InputPart::Text(TextPart::new("text")),
        ])),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(TextPart::new("Describe.").with_openai_prompt_cache_breakpoint()),
            InputPart::File(png(b"png-bytes").with_openai_prompt_cache_breakpoint()),
            InputPart::File(FilePart::image(
                "image/jpeg",
                url("https://example.test/a.jpg"),
            )),
            InputPart::File(pdf(b"%PDF-1.7", Some("report.pdf"))),
            InputPart::File(pdf(b"%PDF-1.4", None)),
            InputPart::File(FilePart::image(
                "image/png",
                FileSource::Text("text-source".into()),
            )),
            InputPart::Custom(CustomPart::new("ignored", json!(null))),
        ])),
        normalized_assistant_with_calls(),
        paired_results(),
        text_tool_results()[0].clone(),
        text_tool_results()[1].clone(),
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![
                AssistantPart::Custom(CustomPart::new("openai.refusal", json!("I can't."))),
                AssistantPart::Custom(CustomPart::new("other.custom", json!("ignored"))),
            ]),
            finish(FinishReason::Stop),
        )),
        HistoryTurn::user(UserMessage::new(vec![])),
        user_text("thanks"),
    ]
}

#[tokio::test]
async fn golden_chat_plain_text() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    model
        .complete(
            Request::new(vec![user_text("hello")]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_golden("chat_plain_text", &last_body(&server).await);
}

#[tokio::test]
async fn golden_chat_options_and_tools() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let mut request = Request::new(vec![system_two_parts(), user_text("look up x")])
        .with_openai_chat_options(chat_options());
    request.tools = tools();
    request.tool_choice = ToolChoice::Tool("lookup".into());
    request.inference = inference();
    request.inference.reasoning_effort = Some("high".into());
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("chat_options_and_tools", &last_body(&server).await);
}

#[tokio::test]
async fn golden_chat_history_media_and_tool_results() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let mut request = Request::new(chat_history_with_media());
    request.tools = tools();
    request.tool_choice = ToolChoice::Required;
    request.inference.reasoning_effort = Some("minimal".into());
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "chat_history_media_and_tool_results",
        &last_body(&server).await,
    );
}

#[tokio::test]
async fn golden_chat_replay() {
    let server = MockServer::start().await;
    mount_once(&server, "/chat/completions", chat_tool_document()).await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let first = model
        .complete(
            Request::new(vec![user_text("look up x")]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let mut request = Request::new(vec![
        user_text("look up x"),
        HistoryTurn::assistant(first.turn),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call_1",
            ToolContent::Text("value".into()),
        )])),
        user_text("and y?"),
    ]);
    request.tools = tools();
    let streamed = model.stream(request, AbortSignal::default()).await.unwrap();
    assert!(matches!(
        streamed.request.replay.decisions[0].disposition,
        oven_sdk::ReplayDisposition::Replayed
    ));
    drop(streamed);
    assert_golden("chat_replay", &last_body(&server).await);
}

#[tokio::test]
async fn golden_chat_structured_output() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = common::official_chat(&server, "gpt-4o-mini");
    let mut request = Request::new(vec![user_text("score it")]);
    request.response_format = ResponseFormat::Json {
        schema: Some(output_schema()),
    };
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("chat_structured_json_schema", &last_body(&server).await);

    let mut request = Request::new(vec![user_text("any json")]);
    request.response_format = ResponseFormat::Json { schema: None };
    request.tools = tools();
    request.tool_choice = ToolChoice::None;
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("chat_json_object", &last_body(&server).await);
}

fn compatible_model(
    server: &MockServer,
    configure: impl FnOnce(&mut oven_sdk_openai::OpenAiCompatibleChatSettings),
) -> OpenAiCompatibleChatModel {
    let mut config = common::compatible_config(server, "fixture-model");
    configure(&mut config.settings);
    OpenAiCompatibleChatModel::new(config).unwrap()
}

async fn golden_compatible_reasoning(name: &str, field: ReasoningField) {
    let server = MockServer::start().await;
    mount_once(&server, "/chat/completions", chat_reasoning_document()).await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = compatible_model(&server, |settings| {
        settings.reasoning_field = field;
    });
    let first = model
        .complete(Request::new(vec![user_text("go")]), AbortSignal::default())
        .await
        .unwrap();
    let mut request = Request::new(vec![
        system_two_parts(),
        user_text("go"),
        HistoryTurn::assistant(first.turn),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call_1",
            ToolContent::Text("tool says hi".into()),
        )])),
        assistant_text("no reasoning here"),
        normalized_assistant_with_calls(),
        paired_results(),
        text_tool_results()[0].clone(),
        text_tool_results()[1].clone(),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(TextPart::new("watch")),
            InputPart::File(FilePart::video(
                "video/mp4",
                FileSource::Bytes(bytes::Bytes::from_static(b"mp4")),
            )),
        ])),
    ])
    .with_compatible_chat_options(CompatibleChatOptions {
        extra_body: json!({"thinking":{"type":"enabled"},"seed":7})
            .as_object()
            .unwrap()
            .clone(),
    });
    request.tools = tools();
    request.inference = inference();
    let streamed = model.stream(request, AbortSignal::default()).await.unwrap();
    assert!(matches!(
        streamed.request.replay.decisions[0].disposition,
        oven_sdk::ReplayDisposition::Replayed
    ));
    drop(streamed);
    assert_golden(name, &last_body(&server).await);
}

#[tokio::test]
async fn golden_compatible_chat_reasoning_content() {
    golden_compatible_reasoning(
        "compatible_chat_reasoning_content",
        ReasoningField::ReasoningContent,
    )
    .await;
}

#[tokio::test]
async fn golden_compatible_chat_reasoning_field() {
    golden_compatible_reasoning("compatible_chat_reasoning", ReasoningField::Reasoning).await;
}

#[tokio::test]
async fn golden_compatible_chat_no_reasoning_field() {
    golden_compatible_reasoning("compatible_chat_no_reasoning", ReasoningField::None).await;
}

#[tokio::test]
async fn golden_compatible_chat_profile_variants() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    let model = compatible_model(&server, |settings| {
        settings.system_message_role = SystemMessageRole::Developer;
        settings.max_tokens_field = MaxTokensField::MaxCompletionTokens;
        settings.structured_output = StructuredOutputSupport::JsonObject;
    });
    let mut request = Request::new(vec![system_two_parts(), user_text("score it")]);
    request.response_format = ResponseFormat::Json {
        schema: Some(output_schema()),
    };
    request.inference.max_output_tokens = Some(99);
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "compatible_chat_profile_variants",
        &last_body(&server).await,
    );

    let model = compatible_model(&server, |settings| {
        settings.system_message_role = SystemMessageRole::Omit;
    });
    let request = Request::new(vec![system_two_parts(), user_text("hi")]);
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("compatible_chat_omitted_fields", &last_body(&server).await);
}
