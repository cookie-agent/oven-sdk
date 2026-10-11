//! Pins the exact JSON bodies sent for representative Azure requests.
//!
//! Expected bodies live in `tests/golden/*.json` and are compared as parsed
//! `serde_json::Value`s, so key order and whitespace are irrelevant. Set
//! `OVEN_SDK_BLESS_GOLDEN=1` to rewrite them after an intentional wire change.

mod common;

use std::path::PathBuf;

use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompactionRequest, CompletedTurn, ContentValue,
    CustomPart, FilePart, FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions,
    InputPart, JsonSchema, LanguageModel, ReasoningPart, Request, ResponseFormat, SystemMessage,
    SystemPart, TextPart, ToolCallPart, ToolChoice, ToolContent, ToolDefinition, ToolMessage,
    ToolResultPart, UserMessage,
};
use oven_sdk_azure::{
    AzureApiRoute, AzureOpenAiChatOptions, AzureOpenAiChatRequestExt, AzureOpenAiCompactionOptions,
    AzureOpenAiCompactionRequestExt, AzureOpenAiPromptCacheBreakpointExt,
    AzureOpenAiPromptCacheMode, AzureOpenAiPromptCacheOptions, AzureOpenAiPromptCacheRetention,
    AzureOpenAiPromptCacheTtl, AzureOpenAiResponsesOptions, AzureOpenAiResponsesRequestExt,
    AzureReasoningField,
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
                .insert_header("apim-request-id", "req_first"),
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
        .insert("azure_openai".into(), json!({"strict":true}));
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
        SystemPart::Text(
            TextPart::new("Stable prefix.").with_azure_openai_prompt_cache_breakpoint(),
        ),
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

const RESPONSES: &str = "/openai/v1/responses";
const CHAT: &str = "/openai/v1/chat/completions";

fn responses_options() -> AzureOpenAiResponsesOptions {
    AzureOpenAiResponsesOptions {
        include: vec!["message.output_text.logprobs".into()],
        user: Some("user-123".into()),
        service_tier: Some("flex".into()),
        verbosity: Some("low".into()),
        reasoning_summary: Some("auto".into()),
        reasoning_mode: None,
        truncation: Some("auto".into()),
        parallel_tool_calls: Some(false),
        prompt_cache_key: Some("cache-key".into()),
        prompt_cache_retention: Some(AzureOpenAiPromptCacheRetention::TwentyFourHours),
        prompt_cache_options: Some(AzureOpenAiPromptCacheOptions {
            mode: AzureOpenAiPromptCacheMode::Explicit,
            ttl: AzureOpenAiPromptCacheTtl::ThirtyMinutes,
        }),
    }
}

fn parallel_off() -> AzureOpenAiResponsesOptions {
    AzureOpenAiResponsesOptions {
        parallel_tool_calls: Some(false),
        ..AzureOpenAiResponsesOptions::default()
    }
}

fn responses_rich_document() -> String {
    let output = json!([
        {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"plan the lookup"}],"encrypted_content":"enc-state-1"},
        {"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Checking.","annotations":[]}]},
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

async fn server_with(endpoint: &str, document: String) -> MockServer {
    let server = MockServer::start().await;
    common::mount(&server, endpoint, document).await;
    server
}

fn responses_model(server: &MockServer) -> oven_sdk_azure::AzureOpenAiResponsesModel {
    common::provider(server, AzureApiRoute::V1)
        .responses("deployment", sampling(common::gpt5()))
        .unwrap()
}

#[tokio::test]
async fn golden_azure_responses_plain_text() {
    let server = server_with(RESPONSES, common::responses_document("ok")).await;
    responses_model(&server)
        .complete(
            Request::new(vec![user_text("hello")]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_golden("azure_responses_plain_text", &last_body(&server).await);
}

#[tokio::test]
async fn golden_azure_responses_options_and_tools() {
    let server = server_with(RESPONSES, common::responses_document("ok")).await;
    let mut request = Request::new(vec![system_two_parts(), user_text("look up x")])
        .with_azure_openai_responses_options(responses_options());
    request.tools = tools();
    request.tool_choice = ToolChoice::Tool("lookup".into());
    request.inference = inference();
    request.inference.reasoning_effort = Some("high".into());
    responses_model(&server)
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "azure_responses_options_and_tools",
        &last_body(&server).await,
    );
}

#[tokio::test]
async fn golden_azure_responses_history_media_and_tool_results() {
    let server = server_with(RESPONSES, common::responses_document("ok")).await;
    let mut request = Request::new(vec![
        system_with_breakpoint(),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(
                TextPart::new("Describe these.").with_azure_openai_prompt_cache_breakpoint(),
            ),
            InputPart::File(png(b"png-bytes").with_azure_openai_prompt_cache_breakpoint()),
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
                AssistantPart::Text(TextPart::new("Partial answer.")),
                AssistantPart::Custom(CustomPart::new("azure.openai.refusal", json!("I can't."))),
            ]),
            finish(FinishReason::Stop),
        )),
        user_text("thanks"),
    ])
    .with_azure_openai_responses_options(parallel_off());
    request.tools = tools();
    request.tool_choice = ToolChoice::Required;
    responses_model(&server)
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "azure_responses_history_media_and_tool_results",
        &last_body(&server).await,
    );
}

#[tokio::test]
async fn golden_azure_responses_reasoning_replay() {
    let server = MockServer::start().await;
    mount_once(&server, RESPONSES, responses_rich_document()).await;
    common::mount(&server, RESPONSES, common::responses_document("ok")).await;
    let model = responses_model(&server);
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
    ])
    .with_azure_openai_responses_options(parallel_off());
    request.tools = tools();
    request.inference.reasoning_effort = Some("medium".into());
    let streamed = model.stream(request, AbortSignal::default()).await.unwrap();
    assert!(matches!(
        streamed.request.replay.decisions[0].disposition,
        oven_sdk::ReplayDisposition::Replayed
    ));
    drop(streamed);
    assert_golden(
        "azure_responses_reasoning_replay",
        &last_body(&server).await,
    );
}

#[tokio::test]
async fn golden_azure_responses_structured_output() {
    let server = server_with(RESPONSES, common::responses_document("ok")).await;
    let model = responses_model(&server);
    let mut request = Request::new(vec![user_text("score it")])
        .with_azure_openai_responses_options(AzureOpenAiResponsesOptions {
            verbosity: Some("high".into()),
            ..AzureOpenAiResponsesOptions::default()
        });
    request.response_format = ResponseFormat::Json {
        schema: Some(output_schema()),
    };
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "azure_responses_structured_json_schema",
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
    assert_golden("azure_responses_json_object", &last_body(&server).await);

    let request = Request::new(vec![user_text("just verbose")])
        .with_azure_openai_responses_options(AzureOpenAiResponsesOptions {
            verbosity: Some("medium".into()),
            ..AzureOpenAiResponsesOptions::default()
        });
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("azure_responses_verbosity_only", &last_body(&server).await);
}

#[tokio::test]
async fn golden_azure_responses_compaction_and_native_context() {
    let server = server_with(RESPONSES, common::responses_document("ok")).await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses/compact"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id":"resp_compact_1",
            "object":"response.compaction",
            "created_at":1_764_967_971_u64,
            "output":[
                {"id":"msg_user_1","type":"message","status":"completed","role":"user","content":[{"type":"input_text","text":"old context"}]},
                {"id":"cmp_1","type":"compaction","encrypted_content":"opaque-encrypted-context","created_by":"azure-openai"}
            ],
            "usage":{"input_tokens":139,"output_tokens":438,"total_tokens":577}
        })))
        .mount(&server)
        .await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5_compaction())
        .unwrap();
    let request = CompactionRequest::new(Request::new(vec![
        system_two_parts(),
        user_text("long history"),
        assistant_text("noted"),
    ]))
    .with_azure_openai_compaction_options(AzureOpenAiCompactionOptions {
        instructions: Some("Keep decisions.".into()),
        prompt_cache_key: Some("compact-key".into()),
        prompt_cache_retention: Some("in_memory".into()),
        service_tier: Some("default".into()),
    });
    let compacted = model
        .compact(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "azure_responses_compaction_request",
        &last_body(&server).await,
    );

    let request =
        Request::new(vec![user_text("continue")]).with_native_context(compacted.native_context);
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("azure_responses_native_context", &last_body(&server).await);
}

// ---------------------------------------------------------------------------
// Chat Completions
// ---------------------------------------------------------------------------

fn chat_options() -> AzureOpenAiChatOptions {
    AzureOpenAiChatOptions {
        user: Some("user-123".into()),
        reasoning_effort: Some("low".into()),
        service_tier: Some("priority".into()),
        verbosity: Some("medium".into()),
        parallel_tool_calls: Some(false),
        prompt_cache_key: Some("chat-key".into()),
        prompt_cache_retention: Some(AzureOpenAiPromptCacheRetention::InMemory),
        prompt_cache_options: Some(AzureOpenAiPromptCacheOptions {
            mode: AzureOpenAiPromptCacheMode::Implicit,
            ttl: AzureOpenAiPromptCacheTtl::ThirtyMinutes,
        }),
    }
}

fn chat_parallel_off() -> AzureOpenAiChatOptions {
    AzureOpenAiChatOptions {
        parallel_tool_calls: Some(false),
        ..AzureOpenAiChatOptions::default()
    }
}

fn chat_reasoning_document() -> String {
    concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking hard\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Calling.\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\n",
        "data: [DONE]\n\n"
    )
    .to_owned()
}

fn sampling(mut setup: common::ModelSetup) -> common::ModelSetup {
    setup.capabilities.features |= oven_sdk::Capability::TEMPERATURE
        | oven_sdk::Capability::TOP_P
        | oven_sdk::Capability::MAX_OUTPUT_TOKENS;
    setup
}

fn chat_model(
    server: &MockServer,
    setup: common::ModelSetup,
) -> oven_sdk_azure::AzureOpenAiChatModel {
    common::provider(server, AzureApiRoute::V1)
        .chat("deployment", setup)
        .unwrap()
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
            InputPart::Text(TextPart::new("Describe.").with_azure_openai_prompt_cache_breakpoint()),
            InputPart::File(png(b"png-bytes").with_azure_openai_prompt_cache_breakpoint()),
            InputPart::File(FilePart::image(
                "image/jpeg",
                url("https://example.test/a.jpg"),
            )),
            InputPart::Custom(CustomPart::new("ignored", json!(null))),
        ])),
        normalized_assistant_with_calls(),
        paired_results(),
        text_tool_results()[0].clone(),
        text_tool_results()[1].clone(),
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![
                AssistantPart::Custom(CustomPart::new("azure.openai.refusal", json!("I can't."))),
                AssistantPart::Custom(CustomPart::new("other.custom", json!("ignored"))),
            ]),
            finish(FinishReason::Stop),
        )),
        HistoryTurn::user(UserMessage::new(vec![])),
        user_text("thanks"),
    ]
}

#[tokio::test]
async fn golden_azure_chat_plain_text() {
    let server = server_with(CHAT, common::chat_document("ok")).await;
    chat_model(&server, common::gpt4o())
        .complete(
            Request::new(vec![user_text("hello")]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert_golden("azure_chat_plain_text", &last_body(&server).await);
}

#[tokio::test]
async fn golden_azure_chat_options_and_tools() {
    let server = server_with(CHAT, common::chat_document("ok")).await;
    let mut request = Request::new(vec![system_two_parts(), user_text("look up x")])
        .with_azure_openai_chat_options(chat_options());
    request.tools = tools();
    request.tool_choice = ToolChoice::Tool("lookup".into());
    request.inference = inference();
    chat_model(&server, sampling(common::gpt4o()))
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("azure_chat_options_and_tools", &last_body(&server).await);
}

#[tokio::test]
async fn golden_azure_chat_reasoning_profile() {
    let server = server_with(CHAT, common::chat_document("ok")).await;
    let mut request = Request::new(vec![system_two_parts(), user_text("look up x")])
        .with_azure_openai_chat_options(chat_parallel_off());
    request.tools = tools();
    request.inference = inference();
    request.inference.reasoning_effort = Some("high".into());
    chat_model(&server, sampling(common::gpt5_chat()))
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden("azure_chat_reasoning_profile", &last_body(&server).await);
}

async fn golden_azure_chat_history(name: &str, field: AzureReasoningField) {
    let server = MockServer::start().await;
    mount_once(&server, CHAT, chat_reasoning_document()).await;
    common::mount(&server, CHAT, common::chat_document("ok")).await;
    let mut setup = common::gpt5_chat();
    setup.completions.reasoning_field = field;
    let model = chat_model(&server, setup);
    let first = model
        .complete(Request::new(vec![user_text("go")]), AbortSignal::default())
        .await
        .unwrap();
    let mut history = chat_history_with_media();
    history.splice(
        0..0,
        [
            user_text("go"),
            HistoryTurn::assistant(first.turn),
            HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
                "call_1",
                ToolContent::Text("tool says hi".into()),
            )])),
        ],
    );
    let mut request = Request::new(history).with_azure_openai_chat_options(chat_parallel_off());
    request.tools = tools();
    request.tool_choice = ToolChoice::Required;
    let streamed = model.stream(request, AbortSignal::default()).await.unwrap();
    assert!(matches!(
        streamed.request.replay.decisions[0].disposition,
        oven_sdk::ReplayDisposition::Replayed
    ));
    drop(streamed);
    assert_golden(name, &last_body(&server).await);
}

#[tokio::test]
async fn golden_azure_chat_history_reasoning_content() {
    golden_azure_chat_history(
        "azure_chat_history_reasoning_content",
        AzureReasoningField::ReasoningContent,
    )
    .await;
}

#[tokio::test]
async fn golden_azure_chat_history_reasoning() {
    golden_azure_chat_history(
        "azure_chat_history_reasoning",
        AzureReasoningField::Reasoning,
    )
    .await;
}

#[tokio::test]
async fn golden_azure_chat_history_no_reasoning() {
    golden_azure_chat_history("azure_chat_history_no_reasoning", AzureReasoningField::None).await;
}

#[tokio::test]
async fn golden_azure_chat_structured_output() {
    let server = server_with(CHAT, common::chat_document("ok")).await;
    let model = chat_model(&server, common::gpt4o());
    let mut request = Request::new(vec![user_text("score it")]);
    request.response_format = ResponseFormat::Json {
        schema: Some(output_schema()),
    };
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert_golden(
        "azure_chat_structured_json_schema",
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
    assert_golden("azure_chat_json_object", &last_body(&server).await);
}
