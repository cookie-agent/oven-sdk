//! Golden request bodies: every case pins the complete JSON body sent to the provider.
//!
//! Regenerate with `OVEN_BLESS_GOLDEN=1 cargo test -p oven-sdk-google --test golden`
//! only when a wire change is intended, and review the diff against the provider docs.

mod common;

use std::path::PathBuf;

use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, CompleteResult, CompletedTurn,
    ContentValue, CustomPart, FilePart, FileSource, Finish, FinishReason, HistoryTurn,
    InferenceOptions, InputPart, JsonSchema, LanguageModel, ModelId, NativeReplayArtifact,
    ReasoningPart, ReplayCapability, ReplayDisposition, ReplayPolicy, Request, ResponseFormat,
    SourcePart, SystemMessage, SystemPart, TextPart, ToolCallPart, ToolChoice, ToolContent,
    ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};
use oven_sdk_google::{
    GOOGLE_GENERATE_CONTENT_ADAPTER_ID, GoogleModel, GoogleProviderTool, GoogleRequestExt,
    GoogleRequestOptions, GoogleSafetySetting, GoogleThinkingConfig, GoogleToolExt,
    GoogleToolOptions, GoogleToolSettings,
};
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

const MODEL: &str = "gemini-3.5-flash";

fn response() -> String {
    format!(
        "data: {}\n\n",
        json!({
            "candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}
        })
    )
}

async fn server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(response()),
        )
        .mount(&server)
        .await;
    server
}

async fn capture_result(
    server: &MockServer,
    model: &impl LanguageModel,
    request: Request,
) -> (Value, CompleteResult) {
    let result = model
        .complete(request, AbortSignal::default())
        .await
        .expect("golden request succeeds");
    let requests = server.received_requests().await.unwrap();
    let body = serde_json::from_slice(&requests.last().expect("request was sent").body).unwrap();
    (body, result)
}

async fn capture(server: &MockServer, model: &impl LanguageModel, request: Request) -> Value {
    capture_result(server, model, request).await.0
}

fn dispositions(result: &CompleteResult) -> Vec<ReplayDisposition> {
    result
        .request
        .replay
        .decisions
        .iter()
        .map(|decision| decision.disposition.clone())
        .collect()
}

fn assert_golden(name: &str, actual: &Value) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.json"));
    if std::env::var_os("OVEN_BLESS_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_string_pretty(actual).unwrap() + "\n").unwrap();
        return;
    }
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing golden {name}")),
    )
    .unwrap();
    assert_eq!(actual, &expected, "request body differs from golden {name}");
}

fn text(value: &str) -> TextPart {
    TextPart::new(value)
}

fn user(value: &str) -> HistoryTurn {
    HistoryTurn::user(UserMessage::new(vec![InputPart::Text(text(value))]))
}

fn assistant(parts: Vec<AssistantPart>, replay: Option<NativeReplayArtifact>) -> HistoryTurn {
    let reason = if parts
        .iter()
        .any(|part| matches!(part, AssistantPart::ToolCall(_)))
    {
        FinishReason::ToolCalls
    } else {
        FinishReason::Stop
    };
    let mut finish = Finish::new(Default::default(), reason);
    finish.native_replay = replay;
    HistoryTurn::assistant(CompletedTurn::new(AssistantMessage::new(parts), finish))
}

fn tool_call(id: &str, name: &str, input: Value) -> AssistantPart {
    AssistantPart::ToolCall(ToolCallPart::new(id, name, input))
}

fn tool_result(id: &str, content: ToolContent) -> ToolResultPart {
    ToolResultPart::new(id, content)
}

fn tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "lookup",
            "Find a record",
            JsonSchema::new(json!({
                "type":"object",
                "properties":{
                    "id":{"type":"string","description":"Record id"},
                    "filters":{"type":"array","items":{"type":"string"}}
                },
                "required":["id"]
            }))
            .unwrap(),
        ),
        ToolDefinition::new(
            "write_file",
            "Write a file",
            JsonSchema::new(json!({"type":"object","properties":{"path":{"type":"string"}}}))
                .unwrap(),
        ),
    ]
}

fn strict_tools() -> Vec<ToolDefinition> {
    let mut tools = tools();
    tools[0] = tools[0]
        .clone()
        .with_google_tool_options(GoogleToolOptions { strict: true });
    tools
}

fn inference(max: Option<u64>, temperature: Option<f64>, top_p: Option<f64>) -> InferenceOptions {
    let mut inference = InferenceOptions::new();
    inference.max_output_tokens = max;
    inference.temperature = temperature;
    inference.top_p = top_p;
    inference
}

fn effort(label: &str) -> InferenceOptions {
    let mut inference = InferenceOptions::new();
    inference.reasoning_effort = Some(label.into());
    inference
}

fn gemini(server: &MockServer) -> GoogleModel {
    common::model(server.uri(), MODEL)
}

fn replay_artifact(
    model: &GoogleModel,
    payload: Value,
    source_model: Option<&str>,
) -> NativeReplayArtifact {
    let artifact = NativeReplayArtifact::new(
        AdapterId::new(GOOGLE_GENERATE_CONTENT_ADAPTER_ID),
        model.native_context_scope().clone(),
        payload,
    )
    .unwrap();
    match source_model {
        Some(id) => artifact
            .with_source_wire_model_id(ModelId::new(id))
            .unwrap(),
        None => artifact,
    }
}

fn all_provider_tools() -> Vec<GoogleProviderTool> {
    vec![
        GoogleProviderTool::GoogleSearch,
        GoogleProviderTool::UrlContext,
        GoogleProviderTool::CodeExecution,
        GoogleProviderTool::FileSearch {
            stores: vec![
                "fileSearchStores/store-a".into(),
                "fileSearchStores/b".into(),
            ],
        },
        GoogleProviderTool::GoogleMaps,
    ]
}

fn provider_tool_options() -> GoogleRequestOptions {
    GoogleRequestOptions {
        provider_tools: all_provider_tools(),
        ..Default::default()
    }
}

#[tokio::test]
async fn golden_plain_text() {
    let server = server().await;
    let body = capture(
        &server,
        &gemini(&server),
        Request::new(vec![user("Hello there")]),
    )
    .await;
    assert_golden("plain_text", &body);
}

#[tokio::test]
async fn golden_system_instruction_generation_config_and_options() {
    let server = server().await;
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![
            SystemPart::Text(text("You are terse.")),
            SystemPart::Text(text("")),
        ])),
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(text(
            "Answer in \"quotes\" \u{1F600}\n",
        ))])),
        HistoryTurn::user(UserMessage::new(vec![InputPart::Text(text(""))])),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(text("first")),
            InputPart::Text(text("")),
            InputPart::Text(text("second\twith\u{0001}control")),
        ])),
    ])
    .with_inference(inference(Some(1000), Some(0.7), Some(0.9)))
    .with_google_options(GoogleRequestOptions {
        top_k: Some(40),
        stop_sequences: vec!["END".into(), "STOP".into()],
        seed: Some(7),
        presence_penalty: Some(0.5),
        frequency_penalty: Some(-0.25),
        service_tier: Some("flex".into()),
        cached_content: Some("cachedContents/cache-123".into()),
        safety_settings: vec![
            GoogleSafetySetting {
                category: "HARM_CATEGORY_HARASSMENT".into(),
                threshold: "BLOCK_ONLY_HIGH".into(),
            },
            GoogleSafetySetting {
                category: "HARM_CATEGORY_DANGEROUS_CONTENT".into(),
                threshold: "BLOCK_NONE".into(),
            },
        ],
        ..Default::default()
    });
    let body = capture(&server, &gemini(&server), request).await;
    assert_golden("system_instruction_generation_config_and_options", &body);
}

#[tokio::test]
async fn golden_tool_choice_modes() {
    let server = server().await;
    let model = gemini(&server);
    for (name, choice) in [
        ("tools_auto_choice", ToolChoice::Auto),
        ("tools_required_choice", ToolChoice::Required),
        ("tools_specific_choice", ToolChoice::Tool("lookup".into())),
        ("tools_none_choice", ToolChoice::None),
    ] {
        let request = Request::new(vec![user("Use a tool")])
            .with_tools(tools())
            .with_tool_choice(choice);
        let body = capture(&server, &model, request).await;
        assert_golden(name, &body);
    }
}

#[tokio::test]
async fn golden_strict_tool_choice_modes() {
    let server = server().await;
    let model = gemini(&server);
    for (name, choice) in [
        ("strict_tools_auto_choice", ToolChoice::Auto),
        ("strict_tools_required_choice", ToolChoice::Required),
        (
            "strict_tools_specific_choice",
            ToolChoice::Tool("write_file".into()),
        ),
        ("strict_tools_none_choice", ToolChoice::None),
    ] {
        let request = Request::new(vec![user("Use a strict tool")])
            .with_tools(strict_tools())
            .with_tool_choice(choice);
        let body = capture(&server, &model, request).await;
        assert_golden(name, &body);
    }
}

#[tokio::test]
async fn golden_mixed_client_and_provider_tool_choice_modes() {
    let server = server().await;
    let model = gemini(&server);
    for (name, choice) in [
        ("mixed_tools_auto_choice", ToolChoice::Auto),
        ("mixed_tools_required_choice", ToolChoice::Required),
        (
            "mixed_tools_specific_choice",
            ToolChoice::Tool("lookup".into()),
        ),
        ("mixed_tools_none_choice", ToolChoice::None),
    ] {
        let request = Request::new(vec![user("Search and look up")])
            .with_tools(strict_tools())
            .with_tool_choice(choice)
            .with_google_options(provider_tool_options());
        let body = capture(&server, &model, request).await;
        assert_golden(name, &body);
    }
}

#[tokio::test]
async fn golden_provider_tools_only() {
    let server = server().await;
    let request =
        Request::new(vec![user("Search the web")]).with_google_options(provider_tool_options());
    let body = capture(&server, &gemini(&server), request).await;
    assert_golden("provider_tools_only", &body);
}

#[tokio::test]
async fn golden_tool_results_all_content_kinds() {
    let server = server().await;
    let mut error_text = tool_result("call-3", ToolContent::Text("boom".into()));
    error_text.is_error = true;
    let mut error_json = tool_result("call-4", ToolContent::Json(json!({"code":500})));
    error_json.is_error = true;
    let request = Request::new(vec![
        user("Run the tools"),
        assistant(
            vec![
                AssistantPart::Text(text("Calling tools.")),
                tool_call("call-1", "lookup", json!({"id":"a"})),
                tool_call(
                    "call-2",
                    "lookup",
                    json!({"id":"b","nested":{"list":[1,2.5,null,true]}}),
                ),
                tool_call("call-3", "lookup", json!({})),
                tool_call("call-4", "lookup", json!({})),
                tool_call("call-5", "write_file", json!({"path":"a.txt"})),
                tool_call("call-6", "write_file", json!({"path":"b.txt"})),
                tool_call("call-7", "lookup", json!({})),
                tool_call("call-8", "lookup", json!({})),
            ],
            None,
        ),
        HistoryTurn::tool(ToolMessage::new(vec![
            tool_result("call-1", ToolContent::Text("plain result".into())),
            tool_result(
                "call-2",
                ToolContent::Json(json!({"answer":42,"items":["x","y"]})),
            ),
            error_text,
            error_json,
            tool_result(
                "call-5",
                ToolContent::Denied {
                    reason: Some("user said no".into()),
                },
            ),
            tool_result("call-6", ToolContent::Denied { reason: None }),
            tool_result(
                "call-7",
                ToolContent::Mixed(vec![
                    ContentValue::Text("caption".into()),
                    ContentValue::Json(json!({"k":"v"})),
                ]),
            ),
            tool_result("call-8", ToolContent::Json(json!("bare string json"))),
        ])),
        user("Thanks"),
    ])
    .with_tools(tools());
    let body = capture(&server, &gemini(&server), request).await;
    assert_golden("tool_results_all_content_kinds", &body);
}

#[tokio::test]
async fn golden_media_inputs() {
    let server = server().await;
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(text("Describe these")),
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(bytes::Bytes::from_static(b"\x89PNG\r\n")),
        )),
        InputPart::File(FilePart::image(
            "image/jpeg",
            FileSource::Url("https://example.test/image.jpg".parse().unwrap()),
        )),
        InputPart::File(FilePart::image(
            "image/webp",
            FileSource::ProviderReference {
                provider: oven_sdk::ProviderId::new("google"),
                id: "files/uploaded-image".into(),
            },
        )),
        InputPart::File(FilePart::document(
            "application/pdf",
            FileSource::Bytes(bytes::Bytes::from_static(b"%PDF-1.7")),
        )),
        InputPart::File(FilePart::document(
            "application/pdf",
            FileSource::Url("https://example.test/file.pdf".parse().unwrap()),
        )),
        InputPart::File(FilePart::document(
            "application/pdf",
            FileSource::Text("pdf as text".into()),
        )),
        InputPart::File(FilePart::document(
            "text/plain",
            FileSource::Text("plain document\nline two".into()),
        )),
        InputPart::File(FilePart::audio(
            "audio/mp3",
            FileSource::Bytes(bytes::Bytes::from_static(b"ID3")),
        )),
        InputPart::File(FilePart::video(
            "video/mp4",
            FileSource::Url("https://example.test/clip.mp4".parse().unwrap()),
        )),
        InputPart::Text(text("Thanks")),
    ]))]);
    let body = capture(&server, &gemini(&server), request).await;
    assert_golden("media_inputs", &body);
}

#[tokio::test]
async fn golden_native_replay_same_model_keeps_thought_signatures() {
    let server = server().await;
    let model = gemini(&server);
    let first = replay_artifact(
        &model,
        json!({"role":"model","parts":[
            {"text":"Let me think.\n","thought":true,"thoughtSignature":"sig-thought"},
            {"text":"Checking."},
            {"functionCall":{"id":"call-1","name":"lookup","args":{"id":"a"}},"thoughtSignature":"sig-call"},
            {"functionCall":{"id":"call-2","name":"lookup","args":{"id":"b"}}}
        ]}),
        Some(MODEL),
    );
    let second = replay_artifact(
        &model,
        json!({"role":"model","parts":[
            {"text":"Found both.","thoughtSignature":"sig-text"}
        ]}),
        Some(MODEL),
    );
    let request = Request::new(vec![
        user("Find a and b"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("Let me think.\n")),
                AssistantPart::Text(text("Checking.")),
                tool_call("call-1", "lookup", json!({"id":"a"})),
                tool_call("call-2", "lookup", json!({"id":"b"})),
            ],
            Some(first),
        ),
        HistoryTurn::tool(ToolMessage::new(vec![
            tool_result("call-1", ToolContent::Text("record a".into())),
            tool_result("call-2", ToolContent::Text("record b".into())),
        ])),
        assistant(vec![AssistantPart::Text(text("Found both."))], Some(second)),
        user("Continue"),
    ])
    .with_tools(tools());
    let (body, result) = capture_result(&server, &model, request).await;
    assert_eq!(
        dispositions(&result),
        vec![ReplayDisposition::Replayed, ReplayDisposition::Replayed]
    );
    assert_golden("native_replay_same_model", &body);
}

#[tokio::test]
async fn golden_native_replay_other_model_strips_thought_signatures() {
    let server = server().await;
    let model = gemini(&server);
    let artifact = replay_artifact(
        &model,
        json!({"role":"model","parts":[
            {"text":"hmm","thought":true,"thoughtSignature":"sig-thought"},
            {"text":"Answer","thoughtSignature":"sig-text"}
        ]}),
        Some("gemini-2.5-pro"),
    );
    let request = Request::new(vec![
        user("Question"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("hmm")),
                AssistantPart::Text(text("Answer")),
            ],
            Some(artifact),
        ),
        user("Again"),
    ]);
    let (body, result) = capture_result(&server, &model, request).await;
    assert_eq!(dispositions(&result), vec![ReplayDisposition::Replayed]);
    assert!(
        result
            .turn
            .warnings
            .iter()
            .any(|warning| warning.contains("native reasoning excluded"))
    );
    assert_golden("native_replay_other_model", &body);
}

#[tokio::test]
async fn golden_native_replay_server_tool_parts() {
    let server = server().await;
    let model = gemini(&server);
    let artifact = replay_artifact(
        &model,
        json!({"role":"model","parts":[
            {"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","toolName":"google_search","args":{"q":"rust"},"id":"s1"},"thoughtSignature":"sig-search"},
            {"toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"items":[]},"id":"s1"}},
            {"executableCode":{"language":"PYTHON","code":"print(1)","id":"code-1"},"thoughtSignature":"sig-code"},
            {"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1","id":"code-1"}},
            {"text":"Searched and ran code."},
            {"functionCall":{"id":"call-1","name":"lookup","args":{"id":"x"}}}
        ]}),
        Some(MODEL),
    );
    let custom = |kind: &str, data: Value| AssistantPart::Custom(CustomPart::new(kind, data));
    let request = Request::new(vec![
        user("Search, compute, then look up"),
        assistant(
            vec![
                custom(
                    "google.server_tool_call",
                    json!({"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","toolName":"google_search","args":{"q":"rust"},"id":"s1"}}),
                ),
                custom(
                    "google.server_tool_result",
                    json!({"toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"items":[]},"id":"s1"}}),
                ),
                custom(
                    "google.server_tool_call",
                    json!({"executableCode":{"language":"PYTHON","code":"print(1)","id":"code-1"}}),
                ),
                custom(
                    "google.server_tool_result",
                    json!({"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1","id":"code-1"}}),
                ),
                AssistantPart::Text(text("Searched and ran code.")),
                tool_call("call-1", "lookup", json!({"id":"x"})),
            ],
            Some(artifact),
        ),
        HistoryTurn::tool(ToolMessage::new(vec![tool_result(
            "call-1",
            ToolContent::Json(json!({"found":true})),
        )])),
    ])
    .with_tools(tools())
    .with_google_options(provider_tool_options());
    let (body, result) = capture_result(&server, &model, request).await;
    assert_eq!(dispositions(&result), vec![ReplayDisposition::Replayed]);
    assert_golden("native_replay_server_tool_parts", &body);
}

#[tokio::test]
async fn golden_native_replay_from_vertex_envelope() {
    let server = server().await;
    let model = gemini(&server);
    let artifact = NativeReplayArtifact::new(
        AdapterId::new("oven.google.vertex.generate-content"),
        model.native_context_scope().clone(),
        json!({
            "format":"oven.google.vertex.generate-content.assistant.v4",
            "content":{"role":"model","parts":[
                {"text":"Looking."},
                {"functionCall":{"id":"provider-call","name":"lookup","args":{"id":"v"}}}
            ]}
        }),
    )
    .unwrap();
    let mut call = ToolCallPart::new("local-call", "lookup", json!({"id":"v"}));
    call.provider_item_id = Some("provider-call".into());
    let request = Request::new(vec![
        user("Look up v"),
        assistant(
            vec![
                AssistantPart::Text(text("Looking.")),
                AssistantPart::ToolCall(call),
            ],
            Some(artifact),
        ),
        HistoryTurn::tool(ToolMessage::new(vec![tool_result(
            "local-call",
            ToolContent::Text("record v".into()),
        )])),
    ])
    .with_tools(tools());
    let (body, result) = capture_result(&server, &model, request).await;
    assert_eq!(dispositions(&result), vec![ReplayDisposition::Replayed]);
    assert_golden("native_replay_from_vertex_envelope", &body);
}

#[tokio::test]
async fn golden_invalid_replay_reconstructs_current_turn_with_sentinel() {
    let server = server().await;
    let model = gemini(&server);
    // The payload text differs from the normalized text, so it is discarded.
    let artifact = replay_artifact(
        &model,
        json!({"role":"model","parts":[
            {"text":"different","thoughtSignature":"sig"}
        ]}),
        Some(MODEL),
    );
    let mut source = SourcePart::new();
    source.url = Some("https://example.test/source".parse().unwrap());
    let request = Request::new(vec![
        user("Do it"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("private reasoning")),
                AssistantPart::Text(text("Working.")),
                AssistantPart::Text(text("")),
                AssistantPart::Source(source),
                AssistantPart::Custom(CustomPart::new(
                    "google.server_tool_call",
                    json!({"toolCall":{"toolType":"URL_CONTEXT","args":{},"id":"u1"}}),
                )),
                AssistantPart::Custom(CustomPart::new("other.custom", json!({"x":1}))),
                tool_call("call-1", "lookup", json!({"id":"a"})),
                tool_call("call-2", "write_file", json!({"path":"out.txt"})),
            ],
            Some(artifact),
        ),
        HistoryTurn::tool(ToolMessage::new(vec![
            tool_result("call-1", ToolContent::Text("a".into())),
            tool_result("call-2", ToolContent::Text("written".into())),
        ])),
    ])
    .with_tools(tools());
    let (body, result) = capture_result(&server, &model, request).await;
    let decisions = dispositions(&result);
    assert!(matches!(
        decisions.as_slice(),
        [
            ReplayDisposition::DiscardedInvalidPayload { .. },
            ReplayDisposition::ReconstructedNormalized
        ]
    ));
    assert_golden("invalid_replay_reconstructed_current_turn_sentinel", &body);
}

#[tokio::test]
async fn golden_normalized_multi_turn_history() {
    let server = server().await;
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(text(
            "Use tools when needed.",
        ))])),
        user("Step one"),
        assistant(
            vec![
                AssistantPart::Text(text("Looking up.")),
                tool_call("call-1", "lookup", json!({"id":"1"})),
            ],
            None,
        ),
        HistoryTurn::tool(ToolMessage::new(vec![tool_result(
            "call-1",
            ToolContent::Text("one".into()),
        )])),
        assistant(
            vec![tool_call("call-2", "write_file", json!({"path":"one.txt"}))],
            None,
        ),
        HistoryTurn::tool(ToolMessage::new(vec![tool_result(
            "call-2",
            ToolContent::Json(json!({"ok":true})),
        )])),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("done with step one")),
                AssistantPart::Text(text("Step one finished.")),
            ],
            None,
        ),
        user("Step two"),
        assistant(
            vec![
                tool_call("call-3", "lookup", json!({"id":"2"})),
                tool_call("call-4", "lookup", json!({"id":"3"})),
            ],
            None,
        ),
        HistoryTurn::tool(ToolMessage::new(vec![
            tool_result("call-3", ToolContent::Text("two".into())),
            tool_result("call-4", ToolContent::Text("three".into())),
        ])),
        assistant(
            vec![tool_call("call-5", "write_file", json!({"path":"two.txt"}))],
            None,
        ),
        HistoryTurn::tool(ToolMessage::new(vec![tool_result(
            "call-5",
            ToolContent::Text("written".into()),
        )])),
    ])
    .with_tools(tools())
    .with_tool_choice(ToolChoice::Auto);
    let body = capture(&server, &gemini(&server), request).await;
    assert_golden("normalized_multi_turn_history", &body);
}

#[tokio::test]
async fn golden_replay_policy_never_reconstructs_without_sentinel_when_disabled() {
    let server = server().await;
    let mut capabilities = common::full_capabilities();
    capabilities.replay.policy = ReplayPolicy::Never;
    capabilities.replay.capability = ReplayCapability::Unsupported;
    capabilities.replay.reasoning = false;
    let model = common::model_with(
        server.uri(),
        MODEL,
        &format!("models/{MODEL}"),
        capabilities,
        common::level_thinking(),
        GoogleToolSettings {
            current_turn_signature_sentinel: false,
            ..common::default_tools()
        },
    );
    let artifact = replay_artifact(
        &model,
        json!({"role":"model","parts":[
            {"text":"Calling.","thoughtSignature":"sig"},
            {"functionCall":{"id":"call-1","name":"lookup","args":{"id":"n"}}}
        ]}),
        Some(MODEL),
    );
    let request = Request::new(vec![
        user("Never replay"),
        assistant(
            vec![
                AssistantPart::Text(text("Calling.")),
                tool_call("call-1", "lookup", json!({"id":"n"})),
            ],
            Some(artifact),
        ),
        HistoryTurn::tool(ToolMessage::new(vec![tool_result(
            "call-1",
            ToolContent::Text("n".into()),
        )])),
    ])
    .with_tools(tools());
    let (body, result) = capture_result(&server, &model, request).await;
    assert_eq!(
        dispositions(&result),
        vec![ReplayDisposition::ReconstructedNormalized]
    );
    assert_golden("replay_policy_never_without_sentinel", &body);
}

#[tokio::test]
async fn golden_structured_output_with_schema_and_thinking_level() {
    let server = server().await;
    let schema = JsonSchema::new(json!({
        "type":"object",
        "properties":{
            "answer":{"type":"string"},
            "score":{"type":"number","minimum":0,"maximum":1},
            "tags":{"type":"array","items":{"type":"string"}}
        },
        "required":["answer"],
        "propertyOrdering":["answer","score","tags"]
    }))
    .unwrap();
    let request = Request::new(vec![user("Give JSON")])
        .with_response_format(ResponseFormat::structured(schema))
        .with_inference(inference(Some(2048), None, None))
        .with_google_options(GoogleRequestOptions {
            thinking_config: Some(GoogleThinkingConfig {
                thinking_level: Some("HIGH".into()),
                include_thoughts: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        });
    let body = capture(&server, &gemini(&server), request).await;
    assert_golden("structured_output_schema_thinking_level", &body);
}

#[tokio::test]
async fn golden_json_mode_without_schema() {
    let server = server().await;
    let request = Request::new(vec![user("Any JSON")])
        .with_response_format(ResponseFormat::Json { schema: None });
    let body = capture(&server, &gemini(&server), request).await;
    assert_golden("json_mode_without_schema", &body);
}

#[tokio::test]
async fn golden_reasoning_effort_maps_to_thinking_level() {
    let server = server().await;
    let request = Request::new(vec![user("Think")]).with_inference(effort("medium"));
    let body = capture(&server, &gemini(&server), request).await;
    assert_golden("reasoning_effort_thinking_level", &body);
}

#[tokio::test]
async fn golden_reasoning_effort_maps_to_thinking_budget() {
    let server = server().await;
    let model = common::budget_model(server.uri(), MODEL);
    let request = Request::new(vec![user("Think")]).with_inference(effort("high"));
    let body = capture(&server, &model, request).await;
    assert_golden("reasoning_effort_thinking_budget", &body);
}

#[tokio::test]
async fn golden_explicit_thinking_budget() {
    let server = server().await;
    let model = common::budget_model(server.uri(), MODEL);
    let request = Request::new(vec![user("Do not think")])
        .with_inference(inference(None, Some(0.0), None))
        .with_google_options(GoogleRequestOptions {
            thinking_config: Some(GoogleThinkingConfig {
                thinking_budget: Some(0),
                include_thoughts: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        });
    let body = capture(&server, &model, request).await;
    assert_golden("explicit_thinking_budget", &body);
}
