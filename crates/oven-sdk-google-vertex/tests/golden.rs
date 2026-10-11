//! Golden request bodies: every case pins the complete JSON body sent to the provider.
//!
//! Regenerate with `OVEN_BLESS_GOLDEN=1 cargo test -p oven-sdk-google-vertex --test golden`
//! only when a wire change is intended, and review the diff against the provider docs.

mod support;

use std::path::PathBuf;

use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, CompleteResult, CompletedTurn,
    ContentValue, CustomPart, FilePart, FileSource, Finish, FinishReason, HistoryTurn,
    InferenceOptions, InputPart, JsonSchema, LanguageModel, ModelId, NativeReplayArtifact,
    ReasoningPart, ReplayCapability, ReplayDisposition, ReplayPolicy, Request, ResponseFormat,
    SourcePart, SystemMessage, SystemPart, TextPart, ToolCallPart, ToolChoice, ToolContent,
    ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};
use oven_sdk_google_vertex::{
    GOOGLE_VERTEX_GENERATE_CONTENT_ADAPTER_ID, GoogleVertexModel, GoogleVertexProviderTool,
    GoogleVertexRequestExt, GoogleVertexRequestOptions, GoogleVertexResource,
    GoogleVertexSafetySetting, GoogleVertexThinkingConfig, GoogleVertexThinkingMode,
    GoogleVertexToolExt, GoogleVertexToolOptions,
};
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

const MODEL: &str = "gemini-future";
const WIRE_MODEL: &str = "resource-model-v1";
const REPLAY_FORMAT: &str = "oven.google.vertex.generate-content.assistant.v4";

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

fn provider_call(id: &str, provider_id: &str, name: &str, input: Value) -> ToolCallPart {
    let mut call = ToolCallPart::new(id, name, input);
    call.provider_item_id = Some(provider_id.into());
    call
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
                    "filters":{"type":"array","items":{"type":"string"}},
                    "query":{"$ref":"#/$defs/query"}
                },
                "required":["id"],
                "$defs":{"query":{"type":"string"}}
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
        .with_google_vertex_tool_options(GoogleVertexToolOptions { strict: true });
    tools
}

fn inference(max: Option<u64>, temperature: Option<f64>, top_p: Option<f64>) -> InferenceOptions {
    let mut inference = InferenceOptions::new();
    inference.max_output_tokens = max;
    inference.temperature = temperature;
    inference.top_p = top_p;
    inference
}

fn publisher_resource() -> GoogleVertexResource {
    GoogleVertexResource::PublisherModel {
        publisher: "google".into(),
        model: WIRE_MODEL.into(),
    }
}

fn vertex(server: &MockServer) -> GoogleVertexModel {
    support::full_model(&server.uri(), MODEL, publisher_resource(), false)
}

fn partial_args_vertex(server: &MockServer) -> GoogleVertexModel {
    support::full_model(&server.uri(), MODEL, publisher_resource(), true)
}

fn replay_artifact(
    model: &GoogleVertexModel,
    parts: Value,
    source_model: Option<&str>,
) -> NativeReplayArtifact {
    let artifact = NativeReplayArtifact::new(
        AdapterId::new(GOOGLE_VERTEX_GENERATE_CONTENT_ADAPTER_ID),
        model.native_context_scope().clone(),
        json!({"format":REPLAY_FORMAT,"content":{"role":"model","parts":parts}}),
    )
    .unwrap();
    match source_model {
        Some(id) => artifact
            .with_source_wire_model_id(ModelId::new(id))
            .unwrap(),
        None => artifact,
    }
}

fn all_provider_tools() -> Vec<GoogleVertexProviderTool> {
    vec![
        GoogleVertexProviderTool::GoogleSearch,
        GoogleVertexProviderTool::UrlContext,
        GoogleVertexProviderTool::CodeExecution,
        GoogleVertexProviderTool::VertexRagStore {
            rag_corpus: "projects/project/locations/global/ragCorpora/corpus-a".into(),
            top_k: Some(5),
        },
        GoogleVertexProviderTool::VertexRagStore {
            rag_corpus: "projects/project/locations/global/ragCorpora/corpus-b".into(),
            top_k: None,
        },
        GoogleVertexProviderTool::GoogleMaps,
    ]
}

fn provider_tool_options() -> GoogleVertexRequestOptions {
    GoogleVertexRequestOptions {
        provider_tools: all_provider_tools(),
        ..Default::default()
    }
}

#[tokio::test]
async fn golden_plain_text() {
    let server = server().await;
    let body = capture(
        &server,
        &vertex(&server),
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
    .with_google_vertex_options(GoogleVertexRequestOptions {
        top_k: Some(40.0),
        stop_sequences: vec!["END".into(), "STOP".into()],
        seed: Some(7),
        presence_penalty: Some(0.5),
        frequency_penalty: Some(-0.25),
        thinking_config: Some(GoogleVertexThinkingConfig {
            thinking_level: Some("HIGH".into()),
            include_thoughts: Some(true),
            ..Default::default()
        }),
        cached_content: Some("projects/project/locations/global/cachedContents/cache-1".into()),
        safety_settings: vec![
            GoogleVertexSafetySetting {
                category: "HARM_CATEGORY_HARASSMENT".into(),
                threshold: "BLOCK_ONLY_HIGH".into(),
                method: Some("PROBABILITY".into()),
            },
            GoogleVertexSafetySetting {
                category: "HARM_CATEGORY_DANGEROUS_CONTENT".into(),
                threshold: "BLOCK_NONE".into(),
                method: None,
            },
        ],
        // Header-only options must not leak into the body.
        shared_request_type: Some("dedicated".into()),
        request_type: Some("shared".into()),
        ..Default::default()
    });
    let body = capture(&server, &vertex(&server), request).await;
    assert_golden("system_instruction_generation_config_and_options", &body);
}

#[tokio::test]
async fn golden_tool_choice_modes() {
    let server = server().await;
    let model = vertex(&server);
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
    let model = vertex(&server);
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
async fn golden_mixed_client_and_provider_tools() {
    let server = server().await;
    let model = vertex(&server);
    for (name, choice) in [
        ("mixed_tools_auto_choice", ToolChoice::Auto),
        (
            "mixed_tools_specific_choice",
            ToolChoice::Tool("lookup".into()),
        ),
    ] {
        let request = Request::new(vec![user("Search and look up")])
            .with_tools(tools())
            .with_tool_choice(choice)
            .with_google_vertex_options(provider_tool_options());
        let body = capture(&server, &model, request).await;
        assert_golden(name, &body);
    }
}

#[tokio::test]
async fn golden_provider_tools_only() {
    let server = server().await;
    let request = Request::new(vec![user("Search the web")])
        .with_google_vertex_options(provider_tool_options());
    let body = capture(&server, &vertex(&server), request).await;
    assert_golden("provider_tools_only", &body);
}

#[tokio::test]
async fn golden_stream_function_call_arguments() {
    let server = server().await;
    let model = partial_args_vertex(&server);
    let without_tools = capture(&server, &model, Request::new(vec![user("No tools")])).await;
    assert_golden("partial_args_without_tools", &without_tools);
    let with_tools = capture(
        &server,
        &model,
        Request::new(vec![user("Use lookup")])
            .with_tools(strict_tools())
            .with_tool_choice(ToolChoice::Tool("lookup".into())),
    )
    .await;
    assert_golden("partial_args_with_strict_specific_tool", &with_tools);
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
                AssistantPart::ToolCall(provider_call(
                    "call-1",
                    "provider-1",
                    "lookup",
                    json!({"id":"a"}),
                )),
                tool_call(
                    "call-2",
                    "lookup",
                    json!({"id":"b","nested":{"list":[1,2.5,null,true]}}),
                ),
                tool_call("call-3", "lookup", json!({})),
                AssistantPart::ToolCall(provider_call("call-4", "provider-4", "lookup", json!({}))),
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
    let body = capture(&server, &vertex(&server), request).await;
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
            FileSource::Url("gs://bucket/image.webp".parse().unwrap()),
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
            FileSource::Url("gs://bucket/file.pdf".parse().unwrap()),
        )),
        InputPart::File(FilePart::document(
            "text/plain",
            FileSource::Text("plain document\nline two".into()),
        )),
        InputPart::File(FilePart::audio(
            "audio/mpeg",
            FileSource::Bytes(bytes::Bytes::from_static(b"ID3")),
        )),
        InputPart::File(FilePart::video(
            "video/mp4",
            FileSource::Url("https://example.test/clip.mp4".parse().unwrap()),
        )),
        InputPart::File(FilePart::video(
            "video/mp4",
            FileSource::Url("gs://bucket/clip.mp4".parse().unwrap()),
        )),
        InputPart::Text(text("Thanks")),
    ]))]);
    let body = capture(&server, &vertex(&server), request).await;
    assert_golden("media_inputs", &body);
}

#[tokio::test]
async fn golden_native_replay_same_model_keeps_thought_signatures() {
    let server = server().await;
    let model = vertex(&server);
    let signed_call = json!({
        "functionCall":{"id":"provider-1","name":"lookup","args":{"id":"a"}},
        "thoughtSignature":"sig-call"
    });
    let first = replay_artifact(
        &model,
        json!([
            {"text":"Let me think.\n","thought":true,"thoughtSignature":"sig-thought"},
            {"text":"Checking."},
            signed_call.clone(),
            {"functionCall":{"id":"provider-2","name":"lookup","args":{"id":"b"}}},
            {"text":"","thoughtSignature":"sig-tail"}
        ]),
        Some(WIRE_MODEL),
    );
    let mut witnessed = provider_call("call-1", "provider-1", "lookup", json!({"id":"a"}));
    oven_sdk::replay::mark_required_vertex_signature(&mut witnessed, &signed_call);
    let second = replay_artifact(
        &model,
        json!([{"text":"Found both.","thoughtSignature":"sig-text"}]),
        Some(WIRE_MODEL),
    );
    let request = Request::new(vec![
        user("Find a and b"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("Let me think.\n")),
                AssistantPart::Text(text("Checking.")),
                AssistantPart::ToolCall(witnessed),
                AssistantPart::ToolCall(provider_call(
                    "call-2",
                    "provider-2",
                    "lookup",
                    json!({"id":"b"}),
                )),
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
    let model = vertex(&server);
    let artifact = replay_artifact(
        &model,
        json!([
            {"text":"hmm","thought":true,"thoughtSignature":"sig-thought"},
            {"text":"Answer","thoughtSignature":"sig-text"}
        ]),
        Some("other-wire-model"),
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
async fn golden_native_replay_from_gemini_api_envelope() {
    let server = server().await;
    let model = vertex(&server);
    let artifact = NativeReplayArtifact::new(
        AdapterId::new("oven.google.generate-content"),
        model.native_context_scope().clone(),
        json!({"role":"model","parts":[
            {"text":"Looking."},
            {"functionCall":{"id":"native-call","name":"lookup","args":{"id":"g"}}}
        ]}),
    )
    .unwrap();
    let request = Request::new(vec![
        user("Look up g"),
        assistant(
            vec![
                AssistantPart::Text(text("Looking.")),
                tool_call("native-call", "lookup", json!({"id":"g"})),
            ],
            Some(artifact),
        ),
        HistoryTurn::tool(ToolMessage::new(vec![tool_result(
            "native-call",
            ToolContent::Text("record g".into()),
        )])),
    ])
    .with_tools(tools());
    let (body, result) = capture_result(&server, &model, request).await;
    assert_eq!(dispositions(&result), vec![ReplayDisposition::Replayed]);
    assert_golden("native_replay_from_gemini_api_envelope", &body);
}

#[tokio::test]
async fn golden_invalid_replay_reconstructs_normalized_parts() {
    let server = server().await;
    let model = vertex(&server);
    // The payload text differs from the normalized text, so it is discarded.
    let artifact = replay_artifact(
        &model,
        json!([{"text":"different","thoughtSignature":"sig"}]),
        Some(WIRE_MODEL),
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
                AssistantPart::Custom(CustomPart::new("other.custom", json!({"x":1}))),
                AssistantPart::ToolCall(provider_call(
                    "call-1",
                    "provider-1",
                    "lookup",
                    json!({"id":"a"}),
                )),
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
    assert_golden("invalid_replay_reconstructed", &body);
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
                AssistantPart::ToolCall(provider_call(
                    "call-1",
                    "provider-1",
                    "lookup",
                    json!({"id":"1"}),
                )),
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
    ])
    .with_tools(tools())
    .with_tool_choice(ToolChoice::Auto);
    let body = capture(&server, &vertex(&server), request).await;
    assert_golden("normalized_multi_turn_history", &body);
}

#[tokio::test]
async fn golden_replay_policy_never_reconstructs() {
    let server = server().await;
    let mut config = support::full_config(&server.uri(), MODEL, publisher_resource(), false);
    config.model.capabilities.replay.policy = ReplayPolicy::Never;
    config.model.capabilities.replay.capability = ReplayCapability::Unsupported;
    config.model.capabilities.replay.reasoning = false;
    let model = GoogleVertexModel::new(config).unwrap();
    let artifact = replay_artifact(
        &model,
        json!([
            {"text":"Calling.","thoughtSignature":"sig"},
            {"functionCall":{"id":"provider-1","name":"lookup","args":{"id":"n"}}}
        ]),
        Some(WIRE_MODEL),
    );
    let request = Request::new(vec![
        user("Never replay"),
        assistant(
            vec![
                AssistantPart::Text(text("Calling.")),
                AssistantPart::ToolCall(provider_call(
                    "call-1",
                    "provider-1",
                    "lookup",
                    json!({"id":"n"}),
                )),
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
    assert_golden("replay_policy_never", &body);
}

#[tokio::test]
async fn golden_structured_output_with_schema() {
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
        .with_inference(inference(Some(2048), None, None));
    let body = capture(&server, &vertex(&server), request).await;
    assert_golden("structured_output_schema", &body);
}

#[tokio::test]
async fn golden_explicit_thinking_budget() {
    let server = server().await;
    let mut config = support::full_config(&server.uri(), MODEL, publisher_resource(), false);
    config.settings.thinking = GoogleVertexThinkingMode::Budget;
    let model = GoogleVertexModel::new(config).unwrap();
    let request = Request::new(vec![user("Do not think")])
        .with_inference(inference(None, Some(0.0), None))
        .with_google_vertex_options(GoogleVertexRequestOptions {
            thinking_config: Some(GoogleVertexThinkingConfig {
                thinking_budget: Some(0),
                include_thoughts: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        });
    let body = capture(&server, &model, request).await;
    assert_golden("explicit_thinking_budget", &body);
}

#[tokio::test]
async fn golden_reasoning_effort_is_not_encoded() {
    let server = server().await;
    let mut inference = InferenceOptions::new();
    inference.reasoning_effort = Some("high".into());
    let request = Request::new(vec![user("Think")]).with_inference(inference);
    let body = capture(&server, &vertex(&server), request).await;
    assert_golden("reasoning_effort_not_encoded", &body);
}
