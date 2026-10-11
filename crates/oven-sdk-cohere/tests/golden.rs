//! Golden request bodies: every case pins the complete JSON body sent to the provider.
//!
//! Regenerate with `OVEN_BLESS_GOLDEN=1 cargo test -p oven-sdk-cohere --test golden`
//! only when a wire change is intended, and review the diff against the provider docs.

mod common;

use std::{collections::BTreeMap, path::PathBuf};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, CompletedTurn, ContentValue,
    CustomPart, FilePart, FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions,
    InputPart, JsonSchema, LanguageModel, ModelId, NativeContextScope, NativeReplayArtifact,
    ProviderId, ReasoningPart, ReplayCapability, ReplayDeclaration, ReplayDisposition,
    ReplayPolicy, Request, ResourceId, ResponseFormat, SourcePart, SystemMessage, SystemPart,
    TextPart, ToolCallPart, ToolChoice, ToolContent, ToolDefinition, ToolMessage, ToolResultPart,
    UserMessage,
};
use oven_sdk_cohere::{
    CohereDocument, CohereModel, CohereRequestExt, CohereRequestOptions, CohereSettings,
    CohereThinking,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use wiremock::MockServer;

const MODEL: &str = "command-a-reasoning-08-2025";
const REPLAY_FORMAT: &str = "cohere.v2.chat.message.v2";
const FINGERPRINT_KIND: &str = "cohere.v2.chat.replay_fingerprint";

async fn server() -> MockServer {
    let server = MockServer::start().await;
    common::mount(&server, common::text_stream("ok")).await;
    server
}

async fn capture(server: &MockServer, model: &CohereModel, request: Request) -> Value {
    capture_with_replay(server, model, request).await.0
}

async fn capture_with_replay(
    server: &MockServer,
    model: &CohereModel,
    request: Request,
) -> (Value, Vec<ReplayDisposition>) {
    let result = model
        .complete(request, AbortSignal::default())
        .await
        .expect("golden request succeeds");
    let decisions = result
        .request
        .replay
        .decisions
        .into_iter()
        .map(|decision| decision.disposition)
        .collect();
    let requests = server.received_requests().await.unwrap();
    let body = serde_json::from_slice(&requests.last().expect("request was sent").body).unwrap();
    (body, decisions)
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

fn model(server: &MockServer) -> CohereModel {
    common::model(server, MODEL)
}

fn model_with_settings(server: &MockServer, settings: CohereSettings) -> CohereModel {
    common::model_with(server, MODEL, common::capabilities(), settings)
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

fn tool_plan(value: &str) -> AssistantPart {
    let mut plan = ReasoningPart::new(value);
    plan.metadata = Some(BTreeMap::from([("cohere.kind".into(), json!("tool_plan"))]));
    AssistantPart::Reasoning(plan)
}

fn citation_source(citation: Value) -> AssistantPart {
    let mut source = SourcePart::new();
    source.id = Some("doc_1".into());
    source.metadata = Some(citation.as_object().unwrap().clone().into_iter().collect());
    AssistantPart::Source(source)
}

fn tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "lookup",
            "Find a record",
            JsonSchema::new(json!({
                "type":"object",
                "properties":{"id":{"type":"string","description":"Record id"}},
                "required":["id"],
                "additionalProperties":false
            }))
            .unwrap(),
        ),
        ToolDefinition::new(
            "write_file",
            "Write a file",
            JsonSchema::new(json!({
                "type":"object",
                "properties":{"path":{"type":"string"}},
                "required":["path"]
            }))
            .unwrap(),
        ),
    ]
}

fn inference(max: Option<u64>, temperature: Option<f64>, top_p: Option<f64>) -> InferenceOptions {
    let mut inference = InferenceOptions::new();
    inference.max_output_tokens = max;
    inference.temperature = temperature;
    inference.top_p = top_p;
    inference
}

fn fingerprint(message: &Value) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(serde_json::to_vec(message).unwrap()))
}

/// Builds a valid Cohere replay artifact plus the normalized fingerprint part that pairs it.
fn replay(message: Value) -> (NativeReplayArtifact, AssistantPart) {
    let fingerprint = fingerprint(&message);
    let scope = NativeContextScope::new(
        ProviderId::new("cohere"),
        ModelId::new(MODEL),
        ResourceId::new("cohere.v2.chat.replay_scope.v2.sha256.golden").unwrap(),
    )
    .unwrap();
    let artifact = NativeReplayArtifact::capture(
        AdapterId::new(oven_sdk_cohere::COHERE_V2_CHAT_ADAPTER_ID),
        scope,
        json!({
            "format": REPLAY_FORMAT,
            "binding": {"version":"cohere.v2.chat.replay_scope.v2","sha256":"golden"},
            "message": message,
            "fingerprint": fingerprint,
        }),
    )
    .unwrap();
    (
        artifact,
        AssistantPart::Custom(CustomPart::new(FINGERPRINT_KIND, json!(fingerprint))),
    )
}

#[tokio::test]
async fn golden_plain_text() {
    let server = server().await;
    let body = capture(
        &server,
        &model(&server),
        Request::new(vec![user("Hello there")]),
    )
    .await;
    assert_golden("plain_text", &body);
}

#[tokio::test]
async fn golden_system_prompt_and_text_concatenation() {
    let server = server().await;
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![
            SystemPart::Text(text("You are terse.")),
            SystemPart::Text(text("")),
            SystemPart::Custom(CustomPart::new("note", json!({"ignored":true}))),
            SystemPart::Text(text("Line two")),
        ])),
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(text(
            "Answer in \"quotes\" \u{1F600}\n",
        ))])),
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(text(""))])),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(text("first")),
            InputPart::Text(text("")),
            InputPart::Custom(CustomPart::new("note", json!({}))),
            InputPart::Text(text("second\twith\u{0001}control")),
        ])),
    ])
    .with_inference(inference(Some(1000), Some(0.7), Some(0.9)));
    let body = capture(&server, &model(&server), request).await;
    assert_golden("system_prompt_and_text_concatenation", &body);
}

#[tokio::test]
async fn golden_tool_choice_modes() {
    let server = server().await;
    let model = model(&server);
    for (name, choice) in [
        ("tools_with_auto_choice", ToolChoice::Auto),
        ("tools_with_required_choice", ToolChoice::Required),
        (
            "tools_with_specific_choice",
            ToolChoice::Tool("write_file".into()),
        ),
        ("tools_with_none_choice", ToolChoice::None),
    ] {
        let body = capture(
            &server,
            &model,
            Request::new(vec![user("Use a tool")])
                .with_tools(tools())
                .with_tool_choice(choice),
        )
        .await;
        assert_golden(name, &body);
    }
}

#[tokio::test]
async fn golden_strict_tools() {
    let server = server().await;
    let model = model_with_settings(
        &server,
        CohereSettings {
            strict_tools: true,
            ..Default::default()
        },
    );
    let body = capture(
        &server,
        &model,
        Request::new(vec![user("Use a tool")])
            .with_tools(tools())
            .with_tool_choice(ToolChoice::Required),
    )
    .await;
    assert_golden("tools_strict_required", &body);
}

#[tokio::test]
async fn golden_tool_results_and_normalized_history() {
    let server = server().await;
    let mut error = ToolResultPart::new("call-3", ToolContent::Text("boom".into()));
    error.is_error = true;
    let mut raw = ToolCallPart::new("call-2", "lookup", json!({"id":"b"}));
    raw.raw_input = Some("{\"id\": \"b\"}".into());
    let mut nameless = ToolCallPart::new("call-6", "lookup", json!({}));
    nameless.metadata = Some(BTreeMap::from([(
        "cohere.name_missing".into(),
        json!(true),
    )]));
    let request = Request::new(vec![
        user("Run the tools"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("private thinking")),
                tool_plan("I will call "),
                tool_plan("several tools."),
                AssistantPart::Text(text("Calling tools.")),
                AssistantPart::Text(text("")),
                tool_call(
                    "call-1",
                    "lookup",
                    json!({"id":"a","nested":{"list":[1,2.5,null,true]}}),
                ),
                AssistantPart::ToolCall(raw),
                tool_call("call-3", "lookup", json!("not an object")),
                tool_call("call-4", "lookup", json!({})),
                tool_call("call-5", "lookup", json!({})),
                AssistantPart::ToolCall(nameless),
                citation_source(json!({
                    "start":0,"end":7,"text":"Calling","type":"TEXT_CONTENT",
                    "sources":[{"type":"document","id":"doc_1","document":{"title":"Doc"}}]
                })),
            ],
            None,
        ),
        HistoryTurn::tool(ToolMessage::new(vec![
            ToolResultPart::new("call-1", ToolContent::Text("plain result".into())),
            ToolResultPart::new(
                "call-2",
                ToolContent::Json(json!({"answer":42,"items":["x","y"]})),
            ),
            error,
            ToolResultPart::new(
                "call-4",
                ToolContent::Denied {
                    reason: Some("user said no".into()),
                },
            ),
            ToolResultPart::new("call-5", ToolContent::Denied { reason: None }),
            ToolResultPart::new(
                "call-6",
                ToolContent::Mixed(vec![
                    ContentValue::Text("caption".into()),
                    ContentValue::Json(json!({"k":"v"})),
                ]),
            ),
        ])),
        user("Thanks"),
        assistant(
            vec![
                tool_call("call-7", "write_file", json!({"path":"/tmp/x"})),
                AssistantPart::ToolResult(ToolResultPart::new(
                    "call-7",
                    ToolContent::Text("written".into()),
                )),
            ],
            None,
        ),
        user("Done?"),
    ])
    .with_tools(tools())
    .with_tool_choice(ToolChoice::Auto);
    let (body, replay) = capture_with_replay(&server, &model(&server), request).await;
    assert_eq!(
        replay,
        vec![
            ReplayDisposition::NoArtifact,
            ReplayDisposition::ReconstructedNormalized,
            ReplayDisposition::NoArtifact,
            ReplayDisposition::ReconstructedNormalized,
        ]
    );
    assert_golden("tool_results_and_normalized_history", &body);
}

#[tokio::test]
async fn golden_image_inputs() {
    let server = server().await;
    let mut detailed = FilePart::image(
        "image/jpeg",
        FileSource::Url("https://example.test/image.jpg?size=large".parse().unwrap()),
    );
    detailed.metadata = Some(BTreeMap::from([(
        "cohere.image_detail".into(),
        json!("high"),
    )]));
    let request = Request::new(vec![
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(text("Describe these")),
            InputPart::File(FilePart::image(
                "image/png",
                FileSource::Bytes(bytes::Bytes::from_static(b"\x89PNG\r\n")),
            )),
            InputPart::File(detailed),
            InputPart::Text(text("")),
            InputPart::File(FilePart::image(
                "image/gif",
                FileSource::Bytes(bytes::Bytes::from_static(b"GIF89a")),
            )),
            InputPart::Text(text("trailing text")),
        ])),
        HistoryTurn::user(UserMessage::new(vec![InputPart::File(FilePart::image(
            "image/webp",
            FileSource::Url("http://example.test/only.webp".parse().unwrap()),
        ))])),
    ])
    .with_cohere_options(CohereRequestOptions {
        image_detail: Some("low".into()),
        ..Default::default()
    });
    let body = capture(&server, &model(&server), request).await;
    assert_golden("image_inputs", &body);
}

#[tokio::test]
async fn golden_native_replay_with_reasoning() {
    let server = server().await;
    let citation = json!({
        "start":0,"end":6,"text":"answer","type":"TEXT_CONTENT",
        "sources":[{"type":"document","id":"doc_1"}]
    });
    let (first, first_fingerprint) = replay(json!({
        "role":"assistant",
        "content":[
            {"type":"thinking","thinking":"private thought"},
            {"type":"text","text":"answer"}
        ],
        "citations":[citation.clone()]
    }));
    let mut raw = ToolCallPart::new("call-1", "lookup", json!({"id":"a"}));
    raw.raw_input = Some("{\"id\":\"a\"}".into());
    let (second, second_fingerprint) = replay(json!({
        "role":"assistant",
        "tool_plan":"Call lookup",
        "tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"id\":\"a\"}"}}]
    }));
    // Stored without thinking although the normalized turn has reasoning: replay sends the
    // stored message, whereas normalized reconstruction would add a thinking block.
    let (third, third_fingerprint) = replay(json!({
        "role":"assistant",
        "content":[{"type":"text","text":"final"}]
    }));
    let request = Request::new(vec![
        user("Question"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("private thought")),
                AssistantPart::Text(text("answer")),
                citation_source(citation),
                first_fingerprint,
            ],
            Some(first),
        ),
        user("Look it up"),
        assistant(
            vec![
                tool_plan("Call lookup"),
                AssistantPart::ToolCall(raw),
                second_fingerprint,
            ],
            Some(second),
        ),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call-1",
            ToolContent::Text("record a".into()),
        )])),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("hidden")),
                AssistantPart::Text(text("final")),
                third_fingerprint,
            ],
            Some(third),
        ),
        user("Continue"),
    ])
    .with_tools(tools());
    let (body, replay) = capture_with_replay(&server, &model(&server), request).await;
    assert_eq!(
        replay,
        vec![
            ReplayDisposition::Replayed,
            ReplayDisposition::Replayed,
            ReplayDisposition::Replayed,
        ]
    );
    assert_golden("native_replay_with_reasoning", &body);
}

#[tokio::test]
async fn golden_native_replay_without_reasoning_replay() {
    let server = server().await;
    let mut capabilities = common::capabilities();
    capabilities.replay.reasoning = false;
    let model = common::model_with(&server, MODEL, capabilities, CohereSettings::default());
    let message = json!({
        "role":"assistant",
        "content":[
            {"type":"thinking","thinking":"private thought"},
            {"type":"text","text":"answer"}
        ]
    });
    let (valid, valid_fingerprint) = replay(message.clone());
    let (tampered, _) = replay(message);
    let parts = |fingerprint: AssistantPart| {
        vec![
            AssistantPart::Reasoning(ReasoningPart::new("private thought")),
            AssistantPart::Text(text("answer")),
            fingerprint,
        ]
    };
    let request = Request::new(vec![
        user("Question"),
        assistant(parts(valid_fingerprint), Some(valid)),
        user("Again"),
        // Fingerprint part does not pair with the artifact: normalized fallback keeps thinking.
        assistant(
            parts(AssistantPart::Custom(CustomPart::new(
                FINGERPRINT_KIND,
                json!("not-the-fingerprint"),
            ))),
            Some(tampered),
        ),
        user("Once more"),
    ]);
    let (body, replay) = capture_with_replay(&server, &model, request).await;
    assert!(matches!(
        replay.as_slice(),
        [
            ReplayDisposition::Replayed,
            ReplayDisposition::DiscardedInvalidPayload { .. },
            ReplayDisposition::ReconstructedNormalized,
        ]
    ));
    assert_golden("native_replay_without_reasoning_replay", &body);
}

#[tokio::test]
async fn golden_invalid_replay_falls_back_to_normalized_history() {
    let server = server().await;
    let (artifact, _) = replay(json!({
        "role":"assistant",
        "content":[{"type":"text","text":"different native text"}]
    }));
    let (mismatched, mismatched_fingerprint) = replay(json!({
        "role":"assistant",
        "content":[{"type":"text","text":"different native text"}]
    }));
    let request = Request::new(vec![
        user("Question"),
        // Missing fingerprint part.
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("think")),
                AssistantPart::Text(text("normalized text")),
            ],
            Some(artifact),
        ),
        user("Again"),
        // Fingerprint pairs, but the stored message differs from the normalized content.
        assistant(
            vec![
                AssistantPart::Text(text("normalized text")),
                mismatched_fingerprint,
            ],
            Some(mismatched),
        ),
        user("Once more"),
    ]);
    let (body, replay) = capture_with_replay(&server, &model(&server), request).await;
    assert!(matches!(
        replay.as_slice(),
        [
            ReplayDisposition::DiscardedInvalidPayload { .. },
            ReplayDisposition::ReconstructedNormalized,
            ReplayDisposition::DiscardedInvalidPayload { .. },
            ReplayDisposition::ReconstructedNormalized,
        ]
    ));
    assert_golden("invalid_replay_normalized_fallback", &body);
}

#[tokio::test]
async fn golden_replay_policy_never_reconstructs_normalized_history() {
    let server = server().await;
    let mut capabilities = common::capabilities();
    capabilities.replay = ReplayDeclaration {
        policy: ReplayPolicy::Never,
        capability: ReplayCapability::Unsupported,
        reasoning: false,
    };
    let model = common::model_with(&server, MODEL, capabilities, CohereSettings::default());
    let (artifact, fingerprint) = replay(json!({
        "role":"assistant",
        "content":[{"type":"text","text":"Answer"}]
    }));
    let request = Request::new(vec![
        user("Question"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("think")),
                AssistantPart::Text(text("Answer")),
                fingerprint,
            ],
            Some(artifact),
        ),
        user("Again"),
    ]);
    let (body, replay) = capture_with_replay(&server, &model, request).await;
    assert_eq!(replay, vec![ReplayDisposition::ReconstructedNormalized]);
    assert_golden("replay_policy_never_normalized", &body);
}

#[tokio::test]
async fn golden_settings_sampling_and_thinking() {
    let server = server().await;
    let model = model_with_settings(
        &server,
        CohereSettings {
            safety_mode: Some("CONTEXTUAL".into()),
            thinking: Some(CohereThinking {
                enabled: true,
                token_budget: Some(2048),
            }),
            top_k: Some(40),
            seed: Some(7),
            frequency_penalty: Some(0.25),
            presence_penalty: Some(0.5),
            stop_sequences: vec!["END".into(), "\n\nUser:".into()],
            priority: Some(1),
            ..Default::default()
        },
    );
    let request = Request::new(vec![user("Sample")]).with_inference(inference(
        Some(512),
        Some(0.3),
        Some(0.75),
    ));
    let body = capture(&server, &model, request).await;
    assert_golden("settings_sampling_and_thinking", &body);
}

#[tokio::test]
async fn golden_thinking_resolution() {
    let server = server().await;
    let model = model_with_settings(
        &server,
        CohereSettings {
            thinking: Some(CohereThinking {
                enabled: false,
                token_budget: None,
            }),
            reasoning_effort: BTreeMap::from([
                (
                    "high".into(),
                    CohereThinking {
                        enabled: true,
                        token_budget: Some(8192),
                    },
                ),
                (
                    "minimal".into(),
                    CohereThinking {
                        enabled: true,
                        token_budget: None,
                    },
                ),
            ]),
            ..Default::default()
        },
    );
    let body = capture(&server, &model, Request::new(vec![user("Default")])).await;
    assert_golden("thinking_default_disabled", &body);
    for (name, effort) in [
        ("thinking_effort_high", "high"),
        ("thinking_effort_minimal", "minimal"),
    ] {
        let mut request = Request::new(vec![user("Effort")]);
        request.inference.reasoning_effort = Some(effort.into());
        let body = capture(&server, &model, request).await;
        assert_golden(name, &body);
    }
}

#[tokio::test]
async fn golden_documents_and_citation_mode() {
    let server = server().await;
    let request = Request::new(vec![user("Summarize the documents")])
        .with_tools(tools())
        .with_tool_choice(ToolChoice::Auto)
        .with_cohere_options(CohereRequestOptions {
            documents: vec![
                CohereDocument {
                    id: Some("doc_1".into()),
                    data: json!({"title":"Doc","text":"source","meta":{"page":3}})
                        .as_object()
                        .unwrap()
                        .clone(),
                },
                CohereDocument {
                    id: None,
                    data: json!({"text":"anonymous"}).as_object().unwrap().clone(),
                },
            ],
            citation_mode: Some("ACCURATE".into()),
            image_detail: None,
        });
    let body = capture(&server, &model(&server), request).await;
    assert_golden("documents_and_citation_mode", &body);
}

#[tokio::test]
async fn golden_structured_output() {
    let server = server().await;
    let model = model(&server);
    let schema = JsonSchema::new(json!({
        "type":"object",
        "properties":{
            "answer":{"type":"string","description":"The answer"},
            "score":{"type":"number"},
            "when":{"type":"string","format":"date-time"}
        },
        "required":["answer"],
        "additionalProperties":false
    }))
    .unwrap();
    let body = capture(
        &server,
        &model,
        Request::new(vec![user("Give JSON")])
            .with_response_format(ResponseFormat::structured(schema)),
    )
    .await;
    assert_golden("structured_output_schema", &body);
    let body = capture(
        &server,
        &model,
        Request::new(vec![user("Give JSON")])
            .with_response_format(ResponseFormat::Json { schema: None }),
    )
    .await;
    assert_golden("structured_output_json_object", &body);
}
