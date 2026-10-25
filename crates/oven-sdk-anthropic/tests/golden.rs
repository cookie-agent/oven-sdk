//! Golden request bodies: every case pins the complete JSON body sent to the provider.
//!
//! Regenerate with `OVEN_BLESS_GOLDEN=1 cargo test -p oven-sdk-anthropic --test golden`
//! only when a wire change is intended, and review the diff against the provider docs.

mod common;

use std::{collections::BTreeMap, path::PathBuf};

use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, CompletedTurn, ContentValue,
    CustomPart, FilePart, FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions,
    InputPart, JsonSchema, LanguageModel, ModelId, NativeReplayArtifact, ReasoningPart,
    ReplayPolicy, Request, ResponseFormat, SystemMessage, SystemPart, TextPart, ToolCallPart,
    ToolChoice, ToolContent, ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};
use oven_sdk_anthropic::{
    AnthropicAwsRequestExt, AnthropicAwsRequestOptions, AnthropicCacheControl, AnthropicCacheTtl,
    AnthropicCompatibleAuth, AnthropicRequestExt, AnthropicRequestOptions, AnthropicThinking,
    AnthropicToolOptions, MiniMaxMediaExt, MiniMaxMediaOptions, MiniMaxRequestExt,
    MiniMaxRequestOptions, MiniMaxThinking,
};
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

use common::{Anthropic, AnthropicAws, MiniMax};

const MODEL: &str = "claude-sonnet-4-5";

fn response() -> &'static str {
    "event: message_start\ndata: {\"message\":{}}\n\nevent: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\nevent: message_stop\ndata: {}\n\n"
}

async fn server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    server
}

async fn capture(server: &MockServer, model: &impl LanguageModel, request: Request) -> Value {
    model
        .complete(request, AbortSignal::default())
        .await
        .expect("golden request succeeds");
    let requests = server.received_requests().await.unwrap();
    serde_json::from_slice(&requests.last().expect("request was sent").body).unwrap()
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

fn cache(ttl: AnthropicCacheTtl) -> Value {
    serde_json::to_value(AnthropicRequestOptions {
        cache_control: Some(AnthropicCacheControl { ttl }),
        ..Default::default()
    })
    .unwrap()
}

fn cache_metadata(ttl: AnthropicCacheTtl) -> Option<BTreeMap<String, Value>> {
    Some(BTreeMap::from([("anthropic".into(), cache(ttl))]))
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

fn tools() -> Vec<ToolDefinition> {
    let mut strict = ToolDefinition::new(
        "lookup",
        "Find a record",
        JsonSchema::new(json!({
            "type":"object",
            "properties":{"id":{"type":"string","description":"Record id"}},
            "required":["id"],
            "additionalProperties":false
        }))
        .unwrap(),
    );
    strict.provider_options.insert(
        "anthropic".into(),
        serde_json::to_value(AnthropicToolOptions { strict: true }).unwrap(),
    );
    let mut cached = ToolDefinition::new(
        "write_file",
        "Write a file",
        JsonSchema::new(json!({"type":"object","properties":{"path":{"type":"string"}}})).unwrap(),
    );
    cached
        .provider_options
        .insert("anthropic".into(), cache(AnthropicCacheTtl::OneHour));
    vec![strict, cached]
}

fn inference(max: Option<u64>, temperature: Option<f64>, top_p: Option<f64>) -> InferenceOptions {
    let mut inference = InferenceOptions::new();
    inference.max_output_tokens = max;
    inference.temperature = temperature;
    inference.top_p = top_p;
    inference
}

fn anthropic(server: &MockServer) -> oven_sdk_anthropic::AnthropicModel {
    Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model(MODEL)
}

/// A model pinned to API version 2023-06-01, the only one that accepts sampling.
fn anthropic_legacy_sampling(server: &MockServer) -> oven_sdk_anthropic::AnthropicModel {
    Anthropic::builder()
        .base_url(server.uri())
        .legacy_sampling()
        .build()
        .unwrap()
        .model(MODEL)
}

fn replay_artifact(
    server: &MockServer,
    content: Value,
    source_model: Option<&str>,
) -> NativeReplayArtifact {
    let scope = common::expected_native_context_scope(
        "anthropic",
        "oven.anthropic.messages",
        &server.uri(),
        MODEL,
        None,
        None,
    );
    let artifact = NativeReplayArtifact::new(
        AdapterId::new("oven.anthropic.messages"),
        scope,
        json!({
            "format": "oven.anthropic.messages.assistant.v3",
            "message": {"role": "assistant", "content": content},
            "stop_reason": "tool_use",
            "stop_sequence": null
        }),
    )
    .unwrap();
    match source_model {
        Some(model) => artifact
            .with_source_wire_model_id(ModelId::new(model))
            .unwrap(),
        None => artifact,
    }
}

#[tokio::test]
async fn golden_plain_text() {
    let server = server().await;
    let body = capture(
        &server,
        &anthropic(&server),
        Request::new(vec![user("Hello there")]),
    )
    .await;
    assert_golden("plain_text", &body);
}

#[tokio::test]
async fn golden_system_prompt_sampling_and_metadata() {
    let server = server().await;
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![
            SystemPart::Text(text("You are terse.")),
            SystemPart::Text(text("")),
            SystemPart::Custom(CustomPart::new("note", json!({"ignored":true}))),
        ])),
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(text(
            "Answer in \"quotes\" \u{1F600}\n",
        ))])),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(text("first")),
            InputPart::Text(text("")),
            InputPart::Custom(CustomPart::new("note", json!({}))),
            InputPart::Text(text("second\twith\u{0001}control")),
        ])),
    ])
    .with_inference(inference(Some(1000), Some(0.7), None))
    .with_anthropic_options(AnthropicRequestOptions {
        user_id: Some("user-123".into()),
        ..Default::default()
    });
    let body = capture(&server, &anthropic_legacy_sampling(&server), request).await;
    assert_golden("system_prompt_sampling_and_metadata", &body);
}

#[tokio::test]
async fn golden_tools_with_required_choice() {
    let server = server().await;
    let request = Request::new(vec![user("Use a tool")])
        .with_tools(tools())
        .with_tool_choice(ToolChoice::Required)
        .with_inference(inference(None, None, Some(0.9)));
    let body = capture(&server, &anthropic_legacy_sampling(&server), request).await;
    assert_golden("tools_with_required_choice", &body);
}

#[tokio::test]
async fn golden_tools_with_specific_and_none_choice() {
    let server = server().await;
    let model = anthropic(&server);
    let specific = capture(
        &server,
        &model,
        Request::new(vec![user("Use lookup")])
            .with_tools(tools())
            .with_tool_choice(ToolChoice::Tool("lookup".into())),
    )
    .await;
    assert_golden("tools_with_specific_choice", &specific);
    let none = capture(
        &server,
        &model,
        Request::new(vec![user("No tools")])
            .with_tools(tools())
            .with_tool_choice(ToolChoice::None),
    )
    .await;
    assert_golden("tools_with_none_choice", &none);
}

#[tokio::test]
async fn golden_tool_results_and_reconstructed_tool_calls() {
    let server = server().await;
    let mut error = ToolResultPart::new("call-3", ToolContent::Text("boom".into()));
    error.is_error = true;
    let mut cached = ToolResultPart::new("call-6", ToolContent::Text("cached".into()));
    cached.metadata = cache_metadata(AnthropicCacheTtl::FiveMinutes);
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
                tool_call("call-3", "lookup", json!("not an object")),
                tool_call("call-4", "lookup", json!({})),
                tool_call("call-5", "lookup", json!({})),
                tool_call("call-6", "lookup", json!({})),
                tool_call("call-7", "lookup", json!({})),
                tool_call("call-8", "lookup", json!({})),
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
            cached,
            ToolResultPart::new(
                "call-7",
                ToolContent::Mixed(vec![
                    ContentValue::Text("caption".into()),
                    ContentValue::Json(json!({"k":"v"})),
                ]),
            ),
            ToolResultPart::new(
                "call-8",
                ToolContent::Mixed(vec![
                    ContentValue::Text("screenshot".into()),
                    ContentValue::Json(json!({"w":10})),
                    ContentValue::File(FilePart::image(
                        "image/png",
                        FileSource::Bytes(bytes::Bytes::from_static(b"\x89PNG\r\n")),
                    )),
                    ContentValue::File(FilePart::document(
                        "application/pdf",
                        FileSource::Url("https://example.test/report.pdf".parse().unwrap()),
                    )),
                ]),
            ),
        ])),
        user("Thanks"),
    ])
    .with_tools(tools())
    .with_tool_choice(ToolChoice::Auto);
    let body = capture(&server, &anthropic(&server), request).await;
    assert_golden("tool_results_and_reconstructed_tool_calls", &body);
}

#[tokio::test]
async fn golden_media_inputs() {
    let server = server().await;
    let mut cached_image = FilePart::image(
        "image/jpeg",
        FileSource::Bytes(bytes::Bytes::from_static(b"jpeg-bytes")),
    );
    cached_image.metadata = cache_metadata(AnthropicCacheTtl::FiveMinutes);
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(text("Describe these")),
        InputPart::File(cached_image),
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Url("https://example.test/image.png".parse().unwrap()),
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
            "text/plain",
            FileSource::Text("plain document\nline two".into()),
        )),
    ]))]);
    let body = capture(&server, &anthropic(&server), request).await;
    assert_golden("media_inputs", &body);
}

#[tokio::test]
async fn golden_native_replay_with_signed_and_redacted_thinking() {
    let server = server().await;
    let first = replay_artifact(
        &server,
        json!([
            {"type":"thinking","thinking":"Let me think.\n","signature":"sig-abc"},
            {"type":"redacted_thinking","data":"opaque-redacted"},
            {"type":"text","text":"Checking."},
            {"type":"tool_use","id":"toolu_1","name":"lookup","input":{"id":"a"}}
        ]),
        Some(MODEL),
    );
    let mut redacted = ReasoningPart::new("");
    redacted.metadata = Some(BTreeMap::from([(
        "anthropic.redacted".into(),
        json!("opaque-redacted"),
    )]));
    let mut cached_call = ToolCallPart::new("toolu_1", "lookup", json!({"id":"a"}));
    cached_call.metadata = cache_metadata(AnthropicCacheTtl::FiveMinutes);
    let second = replay_artifact(
        &server,
        json!([{"type":"text","text":"Found it.  \n"}]),
        Some(MODEL),
    );
    let mut second_turn = assistant(
        vec![AssistantPart::Text(text("Found it.  \n"))],
        Some(second),
    );
    if let HistoryTurn::Assistant(turn) = &mut second_turn {
        turn.message
            .provider_options
            .insert("anthropic".into(), cache(AnthropicCacheTtl::FiveMinutes));
    }
    let request = Request::new(vec![
        user("Find a"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("Let me think.\n")),
                AssistantPart::Reasoning(redacted),
                AssistantPart::Text(text("Checking.")),
                AssistantPart::ToolCall(cached_call),
            ],
            Some(first),
        ),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "toolu_1",
            ToolContent::Text("record a".into()),
        )])),
        second_turn,
        user("Continue"),
    ])
    .with_tools(tools())
    .with_tool_choice(ToolChoice::Auto)
    .with_anthropic_options(AnthropicRequestOptions {
        thinking: Some(AnthropicThinking::Adaptive {
            display: Some("summarized".into()),
        }),
        ..Default::default()
    });
    let body = capture(&server, &anthropic(&server), request).await;
    assert_golden("native_replay_signed_and_redacted_thinking", &body);
}

#[tokio::test]
async fn golden_native_replay_from_other_model_drops_redacted_thinking() {
    let server = server().await;
    let artifact = replay_artifact(
        &server,
        json!([
            {"type":"thinking","thinking":"hmm","signature":"sig"},
            {"type":"redacted_thinking","data":"opaque"},
            {"type":"text","text":"Answer"}
        ]),
        Some("claude-other"),
    );
    let mut redacted = ReasoningPart::new("");
    redacted.metadata = Some(BTreeMap::from([(
        "anthropic.redacted".into(),
        json!("opaque"),
    )]));
    let request = Request::new(vec![
        user("Question"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("hmm")),
                AssistantPart::Reasoning(redacted),
                AssistantPart::Text(text("Answer")),
            ],
            Some(artifact),
        ),
        user("Again"),
    ]);
    let body = capture(&server, &anthropic(&server), request).await;
    assert_golden("native_replay_other_model", &body);
}

#[tokio::test]
async fn golden_normalized_history_and_trimmed_prefill() {
    let server = server().await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .replay_policy(ReplayPolicy::Never)
        .build()
        .unwrap()
        .model(MODEL);
    let mut marked = TextPart::new("Second answer");
    marked.metadata = cache_metadata(AnthropicCacheTtl::FiveMinutes);
    let request = Request::new(vec![
        user("One"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("private")),
                AssistantPart::Text(text("First answer\n")),
                AssistantPart::Text(text("")),
            ],
            None,
        ),
        user("Two"),
        assistant(vec![AssistantPart::Text(marked)], None),
        user("Three"),
        assistant(
            vec![
                AssistantPart::Text(text("Prefix kept ")),
                AssistantPart::Text(text("prefill   \n\t")),
            ],
            None,
        ),
    ]);
    let body = capture(&server, &model, request).await;
    assert_golden("normalized_history_trimmed_prefill", &body);
}

#[tokio::test]
async fn golden_structured_output_thinking_and_cache_markers() {
    let server = server().await;
    let mut system = TextPart::new("Cached system");
    system.metadata = cache_metadata(AnthropicCacheTtl::OneHour);
    let mut cached_user = UserMessage::new(vec![
        InputPart::Text(text("Give JSON")),
        InputPart::Text(text("")),
    ]);
    cached_user
        .provider_options
        .insert("anthropic".into(), cache(AnthropicCacheTtl::FiveMinutes));
    let schema = JsonSchema::new(json!({
        "type":"object",
        "properties":{"answer":{"type":"string"},"score":{"type":"number"}},
        "required":["answer"],
        "additionalProperties":false
    }))
    .unwrap();
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(system)])),
        HistoryTurn::user(cached_user),
    ])
    .with_response_format(ResponseFormat::structured(schema))
    .with_inference(inference(Some(1000), None, Some(0.95)))
    .with_anthropic_options(AnthropicRequestOptions {
        thinking: Some(AnthropicThinking::Enabled {
            budget_tokens: 2048,
            display: Some("omitted".into()),
        }),
        effort: Some("high".into()),
        cache_control: Some(AnthropicCacheControl {
            ttl: AnthropicCacheTtl::FiveMinutes,
        }),
        user_id: Some("user-9".into()),
        betas: vec!["some-beta".into()],
    });
    let body = capture(&server, &anthropic_legacy_sampling(&server), request).await;
    assert_golden("structured_output_thinking_cache_markers", &body);
}

#[tokio::test]
async fn golden_disabled_thinking() {
    let server = server().await;
    let request =
        Request::new(vec![user("Quick")]).with_anthropic_options(AnthropicRequestOptions {
            thinking: Some(AnthropicThinking::Disabled),
            ..Default::default()
        });
    let body = capture(&server, &anthropic(&server), request).await;
    assert_golden("disabled_thinking", &body);
}

#[tokio::test]
async fn golden_minimax_media_and_options() {
    let server = server().await;
    let model = MiniMax::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("MiniMax-M3");
    let image = FilePart::image(
        "image/png",
        FileSource::Bytes(bytes::Bytes::from_static(b"png")),
    )
    .with_minimax_media_options(MiniMaxMediaOptions {
        detail: Some("high".into()),
        max_long_side_pixel: Some(1024),
        ..Default::default()
    });
    let video = FilePart::video(
        "video/mp4",
        FileSource::Url("https://example.test/v.mp4".parse().unwrap()),
    )
    .with_minimax_media_options(MiniMaxMediaOptions {
        fps: Some(1.5),
        ..Default::default()
    });
    let reference = FilePart::video(
        "video/mp4",
        FileSource::ProviderReference {
            provider: oven_sdk::ProviderId::new("minimax"),
            id: "file-123".into(),
        },
    );
    let request = Request::new(vec![
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(text("Watch")),
            InputPart::File(image),
            InputPart::File(video),
            InputPart::File(reference),
        ])),
        assistant(vec![tool_call("call-1", "lookup", json!({"id":"m"}))], None),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call-1",
            ToolContent::Mixed(vec![ContentValue::Text("only text".into())]),
        )])),
    ])
    .with_tools(tools())
    .with_tool_choice(ToolChoice::Auto)
    .with_inference(inference(Some(2000), Some(0.3), Some(0.8)))
    .with_minimax_options(MiniMaxRequestOptions {
        thinking: Some(MiniMaxThinking::Adaptive),
        service_tier: Some("priority".into()),
        user_id: Some("mm-user".into()),
    });
    let body = capture(&server, &model, request).await;
    assert_golden("minimax_media_and_options", &body);
}

#[tokio::test]
async fn golden_aws_inference_geo() {
    let server = server().await;
    let model = AnthropicAws::builder("us-west-2", "workspace")
        .bearer_key("key")
        .base_url(server.uri())
        .build()
        .unwrap()
        .model(MODEL);
    let request = Request::new(vec![user("Hi AWS")])
        .with_tools(tools())
        .with_tool_choice(ToolChoice::Tool("write_file".into()))
        .with_anthropic_aws_options(AnthropicAwsRequestOptions {
            inference_geo: Some("us".into()),
        });
    let body = capture(&server, &model, request).await;
    assert_golden("aws_inference_geo", &body);
}

#[tokio::test]
async fn golden_compatible_video() {
    let server = server().await;
    let model = common::try_compatible_model(
        &server.uri(),
        "compatible",
        "opaque-video-model",
        "app.compatible.messages",
        AnthropicCompatibleAuth::None,
        common::anthropic_compatible_capabilities(ReplayPolicy::IfValid),
        common::anthropic_protocol(),
    )
    .unwrap();
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(text("Clip")),
        InputPart::File(
            FilePart::video(
                "video/mp4",
                FileSource::Bytes(bytes::Bytes::from_static(b"video")),
            )
            .with_minimax_media_options(MiniMaxMediaOptions {
                fps: Some(2.5),
                ..Default::default()
            }),
        ),
        InputPart::File(FilePart::image(
            "image/webp",
            FileSource::Bytes(bytes::Bytes::from_static(b"webp")),
        )),
    ]))]);
    let body = capture(&server, &model, request).await;
    assert_golden("compatible_video", &body);
}
