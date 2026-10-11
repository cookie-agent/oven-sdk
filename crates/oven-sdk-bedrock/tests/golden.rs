//! Golden request bodies: every case pins the complete JSON body sent to the provider.
//!
//! Regenerate with `OVEN_BLESS_GOLDEN=1 cargo test -p oven-sdk-bedrock --test golden`
//! only when a wire change is intended, and review the diff against the provider docs.

mod support;

use std::{collections::BTreeMap, path::PathBuf, time::SystemTime};

use oven_sdk::{
    AbortSignal, AdapterId, AssistantMessage, AssistantPart, Capability, CompletedTurn,
    ContentValue, FilePart, FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions,
    InputPart, JsonSchema, LanguageModel, MediaInputSupport, MediaSourceSupport, Modality,
    ModelConfig, ModelId, NativeReplayArtifact, ReasoningPart, ReplayCapability, ReplayDeclaration,
    ReplayDisposition, ReplayPolicy, Request, ResponseFormat, SystemMessage, SystemPart, TextPart,
    ToolCallPart, ToolChoice, ToolContent, ToolDefinition, ToolMessage, ToolResultPart,
    UserMessage,
};
use oven_sdk_bedrock::{
    BedrockAuth, BedrockCachePoint, BedrockCacheStrategy, BedrockCacheTtl, BedrockConverseSettings,
    BedrockGuardrailConfig, BedrockMessageCachePoint, BedrockModel, BedrockReasoningWireFormat,
    BedrockRequestExt, BedrockRequestOptions,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use support::FixtureKind;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path_regex},
};

const MODEL: &str = "anthropic.claude-sonnet-4-6";
const REPLAY_FORMAT: &str = "oven.bedrock.converse.assistant.v2";

async fn server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex("/converse-stream$"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("ok")))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path_regex("/converse$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},
            "stopReason":"end_turn",
            "usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2},
            "metrics":{"latencyMs":1}
        })))
        .mount(&server)
        .await;
    server
}

async fn last_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.unwrap();
    serde_json::from_slice(&requests.last().expect("request was sent").body).unwrap()
}

/// Streams through `ConverseStream` (the `LanguageModel` path) and returns the sent body.
async fn capture(server: &MockServer, model: &BedrockModel, request: Request) -> Value {
    capture_with_replay(server, model, request).await.0
}

async fn capture_with_replay(
    server: &MockServer,
    model: &BedrockModel,
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
    (last_body(server).await, decisions)
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

fn model_with(
    server: &MockServer,
    model_id: &str,
    kind: FixtureKind,
    configure: impl FnOnce(&mut ModelConfig<BedrockAuth, BedrockConverseSettings>),
) -> BedrockModel {
    let mut config = support::config(
        &server.uri(),
        model_id,
        kind,
        BedrockAuth::Static(support::credentials()),
    );
    configure(&mut config);
    BedrockModel::new(config).expect("golden model configuration")
}

fn model(server: &MockServer, kind: FixtureKind) -> BedrockModel {
    model_with(server, MODEL, kind, |_| {})
}

fn with_pdf(config: &mut ModelConfig<BedrockAuth, BedrockConverseSettings>) {
    let capabilities = &mut config.model.capabilities;
    capabilities.modalities.input.insert(Modality::pdf());
    capabilities.media.input.insert(
        Modality::pdf(),
        MediaInputSupport::new(
            ["application/pdf".to_owned()],
            MediaSourceSupport::INLINE_BYTES,
        )
        .unwrap(),
    );
}

fn with_caching(config: &mut ModelConfig<BedrockAuth, BedrockConverseSettings>) {
    config.model.capabilities.features |= Capability::PROMPT_CACHING;
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
            JsonSchema::new(json!({"type":"object","properties":{"path":{"type":"string"}}}))
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

fn replay_artifact(
    model: &BedrockModel,
    content: Value,
    source_model: Option<&str>,
) -> NativeReplayArtifact {
    let artifact = NativeReplayArtifact::new(
        AdapterId::new(oven_sdk_bedrock::BEDROCK_CONVERSE_ADAPTER_ID),
        model.native_context_scope().clone(),
        json!({"format": REPLAY_FORMAT, "assistant_content": content}),
    )
    .unwrap();
    match source_model {
        Some(id) => artifact
            .with_source_wire_model_id(ModelId::new(id))
            .unwrap(),
        None => artifact,
    }
}

fn png() -> FilePart {
    FilePart::image(
        "image/png",
        FileSource::Bytes(bytes::Bytes::from_static(b"\x89PNG\r\n")),
    )
}

fn pdf(filename: Option<&str>) -> FilePart {
    let mut file = FilePart::document(
        "application/pdf",
        FileSource::Bytes(bytes::Bytes::from_static(b"%PDF-1.7")),
    );
    file.filename = filename.map(str::to_owned);
    file
}

#[tokio::test]
async fn golden_plain_text() {
    let server = server().await;
    let body = capture(
        &server,
        &model(&server, FixtureKind::Text),
        Request::new(vec![user("Hello there")]),
    )
    .await;
    assert_golden("plain_text", &body);
}

#[tokio::test]
async fn golden_system_prompt_inference_and_merged_user_turns() {
    let server = server().await;
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![
            SystemPart::Text(text("You are terse.")),
            SystemPart::Text(text("")),
        ])),
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(text(
            "Answer in \"quotes\" \u{1F600}\n",
        ))])),
        HistoryTurn::user(UserMessage::new(vec![
            InputPart::Text(text("first")),
            InputPart::Text(text("")),
            InputPart::Text(text("second\twith\u{0001}control")),
        ])),
        user("consecutive user turn"),
    ])
    .with_inference(inference(Some(1000), Some(0.7), Some(0.9)))
    .with_bedrock_options(BedrockRequestOptions {
        stop_sequences: vec!["STOP".into(), "\n\nHuman:".into()],
        ..Default::default()
    });
    let body = capture(&server, &model(&server, FixtureKind::Text), request).await;
    assert_golden("system_prompt_inference_and_merged_user_turns", &body);
}

#[tokio::test]
async fn golden_tool_choice_modes() {
    let server = server().await;
    let model = model(&server, FixtureKind::MediaTools);
    for (name, choice) in [
        ("tools_with_auto_choice", ToolChoice::Auto),
        ("tools_with_required_choice", ToolChoice::Required),
        (
            "tools_with_specific_choice",
            ToolChoice::Tool("write_file".into()),
        ),
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
    // `ToolChoice::None` with tools is rejected; without tools it encodes no toolConfig.
    let body = capture(
        &server,
        &model,
        Request::new(vec![user("No tools")]).with_tool_choice(ToolChoice::None),
    )
    .await;
    assert_golden("tool_choice_none_without_tools", &body);
}

#[tokio::test]
async fn golden_tool_results_and_reconstructed_tool_calls() {
    let server = server().await;
    let model = model_with(&server, MODEL, FixtureKind::MediaTools, with_pdf);
    let mut error = ToolResultPart::new("call-3", ToolContent::Text("boom".into()));
    error.is_error = true;
    let request = Request::new(vec![
        user("Run the tools"),
        assistant(
            vec![
                AssistantPart::Text(text("Calling tools.")),
                AssistantPart::Text(text("")),
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
            ToolResultPart::new(
                "call-7",
                ToolContent::Mixed(vec![
                    ContentValue::Text("screenshot".into()),
                    ContentValue::File(png()),
                    ContentValue::File(pdf(Some("tool report.pdf"))),
                ]),
            ),
        ])),
        user("Thanks"),
        assistant(
            vec![
                AssistantPart::Text(text("One more.")),
                tool_call("call-8", "write_file", json!({"path":"/tmp/x"})),
                AssistantPart::ToolResult(ToolResultPart::new(
                    "call-8",
                    ToolContent::Text("written".into()),
                )),
            ],
            None,
        ),
        user("Done?"),
    ])
    .with_tools(tools())
    .with_tool_choice(ToolChoice::Auto);
    let (body, replay) = capture_with_replay(&server, &model, request).await;
    assert_eq!(
        replay,
        vec![
            ReplayDisposition::NoArtifact,
            ReplayDisposition::ReconstructedNormalized,
            ReplayDisposition::NoArtifact,
            ReplayDisposition::ReconstructedNormalized,
        ]
    );
    assert_golden("tool_results_and_reconstructed_tool_calls", &body);
}

#[tokio::test]
async fn golden_media_inputs() {
    let server = server().await;
    let model = model_with(&server, MODEL, FixtureKind::MediaTools, with_pdf);
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(text("Describe these")),
        InputPart::File(png()),
        InputPart::File(FilePart::image(
            "image/jpeg",
            FileSource::Bytes(bytes::Bytes::from_static(b"jpeg-bytes")),
        )),
        InputPart::File(FilePart::image(
            "image/gif",
            FileSource::Bytes(bytes::Bytes::from_static(b"GIF89a")),
        )),
        InputPart::File(FilePart::image(
            "image/webp",
            FileSource::Bytes(bytes::Bytes::from_static(b"RIFFwebp")),
        )),
        InputPart::File(pdf(Some("Quarterly report_(final)  v2.final.pdf"))),
        InputPart::File(pdf(None)),
        InputPart::File(pdf(Some(".pdf"))),
        InputPart::File(FilePart::video(
            "video/mp4",
            FileSource::Bytes(bytes::Bytes::from_static(b"mp4-bytes")),
        )),
        InputPart::Text(text("trailing text")),
    ]))]);
    let body = capture(&server, &model, request).await;
    assert_golden("media_inputs", &body);
}

#[tokio::test]
async fn golden_native_replay_signed_and_redacted_reasoning() {
    let server = server().await;
    let model = model(&server, FixtureKind::SignedReasoning);
    let first = replay_artifact(
        &model,
        json!([
            {"reasoningContent":{"reasoningText":{"text":"Let me think.\n","signature":"sig-abc"}}},
            {"reasoningContent":{"redactedContent":"b3BhcXVlLXJlZGFjdGVk"}},
            {"text":"Checking."},
            {"toolUse":{"toolUseId":"tooluse_1","name":"lookup","input":{"id":"a"}}}
        ]),
        Some(MODEL),
    );
    let second = replay_artifact(
        &model,
        json!([
            {"reasoningContent":{"reasoningText":{"text":"unsigned thought"}}},
            {"text":"Found it.  \n"}
        ]),
        Some(MODEL),
    );
    let request = Request::new(vec![
        user("Find a"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("Let me think.\n")),
                AssistantPart::Text(text("Checking.")),
                tool_call("tooluse_1", "lookup", json!({"id":"a"})),
            ],
            Some(first),
        ),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "tooluse_1",
            ToolContent::Text("record a".into()),
        )])),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("unsigned thought")),
                AssistantPart::Text(text("Found it.  \n")),
            ],
            Some(second),
        ),
        user("Continue"),
    ])
    .with_tools(tools())
    .with_tool_choice(ToolChoice::Auto);
    let (body, replay) = capture_with_replay(&server, &model, request).await;
    assert_eq!(
        replay,
        vec![ReplayDisposition::Replayed, ReplayDisposition::Replayed]
    );
    assert_golden("native_replay_signed_and_redacted_reasoning", &body);
}

#[tokio::test]
async fn golden_native_replay_from_other_or_unknown_model_omits_redacted() {
    let server = server().await;
    let model = model(&server, FixtureKind::SignedReasoning);
    let content = json!([
        {"reasoningContent":{"reasoningText":{"text":"hmm","signature":"sig"}}},
        {"reasoningContent":{"redactedContent":"opaque"}},
        {"text":"Answer"}
    ]);
    let parts = || {
        vec![
            AssistantPart::Reasoning(ReasoningPart::new("hmm")),
            AssistantPart::Text(text("Answer")),
        ]
    };
    let request = Request::new(vec![
        user("Question"),
        assistant(
            parts(),
            Some(replay_artifact(
                &model,
                content.clone(),
                Some("anthropic.claude-other"),
            )),
        ),
        user("Again"),
        assistant(parts(), Some(replay_artifact(&model, content, None))),
        user("Once more"),
    ]);
    let (body, replay) = capture_with_replay(&server, &model, request).await;
    assert_eq!(
        replay,
        vec![ReplayDisposition::Replayed, ReplayDisposition::Replayed]
    );
    assert_golden("native_replay_other_or_unknown_model", &body);
}

#[tokio::test]
async fn golden_native_replay_unsigned_target_omits_redacted() {
    let server = server().await;
    let model = model(&server, FixtureKind::UnsignedReasoning);
    let artifact = replay_artifact(
        &model,
        json!([
            {"reasoningContent":{"reasoningText":{"text":"plain thought","signature":"kept-anyway"}}},
            {"reasoningContent":{"redactedContent":"opaque"}},
            {"text":"Answer"}
        ]),
        Some(MODEL),
    );
    let request = Request::new(vec![
        user("Question"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("plain thought")),
                AssistantPart::Text(text("Answer")),
            ],
            Some(artifact),
        ),
        user("Again"),
    ]);
    let (body, replay) = capture_with_replay(&server, &model, request).await;
    assert_eq!(replay, vec![ReplayDisposition::Replayed]);
    assert_golden("native_replay_unsigned_target", &body);
}

#[tokio::test]
async fn golden_native_replay_reasoning_unsupported_target() {
    let server = server().await;
    let model = model(&server, FixtureKind::MediaTools);
    let artifact = replay_artifact(
        &model,
        json!([
            {"reasoningContent":{"redactedContent":"opaque"}},
            {"text":"Answer"}
        ]),
        Some(MODEL),
    );
    let request = Request::new(vec![
        user("Question"),
        assistant(vec![AssistantPart::Text(text("Answer"))], Some(artifact)),
        user("Again"),
    ]);
    let (body, replay) = capture_with_replay(&server, &model, request).await;
    assert_eq!(replay, vec![ReplayDisposition::Replayed]);
    assert_golden("native_replay_reasoning_unsupported_target", &body);
}

#[tokio::test]
async fn golden_invalid_replay_falls_back_to_normalized_history() {
    let server = server().await;
    let model = model(&server, FixtureKind::SignedReasoning);
    let mismatched = replay_artifact(
        &model,
        json!([
            {"reasoningContent":{"reasoningText":{"text":"think","signature":"sig"}}},
            {"text":"different native text"}
        ]),
        Some(MODEL),
    );
    let request = Request::new(vec![
        user("Question"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("think")),
                AssistantPart::Text(text("normalized text")),
                tool_call("call-1", "lookup", json!({"id":"z"})),
            ],
            Some(mismatched),
        ),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call-1",
            ToolContent::Json(json!({"found":false})),
        )])),
    ])
    .with_tools(tools());
    let (body, replay) = capture_with_replay(&server, &model, request).await;
    assert!(matches!(
        replay.as_slice(),
        [
            ReplayDisposition::DiscardedInvalidPayload { .. },
            ReplayDisposition::ReconstructedNormalized
        ]
    ));
    assert_golden("invalid_replay_normalized_fallback", &body);
}

#[tokio::test]
async fn golden_replay_policy_never_reconstructs_normalized_history() {
    let server = server().await;
    let model = model_with(&server, MODEL, FixtureKind::UnsignedReasoning, |config| {
        config.model.capabilities.replay = ReplayDeclaration {
            policy: ReplayPolicy::Never,
            capability: ReplayCapability::Unsupported,
            reasoning: false,
        };
    });
    let artifact = replay_artifact(
        &model,
        json!([
            {"reasoningContent":{"reasoningText":{"text":"think","signature":"sig"}}},
            {"text":"Answer"}
        ]),
        Some(MODEL),
    );
    let request = Request::new(vec![
        user("Question"),
        assistant(
            vec![
                AssistantPart::Reasoning(ReasoningPart::new("think")),
                AssistantPart::Text(text("Answer")),
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
async fn golden_structured_output() {
    let server = server().await;
    let schema = JsonSchema::new(json!({
        "type":"object",
        "properties":{"answer":{"type":"string"},"score":{"type":"number"}},
        "required":["answer"],
        "additionalProperties":false
    }))
    .unwrap();
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(text(
            "Reply in JSON",
        ))])),
        user("Give JSON"),
    ])
    .with_response_format(ResponseFormat::structured(schema))
    .with_inference(inference(Some(500), None, None));
    let body = capture(
        &server,
        &model(&server, FixtureKind::SignedReasoning),
        request,
    )
    .await;
    assert_golden("structured_output", &body);
}

fn reasoning_model(server: &MockServer, wire: BedrockReasoningWireFormat) -> BedrockModel {
    let kind = if wire == BedrockReasoningWireFormat::AnthropicThinking {
        FixtureKind::SignedReasoning
    } else {
        FixtureKind::UnsignedReasoning
    };
    model_with(server, "opaque-reasoning-model", kind, |config| {
        config.settings.reasoning_wire_format = wire;
    })
}

#[tokio::test]
async fn golden_reasoning_wire_formats() {
    let server = server().await;
    let anthropic = reasoning_model(&server, BedrockReasoningWireFormat::AnthropicThinking);
    let cases = [
        (
            "reasoning_anthropic_adaptive_display_effort",
            &anthropic,
            Request::new(vec![user("Think")]).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("adaptive".into()),
                reasoning_display: Some("summarized".into()),
                max_reasoning_effort: Some("high".into()),
                additional_model_request_fields: Some(json!({"top_k":40})),
                ..Default::default()
            }),
        ),
        (
            "reasoning_anthropic_budget_and_inference_effort",
            &anthropic,
            {
                let mut request = Request::new(vec![user("Think")])
                    .with_inference(inference(Some(4096), None, None))
                    .with_bedrock_options(BedrockRequestOptions {
                        reasoning_budget_tokens: Some(2048),
                        ..Default::default()
                    });
                request.inference.reasoning_effort = Some("medium".into());
                request
            },
        ),
        (
            "reasoning_anthropic_disabled",
            &anthropic,
            Request::new(vec![user("Quick")])
                .with_inference(inference(None, Some(0.2), None))
                .with_bedrock_options(BedrockRequestOptions {
                    reasoning_type: Some("disabled".into()),
                    ..Default::default()
                }),
        ),
    ];
    for (name, model, request) in cases {
        let body = capture(&server, model, request).await;
        assert_golden(name, &body);
    }

    let openai = reasoning_model(&server, BedrockReasoningWireFormat::OpenAiReasoningEffort);
    let mut request =
        Request::new(vec![user("Think")]).with_bedrock_options(BedrockRequestOptions {
            additional_model_request_fields: Some(json!({"custom":{"nested":[1,2]}})),
            ..Default::default()
        });
    request.inference.reasoning_effort = Some("low".into());
    let body = capture(&server, &openai, request).await;
    assert_golden("reasoning_openai_effort", &body);

    let generic = reasoning_model(&server, BedrockReasoningWireFormat::BedrockReasoningConfig);
    let body = capture(
        &server,
        &generic,
        Request::new(vec![user("Think")]).with_bedrock_options(BedrockRequestOptions {
            reasoning_type: Some("enabled".into()),
            reasoning_budget_tokens: Some(1024),
            max_reasoning_effort: Some("future-effort".into()),
            ..Default::default()
        }),
    )
    .await;
    assert_golden("reasoning_bedrock_config", &body);
}

fn full_options() -> BedrockRequestOptions {
    BedrockRequestOptions {
        additional_model_request_fields: Some(json!({
            "top_k": 40,
            "anthropic_beta": ["some-beta"],
            "nested": {"deep": {"value": null}}
        })),
        additional_model_response_field_paths: vec!["/stop_sequence".into(), "/usage/x".into()],
        service_tier: Some("priority".into()),
        performance_latency: Some("optimized".into()),
        request_metadata: BTreeMap::from([
            ("tenant".into(), "acme".into()),
            ("trace".into(), "abc-123".into()),
        ]),
        guardrail: Some(BedrockGuardrailConfig {
            guardrail_identifier: "gr-123".into(),
            guardrail_version: "DRAFT".into(),
            trace: Some("enabled_full".into()),
            stream_processing_mode: Some("async".into()),
        }),
        s3: None,
        cache: Some(BedrockCacheStrategy {
            system: Some(BedrockCachePoint {
                ttl: Some(BedrockCacheTtl::OneHour),
            }),
            tools: Some(BedrockCachePoint {
                ttl: Some(BedrockCacheTtl::OneHour),
            }),
            messages: vec![
                BedrockMessageCachePoint {
                    history_index: 4,
                    cache_point: BedrockCachePoint::default(),
                },
                BedrockMessageCachePoint {
                    history_index: 2,
                    cache_point: BedrockCachePoint {
                        ttl: Some(BedrockCacheTtl::FiveMinutes),
                    },
                },
            ],
        }),
        stop_sequences: vec!["END".into()],
        reasoning_type: None,
        reasoning_budget_tokens: None,
        reasoning_display: None,
        max_reasoning_effort: None,
    }
}

fn full_options_history() -> Vec<HistoryTurn> {
    vec![
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(text(
            "Stable system prompt",
        ))])),
        user("Question one"),
        assistant(
            vec![
                AssistantPart::Text(text("Looking")),
                tool_call("call-1", "lookup", json!({"id":"1"})),
            ],
            None,
        ),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call-1",
            ToolContent::Text("one".into()),
        )])),
        user("Question two"),
    ]
}

#[tokio::test]
async fn golden_provider_options_and_cache_points_stream() {
    let server = server().await;
    let model = model_with(&server, MODEL, FixtureKind::SignedReasoning, with_caching);
    let request = Request::new(full_options_history())
        .with_tools(tools())
        .with_tool_choice(ToolChoice::Required)
        .with_inference(inference(Some(2000), None, Some(0.5)))
        .with_bedrock_options(full_options());
    let body = capture(&server, &model, request).await;
    assert_golden("provider_options_and_cache_points_stream", &body);
}

#[tokio::test]
async fn golden_provider_options_and_cache_points_converse() {
    let server = server().await;
    let model = model_with(&server, MODEL, FixtureKind::SignedReasoning, with_caching);
    let mut options = full_options();
    options
        .guardrail
        .as_mut()
        .unwrap()
        .stream_processing_mode
        .take();
    let request = Request::new(full_options_history())
        .with_tools(tools())
        .with_tool_choice(ToolChoice::Required)
        .with_inference(inference(Some(2000), None, Some(0.5)))
        .with_bedrock_options(options);
    model
        .converse(request, AbortSignal::default())
        .await
        .expect("converse succeeds");
    let requests = server.received_requests().await.unwrap();
    let sent = requests.last().unwrap();
    assert!(sent.url.path().ends_with("/converse"));
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_golden("provider_options_and_cache_points_converse", &body);
}

fn amz_date_to_system_time(value: &str) -> SystemTime {
    let number = |range: std::ops::Range<usize>| value[range].parse::<u32>().unwrap();
    let date = time::Date::from_calendar_date(
        number(0..4) as i32,
        time::Month::try_from(number(4..6) as u8).unwrap(),
        number(6..8) as u8,
    )
    .unwrap();
    let datetime = date
        .with_hms(
            number(9..11) as u8,
            number(11..13) as u8,
            number(13..15) as u8,
        )
        .unwrap()
        .assume_utc();
    SystemTime::from(datetime)
}

#[tokio::test]
async fn sigv4_signs_exactly_the_sent_body_bytes() {
    let server = server().await;
    let model = model_with(
        &server,
        "amazon.nova-pro-v1:0",
        FixtureKind::MediaTools,
        |_| {},
    );
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(text("unicode \u{1F600} and \"quotes\"\n")),
        InputPart::File(png()),
    ]))])
    .with_tools(tools());
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let sent = requests.last().unwrap();

    let body_hash = hex::encode(Sha256::digest(&sent.body));
    assert_eq!(
        sent.headers["x-amz-content-sha256"].to_str().unwrap(),
        body_hash
    );

    let authorization = sent.headers["authorization"].to_str().unwrap();
    let signed_headers = authorization
        .split(", ")
        .find_map(|part| part.strip_prefix("SignedHeaders="))
        .expect("signed header list");
    let mut headers = HeaderMap::new();
    for name in signed_headers.split(';') {
        headers.insert(
            HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_bytes(sent.headers[name].as_bytes()).unwrap(),
        );
    }
    // wiremock rewrites the captured URL authority; rebuild it from the signed Host header.
    let url = url::Url::parse(&format!(
        "http://{}{}",
        sent.headers["host"].to_str().unwrap(),
        sent.url.path()
    ))
    .unwrap();
    assert_eq!(url, model.converse_stream_url().unwrap());
    let amz_date = sent.headers["x-amz-date"].to_str().unwrap();
    oven_sdk_bedrock::sigv4::sign_at(
        "POST",
        &url,
        &sent.body,
        &mut headers,
        "us-east-1",
        &support::credentials(),
        amz_date_to_system_time(amz_date),
    )
    .unwrap();
    assert_eq!(headers["authorization"].to_str().unwrap(), authorization);
}
