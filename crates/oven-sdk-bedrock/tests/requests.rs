mod support;

use oven_sdk::{
    AbortSignal, Capability, FilePart, FileSource, HistoryTurn, InputPart, JsonSchema,
    LanguageModel, Request, ResponseFormat, SystemMessage, SystemPart, TextPart, ToolChoice,
    ToolDefinition, UserMessage,
};
use oven_sdk_bedrock::{
    BedrockAuth, BedrockCachePoint, BedrockCacheStrategy, BedrockCacheTtl, BedrockGuardrailConfig,
    BedrockMessageCachePoint, BedrockModel, BedrockReasoningWireFormat, BedrockRequestExt,
    BedrockRequestOptions,
};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, body_partial_json, header_regex, method, path},
};

#[tokio::test]
async fn converse_encodes_system_tools_and_message_cache_points() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "system":[
                {"text":"stable system"},
                {"cachePoint":{"type":"default","ttl":"1h"}}
            ],
            "toolConfig":{"tools":[
                {"toolSpec":{"name":"lookup"}},
                {"cachePoint":{"type":"default","ttl":"1h"}}
            ]},
            "messages":[{"role":"user","content":[
                {"text":"variable question"},
                {"cachePoint":{"type":"default"}}
            ]}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},
            "stopReason":"end_turn",
            "usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3},
            "metrics":{"latencyMs":1}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let mut config = support::config(
        &server.uri(),
        "anthropic.claude-sonnet-4-6",
        support::FixtureKind::MediaTools,
        BedrockAuth::Static(support::credentials()),
    );
    config.model.capabilities.features |= Capability::PROMPT_CACHING;
    let model = BedrockModel::new(config).unwrap();
    let one_hour = BedrockCachePoint {
        ttl: Some(BedrockCacheTtl::OneHour),
    };
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(TextPart::new(
            "stable system",
        ))])),
        HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
            "variable question",
        ))])),
    ])
    .with_tools(vec![ToolDefinition::new(
        "lookup",
        "lookup",
        JsonSchema::new(json!({"type":"object"})).unwrap(),
    )])
    .with_bedrock_options(BedrockRequestOptions {
        cache: Some(BedrockCacheStrategy {
            system: Some(one_hour.clone()),
            tools: Some(one_hour),
            messages: vec![BedrockMessageCachePoint {
                history_index: 1,
                cache_point: BedrockCachePoint::default(),
            }],
        }),
        ..Default::default()
    });

    assert_eq!(
        model
            .converse(request, AbortSignal::default())
            .await
            .unwrap()
            .turn
            .text(),
        "ok"
    );
}

#[tokio::test]
async fn invalid_cache_strategies_are_rejected_before_network() {
    let server = MockServer::start().await;
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("prompt")),
    ]))])
    .with_bedrock_options(BedrockRequestOptions {
        cache: Some(BedrockCacheStrategy {
            messages: vec![BedrockMessageCachePoint {
                history_index: 0,
                cache_point: BedrockCachePoint::default(),
            }],
            ..Default::default()
        }),
        ..Default::default()
    });
    let unsupported = support::model(&server.uri(), "opaque", support::FixtureKind::MediaTools);
    assert!(unsupported.validate_request(&request).is_err());

    let mut config = support::config(
        &server.uri(),
        "opaque",
        support::FixtureKind::MediaTools,
        BedrockAuth::Static(support::credentials()),
    );
    config.model.capabilities.features |= Capability::PROMPT_CACHING;
    let caching = BedrockModel::new(config).unwrap();
    let five_points = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("prompt")),
    ]))])
    .with_bedrock_options(BedrockRequestOptions {
        cache: Some(BedrockCacheStrategy {
            messages: (0..5)
                .map(|_| BedrockMessageCachePoint {
                    history_index: 0,
                    cache_point: BedrockCachePoint::default(),
                })
                .collect(),
            ..Default::default()
        }),
        ..Default::default()
    });
    let missing_tools = Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
        cache: Some(BedrockCacheStrategy {
            tools: Some(BedrockCachePoint::default()),
            ..Default::default()
        }),
        ..Default::default()
    });
    assert!(caching.validate_request(&five_points).is_err());
    assert!(caching.validate_request(&missing_tools).is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn converse_signs_exact_resource_and_encodes_media_tools_and_open_labels() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/model/amazon.nova-pro-v1%3A0/converse"))
        .and(header_regex("authorization", "^AWS4-HMAC-SHA256 "))
        .and(body_partial_json(json!({
            "messages":[{"role":"user","content":[
                {"text":"inspect"},
                {"image":{"format":"png","source":{"bytes":"AQID"}}}
            ]}],
            "toolConfig":{"toolChoice":{"tool":{"name":"lookup"}}},
            "serviceTier":{"type":"future-tier"},
            "performanceConfig":{"latency":"optimized"}
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-amzn-requestid", "request-1")
                .set_body_json(json!({
                    "output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},
                    "stopReason":"end_turn",
                    "usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3},
                    "metrics":{"latencyMs":1}
                })),
        )
        .expect(1)
        .mount(&server)
        .await;
    let model = support::model(
        &server.uri(),
        "amazon.nova-pro-v1:0",
        support::FixtureKind::MediaTools,
    );
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("inspect")),
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(vec![1, 2, 3].into()),
        )),
    ]))])
    .with_tools(vec![ToolDefinition::new(
        "lookup",
        "lookup",
        JsonSchema::new(json!({"type":"object"})).unwrap(),
    )])
    .with_tool_choice(ToolChoice::Tool("lookup".into()))
    .with_bedrock_options(BedrockRequestOptions {
        service_tier: Some("future-tier".into()),
        performance_latency: Some("optimized".into()),
        ..Default::default()
    });
    let result = model
        .converse(request, AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(result.turn.text(), "ok");
    assert_eq!(result.response.request_id.as_deref(), Some("request-1"));
}

#[tokio::test]
async fn native_structured_output_uses_output_config_without_synthetic_tool() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_json(json!({
            "messages":[],
            "outputConfig":{"textFormat":{"type":"json_schema","structure":{"jsonSchema":{"schema":"{\"type\":\"object\"}"}}}}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("{}")))
        .mount(&server)
        .await;
    let model = support::model(
        &server.uri(),
        "anthropic.claude-sonnet-4-6",
        support::FixtureKind::SignedReasoning,
    );
    let request = Request::new(Vec::new()).with_response_format(ResponseFormat::structured(
        JsonSchema::new(json!({"type":"object"})).unwrap(),
    ));
    assert_eq!(
        model
            .complete(request, AbortSignal::default())
            .await
            .unwrap()
            .turn
            .text(),
        "{}"
    );
}

#[tokio::test]
async fn unsupported_url_is_rejected_before_network() {
    let server = MockServer::start().await;
    let model = support::model(
        &server.uri(),
        "amazon.nova-pro-v1:0",
        support::FixtureKind::MediaTools,
    );
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("inspect")),
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Url("https://example.com/image.png".parse().unwrap()),
        )),
    ]))]);
    assert!(model.stream(request, AbortSignal::default()).await.is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn invalid_media_boundaries_are_rejected_before_network() {
    let server = MockServer::start().await;
    let model = support::model(
        &server.uri(),
        "amazon.nova-pro-v1:0",
        support::FixtureKind::MediaTools,
    );
    let requests = [
        Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
            InputPart::File(FilePart::image(
                "image/png",
                FileSource::Bytes(Vec::new().into()),
            )),
        ]))]),
        Request::new(vec![HistoryTurn::user(UserMessage::new(
            (0..21)
                .map(|_| {
                    InputPart::File(FilePart::image(
                        "image/png",
                        FileSource::Bytes(vec![1].into()),
                    ))
                })
                .collect(),
        ))]),
        Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
            InputPart::File(FilePart::image(
                "image/png",
                FileSource::Bytes(vec![0; 15 * 1024 * 1024 / 4 + 1].into()),
            )),
        ]))]),
        Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
            InputPart::File(FilePart::image(
                "image/png",
                FileSource::Url("s3://ab/image.png".parse().unwrap()),
            )),
        ]))])
        .with_bedrock_options(BedrockRequestOptions {
            s3: Some(oven_sdk_bedrock::BedrockS3LocationOptions {
                bucket_owner: Some("123".into()),
            }),
            ..Default::default()
        }),
    ];
    for request in requests {
        assert!(model.stream(request, AbortSignal::default()).await.is_err());
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn converse_rejects_missing_or_non_assistant_output_role() {
    for response in [
        json!({
            "output":{"message":{"content":[{"text":"wrong"}]}},
            "stopReason":"end_turn"
        }),
        json!({
            "output":{"message":{"role":"user","content":[{"text":"wrong"}]}},
            "stopReason":"end_turn"
        }),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response))
            .mount(&server)
            .await;
        let model = support::model(
            &server.uri(),
            "anthropic.claude-sonnet-4-6",
            support::FixtureKind::SignedReasoning,
        );
        assert!(
            model
                .converse(Request::new(Vec::new()), AbortSignal::default())
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn stream_processing_mode_is_stream_only_and_exact_on_wire() {
    let options = BedrockRequestOptions {
        guardrail: Some(BedrockGuardrailConfig {
            guardrail_identifier: "guardrail".into(),
            guardrail_version: "DRAFT".into(),
            trace: Some("enabled".into()),
            stream_processing_mode: Some("async".into()),
        }),
        ..Default::default()
    };

    let rejected_server = MockServer::start().await;
    let rejected = support::model(
        &rejected_server.uri(),
        "anthropic.claude-sonnet-4-6",
        support::FixtureKind::SignedReasoning,
    );
    assert!(
        rejected
            .converse(
                Request::new(Vec::new()).with_bedrock_options(options.clone()),
                AbortSignal::default(),
            )
            .await
            .is_err()
    );
    assert!(
        rejected_server
            .received_requests()
            .await
            .unwrap()
            .is_empty()
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_json(json!({
            "messages":[],
            "guardrailConfig":{
                "guardrailIdentifier":"guardrail",
                "guardrailVersion":"DRAFT",
                "trace":"enabled",
                "streamProcessingMode":"async"
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("ok")))
        .mount(&server)
        .await;
    let streaming = support::model(
        &server.uri(),
        "anthropic.claude-sonnet-4-6",
        support::FixtureKind::SignedReasoning,
    );
    assert_eq!(
        streaming
            .complete(
                Request::new(Vec::new()).with_bedrock_options(options),
                AbortSignal::default(),
            )
            .await
            .unwrap()
            .turn
            .text(),
        "ok"
    );
}

#[tokio::test]
async fn anthropic_looking_model_ids_do_not_select_reasoning_wire_format() {
    for model_id in ["anthropic.must-remain-opaque", "opaque-resource"] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_partial_json(json!({
                "additionalModelRequestFields":{"reasoning_effort":"high"}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("ok")))
            .mount(&server)
            .await;
        let model = support::model(
            &server.uri(),
            model_id,
            support::FixtureKind::UnsignedReasoning,
        );
        let mut request = Request::new(Vec::new());
        request.inference.reasoning_effort = Some("high".into());
        model
            .complete(request, AbortSignal::default())
            .await
            .unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert!(
            body.pointer("/additionalModelRequestFields/thinking")
                .is_none()
        );
    }
}

fn reasoning_model(server: &MockServer, wire: BedrockReasoningWireFormat) -> BedrockModel {
    let kind = if wire == BedrockReasoningWireFormat::AnthropicThinking {
        support::FixtureKind::SignedReasoning
    } else {
        support::FixtureKind::UnsignedReasoning
    };
    let mut config = support::config(
        &server.uri(),
        "opaque-reasoning-model",
        kind,
        BedrockAuth::Static(support::credentials()),
    );
    config.settings.reasoning_wire_format = wire;
    BedrockModel::new(config).unwrap()
}

#[tokio::test]
async fn reasoning_wire_omits_absent_display_budget_and_max_tokens() {
    let anthropic = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_json(json!({
            "messages":[],
            "additionalModelRequestFields":{"thinking":{"type":"adaptive"}}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("ok")))
        .mount(&anthropic)
        .await;
    reasoning_model(&anthropic, BedrockReasoningWireFormat::AnthropicThinking)
        .complete(
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("adaptive".into()),
                ..Default::default()
            }),
            AbortSignal::default(),
        )
        .await
        .unwrap();

    let generic = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_json(json!({
            "messages":[],
            "additionalModelRequestFields":{"reasoningConfig":{
                "type":"future-reasoning-mode",
                "maxReasoningEffort":"future-effort"
            }}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(support::text_stream("ok")))
        .mount(&generic)
        .await;
    reasoning_model(&generic, BedrockReasoningWireFormat::BedrockReasoningConfig)
        .complete(
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("future-reasoning-mode".into()),
                max_reasoning_effort: Some("future-effort".into()),
                ..Default::default()
            }),
            AbortSignal::default(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn incompatible_reasoning_controls_are_rejected_before_network() {
    let server = MockServer::start().await;
    let cases = [
        (
            BedrockReasoningWireFormat::AnthropicThinking,
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("adaptive".into()),
                reasoning_budget_tokens: Some(1_024),
                ..Default::default()
            }),
        ),
        (
            BedrockReasoningWireFormat::AnthropicThinking,
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("enabled".into()),
                reasoning_display: Some("summarized".into()),
                ..Default::default()
            }),
        ),
        (
            BedrockReasoningWireFormat::AnthropicThinking,
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("enabled".into()),
                ..Default::default()
            }),
        ),
        (
            BedrockReasoningWireFormat::AnthropicThinking,
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("disabled".into()),
                max_reasoning_effort: Some("high".into()),
                ..Default::default()
            }),
        ),
        (
            BedrockReasoningWireFormat::OpenAiReasoningEffort,
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("enabled".into()),
                ..Default::default()
            }),
        ),
        (
            BedrockReasoningWireFormat::BedrockReasoningConfig,
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("adaptive".into()),
                ..Default::default()
            }),
        ),
        (
            BedrockReasoningWireFormat::BedrockReasoningConfig,
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("future-mode".into()),
                reasoning_budget_tokens: Some(100),
                ..Default::default()
            }),
        ),
        (
            BedrockReasoningWireFormat::BedrockReasoningConfig,
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_display: Some("summarized".into()),
                ..Default::default()
            }),
        ),
        (
            BedrockReasoningWireFormat::BedrockReasoningConfig,
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_type: Some("disabled".into()),
                max_reasoning_effort: Some("high".into()),
                ..Default::default()
            }),
        ),
        (
            BedrockReasoningWireFormat::BedrockReasoningConfig,
            Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                reasoning_budget_tokens: Some(0),
                ..Default::default()
            }),
        ),
        (BedrockReasoningWireFormat::BedrockReasoningConfig, {
            let mut request =
                Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                    max_reasoning_effort: Some("high".into()),
                    ..Default::default()
                });
            request.inference.reasoning_effort = Some("medium".into());
            request
        }),
        (BedrockReasoningWireFormat::AnthropicThinking, {
            let mut request =
                Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
                    reasoning_type: Some("enabled".into()),
                    ..Default::default()
                });
            request.inference.temperature = Some(0.5);
            request
        }),
    ];
    for (wire, request) in cases {
        assert!(
            reasoning_model(&server, wire)
                .stream(request, AbortSignal::default())
                .await
                .is_err()
        );
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn additional_model_fields_cannot_bypass_typed_reasoning_or_output_controls() {
    let server = MockServer::start().await;
    let model = support::model(
        &server.uri(),
        "opaque",
        support::FixtureKind::SignedReasoning,
    );
    for key in [
        "thinking",
        "THINKING",
        "output_config",
        "outputConfig",
        "OUTPUT-CONFIG",
        "reasoning_effort",
        "reasoningEffort",
        "reasoningConfig",
        "reasoning_config",
        "REASONING-CONFIG",
    ] {
        let request = Request::new(Vec::new()).with_bedrock_options(BedrockRequestOptions {
            additional_model_request_fields: Some(json!({(key):null})),
            ..Default::default()
        });
        assert!(
            model.stream(request, AbortSignal::default()).await.is_err(),
            "reserved key {key} must fail before dispatch"
        );
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}
