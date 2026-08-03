use std::collections::BTreeMap;

use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, Capability, CompletedTurn, FilePart, FileSource,
    Finish, FinishReason, HistoryTurn, InputPart, JsonSchema, LanguageModel, Request,
    ResponseFormat, SystemMessage, SystemPart, TextPart, ToolCallPart, ToolChoice, ToolContent,
    ToolDefinition, ToolMessage, ToolResultPart, UserMessage,
};
use oven_sdk_anthropic::{
    AnthropicAwsCredentials, AnthropicCacheControl, AnthropicCacheTtl, AnthropicRequestExt,
    AnthropicRequestOptions, AnthropicThinking, AnthropicToolOptions,
};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::path};

fn response() -> &'static str {
    "event: message_start\ndata: {\"message\":{}}\n\nevent: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\nevent: message_stop\ndata: {}\n\n"
}

#[tokio::test]
async fn messages_path_is_appended_with_url_path_semantics() {
    let server = MockServer::start().await;
    Mock::given(path("/custom/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(format!("{}/custom/v1/", server.uri()))
        .build()
        .unwrap()
        .model("future-id");
    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
}

#[tokio::test]
async fn encoded_endpoint_segments_are_not_reparsed_or_double_encoded() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(format!("{}/tenant%2Fresource/v1/", server.uri()))
        .build()
        .unwrap()
        .model("future-id");
    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(
        server.received_requests().await.unwrap()[0].url.path(),
        "/tenant%2Fresource/v1/messages"
    );
}

#[tokio::test]
async fn request_encodes_headers_tools_cache_thinking_and_structured_output() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .api_key("secret")
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-4-5");
    let schema = JsonSchema::new(serde_json::json!({"type":"object"})).unwrap();
    let mut tool = ToolDefinition::new("lookup", "find", schema.clone());
    tool.provider_options.insert(
        "anthropic".into(),
        serde_json::to_value(AnthropicToolOptions { strict: true }).unwrap(),
    );
    let request = Request::new(vec![
        HistoryTurn::system(SystemMessage::new(vec![SystemPart::Text(TextPart::new(
            "system",
        ))])),
        HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
            "user",
        ))])),
        HistoryTurn::user(UserMessage::new(Vec::new())),
    ])
    .with_tools(vec![tool])
    .with_tool_choice(ToolChoice::Auto)
    .with_response_format(ResponseFormat::structured(schema))
    .with_anthropic_options(AnthropicRequestOptions {
        thinking: Some(AnthropicThinking::Enabled {
            budget_tokens: 1024,
            display: None,
        }),
        effort: Some("medium".into()),
        cache_control: Some(AnthropicCacheControl {
            ttl: AnthropicCacheTtl::FiveMinutes,
        }),
        betas: vec![
            "test-beta".into(),
            "test-beta".into(),
            "prompt-caching-2024-07-31".into(),
        ],
        ..Default::default()
    });
    oven_sdk::LanguageModel::complete(&model, request, AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(
        request.headers.get("anthropic-version").unwrap(),
        "2023-06-01"
    );
    assert_eq!(request.headers.get("x-api-key").unwrap(), "secret");
    assert_eq!(request.headers.get("anthropic-beta").unwrap(), "test-beta");
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["tools"][0]["strict"], true);
    assert_eq!(body["tools"][0]["name"], "lookup");
    assert_eq!(body["tools"][0]["description"], "find");
    assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    assert_eq!(body["tool_choice"]["type"], "auto");
    assert_eq!(body["cache_control"]["ttl"], "5m");
    assert_eq!(body["output_config"]["effort"], "medium");
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    assert_eq!(body["thinking"]["budget_tokens"], 1024);
    assert_eq!(body["system"][0]["text"], "system");
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
}

#[test]
fn strict_tools_and_structured_output_are_explicit_capability_gated() {
    let mut capabilities = common::anthropic_capabilities(oven_sdk::ReplayPolicy::IfValid);
    capabilities.features.remove(Capability::STRUCTURED_OUTPUT);
    let model = Anthropic::builder()
        .capabilities(capabilities)
        .build()
        .unwrap()
        .model("future-model-id");
    let schema = JsonSchema::new(serde_json::json!({"type":"object"})).unwrap();
    let request = Request::new(Vec::new()).with_response_format(ResponseFormat::structured(schema));
    assert!(model.validate_request(&request).is_err());
    assert!(
        !model
            .capabilities()
            .features
            .contains(Capability::STRUCTURED_OUTPUT)
    );
}

#[test]
fn cache_breakpoint_limit_is_pre_network_validation() {
    let model = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-opus-4-5");
    let cache = serde_json::to_value(AnthropicRequestOptions {
        cache_control: Some(AnthropicCacheControl {
            ttl: AnthropicCacheTtl::FiveMinutes,
        }),
        ..Default::default()
    })
    .unwrap();
    let mut request = Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
        cache_control: Some(AnthropicCacheControl {
            ttl: AnthropicCacheTtl::FiveMinutes,
        }),
        ..Default::default()
    });
    for index in 0..4 {
        let mut tool = ToolDefinition::new(
            format!("tool{index}"),
            "tool",
            JsonSchema::new(serde_json::json!({})).unwrap(),
        );
        tool.provider_options = BTreeMap::from([("anthropic".into(), cache.clone())]);
        request.tools.push(tool);
    }
    assert!(model.validate_request(&request).is_err());
}

#[tokio::test]
async fn effort_labels_are_forwarded_unchanged() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-4-5");
    let labels = ["low", "medium", "high", "xhigh", "max", "future-effort"];

    for label in labels {
        oven_sdk::LanguageModel::complete(
            &model,
            Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
                effort: Some(label.into()),
                ..Default::default()
            }),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    }

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), labels.len());
    for (request, label) in requests.iter().zip(labels) {
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["output_config"]["effort"], label);
    }
}

#[tokio::test]
async fn normalized_reasoning_effort_is_not_anthropic_effort_fallback() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");

    let mut request = Request::new(Vec::new());
    request.inference.reasoning_effort = Some("normalized-ignored".into());
    oven_sdk::LanguageModel::complete(&model, request, AbortSignal::default())
        .await
        .unwrap();

    let request = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert!(body.get("output_config").is_none());
}

#[tokio::test]
async fn media_sources_and_tool_choice_are_lowered_to_messages_wire_format() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let tool = ToolDefinition::new(
        "lookup",
        "find",
        JsonSchema::new(serde_json::json!({"type":"object"})).unwrap(),
    );
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(bytes::Bytes::from_static(b"png")),
        )),
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Url("https://example.test/image.png".parse().unwrap()),
        )),
        InputPart::File(FilePart::document(
            "application/pdf",
            FileSource::Bytes(bytes::Bytes::from_static(b"pdf")),
        )),
        InputPart::File(FilePart::document(
            "text/plain",
            FileSource::Text("document text".into()),
        )),
        InputPart::File(FilePart::document(
            "application/pdf",
            FileSource::Url("https://example.test/file.pdf".parse().unwrap()),
        )),
    ]))])
    .with_tools(vec![tool])
    .with_tool_choice(ToolChoice::Auto);
    oven_sdk::LanguageModel::complete(&model, request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    let content = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(body["tool_choice"], serde_json::json!({"type":"auto"}));
    assert_eq!(content[0]["source"]["type"], "base64");
    assert_eq!(content[1]["source"]["type"], "url");
    assert_eq!(content[2]["type"], "document");
    assert_eq!(content[3]["source"]["type"], "text");
    assert_eq!(content[4]["source"]["type"], "url");
}

#[tokio::test]
async fn audio_and_video_are_rejected_before_http_dispatch() {
    let server = MockServer::start().await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-opus-4-8");
    for part in [
        FilePart::audio("audio/wav", FileSource::Bytes(bytes::Bytes::new())),
        FilePart::video("video/mp4", FileSource::Bytes(bytes::Bytes::new())),
    ] {
        let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
            InputPart::File(part),
        ]))]);
        assert!(model.stream(request, AbortSignal::default()).await.is_err());
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn tool_choice_specific_and_none_have_the_documented_wire_behavior() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let tool = || {
        ToolDefinition::new(
            "lookup",
            "find",
            JsonSchema::new(serde_json::json!({"type":"object"})).unwrap(),
        )
    };
    for choice in [ToolChoice::Tool("lookup".into()), ToolChoice::None] {
        oven_sdk::LanguageModel::complete(
            &model,
            Request::new(Vec::new())
                .with_tools(vec![tool()])
                .with_tool_choice(choice),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    }
    let requests = server.received_requests().await.unwrap();
    let specific: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let none: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        specific["tool_choice"],
        serde_json::json!({"type":"tool","name":"lookup"})
    );
    assert!(none.get("tools").is_none());
    assert!(none.get("tool_choice").is_none());
}

#[test]
fn sampling_behavior_is_selected_only_by_explicit_protocol_settings() {
    let mut strict = common::anthropic_protocol();
    strict.reject_non_default_sampling = true;
    let strict = Anthropic::builder()
        .protocol(strict)
        .build()
        .unwrap()
        .model("future-model-id");
    let mut non_default = Request::new(Vec::new());
    non_default.inference.temperature = Some(0.2);
    assert!(strict.validate_request(&non_default).is_err());
    let mut defaults = Request::new(Vec::new());
    defaults.inference.temperature = Some(1.0);
    defaults.inference.top_p = Some(1.0);
    assert!(strict.validate_request(&defaults).is_ok());

    let flexible = Anthropic::builder()
        .build()
        .unwrap()
        .model("same-behavior-with-any-id");
    let active = |top_p| {
        let mut request =
            Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
                thinking: Some(AnthropicThinking::Adaptive { display: None }),
                ..Default::default()
            });
        request.inference.top_p = Some(top_p);
        request
    };
    assert!(flexible.validate_request(&active(0.95)).is_ok());
    assert!(flexible.validate_request(&active(1.0)).is_ok());
    assert!(flexible.validate_request(&active(0.949)).is_err());
    let mut temperature = active(1.0);
    temperature.inference.top_p = None;
    temperature.inference.temperature = Some(1.0);
    assert!(flexible.validate_request(&temperature).is_err());
}

#[test]
fn explicit_thinking_settings_and_budget_rules_are_validated_locally() {
    let request = |thinking| {
        Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
            thinking: Some(thinking),
            ..Default::default()
        })
    };
    let mut extended = common::anthropic_protocol();
    extended.thinking = oven_sdk_anthropic::AnthropicThinkingSupport::Extended;
    let extended = Anthropic::builder()
        .protocol(extended)
        .build()
        .unwrap()
        .model("any-id");
    assert!(
        extended
            .validate_request(&request(AnthropicThinking::Enabled {
                budget_tokens: 1024,
                display: None,
            }))
            .is_ok()
    );
    assert!(
        extended
            .validate_request(&request(AnthropicThinking::Adaptive { display: None }))
            .is_err()
    );

    let mut adaptive = common::anthropic_protocol();
    adaptive.thinking = oven_sdk_anthropic::AnthropicThinkingSupport::Adaptive;
    adaptive.thinking_disable_allowed = false;
    let adaptive = Anthropic::builder()
        .protocol(adaptive)
        .build()
        .unwrap()
        .model("another-id");
    assert!(
        adaptive
            .validate_request(&request(AnthropicThinking::Adaptive { display: None }))
            .is_ok()
    );
    assert!(
        adaptive
            .validate_request(&request(AnthropicThinking::Enabled {
                budget_tokens: 1024,
                display: None,
            }))
            .is_err()
    );
    assert!(
        adaptive
            .validate_request(&request(AnthropicThinking::Disabled))
            .is_err()
    );

    assert!(
        extended
            .validate_request(&request(AnthropicThinking::Enabled {
                budget_tokens: 1023,
                display: None,
            }))
            .is_err()
    );

    let mut over_limit = request(AnthropicThinking::Enabled {
        budget_tokens: 1024,
        display: None,
    });
    over_limit.inference.max_output_tokens = Some(64_000);
    assert!(extended.validate_request(&over_limit).is_err());

    let final_text_prefill = Request::new(vec![HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::Text(TextPart::new("prefill"))]),
        Finish::new(Default::default(), FinishReason::Stop),
    ))])
    .with_anthropic_options(AnthropicRequestOptions {
        thinking: Some(AnthropicThinking::Enabled {
            budget_tokens: 1024,
            display: None,
        }),
        ..Default::default()
    });
    assert!(extended.validate_request(&final_text_prefill).is_err());
    let disabled_max = Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
        thinking: Some(AnthropicThinking::Disabled),
        effort: Some("max".into()),
        ..Default::default()
    });
    let mut effort_restricted = common::anthropic_protocol();
    effort_restricted
        .thinking_disable_forbidden_efforts
        .insert("max".into());
    let effort_restricted = Anthropic::builder()
        .protocol(effort_restricted)
        .build()
        .unwrap()
        .model("future-id");
    assert!(effort_restricted.validate_request(&disabled_max).is_err());

    let mut zero = Request::new(Vec::new());
    zero.inference.max_output_tokens = Some(0);
    assert!(extended.validate_request(&zero).is_err());

    let tool = ToolDefinition::new(
        "lookup",
        "lookup",
        JsonSchema::new(serde_json::json!({"type":"object"})).unwrap(),
    );
    let forced = request(AnthropicThinking::Enabled {
        budget_tokens: 1024,
        display: None,
    })
    .with_tools(vec![tool])
    .with_tool_choice(ToolChoice::Required);
    assert!(extended.validate_request(&forced).is_err());

    let reasoning_prefill = Request::new(vec![HistoryTurn::assistant(CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::Reasoning(
            oven_sdk::ReasoningPart::new("prefill"),
        )]),
        Finish::new(Default::default(), FinishReason::Stop),
    ))])
    .with_anthropic_options(AnthropicRequestOptions {
        thinking: Some(AnthropicThinking::Adaptive { display: None }),
        ..Default::default()
    });
    assert!(adaptive.validate_request(&reasoning_prefill).is_err());
}

#[test]
fn first_party_sampling_rules_apply_to_direct_and_aws_but_not_minimax() {
    let direct = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-opus-4-8");
    let aws = AnthropicAws::builder("us-west-2", "workspace")
        .static_credentials(AnthropicAwsCredentials {
            access_key_id: "id".into(),
            secret_access_key: oven_sdk::SecretString::new("secret"),
            session_token: None,
        })
        .build()
        .unwrap()
        .model("claude-opus-4-8");
    for model in [&direct as &dyn LanguageModel, &aws as &dyn LanguageModel] {
        let mut too_hot = Request::new(Vec::new());
        too_hot.inference.temperature = Some(1.01);
        assert!(model.validate_request(&too_hot).is_err());
        let mut both = Request::new(Vec::new());
        both.inference.temperature = Some(0.5);
        both.inference.top_p = Some(0.9);
        assert!(model.validate_request(&both).is_err());
    }
}

#[tokio::test]
async fn enabled_thinking_adds_budget_without_exceeding_model_limit() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut request = Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
        thinking: Some(AnthropicThinking::Enabled {
            budget_tokens: 1024,
            display: None,
        }),
        ..Default::default()
    });
    request.inference.max_output_tokens = Some(2000);
    model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["max_tokens"], 3024);
}

#[tokio::test]
async fn ordinary_assistant_text_is_exact_and_only_true_prefill_is_trimmed() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let completed = |text| {
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![AssistantPart::Text(TextPart::new(text))]),
            Finish::new(Default::default(), FinishReason::Stop),
        ))
    };
    model
        .complete(
            Request::new(vec![
                completed("ordinary\n"),
                HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
                    "next",
                ))])),
            ]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    model
        .complete(
            Request::new(vec![completed("prefill\n")]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let ordinary: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let prefill: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(ordinary["messages"][0]["content"][0]["text"], "ordinary\n");
    assert_eq!(prefill["messages"][0]["content"][0]["text"], "prefill");
}

#[tokio::test]
async fn older_model_sampling_and_open_thinking_display_have_expected_wire_encoding() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response()))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-6");
    let mut temperature = Request::new(Vec::new());
    temperature.inference.temperature = Some(0.3);
    oven_sdk::LanguageModel::complete(&model, temperature, AbortSignal::default())
        .await
        .unwrap();
    let mut disabled = Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
        thinking: Some(AnthropicThinking::Disabled),
        ..Default::default()
    });
    disabled.inference.top_p = Some(0.8);
    oven_sdk::LanguageModel::complete(&model, disabled, AbortSignal::default())
        .await
        .unwrap();
    let mut adaptive_request =
        Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
            thinking: Some(AnthropicThinking::Adaptive {
                display: Some("future-provider-label".into()),
            }),
            ..Default::default()
        });
    adaptive_request.inference.top_p = Some(0.95);
    oven_sdk::LanguageModel::complete(&model, adaptive_request, AbortSignal::default())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let temperature: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let disabled: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    let adaptive: serde_json::Value = serde_json::from_slice(&requests[2].body).unwrap();
    assert_eq!(temperature["temperature"], 0.3);
    assert!(temperature.get("thinking").is_none());
    assert_eq!(disabled["top_p"], 0.8);
    assert_eq!(disabled["thinking"], serde_json::json!({"type":"disabled"}));
    assert_eq!(
        adaptive["thinking"],
        serde_json::json!({"type":"adaptive","display":"future-provider-label"})
    );
    assert!(adaptive.get("temperature").is_none());
    assert_eq!(adaptive["top_p"], 0.95);
}

#[test]
fn consecutive_tool_result_turns_are_rejected_by_the_core_history_contract() {
    let model = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let first = ToolCallPart::new("call_1", "lookup", serde_json::json!({}));
    let second = ToolCallPart::new("call_2", "lookup", serde_json::json!({}));
    let assistant = oven_sdk::CompletedTurn::new(
        oven_sdk::AssistantMessage::new(vec![
            oven_sdk::AssistantPart::ToolCall(first),
            oven_sdk::AssistantPart::ToolCall(second),
        ]),
        oven_sdk::Finish::new(Default::default(), oven_sdk::FinishReason::ToolCalls),
    );
    let request = Request::new(vec![
        HistoryTurn::assistant(assistant),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call_1",
            ToolContent::Text("one".into()),
        )])),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call_2",
            ToolContent::Text("two".into()),
        )])),
    ]);
    assert!(model.validate_request(&request).is_err());
}
mod common;

use common::{Anthropic, AnthropicAws};
