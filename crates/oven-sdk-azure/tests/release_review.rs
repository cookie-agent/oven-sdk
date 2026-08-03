mod common;

use bytes::Bytes;
use oven_sdk::{
    AbortSignal, AssistantPart, FilePart, FileSource, HistoryTurn, InputPart, JsonSchema,
    LanguageModel, ReplayDisposition, Request, ResponseFormat, ToolDefinition, UserMessage,
};
use oven_sdk_azure::{
    AzureApiRoute, AzureApiVersion, AzureOpenAiChatModel, AzureOpenAiChatOptions,
    AzureOpenAiChatRequestExt, AzureOpenAiOptions, AzureOpenAiResponsesModel,
    AzureOpenAiResponsesOptions, AzureOpenAiResponsesRequestExt, AzureOpenAiRevision,
};
use wiremock::MockServer;

fn setup(model: &str, version: &str, deployment_type: &str) -> common::ModelSetup {
    let mut setup = common::gpt4o();
    setup.revision = Some(AzureOpenAiRevision {
        model: model.into(),
        version: version.into(),
        deployment_type: deployment_type.into(),
    });
    setup
}

fn strict_schema() -> JsonSchema {
    JsonSchema::new(serde_json::json!({
        "type":"object",
        "properties":{"value":{"type":"string"}},
        "required":["value"],
        "additionalProperties":false
    }))
    .unwrap()
}

fn discarded_and_reconstructed(result: &oven_sdk::CompleteResult) -> bool {
    matches!(
        result.request.replay.decisions.as_slice(),
        [
            oven_sdk::ReplayDecision {
                disposition: ReplayDisposition::DiscardedForeignScope { .. },
                ..
            },
            oven_sdk::ReplayDecision {
                disposition: ReplayDisposition::ReconstructedNormalized,
                ..
            }
        ]
    )
}

#[tokio::test]
async fn replay_binds_route_shape_model_id_and_complete_caller_identity() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/chat/completions",
        common::chat_document("ok"),
    )
    .await;
    common::mount(
        &server,
        "/openai/v1/responses",
        common::responses_document("ok"),
    )
    .await;
    let first = common::provider(&server, AzureApiRoute::V1)
        .chat("deployment-a", common::gpt4o())
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();

    let request = || Request::new(vec![HistoryTurn::assistant(first.turn.clone())]);
    let deployment_switch = common::provider(&server, AzureApiRoute::V1)
        .chat("deployment-b", common::gpt4o())
        .unwrap()
        .complete(request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(discarded_and_reconstructed(&deployment_switch));

    let route_switch = common::provider(&server, AzureApiRoute::V1Preview)
        .chat("deployment-a", common::gpt4o())
        .unwrap()
        .complete(request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(discarded_and_reconstructed(&route_switch));

    let version_switch = common::provider(&server, AzureApiRoute::V1)
        .chat(
            "deployment-a",
            setup("gpt-4o-mini", "different-version", "standard"),
        )
        .unwrap()
        .complete(request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(discarded_and_reconstructed(&version_switch));

    let type_switch = common::provider(&server, AzureApiRoute::V1)
        .chat(
            "deployment-a",
            setup("gpt-4o-mini", "2024-07-18", "global_standard"),
        )
        .unwrap()
        .complete(request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(discarded_and_reconstructed(&type_switch));

    let model_switch = common::provider(&server, AzureApiRoute::V1)
        .chat("deployment-a", common::gpt5_chat())
        .unwrap()
        .complete(request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(discarded_and_reconstructed(&model_switch));

    let shape_switch = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment-a", common::gpt5())
        .unwrap()
        .complete(request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(matches!(
        shape_switch.request.replay.decisions.first(),
        Some(oven_sdk::ReplayDecision {
            disposition: ReplayDisposition::DiscardedForeignAdapter { .. },
            ..
        })
    ));
}

fn encrypted_reasoning_document(secret: &str) -> String {
    format!(
        concat!(
            "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp\"}}}}\n\n",
            "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[],\"encrypted_content\":{secret:?}}}}}\n\n",
            "data: {{\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":0,\"delta\":\"summary\"}}\n\n",
            "data: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[{{\"type\":\"summary_text\",\"text\":\"summary\"}}],\"encrypted_content\":{secret:?}}}}}\n\n",
            "data: {{\"type\":\"response.completed\",\"response\":{{\"status\":\"completed\",\"output\":[{{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[{{\"type\":\"summary_text\",\"text\":\"summary\"}}],\"encrypted_content\":{secret:?}}}]}}}}\n\n"
        ),
        secret = secret
    )
}

#[tokio::test]
async fn encrypted_reasoning_never_crosses_caller_identity() {
    let server = MockServer::start().await;
    let secret = "encrypted-secret-payload";
    common::mount(
        &server,
        "/openai/v1/responses",
        encrypted_reasoning_document(secret),
    )
    .await;
    let first = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(
        first
            .turn
            .message
            .content
            .iter()
            .any(|part| matches!(part, AssistantPart::Reasoning(_)))
    );

    let second = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", {
            let mut value = common::gpt5();
            value.revision.as_mut().unwrap().version = "different-version".into();
            value
        })
        .unwrap()
        .complete(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(discarded_and_reconstructed(&second));
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert!(!body.to_string().contains(secret));
    assert!(
        !body["input"].as_array().unwrap().iter().any(|item| {
            item.get("type").and_then(serde_json::Value::as_str) == Some("reasoning")
        })
    );
}

fn refusal_document(streamed: bool, done_value: &str, terminal_value: &str) -> String {
    let delta = if streamed {
        "data: {\"type\":\"response.refusal.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"blocked\"}\n\n"
    } else {
        ""
    };
    let done = if streamed {
        format!(
            "data: {{\"type\":\"response.refusal.done\",\"output_index\":0,\"content_index\":0,\"refusal\":{done_value:?}}}\n\n"
        )
    } else {
        String::new()
    };
    format!(
        concat!(
            "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp\"}}}}\n\n",
            "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"content\":[]}}}}\n\n",
            "{delta}{done}",
            "data: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"content\":[{{\"type\":\"refusal\",\"refusal\":{terminal_value:?}}}]}}}}\n\n",
            "data: {{\"type\":\"response.completed\",\"response\":{{\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\"id\":\"msg\",\"role\":\"assistant\",\"content\":[{{\"type\":\"refusal\",\"refusal\":{terminal_value:?}}}]}}]}}}}\n\n"
        ),
        delta = delta,
        done = done,
        terminal_value = terminal_value
    )
}

#[tokio::test]
async fn responses_refusal_stream_full_item_and_replay_are_semantically_exact() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/responses",
        refusal_document(true, "blocked", "blocked"),
    )
    .await;
    let model = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap();
    let first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(
        first
            .turn
            .message
            .content
            .iter()
            .filter(|part| matches!(part, AssistantPart::Custom(custom) if custom.kind == "azure.openai.refusal" && custom.data == "blocked"))
            .count(),
        1
    );
    let second = model
        .complete(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        second.request.replay.decisions.first(),
        Some(oven_sdk::ReplayDecision {
            disposition: ReplayDisposition::Replayed,
            ..
        })
    ));
    let requests = server.received_requests().await.unwrap();
    let replay_body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert!(replay_body["input"].as_array().unwrap().iter().any(|item| {
        item.pointer("/content/0/refusal")
            .and_then(serde_json::Value::as_str)
            == Some("blocked")
    }));

    let full_server = MockServer::start().await;
    common::mount(
        &full_server,
        "/openai/v1/responses",
        refusal_document(false, "", "full refusal"),
    )
    .await;
    let full = common::provider(&full_server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(full.turn.message.content.iter().any(
        |part| matches!(part, AssistantPart::Custom(custom) if custom.data == "full refusal")
    ));
}

#[tokio::test]
async fn responses_refusal_done_is_terminally_authoritative() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        "/openai/v1/responses",
        refusal_document(true, "different", "blocked"),
    )
    .await;
    let error = common::provider(&server, AzureApiRoute::V1)
        .responses("deployment", common::gpt5())
        .unwrap()
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::InvalidResponse);
}

fn models(server: &MockServer) -> (AzureOpenAiChatModel, AzureOpenAiResponsesModel) {
    let provider = common::provider(server, AzureApiRoute::V1);
    (
        provider.chat("deployment", common::gpt4o()).unwrap(),
        provider.responses("deployment", common::gpt5()).unwrap(),
    )
}

#[test]
fn strict_json_schema_rules_apply_to_chat_responses_and_tools() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::start());
    let (chat, responses) = models(&server);
    let invalid = [
        serde_json::json!({"type":"array","items":{"type":"string"}}),
        serde_json::json!({"type":"object","properties":{},"required":[]}),
        serde_json::json!({"type":"object","properties":{"x":{"type":"string"}},"required":[],"additionalProperties":false}),
        serde_json::json!({"type":"object","properties":{},"required":[],"additionalProperties":false,"format":"bad"}),
    ];
    for schema in invalid {
        let request = Request::new(Vec::new())
            .with_response_format(ResponseFormat::structured(JsonSchema::new(schema).unwrap()));
        assert!(chat.validate_request(&request).is_err());
        assert!(responses.validate_request(&request).is_err());
    }

    let mut nested = serde_json::json!({"type":"string"});
    for index in (0..6).rev() {
        let name = format!("p{index}");
        let mut properties = serde_json::Map::new();
        properties.insert(name.clone(), nested);
        nested = serde_json::json!({
            "type":"object",
            "properties":properties,
            "required":[name],
            "additionalProperties":false
        });
    }
    let deep = Request::new(Vec::new())
        .with_response_format(ResponseFormat::structured(JsonSchema::new(nested).unwrap()));
    assert!(chat.validate_request(&deep).is_err());
    assert!(responses.validate_request(&deep).is_err());

    let properties = (0..101)
        .map(|index| (format!("p{index}"), serde_json::json!({"type":"string"})))
        .collect::<serde_json::Map<_, _>>();
    let required = properties.keys().cloned().collect::<Vec<_>>();
    let too_many = Request::new(Vec::new()).with_response_format(ResponseFormat::structured(
        JsonSchema::new(serde_json::json!({
            "type":"object","properties":properties,"required":required,"additionalProperties":false
        }))
        .unwrap(),
    ));
    assert!(chat.validate_request(&too_many).is_err());
    assert!(responses.validate_request(&too_many).is_err());

    let mut strict_tool = ToolDefinition::new("tool", "strict", strict_schema());
    strict_tool
        .provider_options
        .insert("azure_openai".into(), serde_json::json!({"strict":true}));
    let incompatible = Request::new(Vec::new()).with_tools(vec![strict_tool.clone()]);
    assert!(chat.validate_request(&incompatible).is_err());
    assert!(responses.validate_request(&incompatible).is_err());
    let parallel_true =
        incompatible
            .clone()
            .with_azure_openai_chat_options(AzureOpenAiChatOptions {
                parallel_tool_calls: Some(true),
                ..Default::default()
            });
    assert!(chat.validate_request(&parallel_true).is_err());
    let valid_chat = incompatible
        .clone()
        .with_azure_openai_chat_options(AzureOpenAiChatOptions {
            parallel_tool_calls: Some(false),
            ..Default::default()
        });
    assert!(chat.validate_request(&valid_chat).is_ok());
    let valid_responses =
        incompatible.with_azure_openai_responses_options(AzureOpenAiResponsesOptions {
            parallel_tool_calls: Some(false),
            ..Default::default()
        });
    assert!(responses.validate_request(&valid_responses).is_ok());

    let mut malformed_tool = ToolDefinition::new(
        "tool",
        "strict",
        JsonSchema::new(serde_json::json!({
            "type":"object","properties":{},"required":[],"additionalProperties":true
        }))
        .unwrap(),
    );
    malformed_tool
        .provider_options
        .insert("azure_openai".into(), serde_json::json!({"strict":true}));
    let malformed = Request::new(Vec::new())
        .with_tools(vec![malformed_tool])
        .with_azure_openai_chat_options(AzureOpenAiChatOptions {
            parallel_tool_calls: Some(false),
            ..Default::default()
        });
    assert!(chat.validate_request(&malformed).is_err());
}

fn media_request(files: Vec<FilePart>) -> Request {
    Request::new(vec![HistoryTurn::user(UserMessage::new(
        files.into_iter().map(InputPart::File).collect(),
    ))])
}

fn still_gif() -> Vec<u8> {
    vec![
        71, 73, 70, 56, 57, 97, 1, 0, 1, 0, 128, 0, 0, 0, 0, 0, 255, 255, 255, 44, 0, 0, 0, 0, 1,
        0, 1, 0, 0, 2, 2, 68, 1, 0, 59,
    ]
}

fn animated_gif() -> Vec<u8> {
    let still = still_gif();
    let mut animated = still[..still.len() - 1].to_vec();
    animated.extend_from_slice(&still[19..still.len() - 1]);
    animated.push(59);
    animated
}

#[test]
fn exact_image_mimes_sizes_counts_and_animation_are_preflighted() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::start());
    let (chat, responses) = models(&server);
    for (media_type, data) in [
        ("image/png", vec![1]),
        ("image/jpeg", vec![1]),
        ("image/webp", vec![1]),
        ("image/gif", still_gif()),
    ] {
        let request = media_request(vec![FilePart::image(
            media_type,
            FileSource::Bytes(Bytes::from(data)),
        )]);
        assert!(chat.validate_request(&request).is_ok());
        assert!(responses.validate_request(&request).is_ok());
    }
    for media_type in ["image/svg+xml", "image/tiff"] {
        let request = media_request(vec![FilePart::image(
            media_type,
            FileSource::Bytes(Bytes::from_static(b"x")),
        )]);
        assert!(chat.validate_request(&request).is_err());
        assert!(responses.validate_request(&request).is_err());
    }
    let animated = media_request(vec![FilePart::image(
        "image/gif",
        FileSource::Bytes(Bytes::from(animated_gif())),
    )]);
    assert!(chat.validate_request(&animated).is_err());
    let gif_url = media_request(vec![FilePart::image(
        "image/gif",
        FileSource::Url("https://example.test/image.gif".parse().unwrap()),
    )]);
    assert!(chat.validate_request(&gif_url).is_err());
    let oversized = media_request(vec![FilePart::image(
        "image/png",
        FileSource::Bytes(Bytes::from(vec![0; 20 * 1024 * 1024 + 1])),
    )]);
    assert!(chat.validate_request(&oversized).is_err());
    let too_many = media_request(
        (0..11)
            .map(|_| FilePart::image("image/png", FileSource::Bytes(Bytes::from_static(b"x"))))
            .collect(),
    );
    assert!(chat.validate_request(&too_many).is_err());
}

#[test]
fn dated_api_versions_validate_calendar_suffix_and_typed_cutoff() {
    for valid in [
        "2024-02-29",
        "2025-03-01-preview",
        "2025-03-01",
        "2025-04-01-preview",
    ] {
        assert!(AzureApiVersion::new(valid).is_ok(), "{valid}");
    }
    for invalid in [
        "2023-02-29",
        "2025-00-01",
        "2025-13-01",
        "2025-04-31",
        "2025-01-00",
        "2025-01-01-Preview",
        "2025-01-01-preview-preview",
        "2025-01-01-beta",
    ] {
        assert!(AzureApiVersion::new(invalid).is_err(), "{invalid}");
    }
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let server = runtime.block_on(MockServer::start());
    assert!(
        common::provider(
            &server,
            AzureApiRoute::Dated(AzureApiVersion::new("2025-02-28-preview").unwrap()),
        )
        .responses("deployment", common::gpt5())
        .is_err()
    );
    assert!(
        common::provider(
            &server,
            AzureApiRoute::Dated(AzureApiVersion::new("2025-03-01-preview").unwrap()),
        )
        .responses("deployment", common::gpt5())
        .is_ok()
    );
}

#[test]
fn request_extensions_replace_malformed_values_without_panicking_and_merge_current_shape() {
    let mut malformed = Request::new(Vec::new());
    malformed.provider_options.insert(
        "azure_openai".into(),
        serde_json::json!({"chat":"not-an-object","secret":"do-not-parse"}),
    );
    let replaced = malformed.with_azure_openai_chat_options(AzureOpenAiChatOptions {
        user: Some("user".into()),
        ..Default::default()
    });
    let decoded: AzureOpenAiOptions =
        serde_json::from_value(replaced.provider_options["azure_openai"].clone()).unwrap();
    assert_eq!(decoded.chat.unwrap().user.as_deref(), Some("user"));
    assert!(decoded.responses.is_none());

    let merged = replaced.with_azure_openai_responses_options(AzureOpenAiResponsesOptions {
        user: Some("response-user".into()),
        ..Default::default()
    });
    let decoded: AzureOpenAiOptions =
        serde_json::from_value(merged.provider_options["azure_openai"].clone()).unwrap();
    assert_eq!(decoded.chat.unwrap().user.as_deref(), Some("user"));
    assert_eq!(
        decoded.responses.unwrap().user.as_deref(),
        Some("response-user")
    );
}
