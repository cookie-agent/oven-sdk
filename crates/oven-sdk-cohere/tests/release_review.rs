mod common;

use std::{collections::BTreeMap, time::Duration};

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, ContentValue, CustomPart,
    FilePart, FileSource, Finish, FinishReason, HistoryTurn, InferenceOptions, JsonSchema,
    LanguageModel, ModelErrorKind, NativeReplayArtifact, ReplayCapability, ReplayDisposition,
    ReplayPolicy, Request, ResponseFormat, StreamPart, ToolCallPart, ToolContent, ToolDefinition,
    ToolResultPart, Usage,
};
use oven_sdk_cohere::{CohereSettings, CohereThinking, CohereTimeouts};
use serde_json::{Map, Value, json};
use wiremock::MockServer;

fn inference(top_p: Option<f64>, effort: Option<&str>) -> InferenceOptions {
    let mut inference = InferenceOptions::new();
    inference.top_p = top_p;
    inference.reasoning_effort = effort.map(str::to_owned);
    inference
}

fn completed(parts: Vec<AssistantPart>) -> CompletedTurn {
    CompletedTurn::new(
        AssistantMessage::new(parts),
        Finish::new(Usage::default(), FinishReason::Stop),
    )
}

fn thinking_stream(thinking: &str, text: &str) -> String {
    let events = [
        json!({"type":"message-start","id":"gen_reasoning","delta":{"message":{"role":"assistant"}}}),
        json!({"type":"content-start","index":0,"delta":{"message":{"content":{"type":"thinking","thinking":""}}}}),
        json!({"type":"content-delta","index":0,"delta":{"message":{"content":{"thinking":thinking}}}}),
        json!({"type":"content-end","index":0}),
        json!({"type":"content-start","index":1,"delta":{"message":{"content":{"type":"text","text":""}}}}),
        json!({"type":"content-delta","index":1,"delta":{"message":{"content":{"text":text}}}}),
        json!({"type":"content-end","index":1}),
        json!({"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":2}}}}),
    ];
    events
        .into_iter()
        .map(|value| {
            format!(
                "event: {}\ndata: {value}\n\n",
                value["type"].as_str().unwrap()
            )
        })
        .collect()
}

fn strict_schema(nested_fields: usize, definition_fields: usize) -> JsonSchema {
    fn object_with_fields(count: usize) -> Value {
        let mut properties = Map::new();
        for index in 0..count {
            properties.insert(format!("field_{index}"), json!({"type":"string"}));
        }
        json!({
            "type":"object",
            "properties":properties,
            "required":["field_0"]
        })
    }

    JsonSchema::new(json!({
        "type":"object",
        "properties":{
            "values":{
                "type":"array",
                "items":object_with_fields(nested_fields)
            }
        },
        "required":["values"],
        "$defs":{"record":object_with_fields(definition_fields)}
    }))
    .unwrap()
}

#[tokio::test]
async fn reasoning_effort_is_explicitly_mapped_and_sampling_is_bounded() {
    let server = MockServer::start().await;
    common::mount(&server, common::text_stream("ok")).await;
    let settings = CohereSettings {
        reasoning_effort: BTreeMap::from([(
            "low".into(),
            CohereThinking {
                enabled: true,
                token_budget: Some(64),
            },
        )]),
        ..CohereSettings::default()
    };
    let model = common::model_with(&server, "opaque", common::capabilities(), settings);
    model
        .complete(
            Request::new(Vec::new()).with_inference(inference(Some(0.5), Some("low"))),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["p"], 0.5);
    assert_eq!(
        body["thinking"],
        json!({"type":"enabled","token_budget":64})
    );
    for value in [0.01, 0.99] {
        model
            .validate_request(
                &Request::new(Vec::new()).with_inference(inference(Some(value), None)),
            )
            .unwrap();
    }

    for request in [
        Request::new(Vec::new()).with_inference(inference(None, Some("medium"))),
        Request::new(Vec::new()).with_inference(inference(Some(0.009), None)),
        Request::new(Vec::new()).with_inference(inference(Some(1.0), None)),
    ] {
        assert!(model.stream(request, AbortSignal::default()).await.is_err());
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn structured_output_and_strict_tool_schemas_enforce_the_cohere_subset() {
    let server = MockServer::start().await;
    let model = common::model(&server, "opaque");
    let valid = JsonSchema::new(json!({
        "type":"object",
        "properties":{
            "records":{"type":"array","items":{"$ref":"#/$defs/record"}}
        },
        "required":["records"],
        "$defs":{
            "record":{
                "type":"object",
                "properties":{
                    "id":{"type":"string","format":"uuid"},
                    "kind":{"anyOf":[{"const":"a"},{"const":"b"}]}
                },
                "required":["id"]
            }
        }
    }))
    .unwrap();
    model
        .validate_request(
            &Request::new(Vec::new()).with_response_format(ResponseFormat::structured(valid)),
        )
        .unwrap();

    for schema in [
        json!({"type":"array","items":{"type":"string"}}),
        json!({"type":"object","properties":{"x":{"type":"number","maximum":1}},"required":["x"]}),
        json!({"type":"object","properties":{"x":{"type":"string","format":"email"}},"required":["x"]}),
        json!({"type":"object","properties":{"x":{"type":"string","pattern":"^x$"}},"required":["x"]}),
        json!({"type":"object","properties":{"x":{"type":"string","default":"x"}},"required":["x"]}),
        json!({"type":"object","properties":{"x":{"type":"object","properties":{"y":{"type":"string"}}}},"required":["x"]}),
    ] {
        let request = Request::new(Vec::new())
            .with_response_format(ResponseFormat::structured(JsonSchema::new(schema).unwrap()));
        assert!(model.validate_request(&request).is_err());
    }

    let settings = CohereSettings {
        strict_tools: true,
        ..CohereSettings::default()
    };
    let strict = common::model_with(&server, "opaque", common::capabilities(), settings);
    let valid_tool = ToolDefinition::new("valid", "valid", strict_schema(100, 99));
    strict
        .validate_request(&Request::new(Vec::new()).with_tools(vec![valid_tool]))
        .unwrap();
    let oversized = ToolDefinition::new("oversized", "oversized", strict_schema(100, 100));
    assert!(
        strict
            .validate_request(&Request::new(Vec::new()).with_tools(vec![oversized]))
            .is_err()
    );
    let nested_unsupported = ToolDefinition::new(
        "bad",
        "bad",
        JsonSchema::new(json!({
            "type":"object",
            "properties":{"values":{"type":"array","items":{"type":"string","maxLength":2}}},
            "required":["values"]
        }))
        .unwrap(),
    );
    assert!(
        strict
            .validate_request(&Request::new(Vec::new()).with_tools(vec![nested_unsupported]))
            .is_err()
    );
}

#[tokio::test]
async fn inline_tool_results_are_encoded_and_unsupported_assistant_parts_are_rejected() {
    let server = MockServer::start().await;
    common::mount(&server, common::tool_stream()).await;
    let model = common::model(&server, "opaque");
    let mut turn = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    turn.message
        .content
        .push(AssistantPart::ToolResult(ToolResultPart::new(
            "call_1",
            ToolContent::Mixed(vec![
                oven_sdk::ContentValue::Text("plain".into()),
                oven_sdk::ContentValue::Json(json!({"ok":true})),
            ]),
        )));
    turn.message
        .content
        .push(AssistantPart::ToolResult(ToolResultPart::new(
            "call_2",
            ToolContent::Text("second".into()),
        )));
    let replayed = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        replayed.request.replay.decisions.as_slice(),
        [first, second]
            if matches!(first.disposition, ReplayDisposition::DiscardedInvalidPayload { .. })
                && second.disposition == ReplayDisposition::ReconstructedNormalized
    ));
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(body["messages"][0]["role"], "assistant");
    assert_eq!(body["messages"][1]["role"], "tool");
    assert_eq!(body["messages"][1]["tool_call_id"], "call_1");
    assert_eq!(body["messages"][1]["content"][1]["text"], "{\"ok\":true}");

    let file_result = completed(vec![
        AssistantPart::ToolCall(ToolCallPart::new("call_1", "lookup", json!({}))),
        AssistantPart::ToolResult(ToolResultPart::new(
            "call_1",
            ToolContent::Mixed(vec![ContentValue::File(FilePart::image(
                "image/png",
                FileSource::Bytes(bytes::Bytes::from_static(b"png")),
            ))]),
        )),
    ]);
    let error = model
        .validate_request(&Request::new(vec![HistoryTurn::assistant(file_result)]))
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::Unsupported);
    assert_eq!(
        error.diagnostics.stage,
        oven_sdk::ErrorStage::RequestValidation
    );

    let unsupported = completed(vec![AssistantPart::Custom(CustomPart::new(
        "caller.custom",
        json!({"secret":"not sent"}),
    ))]);
    assert!(
        model
            .stream(
                Request::new(vec![HistoryTurn::assistant(unsupported)]),
                AbortSignal::default(),
            )
            .await
            .is_err()
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn reasoning_replay_declaration_controls_native_capture_and_replay() {
    let server = MockServer::start().await;
    common::mount(&server, thinking_stream("private thought", "answer")).await;
    let mut capabilities = common::capabilities();
    capabilities.replay.reasoning = false;
    let model = common::model_with(&server, "opaque", capabilities, CohereSettings::default());
    let first = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let artifact = first.turn.finish.native_replay.as_ref().unwrap();
    assert_eq!(
        artifact.payload().get("format").and_then(Value::as_str),
        Some("cohere.v2.chat.message.v2")
    );
    assert!(
        artifact
            .scope()
            .resource_id
            .as_str()
            .starts_with("cohere.v2.chat.replay_scope.v2.sha256.")
    );
    assert_eq!(
        artifact.payload().pointer("/message/content"),
        Some(&json!([{"type":"text","text":"answer"}]))
    );

    let replayed = model
        .stream(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        replayed.request.replay.decisions.as_slice(),
        [decision] if decision.disposition == ReplayDisposition::Replayed
    ));
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        body.pointer("/messages/0/content"),
        Some(&json!([{"type":"text","text":"answer"}]))
    );
}

#[tokio::test]
async fn optional_oversize_replay_is_discarded_but_required_replay_is_fatal() {
    let text = "x".repeat(NativeReplayArtifact::MAX_PAYLOAD_BYTES + 1024);

    let optional_server = MockServer::start().await;
    common::mount(&optional_server, common::text_stream(&text)).await;
    let optional = common::model(&optional_server, "opaque");
    let completed = optional
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(completed.turn.finish.native_replay.is_none());
    assert_eq!(
        completed
            .turn
            .finish
            .provider_metadata
            .get("cohere.replay_capture")
            .and_then(|value| value.get("reason"))
            .and_then(Value::as_str),
        Some("payload_too_large")
    );

    let required_server = MockServer::start().await;
    common::mount(&required_server, common::text_stream(&text)).await;
    let mut capabilities = common::capabilities();
    capabilities.replay.policy = ReplayPolicy::IfValid;
    capabilities.replay.capability = ReplayCapability::Required;
    let required = common::model_with(
        &required_server,
        "opaque",
        capabilities,
        CohereSettings::default(),
    );
    let error = required
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::Replay);
}

#[tokio::test]
async fn timeout_settings_are_scope_bound_and_error_finish_is_always_in_band() {
    let scope_server = MockServer::start().await;
    common::mount(&scope_server, common::text_stream("ok")).await;
    let first_model = common::model(&scope_server, "opaque");
    let first = first_model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let default_timeouts = CohereTimeouts::default();
    let settings = CohereSettings {
        timeouts: CohereTimeouts {
            stream_idle: default_timeouts.stream_idle + Duration::from_secs(1),
            ..default_timeouts
        },
        ..CohereSettings::default()
    };
    let second_model =
        common::model_with(&scope_server, "opaque", common::capabilities(), settings);
    let second = second_model
        .stream(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        second.request.replay.decisions.as_slice(),
        [first, second]
            if matches!(first.disposition, ReplayDisposition::DiscardedForeignScope { .. })
                && second.disposition == ReplayDisposition::ReconstructedNormalized
    ));

    let error_server = MockServer::start().await;
    common::mount(
        &error_server,
        "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"gen_error\",\"delta\":{\"message\":{\"role\":\"assistant\"}}}\n\nevent: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"ERROR\"}}\n\n".into(),
    )
    .await;
    let model = common::model(&error_server, "opaque");
    let mut response = model
        .stream(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let mut saw_error = false;
    let mut saw_finish = false;
    while let Some(item) = response.stream.next().await {
        match item.unwrap() {
            StreamPart::Error { .. } => {
                assert!(!saw_finish);
                saw_error = true;
            }
            StreamPart::Finish { finish } => {
                assert!(saw_error);
                assert_eq!(finish.finish_reason, FinishReason::Error);
                saw_finish = true;
            }
            _ => {}
        }
    }
    assert!(saw_error && saw_finish);
}

#[tokio::test]
async fn truncated_sse_fails_but_event_name_mismatch_is_accepted() {
    let truncated_server = MockServer::start().await;
    common::mount(
        &truncated_server,
        "event: message-start\ndata: {\"type\":\"message-start\",\"delta\":{\"message\":{\"role\":\"assistant\"}}}\n\n".into(),
    )
    .await;
    let truncated = common::model(&truncated_server, "opaque");
    let error = truncated
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::UnexpectedEof);

    let mismatch_server = MockServer::start().await;
    common::mount(
        &mismatch_server,
        concat!(
            "event: content-start\ndata: {\"type\":\"message-start\",\"delta\":{\"message\":{\"role\":\"assistant\"}}}\n\n",
            "event: content-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\"}}\n\n"
        ).into(),
    )
    .await;
    let mismatch = common::model(&mismatch_server, "opaque");
    let completed = mismatch
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert_eq!(completed.turn.finish.finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn missing_tool_id_and_name_use_a_stable_id_and_empty_name() {
    let server = MockServer::start().await;
    common::mount(
        &server,
        concat!(
            "event: message-start\ndata: {\"type\":\"message-start\",\"delta\":{\"message\":{\"role\":\"assistant\"}}}\n\n",
            "event: tool-call-start\ndata: {\"type\":\"tool-call-start\",\"index\":3,\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"{}\"}}}}}\n\n",
            "event: tool-call-end\ndata: {\"type\":\"tool-call-end\",\"index\":3}\n\n",
            "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"TOOL_CALL\"}}\n\n"
        ).into(),
    )
    .await;
    let completed = common::model(&server, "opaque")
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    assert!(completed.turn.message.content.iter().any(|part| {
        matches!(part, AssistantPart::ToolCall(call) if call.id == "google-call-3" && call.name.is_empty())
    }));
    let replay = completed.turn.finish.native_replay.unwrap();
    assert!(
        replay
            .payload()
            .pointer("/message/tool_calls/0/function/name")
            .is_none()
    );
}
