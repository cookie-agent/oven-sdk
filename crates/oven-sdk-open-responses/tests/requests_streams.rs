mod common;

use oven_sdk::{
    AbortSignal, FilePart, FileSource, HeaderOverrides, HistoryTurn, InferenceOptions, InputPart,
    JsonSchema, LanguageModel, Request, ResponseFormat, TextPart, ToolDefinition, UserMessage,
};
use oven_sdk_open_responses::{
    OpenResponsesModel, OpenResponsesRequestExt, OpenResponsesRequestOptions,
};
use reqwest::header::{HeaderMap, HeaderValue};
use wiremock::MockServer;

#[tokio::test]
async fn caller_cookie_suppresses_open_responses_bearer_injection() {
    let server = MockServer::start().await;
    common::mount(&server, common::text_stream("ok")).await;
    let mut config = common::generic_config(&server, "opaque");
    let mut headers = HeaderMap::new();
    headers.insert("cookie", HeaderValue::from_static("caller-session=value"));
    config.provider.headers.static_headers = HeaderOverrides::new(headers);
    let model = OpenResponsesModel::new(config).unwrap();

    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].headers["cookie"], "caller-session=value");
    assert!(requests[0].headers.get("authorization").is_none());
}

#[tokio::test]
async fn generic_request_encodes_standard_tools_schema_reasoning_sources_and_media() {
    let server = MockServer::start().await;
    common::mount(&server, common::text_stream("ok")).await;
    let model = common::generic_model(&server, "exact/model:route");
    let schema = JsonSchema::new(serde_json::json!({
        "type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false
    }))
    .unwrap();
    let mut inference = InferenceOptions::new();
    inference.reasoning_effort = Some("future-effort".into());
    inference.max_output_tokens = Some(64);
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("inspect")),
        InputPart::File(FilePart::image(
            "image/png",
            FileSource::Bytes(bytes::Bytes::from_static(b"png")),
        )),
        InputPart::File(FilePart::document(
            "application/pdf",
            FileSource::Bytes(bytes::Bytes::from_static(b"pdf")),
        )),
    ]))])
    .with_tools(vec![ToolDefinition::new("lookup", "find", schema.clone())])
    .with_response_format(ResponseFormat::structured(schema))
    .with_inference(inference)
    .with_open_responses_options(OpenResponsesRequestOptions {
        service_tier: Some("future-tier".into()),
        truncation: Some("future-policy".into()),
        text_verbosity: Some("future-verbosity".into()),
        ..Default::default()
    });
    let result = model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert!(
        result
            .turn
            .message
            .content
            .iter()
            .any(|part| matches!(part, oven_sdk::AssistantPart::Source(_)))
    );
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["model"], "exact/model:route");
    assert_eq!(body["tools"][0]["strict"], true);
    assert_eq!(body["text"]["format"]["type"], "json_schema");
    assert_eq!(body["reasoning"]["effort"], "future-effort");
    assert_eq!(body["service_tier"], "future-tier");
    assert!(
        body["input"][0]["content"][1]["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
    assert!(
        body["input"][0]["content"][2]["file_data"]
            .as_str()
            .unwrap()
            .starts_with("data:application/pdf;base64,")
    );
}

#[tokio::test]
async fn hugging_face_profile_does_not_rewrite_exact_model_id() {
    let server = MockServer::start().await;
    common::mount(&server, common::text_stream("ok")).await;
    let model = common::hugging_face_model(&server, "org/model:groq");
    model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(body["model"], "org/model:groq");
    assert_eq!(
        model.descriptor().provider_metadata["huggingface.routing"],
        "exact-model-id-suffix"
    );
}
