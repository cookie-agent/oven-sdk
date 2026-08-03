use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, Finish, FinishReason, HistoryTurn,
    JsonSchema, LanguageModel, Request, ResponseFormat, TextPart, ToolDefinition,
};
use oven_sdk_anthropic::{AnthropicRequestExt, AnthropicRequestOptions, AnthropicToolOptions};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn schema_less_json_is_invalid() {
    let model = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    assert!(
        model
            .validate_request(
                &Request::new(Vec::new())
                    .with_response_format(ResponseFormat::Json { schema: None })
            )
            .is_err()
    );
    assert!(JsonSchema::new(serde_json::json!({"type":"object"})).is_ok());
}

#[test]
fn unsupported_schema_keywords_are_rejected_individually() {
    let model = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    for keyword in [
        "oneOf", "anyOf", "allOf", "not", "$ref", "pattern", "minimum",
    ] {
        let schema = JsonSchema::new(serde_json::json!({"type":"object", keyword: []})).unwrap();
        let request =
            Request::new(Vec::new()).with_response_format(ResponseFormat::structured(schema));
        assert!(model.validate_request(&request).is_err(), "{keyword}");
    }
}

#[test]
fn structured_output_rejects_assistant_prefill() {
    let model = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let schema = JsonSchema::new(serde_json::json!({"type":"object"})).unwrap();
    let turn = CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::Text(TextPart::new("prefill"))]),
        Finish::new(Default::default(), FinishReason::Stop),
    );
    assert!(
        model
            .validate_request(
                &Request::new(vec![HistoryTurn::assistant(turn)])
                    .with_response_format(ResponseFormat::structured(schema))
            )
            .is_err()
    );
}

#[tokio::test]
async fn structured_output_omits_obsolete_beta_and_coexists_with_strict_tools() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            "event: message_start\ndata: {\"message\":{}}\n\nevent: message_stop\ndata: {}\n\n",
        ))
        .mount(&server)
        .await;
    let model = Anthropic::builder()
        .base_url(server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let schema = JsonSchema::new(serde_json::json!({"type":"object"})).unwrap();
    let mut tool = ToolDefinition::new("lookup", "find", schema.clone());
    tool.provider_options.insert(
        "anthropic".into(),
        serde_json::to_value(AnthropicToolOptions { strict: true }).unwrap(),
    );
    let request = Request::new(Vec::new())
        .with_tools(vec![tool])
        .with_response_format(ResponseFormat::structured(schema))
        .with_anthropic_options(AnthropicRequestOptions {
            betas: vec!["structured-outputs-2025-11-13".into()],
            ..Default::default()
        });
    oven_sdk::LanguageModel::complete(&model, request, AbortSignal::default())
        .await
        .unwrap();
    let request = &server.received_requests().await.unwrap()[0];
    assert!(request.headers.get("anthropic-beta").is_none());
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["tools"][0]["strict"], true);
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
}
mod common;

use common::Anthropic;
