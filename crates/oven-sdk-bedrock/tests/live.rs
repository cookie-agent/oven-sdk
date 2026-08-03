use std::time::{Duration, Instant};

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, HistoryTurn, InputPart, JsonSchema, LanguageModel, Request, TextPart,
    ToolDefinition, UserMessage,
};
use oven_sdk_bedrock::{
    AwsCredentials, BedrockAuth, BedrockModel, BedrockRequestExt, BedrockRequestOptions,
    BedrockTimeouts,
};

fn live_model(timeouts: Option<BedrockTimeouts>) -> Option<BedrockModel> {
    let access_key_id = std::env::var("AWS_ACCESS_KEY_ID").ok()?;
    let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok()?;
    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into());
    let model_id = std::env::var("BEDROCK_MODEL_ID")
        .unwrap_or_else(|_| "global.anthropic.claude-sonnet-4-6".into());
    let endpoint = std::env::var("BEDROCK_ENDPOINT")
        .unwrap_or_else(|_| format!("https://bedrock-runtime.{region}.amazonaws.com"));
    let mut config = support::config(
        &endpoint,
        &model_id,
        support::FixtureKind::SignedReasoning,
        BedrockAuth::Static(AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token: std::env::var("AWS_SESSION_TOKEN").ok(),
        }),
    );
    config.settings.region = region;
    if let Some(timeouts) = timeouts {
        config.settings.timeouts = timeouts;
    }
    BedrockModel::new(config).ok()
}

fn user_request(text: &str) -> Request {
    Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new(text)),
    ]))])
}

#[tokio::test]
#[ignore = "requires explicit AWS credentials, region, and model"]
async fn live_bedrock_text_stream() {
    let Some(model) = live_model(None) else {
        return;
    };
    let result = model
        .complete(
            user_request("Reply with the word oven."),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(result.turn.finish.native_replay.is_some());
}

#[tokio::test]
#[ignore = "requires a tool-capable Bedrock model"]
async fn live_bedrock_tool_stream() {
    let Some(model) = live_model(None) else {
        return;
    };
    let request =
        user_request("Use the lookup tool for key oven.").with_tools(vec![ToolDefinition::new(
            "lookup",
            "Looks up a key",
            JsonSchema::new(serde_json::json!({
                "type":"object","properties":{"key":{"type":"string"}},"required":["key"]
            }))
            .unwrap(),
        )]);
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
            .any(|part| matches!(part, oven_sdk::AssistantPart::ToolCall(_)))
    );
}

#[tokio::test]
#[ignore = "requires a signed-reasoning Bedrock model"]
async fn live_bedrock_signed_reasoning_replay() {
    let Some(model) = live_model(None) else {
        return;
    };
    let first = user_request("Think carefully and answer: what is 17 * 19?").with_bedrock_options(
        BedrockRequestOptions {
            reasoning_type: Some("enabled".into()),
            reasoning_budget_tokens: Some(1_024),
            ..Default::default()
        },
    );
    let turn = model
        .complete(first, AbortSignal::default())
        .await
        .unwrap()
        .turn;
    assert!(turn.finish.native_replay.is_some());
    let second = Request::new(vec![
        HistoryTurn::assistant(turn),
        HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
            "Now add one to that result.",
        ))])),
    ]);
    let result = model
        .complete(second, AbortSignal::default())
        .await
        .unwrap();
    assert!(!result.turn.text().is_empty());
}

#[tokio::test]
#[ignore = "requires explicit AWS credentials and external network"]
async fn live_bedrock_cancellation() {
    let Some(model) = live_model(None) else {
        return;
    };
    let (signal, registration) = AbortSignal::new();
    let response = model
        .stream(
            user_request("Write a very long detailed history of computing."),
            signal,
        )
        .await;
    let Ok(mut response) = response else {
        return;
    };
    registration.abort();
    let mut aborted = false;
    while let Some(item) = response.stream.next().await {
        if let Err(error) = item {
            aborted = error.kind == oven_sdk::ModelErrorKind::Abort;
            break;
        }
    }
    assert!(aborted);
}

#[tokio::test]
#[ignore = "requires BEDROCK_LONG_STREAM_TEST=1 and a stream lasting over two minutes"]
async fn live_bedrock_stream_exceeding_two_minutes() {
    if std::env::var("BEDROCK_LONG_STREAM_TEST").as_deref() != Ok("1") {
        return;
    }
    let Some(model) = live_model(Some(BedrockTimeouts {
        stream_idle: Duration::from_secs(180),
        ..Default::default()
    })) else {
        return;
    };
    let started = Instant::now();
    let result = model
        .complete(
            user_request("Produce an extremely long, detailed technical monograph."),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(started.elapsed() > Duration::from_secs(120));
    assert!(!result.turn.text().is_empty());
}
mod support;
