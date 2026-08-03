//! Environment-gated live Anthropic checks.

mod common;

use common::{Anthropic, AnthropicAws, MiniMax};

use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, HistoryTurn, InputPart, LanguageModel, Request, TextPart, UserMessage,
};
use oven_sdk_anthropic::AnthropicAwsCredentials;

fn model() -> oven_sdk_anthropic::AnthropicModel {
    Anthropic::builder()
        .api_key(std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set"))
        .build()
        .unwrap()
        .model("claude-haiku-4-5-20251001")
}

fn request(prompt: &str) -> Request {
    Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new(prompt)),
    ]))])
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and external network access"]
async fn live_basic_complete_and_stream() {
    let complete = model()
        .complete(request("Reply with exactly: live"), AbortSignal::default())
        .await
        .unwrap();
    assert!(!complete.turn.text().is_empty());

    let mut stream = model()
        .stream(
            request("Reply with exactly: stream"),
            AbortSignal::default(),
        )
        .await
        .unwrap()
        .stream;
    assert!(stream.next().await.unwrap().is_ok());
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and external network access"]
async fn live_replay_round_trip() {
    let first = model()
        .complete(
            request("Reply with exactly: replay"),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    let follow_up = Request::new(vec![
        HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
            "Reply with exactly: replay",
        ))])),
        HistoryTurn::assistant(first.turn),
        HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
            "Reply with exactly: continued",
        ))])),
    ]);
    let response = model()
        .stream(follow_up, AbortSignal::default())
        .await
        .unwrap();
    assert!(
        response
            .request
            .replay
            .decisions
            .iter()
            .any(|decision| matches!(decision.disposition, oven_sdk::ReplayDisposition::Replayed))
    );
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and external network access"]
async fn live_long_stream() {
    let mut request = request("Count from 1 to 5000, one number per line.");
    request.inference.max_output_tokens = Some(12_000);
    let mut stream = model()
        .stream(request, AbortSignal::default())
        .await
        .unwrap()
        .stream;
    let mut count = 0;
    while let Some(item) = stream.next().await {
        item.unwrap();
        count += 1;
    }
    assert!(count > 1);
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and external network access"]
async fn live_cancellation() {
    let (signal, registration) = AbortSignal::new();
    let mut stream = model()
        .stream(request("Write a very long story."), signal)
        .await
        .unwrap()
        .stream;
    registration.abort();
    while let Some(item) = stream.next().await {
        if let Err(error) = item {
            assert_eq!(error.kind, oven_sdk::ModelErrorKind::Abort);
            return;
        }
    }
    panic!("cancelled live stream did not report an abort");
}

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY and external network access"]
async fn live_minimax_messages() {
    let model = MiniMax::builder()
        .api_key(std::env::var("MINIMAX_API_KEY").expect("MINIMAX_API_KEY must be set"))
        .build()
        .unwrap()
        .model("MiniMax-M3");
    let result = model
        .complete(
            request("Reply with exactly: minimax-live"),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(!result.turn.text().is_empty());
}

#[tokio::test]
#[ignore = "requires Claude Platform on AWS environment credentials and external network access"]
async fn live_anthropic_aws_messages() {
    let region = std::env::var("AWS_REGION").expect("AWS_REGION must be set");
    let workspace = std::env::var("ANTHROPIC_AWS_WORKSPACE_ID")
        .expect("ANTHROPIC_AWS_WORKSPACE_ID must be set");
    let builder = AnthropicAws::builder(region, workspace);
    let factory = if let Ok(key) = std::env::var("ANTHROPIC_AWS_API_KEY") {
        builder.bearer_key(key).build().unwrap()
    } else {
        builder
            .static_credentials(AnthropicAwsCredentials {
                access_key_id: std::env::var("AWS_ACCESS_KEY_ID")
                    .expect("AWS_ACCESS_KEY_ID must be set"),
                secret_access_key: oven_sdk::SecretString::new(
                    std::env::var("AWS_SECRET_ACCESS_KEY")
                        .expect("AWS_SECRET_ACCESS_KEY must be set"),
                ),
                session_token: std::env::var("AWS_SESSION_TOKEN")
                    .ok()
                    .map(oven_sdk::SecretString::new),
            })
            .build()
            .unwrap()
    };
    let result = factory
        .model("claude-sonnet-4-6")
        .complete(
            request("Reply with exactly: aws-live"),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(!result.turn.text().is_empty());
}
