mod common;

use oven_sdk::{
    AbortSignal, HistoryTurn, InputPart, LanguageModel, Request, SecretString, TextPart,
    UserMessage,
};
use oven_sdk_azure::{
    AzureApiRoute, AzureOpenAiAuth, AzureOpenAiChatModel, AzureOpenAiResponsesModel,
};

fn api() -> String {
    let resource = std::env::var("AZURE_OPENAI_RESOURCE").expect("AZURE_OPENAI_RESOURCE");
    format!("https://{resource}.openai.azure.com")
}

fn auth() -> AzureOpenAiAuth {
    AzureOpenAiAuth::ApiKey(SecretString::new(
        std::env::var("AZURE_OPENAI_API_KEY").expect("AZURE_OPENAI_API_KEY"),
    ))
}

fn chat_model(deployment: String, setup: common::ModelSetup) -> AzureOpenAiChatModel {
    AzureOpenAiChatModel::new(common::chat_config(
        api(),
        AzureApiRoute::V1,
        deployment,
        setup,
        auth(),
    ))
    .unwrap()
}

fn responses_model(deployment: String, setup: common::ModelSetup) -> AzureOpenAiResponsesModel {
    AzureOpenAiResponsesModel::new(common::responses_config(
        api(),
        AzureApiRoute::V1,
        deployment,
        setup,
        auth(),
    ))
    .unwrap()
}

fn request() -> Request {
    Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("Reply exactly with pong.")),
    ]))])
}

#[tokio::test]
#[ignore = "requires Azure OpenAI resource, API key, deployment, and network access"]
async fn live_chat() {
    let deployment = std::env::var("AZURE_OPENAI_CHAT_DEPLOYMENT").expect("chat deployment");
    let model_name =
        std::env::var("AZURE_OPENAI_CHAT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
    let mut setup = common::gpt4o();
    setup.revision.as_mut().unwrap().model = model_name;
    setup.revision.as_mut().unwrap().version = std::env::var("AZURE_OPENAI_CHAT_MODEL_VERSION")
        .unwrap_or_else(|_| "caller-supplied-version".into());
    let result = chat_model(deployment, setup)
        .complete(request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(result.turn.text().to_lowercase().contains("pong"));
}

#[tokio::test]
#[ignore = "requires Azure OpenAI resource, API key, deployment, and network access"]
async fn live_responses_and_replay() {
    let deployment =
        std::env::var("AZURE_OPENAI_RESPONSES_DEPLOYMENT").expect("responses deployment");
    let model_name =
        std::env::var("AZURE_OPENAI_RESPONSES_MODEL").unwrap_or_else(|_| "gpt-5-mini".into());
    let mut setup = common::gpt5();
    setup.revision.as_mut().unwrap().model = model_name;
    setup.revision.as_mut().unwrap().version =
        std::env::var("AZURE_OPENAI_RESPONSES_MODEL_VERSION")
            .unwrap_or_else(|_| "caller-supplied-version".into());
    let model = responses_model(deployment, setup);
    let first = model
        .complete(request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(first.turn.text().to_lowercase().contains("pong"));
    let second = model
        .complete(
            Request::new(vec![HistoryTurn::assistant(first.turn)]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(second.turn.finish.native_replay.is_some());
}

#[tokio::test]
#[ignore = "requires Azure OpenAI resource, API key, deployment, and network access"]
async fn live_local_cancellation() {
    let deployment = std::env::var("AZURE_OPENAI_CHAT_DEPLOYMENT").expect("chat deployment");
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    assert!(
        chat_model(deployment, common::gpt4o())
            .stream(request(), signal)
            .await
            .is_err()
    );
}

#[tokio::test]
#[ignore = "requires AZURE_OPENAI_LONG_STREAM_TEST=1 and a configured byte threshold"]
async fn live_long_stream() {
    use futures_util::StreamExt;
    use oven_sdk::StreamPart;

    assert_eq!(
        std::env::var("AZURE_OPENAI_LONG_STREAM_TEST").as_deref(),
        Ok("1")
    );
    let minimum = std::env::var("AZURE_OPENAI_LONG_STREAM_MIN_BYTES")
        .expect("AZURE_OPENAI_LONG_STREAM_MIN_BYTES")
        .parse::<usize>()
        .expect("byte threshold");
    let deployment =
        std::env::var("AZURE_OPENAI_RESPONSES_DEPLOYMENT").expect("responses deployment");
    let mut response = responses_model(deployment, common::gpt5())
        .stream(request(), AbortSignal::default())
        .await
        .unwrap();
    let mut bytes = 0;
    let mut finished = false;
    while let Some(item) = response.stream.next().await {
        match item.unwrap() {
            StreamPart::TextDelta { delta, .. }
            | StreamPart::ReasoningDelta { delta, .. }
            | StreamPart::ToolCallDelta { delta, .. } => bytes += delta.len(),
            StreamPart::Finish { .. } => finished = true,
            _ => {}
        }
    }
    assert!(finished);
    assert!(
        bytes >= minimum,
        "normalized output bytes {bytes} < {minimum}"
    );
}
