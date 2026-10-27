use futures_util::StreamExt;
use oven_sdk::{
    AbortSignal, CompactionRequest, FinishReason, HistoryTurn, InferenceOptions, InputPart,
    LanguageModel, ReplayDisposition, Request, StreamPart, TextPart, UserMessage,
};
use oven_sdk_openai::{OpenAiChatModel, OpenAiCompatibleChatModel, OpenAiResponsesModel};

fn key() -> String {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is required for ignored live tests")
}

fn minimal_user_turn() -> HistoryTurn {
    HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
        "Reply exactly with the word pong.",
    ))]))
}

fn minimal_request() -> Request {
    Request::new(vec![minimal_user_turn()])
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and network access"]
async fn live_chat() {
    let model = OpenAiChatModel::new(common::official_chat_config_at(
        "https://api.openai.com/v1",
        "gpt-4o-mini",
        &key(),
    ))
    .unwrap();
    let result = model
        .complete(minimal_request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(result.turn.text().to_lowercase().contains("pong"));
    assert_eq!(result.turn.finish.finish_reason, FinishReason::Stop);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and network access"]
async fn live_responses() {
    let model = OpenAiResponsesModel::new(common::official_responses_config_at(
        "https://api.openai.com/v1",
        "gpt-5-mini",
        &key(),
    ))
    .unwrap();
    let result = model
        .complete(minimal_request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(result.turn.text().to_lowercase().contains("pong"));
    assert_eq!(result.turn.finish.finish_reason, FinishReason::Stop);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY, native-compaction model access, and network access"]
async fn live_responses_native_compaction_round_trip() {
    let model_id = std::env::var("OPENAI_COMPACTION_MODEL").unwrap_or_else(|_| "gpt-5-mini".into());
    let model = OpenAiResponsesModel::new(common::official_responses_native_config_at(
        "https://api.openai.com/v1",
        &model_id,
        &key(),
    ))
    .unwrap();
    let compacted = model
        .compact(
            CompactionRequest::new(Request::new(vec![minimal_user_turn()])),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(!compacted.native_context.payload().is_null());
    let result = model
        .complete(
            Request::new(vec![minimal_user_turn()]).with_native_context(compacted.native_context),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(result.turn.text().to_lowercase().contains("pong"));
    assert_eq!(result.turn.finish.finish_reason, FinishReason::Stop);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and network access"]
async fn live_compatible_chat() {
    let mut config =
        common::compatible_config_at("https://api.openai.com/v1", "gpt-4o-mini", &key());
    config.provider.id = oven_sdk::ProviderId::new("openai-compatible-live");
    config.settings.adapter_id = oven_sdk::AdapterId::new("live.openai-compatible.chat");
    config.settings.stream_usage = true;
    let model = OpenAiCompatibleChatModel::new(config).unwrap();
    let result = model
        .complete(minimal_request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(result.turn.text().to_lowercase().contains("pong"));
    assert_eq!(result.turn.finish.finish_reason, FinishReason::Stop);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and network access"]
async fn live_replay_round_trip() {
    let model = OpenAiChatModel::new(common::official_chat_config_at(
        "https://api.openai.com/v1",
        "gpt-4o-mini",
        &key(),
    ))
    .unwrap();
    let first = model
        .complete(minimal_request(), AbortSignal::default())
        .await
        .unwrap();
    assert!(first.turn.text().to_lowercase().contains("pong"));
    assert_eq!(first.turn.finish.finish_reason, FinishReason::Stop);
    assert!(first.turn.finish.native_replay.is_some());
    let second = model
        .complete(
            Request::new(vec![
                minimal_user_turn(),
                HistoryTurn::assistant(first.turn),
            ]),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(second.turn.text().to_lowercase().contains("pong"));
    assert_eq!(second.turn.finish.finish_reason, FinishReason::Stop);
    assert!(matches!(
        second.request.replay.decisions.as_slice(),
        [oven_sdk::ReplayDecision {
            disposition: ReplayDisposition::Replayed,
            ..
        }]
    ));
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and network access"]
async fn live_cancellation() {
    let model = OpenAiChatModel::new(common::official_chat_config_at(
        "https://api.openai.com/v1",
        "gpt-4o-mini",
        &key(),
    ))
    .unwrap();
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    assert!(
        model
            .stream(Request::new(Vec::new()), signal)
            .await
            .is_err()
    );
}

#[tokio::test]
#[ignore = "requires OPENAI_LONG_STREAM_TEST=1 and configured duration or byte threshold"]
async fn live_long_stream() {
    assert_eq!(
        std::env::var("OPENAI_LONG_STREAM_TEST").as_deref(),
        Ok("1"),
        "set OPENAI_LONG_STREAM_TEST=1 to opt into this potentially expensive test"
    );
    let minimum_seconds = std::env::var("OPENAI_LONG_STREAM_MIN_SECONDS")
        .ok()
        .map(|value| value.parse::<u64>().expect("valid minimum seconds"));
    let minimum_bytes = std::env::var("OPENAI_LONG_STREAM_MIN_BYTES")
        .ok()
        .map(|value| value.parse::<usize>().expect("valid minimum bytes"));
    assert!(
        minimum_seconds.is_some() || minimum_bytes.is_some(),
        "set OPENAI_LONG_STREAM_MIN_SECONDS or OPENAI_LONG_STREAM_MIN_BYTES"
    );
    let model_id =
        std::env::var("OPENAI_LONG_STREAM_MODEL").unwrap_or_else(|_| "gpt-5-mini".into());
    let prompt = std::env::var("OPENAI_LONG_STREAM_PROMPT").unwrap_or_else(|_| {
        "Write a long, detailed technical essay. Continue until the output limit.".into()
    });
    let mut inference = InferenceOptions::new();
    inference.max_output_tokens = Some(
        std::env::var("OPENAI_LONG_STREAM_MAX_OUTPUT_TOKENS")
            .unwrap_or_else(|_| "16000".into())
            .parse()
            .expect("valid maximum output tokens"),
    );
    let model = OpenAiResponsesModel::new(common::official_responses_config_at(
        "https://api.openai.com/v1",
        &model_id,
        &key(),
    ))
    .unwrap();
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new(prompt)),
    ]))])
    .with_inference(inference);
    let started = std::time::Instant::now();
    let mut response = model.stream(request, AbortSignal::default()).await.unwrap();
    let mut output_bytes = 0_usize;
    let mut finished = false;
    while let Some(item) = response.stream.next().await {
        match item.unwrap() {
            StreamPart::TextDelta { delta, .. }
            | StreamPart::ReasoningDelta { delta, .. }
            | StreamPart::ToolCallDelta { delta, .. } => output_bytes += delta.len(),
            StreamPart::Finish { .. } => finished = true,
            _ => {}
        }
    }
    let elapsed = started.elapsed();
    assert!(finished, "long stream ended without Finish");
    let duration_met =
        minimum_seconds.is_some_and(|seconds| elapsed >= std::time::Duration::from_secs(seconds));
    let bytes_met = minimum_bytes.is_some_and(|bytes| output_bytes >= bytes);
    assert!(
        duration_met || bytes_met,
        "long stream missed configured gate: elapsed={elapsed:?}, output_bytes={output_bytes}"
    );
}
pub mod common;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and network access"]
async fn live_responses_deferred_tool_round_trip() {
    use oven_sdk::{
        AssistantMessage, AssistantPart, CompletedTurn, DeferredToolsCapability, Finish,
        JsonSchema, ToolCallPart, ToolContent, ToolDefinition, ToolMessage, ToolResultPart,
    };
    let mut config = common::official_responses_config_at(
        "https://api.openai.com/v1",
        &std::env::var("OPENAI_LIVE_MODEL").unwrap_or_else(|_| "gpt-5.5".into()),
        &key(),
    );
    config.model.capabilities.deferred_tools = DeferredToolsCapability::Native;
    config.provider.auth.organization = None;
    config.provider.auth.project = None;
    let model = OpenAiResponsesModel::new(config).unwrap();
    let tool = |name: &str, description: &str| {
        ToolDefinition::new(
            name,
            description,
            JsonSchema::new(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            }))
            .unwrap(),
        )
    };
    let tools = || {
        vec![
            tool("tool_search", "Search for more tools by keyword."),
            tool("get_weather", "Get the current weather for a city.").deferred(),
            tool("get_stock_price", "Get the stock price for a ticker.").deferred(),
        ]
    };
    let mut history = vec![
        HistoryTurn::user(UserMessage::new(vec![InputPart::Text(TextPart::new(
            "What is the weather in Paris? Use the tools.",
        ))])),
        HistoryTurn::assistant(CompletedTurn::new(
            AssistantMessage::new(vec![AssistantPart::ToolCall(ToolCallPart::new(
                "call_live_search",
                "tool_search",
                serde_json::json!({"city": "weather"}),
            ))]),
            Finish::new(Default::default(), FinishReason::ToolCalls),
        )),
        HistoryTurn::tool(ToolMessage::new(vec![ToolResultPart::new(
            "call_live_search",
            ToolContent::LoadTools {
                names: vec!["get_weather".into()],
                text: "Loaded tools: get_weather".into(),
            },
        )])),
    ];
    let turn = model
        .complete(
            Request::new(history.clone()).with_tools(tools()),
            AbortSignal::default(),
        )
        .await
        .unwrap()
        .turn;
    let call = turn
        .message
        .content
        .iter()
        .find_map(|part| match part {
            AssistantPart::ToolCall(call) if call.name == "get_weather" => Some(call.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected a get_weather call: {:?}", turn.message.content));
    history.push(HistoryTurn::assistant(turn));
    history.push(HistoryTurn::tool(ToolMessage::new(vec![
        ToolResultPart::new(call.id.clone(), ToolContent::Text("Sunny, 18 C".into())),
    ])));
    // Replaying the call needs its namespace, natively and normalized alike.
    let answer = model
        .complete(
            Request::new(history.clone()).with_tools(tools()),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(!answer.turn.text().is_empty());
    let mut normalized = history;
    for turn in &mut normalized {
        if let HistoryTurn::Assistant(turn) = turn {
            turn.finish.native_replay = None;
        }
    }
    let answer = model
        .complete(
            Request::new(normalized).with_tools(tools()),
            AbortSignal::default(),
        )
        .await
        .unwrap();
    assert!(!answer.turn.text().is_empty());
}
