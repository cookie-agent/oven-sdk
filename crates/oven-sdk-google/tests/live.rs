use oven_sdk::{
    AbortSignal, HistoryTurn, InputPart, LanguageModel, Request, TextPart, UserMessage,
};

mod common;
use common::model_with_key;

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn live_google_text_stream() {
    let key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY");
    let model = model_with_key(
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini-2.5-flash",
        &key,
    );
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("Reply with the word oven.")),
    ]))]);
    let result = model
        .complete(request, AbortSignal::default())
        .await
        .unwrap();
    assert!(!result.turn.text().is_empty());
}
