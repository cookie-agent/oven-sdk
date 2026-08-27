use oven_sdk::{
    AbortSignal, AssistantMessage, AssistantPart, CompletedTurn, ContentValue, FilePart,
    FileSource, Finish, FinishReason, HistoryTurn, InputPart, LanguageModel, ModelErrorKind,
    ProviderId, Request, TextPart, ToolCallPart, ToolContent, ToolMessage, ToolResultPart,
    UserMessage,
};
use oven_sdk_anthropic::{MiniMaxMediaExt, MiniMaxMediaOptions};
use wiremock::MockServer;

fn request_with(file: FilePart) -> Request {
    Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::File(file),
    ]))])
}

fn tool_result_request_with(file: FilePart) -> Request {
    let assistant = CompletedTurn::new(
        AssistantMessage::new(vec![AssistantPart::ToolCall(ToolCallPart::new(
            "call-1",
            "inspect",
            serde_json::json!({}),
        ))]),
        Finish::new(Default::default(), FinishReason::ToolCalls),
    );
    let result = ToolResultPart::new("call-1", ToolContent::Mixed(vec![ContentValue::File(file)]));
    Request::new(vec![
        HistoryTurn::assistant(assistant),
        HistoryTurn::tool(ToolMessage::new(vec![result])),
    ])
}

#[test]
fn anthropic_media_mime_source_capability_and_image_boundaries_are_exact() {
    let known = Anthropic::builder()
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let mut protocol = common::anthropic_protocol();
    protocol.thinking = oven_sdk_anthropic::AnthropicThinkingSupport::None;
    protocol.thinking_disable_allowed = false;
    protocol.effort = false;
    let without_media = Anthropic::builder()
        .capabilities(common::conservative_capabilities())
        .protocol(protocol)
        .build()
        .unwrap()
        .model("future-model");
    let encoded_limit_raw_bytes = 10 * 1024 * 1024 / 4 * 3;

    assert!(
        known
            .validate_request(&request_with(FilePart::image(
                "image/png",
                FileSource::Bytes(vec![0; encoded_limit_raw_bytes].into()),
            )))
            .is_ok()
    );
    assert!(
        known
            .validate_request(&request_with(FilePart::image(
                "image/png",
                FileSource::Bytes(vec![0; encoded_limit_raw_bytes + 1].into()),
            )))
            .is_err()
    );
    let error = known
        .validate_request(&tool_result_request_with(FilePart::image(
            "image/png",
            FileSource::Bytes(vec![0; encoded_limit_raw_bytes + 1].into()),
        )))
        .unwrap_err();
    assert_eq!(error.kind(), ModelErrorKind::InvalidRequest);
    for file in [
        FilePart::image("image/svg+xml", FileSource::Bytes(Vec::new().into())),
        FilePart::image("image/png", FileSource::Text("not binary".into())),
        FilePart::document("application/pdf", FileSource::Text("not a pdf".into())),
        FilePart::document("text/plain", FileSource::Bytes(b"text".to_vec().into())),
        FilePart::document(
            "text/plain",
            FileSource::Url("https://example.test/file.txt".parse().unwrap()),
        ),
        FilePart::image(
            "image/png",
            FileSource::Url("http://example.test/image.png".parse().unwrap()),
        ),
    ] {
        assert!(known.validate_request(&request_with(file)).is_err());
    }
    assert!(
        known
            .validate_request(&request_with(FilePart::document(
                "text/plain",
                FileSource::Text("plain text".into()),
            )))
            .is_ok()
    );
    assert!(
        without_media
            .validate_request(&request_with(FilePart::image(
                "image/png",
                FileSource::Bytes(Vec::new().into()),
            )))
            .is_err()
    );
}

#[test]
fn minimax_media_mime_source_and_file_boundaries_are_exact() {
    let model = MiniMax::builder().build().unwrap().model("MiniMax-M3");
    assert!(
        model
            .validate_request(&request_with(FilePart::image(
                "image/jpeg",
                FileSource::Bytes(vec![0; 10 * 1024 * 1024].into()),
            )))
            .is_ok()
    );
    assert!(
        model
            .validate_request(&request_with(FilePart::image(
                "image/jpeg",
                FileSource::Bytes(vec![0; 10 * 1024 * 1024 + 1].into()),
            )))
            .is_err()
    );
    assert!(
        model
            .validate_request(&request_with(FilePart::video(
                "video/mp4",
                FileSource::Bytes(vec![0; 50 * 1024 * 1024].into()),
            )))
            .is_ok()
    );
    assert!(
        model
            .validate_request(&request_with(FilePart::video(
                "video/mp4",
                FileSource::Bytes(vec![0; 50 * 1024 * 1024 + 1].into()),
            )))
            .is_err()
    );
    for file in [
        FilePart::image("image/png", FileSource::Text("not binary".into())),
        FilePart::video("video/quicktime", FileSource::Bytes(Vec::new().into())),
        FilePart::video(
            "video/mov",
            FileSource::Url("https://example.test/video.mov".parse().unwrap()),
        ),
        FilePart::image(
            "image/png",
            FileSource::Url("http://example.test/image.png".parse().unwrap()),
        ),
        FilePart::image(
            "image/png",
            FileSource::ProviderReference {
                provider: ProviderId::new("minimax"),
                id: "file".into(),
            },
        ),
    ] {
        assert!(model.validate_request(&request_with(file)).is_err());
    }
    assert!(
        model
            .validate_request(&request_with(
                FilePart::image("image/png", FileSource::Bytes(Vec::new().into()))
                    .with_minimax_media_options(MiniMaxMediaOptions {
                        fps: Some(1.0),
                        ..Default::default()
                    }),
            ))
            .is_err()
    );
}

#[tokio::test]
async fn oversized_media_and_serialized_requests_fail_without_dispatch() {
    let direct_server = MockServer::start().await;
    let direct = Anthropic::builder()
        .base_url(direct_server.uri())
        .build()
        .unwrap()
        .model("claude-sonnet-4-5");
    let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
        InputPart::Text(TextPart::new("x".repeat(32 * 1024 * 1024))),
    ]))]);
    let error = direct
        .stream(request, AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::InvalidRequest);
    assert!(direct_server.received_requests().await.unwrap().is_empty());

    let minimax_server = MockServer::start().await;
    let minimax = MiniMax::builder()
        .base_url(minimax_server.uri())
        .build()
        .unwrap()
        .model("MiniMax-M3");
    let request = request_with(FilePart::video(
        "video/mp4",
        FileSource::Bytes(vec![0; 50 * 1024 * 1024].into()),
    ));
    let error = minimax
        .stream(request, AbortSignal::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, ModelErrorKind::InvalidRequest);
    assert!(minimax_server.received_requests().await.unwrap().is_empty());
}
mod common;

use common::{Anthropic, MiniMax};
