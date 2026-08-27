mod support;

use oven_sdk::{AbortSignal, CompactionRequest, LanguageModel, Request, StreamPart};
use oven_sdk_conformance::{
    ToolResultFileKind, ToolResultFilePolicy, assert_capability_honesty,
    assert_compaction_unsupported_before_io, assert_complete_drain, assert_replay_round_trip,
    assert_stream_lifecycle, assert_tool_result_file_policy,
};
use oven_sdk_google_vertex::GoogleVertexResource;
use serde_json::json;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn resource() -> GoogleVertexResource {
    GoogleVertexResource::PublisherModel {
        publisher: "google".into(),
        model: "resource-model-v1".into(),
    }
}

#[tokio::test]
async fn explicit_model_passes_lifecycle_complete_capability_and_replay_conformance() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\n",
                    json!({
                        "candidates":[{"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}],
                        "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}
                    })
                )),
        )
        .mount(&server)
        .await;
    let model = support::full_model(&server.uri(), "future-conformance", resource(), true);
    let report = assert_stream_lifecycle(&model, Request::new(Vec::new()))
        .await
        .unwrap();
    assert!(matches!(
        report.stream.parts.last(),
        Some(StreamPart::Finish { .. })
    ));
    assert_complete_drain(&model, Request::new(Vec::new()))
        .await
        .unwrap();
    assert_capability_honesty(&model).unwrap();
    for kind in [ToolResultFileKind::Image, ToolResultFileKind::Pdf] {
        assert_tool_result_file_policy(&model, kind, ToolResultFilePolicy::Reject).unwrap();
    }
    assert_compaction_unsupported_before_io(
        &model,
        CompactionRequest::new(Request::new(Vec::new())),
    )
    .await
    .unwrap();

    let turn = model
        .complete(Request::new(Vec::new()), AbortSignal::default())
        .await
        .unwrap()
        .turn;
    assert_replay_round_trip(
        &model,
        model.native_context_scope(),
        Request::new(vec![oven_sdk::HistoryTurn::assistant(turn)]),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn pre_dispatch_abort_is_structured() {
    let model = support::full_model("https://example.com", "future-abort", resource(), false);
    let (signal, registration) = AbortSignal::new();
    registration.abort();
    let error = model
        .stream(Request::new(Vec::new()), signal)
        .await
        .unwrap_err();
    assert_eq!(error.kind, oven_sdk::ModelErrorKind::Abort);
}
