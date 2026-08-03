#![allow(dead_code)]

use oven_sdk::{
    ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    MediaCapabilities, MediaInputSupport, MediaSourceSupport, Modalities, Modality,
    ModelCapabilities, ModelConfig, ModelDeclaration, ModelId, ModelLimits, ProviderConfig,
    ProviderId, ReplayCapability, ReplayDeclaration, ReplayPolicy, SecretString,
};
use oven_sdk_open_responses::{
    HuggingFaceTransport, OpenResponsesAuth, OpenResponsesConfig, OpenResponsesModel,
    OpenResponsesSettings, OpenResponsesTimeouts, OpenResponsesTransport,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

pub fn capabilities(include_pdf: bool) -> ModelCapabilities {
    let mut media = MediaCapabilities::default();
    media.input.insert(
        Modality::image(),
        MediaInputSupport::new(
            ["image/png", "image/jpeg", "image/webp", "image/gif"].map(str::to_owned),
            MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
        )
        .unwrap(),
    );
    let mut input = vec![Modality::text(), Modality::image()];
    if include_pdf {
        input.push(Modality::pdf());
        media.input.insert(
            Modality::pdf(),
            MediaInputSupport::new(
                ["application/pdf".into()],
                MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
            )
            .unwrap(),
        );
    }
    ModelCapabilities {
        features: Capability::TOOL_CALLING
            | Capability::PARALLEL_TOOLS
            | Capability::TOOL_INPUT_DELTAS
            | Capability::REASONING
            | Capability::STRUCTURED_OUTPUT
            | Capability::TEMPERATURE
            | Capability::TOP_P
            | Capability::MAX_OUTPUT_TOKENS
            | Capability::USAGE
            | Capability::SOURCES,
        limits: ModelLimits::new(Some(128_000), None, Some(32_000)),
        modalities: Modalities::new(input, [Modality::text()]),
        media,
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        replay: ReplayDeclaration {
            policy: ReplayPolicy::IfValid,
            capability: ReplayCapability::Optional,
            reasoning: false,
        },
    }
}

pub fn generic_model(server: &MockServer, model_id: &str) -> OpenResponsesModel {
    OpenResponsesModel::new(generic_config(server, model_id)).unwrap()
}

pub fn generic_config(server: &MockServer, model_id: &str) -> OpenResponsesConfig {
    config(
        server,
        model_id,
        "generic.open-responses",
        OpenResponsesTransport::Generic {
            profile: "conformance-2026-04-24".into(),
        },
        true,
    )
}

pub fn hugging_face_model(server: &MockServer, model_id: &str) -> OpenResponsesModel {
    OpenResponsesModel::new(hugging_face_config(server, model_id)).unwrap()
}

pub fn hugging_face_config(server: &MockServer, model_id: &str) -> OpenResponsesConfig {
    config(
        server,
        model_id,
        "huggingface",
        OpenResponsesTransport::HuggingFace(HuggingFaceTransport {
            routing: "exact-model-id-suffix".into(),
        }),
        false,
    )
}

fn config(
    server: &MockServer,
    model_id: &str,
    provider_id: &str,
    transport: OpenResponsesTransport,
    include_pdf: bool,
) -> OpenResponsesConfig {
    let provider = ProviderConfig::new(
        ProviderId::new(provider_id),
        ApiEndpoint::parse(format!("{}/v1/responses", server.uri())).unwrap(),
        OpenResponsesAuth::bearer(SecretString::new("secret")),
        HeaderConfig::empty(),
    )
    .unwrap();
    let declaration =
        ModelDeclaration::new(ModelId::new(model_id), capabilities(include_pdf)).unwrap();
    ModelConfig::new(
        provider,
        declaration,
        OpenResponsesSettings {
            transport,
            timeouts: OpenResponsesTimeouts::default(),
            strict_json_schema: true,
            strict_tools: true,
            parallel_tool_calls: true,
            store: false,
            include: Vec::new(),
            reasoning_summary: None,
        },
    )
}

pub async fn mount(server: &MockServer, body: String) {
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req_open_responses_1")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(server)
        .await;
}

pub fn text_stream(text: &str) -> String {
    let item = serde_json::json!({
        "type":"message","id":"msg_1","status":"completed","role":"assistant",
        "content":[{"type":"output_text","text":text,"annotations":[{
            "type":"url_citation","start_index":0,"end_index":2,
            "url":"https://example.com/source","title":"Source"
        }]}]
    });
    let events = vec![
        (
            "response.created",
            serde_json::json!({"type":"response.created","sequence_number":0,"response":{"id":"resp_1","status":"in_progress","model":"opaque"}}),
        ),
        (
            "response.in_progress",
            serde_json::json!({"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_1","status":"in_progress","model":"opaque"}}),
        ),
        (
            "response.output_item.added",
            serde_json::json!({"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"message","id":"msg_1","status":"in_progress","role":"assistant","content":[]}}),
        ),
        (
            "response.content_part.added",
            serde_json::json!({"type":"response.content_part.added","sequence_number":3,"item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}),
        ),
        (
            "response.output_text.delta",
            serde_json::json!({"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_1","output_index":0,"content_index":0,"delta":text}),
        ),
        (
            "response.output_text.done",
            serde_json::json!({"type":"response.output_text.done","sequence_number":5,"item_id":"msg_1","output_index":0,"content_index":0,"text":text}),
        ),
        (
            "response.content_part.done",
            serde_json::json!({"type":"response.content_part.done","sequence_number":6,"item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":text,"annotations":[{"type":"url_citation","start_index":0,"end_index":2,"url":"https://example.com/source","title":"Source"}]}}),
        ),
        (
            "response.output_item.done",
            serde_json::json!({"type":"response.output_item.done","sequence_number":7,"output_index":0,"item":item.clone()}),
        ),
        (
            "response.completed",
            serde_json::json!({"type":"response.completed","sequence_number":8,"response":{"id":"resp_1","status":"completed","model":"opaque","output":[item],"usage":{"input_tokens":3,"input_tokens_details":{"cached_tokens":1},"output_tokens":5,"output_tokens_details":{"reasoning_tokens":2}}}}),
        ),
    ];
    let mut output = String::new();
    for (name, value) in events {
        output.push_str(&format!("event: {name}\ndata: {value}\n\n"));
    }
    output.push_str("data: [DONE]\n\n");
    output
}

pub fn bad_sequence_stream() -> String {
    concat!(
        "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"model\":\"opaque\"}}\n\n",
        "event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"sequence_number\":2,\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"model\":\"opaque\"}}\n\n"
    )
    .into()
}
