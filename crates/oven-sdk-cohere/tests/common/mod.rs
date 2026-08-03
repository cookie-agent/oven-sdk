#![allow(dead_code)]

use oven_sdk::{
    ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    MediaCapabilities, MediaInputSupport, MediaSourceSupport, Modalities, Modality,
    ModelCapabilities, ModelConfig, ModelDeclaration, ModelId, ModelLimits, ProviderConfig,
    ProviderId, ReplayCapability, ReplayDeclaration, ReplayPolicy, SecretString,
};
use oven_sdk_cohere::{CohereAuth, CohereModel, CohereSettings};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

pub fn capabilities() -> ModelCapabilities {
    let mut media = MediaCapabilities::default();
    media.input.insert(
        Modality::image(),
        MediaInputSupport::new(
            ["image/png", "image/jpeg", "image/webp", "image/gif"].map(str::to_owned),
            MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
        )
        .unwrap(),
    );
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
        modalities: Modalities::new([Modality::text(), Modality::image()], [Modality::text()]),
        media,
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        replay: ReplayDeclaration {
            policy: ReplayPolicy::IfValid,
            capability: ReplayCapability::Optional,
            reasoning: true,
        },
    }
}

pub fn model(server: &MockServer, model_id: &str) -> CohereModel {
    model_with(server, model_id, capabilities(), CohereSettings::default())
}

pub fn model_with(
    server: &MockServer,
    model_id: &str,
    capabilities: ModelCapabilities,
    settings: CohereSettings,
) -> CohereModel {
    let provider = ProviderConfig::new(
        ProviderId::new("cohere"),
        ApiEndpoint::parse(format!("{}/v2/chat", server.uri())).unwrap(),
        CohereAuth::bearer(SecretString::new("secret")),
        HeaderConfig::empty(),
    )
    .unwrap();
    let declaration = ModelDeclaration::new(ModelId::new(model_id), capabilities).unwrap();
    CohereModel::new(ModelConfig::new(provider, declaration, settings)).unwrap()
}

pub async fn mount(server: &MockServer, body: String) {
    Mock::given(method("POST"))
        .and(path("/v2/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req_cohere_1")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(server)
        .await;
}

pub fn text_stream(text: &str) -> String {
    let events = vec![
        (
            "message-start",
            serde_json::json!({"type":"message-start","id":"gen_1","delta":{"message":{"role":"assistant"}}}),
        ),
        (
            "content-start",
            serde_json::json!({"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}),
        ),
        (
            "content-delta",
            serde_json::json!({"type":"content-delta","index":0,"delta":{"message":{"content":{"text":text}}}}),
        ),
        (
            "citation-start",
            serde_json::json!({"type":"citation-start","index":0,"delta":{"message":{"citations":{"start":0,"end":2,"text":"ok","type":"TEXT_CONTENT","sources":[{"type":"document","id":"doc_1","document":{"title":"Doc","url":"https://example.com","text":"source"}}]}}}}),
        ),
        (
            "citation-end",
            serde_json::json!({"type":"citation-end","index":0}),
        ),
        (
            "content-end",
            serde_json::json!({"type":"content-end","index":0}),
        ),
        (
            "message-end",
            serde_json::json!({"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":2,"output_tokens":3},"cached_tokens":1}}}),
        ),
    ];
    let mut output = String::new();
    for (name, value) in events {
        output.push_str(&format!("event: {name}\ndata: {value}\n\n"));
    }
    output
}

pub fn tool_stream() -> String {
    concat!(
        "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"gen_tools\",\"delta\":{\"message\":{\"role\":\"assistant\"}}}\n\n",
        "event: tool-plan-delta\ndata: {\"type\":\"tool-plan-delta\",\"delta\":{\"message\":{\"tool_plan\":\"Call both\"}}}\n\n",
        "event: tool-call-start\ndata: {\"type\":\"tool-call-start\",\"index\":0,\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"\"}}}}}\n\n",
        "event: tool-call-delta\ndata: {\"type\":\"tool-call-delta\",\"index\":0,\"delta\":{\"message\":{\"tool_calls\":{\"function\":{\"arguments\":\"{\\\"q\\\":\\\"a\\\"}\"}}}}}\n\n",
        "event: tool-call-end\ndata: {\"type\":\"tool-call-end\",\"index\":0}\n\n",
        "event: tool-call-start\ndata: {\"type\":\"tool-call-start\",\"index\":1,\"delta\":{\"message\":{\"tool_calls\":{\"id\":\"call_2\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}}}}\n\n",
        "event: tool-call-end\ndata: {\"type\":\"tool-call-end\",\"index\":1}\n\n",
        "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"TOOL_CALL\",\"usage\":{\"tokens\":{\"input_tokens\":4,\"output_tokens\":5}}}}\n\n"
    )
    .into()
}
