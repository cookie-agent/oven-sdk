// Each integration test compiles this shared support module independently and
// intentionally uses only the fixtures relevant to that test binary.
#![allow(dead_code)]

use serde_json::Value;

use oven_sdk::{
    ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    MediaCapabilities, MediaInputSupport, MediaSourceSupport, Modalities, Modality,
    ModelCapabilities, ModelConfig, ModelDeclaration, ModelId, ModelLimits, ProviderConfig,
    ProviderId, ReplayCapability, ReplayDeclaration, ReplayPolicy,
};
use oven_sdk_bedrock::{
    AwsCredentials, BedrockAuth, BedrockConverseSettings, BedrockEventStreamLimits, BedrockModel,
    BedrockReasoningWireFormat, BedrockStructuredOutput,
};

#[derive(Clone, Copy)]
pub enum FixtureKind {
    Text,
    MediaTools,
    SignedReasoning,
    UnsignedReasoning,
}

pub fn credentials() -> AwsCredentials {
    AwsCredentials {
        access_key_id: "AKIDEXAMPLE".into(),
        secret_access_key: "secret".into(),
        session_token: Some("session".into()),
    }
}

pub fn model(endpoint: &str, model_id: &str, kind: FixtureKind) -> BedrockModel {
    model_with_auth(endpoint, model_id, kind, BedrockAuth::Static(credentials()))
}

pub fn model_with_auth(
    endpoint: &str,
    model_id: &str,
    kind: FixtureKind,
    auth: BedrockAuth,
) -> BedrockModel {
    BedrockModel::new(config(endpoint, model_id, kind, auth)).expect("test model configuration")
}

pub fn config(
    endpoint: &str,
    model_id: &str,
    kind: FixtureKind,
    auth: BedrockAuth,
) -> ModelConfig<BedrockAuth, BedrockConverseSettings> {
    let (capabilities, reasoning, signed, structured) = declaration(kind);
    let provider = ProviderConfig::new(
        ProviderId::new(oven_sdk_bedrock::BEDROCK_PROVIDER_ID),
        ApiEndpoint::parse(endpoint).expect("endpoint"),
        auth,
        HeaderConfig::empty(),
    )
    .expect("provider");
    let model = ModelDeclaration::new(ModelId::new(model_id), capabilities).expect("declaration");
    let settings = BedrockConverseSettings::new(
        "us-east-1",
        reasoning,
        signed,
        structured,
        BedrockEventStreamLimits::new(16 * 1024 * 1024),
    );
    ModelConfig::new(provider, model, settings)
}

fn declaration(
    kind: FixtureKind,
) -> (
    ModelCapabilities,
    BedrockReasoningWireFormat,
    bool,
    BedrockStructuredOutput,
) {
    let replay = match kind {
        FixtureKind::SignedReasoning => ReplayDeclaration {
            policy: ReplayPolicy::IfValid,
            capability: ReplayCapability::Required,
            reasoning: true,
        },
        FixtureKind::UnsignedReasoning => ReplayDeclaration {
            policy: ReplayPolicy::IfValid,
            capability: ReplayCapability::Optional,
            reasoning: true,
        },
        FixtureKind::Text | FixtureKind::MediaTools => ReplayDeclaration {
            policy: ReplayPolicy::IfValid,
            capability: ReplayCapability::Optional,
            reasoning: false,
        },
    };
    let mut capabilities = ModelCapabilities {
        features: Capability::USAGE
            | Capability::TEMPERATURE
            | Capability::TOP_P
            | Capability::MAX_OUTPUT_TOKENS,
        limits: ModelLimits::new(Some(1_000_000), None, Some(64_000)),
        modalities: Modalities::new([Modality::text()], [Modality::text()]),
        media: MediaCapabilities::default(),
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        replay,
    };
    let (reasoning, signed, structured) = match kind {
        FixtureKind::Text => (
            BedrockReasoningWireFormat::Unsupported,
            false,
            BedrockStructuredOutput::Unsupported,
        ),
        FixtureKind::MediaTools => {
            capabilities.features |= Capability::TOOL_CALLING
                | Capability::PARALLEL_TOOLS
                | Capability::TOOL_INPUT_DELTAS;
            add_media(&mut capabilities);
            (
                BedrockReasoningWireFormat::Unsupported,
                false,
                BedrockStructuredOutput::Unsupported,
            )
        }
        FixtureKind::SignedReasoning => {
            capabilities.features |= Capability::TOOL_CALLING
                | Capability::PARALLEL_TOOLS
                | Capability::TOOL_INPUT_DELTAS
                | Capability::REASONING
                | Capability::STRUCTURED_OUTPUT
                | Capability::SOURCES;
            add_media(&mut capabilities);
            (
                BedrockReasoningWireFormat::AnthropicThinking,
                true,
                BedrockStructuredOutput::JsonSchema,
            )
        }
        FixtureKind::UnsignedReasoning => {
            capabilities.features |= Capability::TOOL_CALLING
                | Capability::PARALLEL_TOOLS
                | Capability::TOOL_INPUT_DELTAS
                | Capability::REASONING;
            (
                BedrockReasoningWireFormat::OpenAiReasoningEffort,
                false,
                BedrockStructuredOutput::Unsupported,
            )
        }
    };
    (capabilities, reasoning, signed, structured)
}

fn add_media(capabilities: &mut ModelCapabilities) {
    capabilities
        .modalities
        .input
        .extend([Modality::image(), Modality::video()]);
    capabilities.media.input.insert(
        Modality::image(),
        MediaInputSupport::new(
            ["image/png", "image/jpeg", "image/gif", "image/webp"].map(str::to_owned),
            MediaSourceSupport::INLINE_BYTES,
        )
        .expect("image media"),
    );
    capabilities.media.input.insert(
        Modality::video(),
        MediaInputSupport::new(
            [
                "video/x-matroska",
                "video/quicktime",
                "video/mp4",
                "video/webm",
                "video/x-flv",
                "video/mpeg",
                "video/mpg",
                "video/wmv",
                "video/3gpp",
            ]
            .map(str::to_owned),
            MediaSourceSupport::INLINE_BYTES,
        )
        .expect("video media"),
    );
}

pub fn frame(event_type: &str, payload: Value) -> Vec<u8> {
    frame_with_headers(
        &[
            (":message-type", "event"),
            (":event-type", event_type),
            (":content-type", "application/json"),
        ],
        serde_json::to_vec(&payload).unwrap(),
    )
}

pub fn frame_with_headers(headers: &[(&str, &str)], payload: Vec<u8>) -> Vec<u8> {
    let mut encoded_headers = Vec::new();
    for (name, value) in headers {
        encoded_headers.push(name.len() as u8);
        encoded_headers.extend_from_slice(name.as_bytes());
        encoded_headers.push(7);
        encoded_headers.extend_from_slice(&(value.len() as u16).to_be_bytes());
        encoded_headers.extend_from_slice(value.as_bytes());
    }
    let total = 16 + encoded_headers.len() + payload.len();
    let mut frame = Vec::new();
    frame.extend_from_slice(&(total as u32).to_be_bytes());
    frame.extend_from_slice(&(encoded_headers.len() as u32).to_be_bytes());
    frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());
    frame.extend_from_slice(&encoded_headers);
    frame.extend_from_slice(&payload);
    frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());
    frame
}

pub fn text_stream(text: &str) -> Vec<u8> {
    let mut body = Vec::new();
    for (event, payload) in [
        ("messageStart", serde_json::json!({"role":"assistant"})),
        (
            "contentBlockDelta",
            serde_json::json!({"contentBlockIndex":0,"delta":{"text":text}}),
        ),
        (
            "contentBlockStop",
            serde_json::json!({"contentBlockIndex":0}),
        ),
        ("messageStop", serde_json::json!({"stopReason":"end_turn"})),
        (
            "metadata",
            serde_json::json!({"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2},"metrics":{"latencyMs":1}}),
        ),
    ] {
        body.extend(frame(event, payload));
    }
    body
}
