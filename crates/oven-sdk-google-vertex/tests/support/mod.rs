// Each integration-test crate compiles this shared helper independently and uses a different subset.
#![allow(dead_code)]

use std::collections::BTreeMap;

use oven_sdk::{
    ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    MediaCapabilities, MediaInputSupport, MediaSourceSupport, Modalities, Modality,
    ModelCapabilities, ModelConfig, ModelDeclaration, ModelId, ModelLimits, ProviderConfig,
    ProviderId, ReplayCapability, ReplayDeclaration, ReplayPolicy, SecretString,
};
use oven_sdk_google_vertex::{
    GoogleVertexMediaSettings, GoogleVertexModel, GoogleVertexResource, GoogleVertexSettings,
    GoogleVertexThinkingMode, GoogleVertexTimeouts, GoogleVertexToolSettings, VertexAuth,
    google_vertex_native_context_scope,
};

pub fn full_model(
    api_origin: &str,
    model_id: &str,
    resource: GoogleVertexResource,
    partial_args: bool,
) -> GoogleVertexModel {
    GoogleVertexModel::new(full_config(api_origin, model_id, resource, partial_args)).unwrap()
}

pub fn full_config(
    api_origin: &str,
    model_id: &str,
    resource: GoogleVertexResource,
    partial_args: bool,
) -> ModelConfig<VertexAuth, GoogleVertexSettings> {
    let provider_id = ProviderId::new("google.vertex");
    let model_id = ModelId::new(model_id);
    let api = ApiEndpoint::parse(format!("{}/v1beta1", api_origin.trim_end_matches('/'))).unwrap();
    let native_context_scope = google_vertex_native_context_scope(
        provider_id.clone(),
        model_id.clone(),
        &api,
        "project",
        "global",
        &resource,
    )
    .unwrap();
    let provider = ProviderConfig::new(
        provider_id,
        api,
        VertexAuth::AccessToken(SecretString::new("token")),
        HeaderConfig::empty(),
    )
    .unwrap();
    let model = ModelDeclaration::new(model_id, full_capabilities(partial_args)).unwrap();
    let settings = GoogleVertexSettings {
        project: "project".into(),
        location: "global".into(),
        resource,
        thinking: GoogleVertexThinkingMode::Level,
        tools: GoogleVertexToolSettings {
            provider_tools: true,
            mixed_client_and_provider_tools: true,
            strict_functions: true,
        },
        stream_function_call_arguments: partial_args,
        media: GoogleVertexMediaSettings {
            max_images: 3_000,
            max_https_images: 10,
            max_documents: 3_000,
            max_audio: 1,
            max_videos: 10,
            max_https_videos: 1,
            max_inline_image_bytes: 7 * 1024 * 1024,
            max_inline_pdf_bytes: 50 * 1024 * 1024,
            max_inline_text_bytes: 7 * 1024 * 1024,
            url_schemes: vec!["https".into(), "gs".into()],
        },
        native_context_scope,
        client: None,
        timeouts: GoogleVertexTimeouts::default(),
    };
    ModelConfig::new(provider, model, settings)
}

pub fn full_capabilities(partial_args: bool) -> ModelCapabilities {
    let mut features = Capability::TOOL_CALLING
        | Capability::PARALLEL_TOOLS
        | Capability::REASONING
        | Capability::STRUCTURED_OUTPUT
        | Capability::TEMPERATURE
        | Capability::TOP_P
        | Capability::MAX_OUTPUT_TOKENS
        | Capability::PROMPT_CACHING
        | Capability::USAGE
        | Capability::PROVIDER_TOOLS
        | Capability::SOURCES;
    if partial_args {
        features |= Capability::TOOL_INPUT_DELTAS;
    }
    let mut media = BTreeMap::new();
    media.insert(
        Modality::image(),
        MediaInputSupport::new(
            [
                "image/png".into(),
                "image/jpeg".into(),
                "image/webp".into(),
                "image/heic".into(),
                "image/heif".into(),
            ],
            MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
        )
        .unwrap(),
    );
    media.insert(
        Modality::pdf(),
        MediaInputSupport::new(
            ["application/pdf".into()],
            MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
        )
        .unwrap(),
    );
    media.insert(
        Modality::text(),
        MediaInputSupport::new(
            ["text/plain".into()],
            MediaSourceSupport::INLINE_BYTES
                | MediaSourceSupport::INLINE_TEXT
                | MediaSourceSupport::URL,
        )
        .unwrap(),
    );
    media.insert(
        Modality::audio(),
        MediaInputSupport::new(
            [
                "audio/x-aac".into(),
                "audio/flac".into(),
                "audio/mp3".into(),
                "audio/m4a".into(),
                "audio/mpeg".into(),
                "audio/mpga".into(),
                "audio/mp4".into(),
                "audio/ogg".into(),
                "audio/pcm".into(),
                "audio/wav".into(),
                "audio/webm".into(),
            ],
            MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
        )
        .unwrap(),
    );
    media.insert(
        Modality::video(),
        MediaInputSupport::new(
            [
                "video/x-flv".into(),
                "video/quicktime".into(),
                "video/mpeg".into(),
                "video/mpegs".into(),
                "video/mpg".into(),
                "video/mp4".into(),
                "video/webm".into(),
                "video/wmv".into(),
                "video/3gpp".into(),
            ],
            MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
        )
        .unwrap(),
    );
    ModelCapabilities {
        features,
        limits: ModelLimits::new(Some(1_048_576), Some(1_048_576), Some(65_536)),
        modalities: Modalities::new(
            [
                Modality::text(),
                Modality::image(),
                Modality::pdf(),
                Modality::audio(),
                Modality::video(),
            ],
            [Modality::text()],
        ),
        media: MediaCapabilities { input: media },
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        replay: ReplayDeclaration {
            policy: ReplayPolicy::IfValid,
            capability: ReplayCapability::Required,
            reasoning: true,
        },
    }
}
