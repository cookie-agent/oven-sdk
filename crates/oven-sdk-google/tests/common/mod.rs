// Each integration test is a separate crate and uses a different subset of these shared helpers.
#![allow(dead_code)]

use std::collections::BTreeMap;

use oven_sdk::{
    ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    MediaCapabilities, MediaInputSupport, MediaSourceSupport, Modalities, Modality,
    ModelCapabilities, ModelConfig, ModelDeclaration, ModelId, ModelLimits, ProviderConfig,
    ProviderId, ReplayCapability, ReplayDeclaration, ReplayPolicy,
};
use oven_sdk_google::{
    GOOGLE_PROVIDER_ID, GoogleApiKeyAuth, GoogleGenerateContentSettings, GoogleModel,
    GoogleThinkingSettings, GoogleTimeouts, GoogleToolSettings,
};

pub fn full_capabilities() -> ModelCapabilities {
    let inline_binary = MediaSourceSupport::INLINE_BYTES
        | MediaSourceSupport::URL
        | MediaSourceSupport::PROVIDER_REFERENCE;
    let inline_document = inline_binary | MediaSourceSupport::INLINE_TEXT;
    let mut media = BTreeMap::new();
    media.insert(
        Modality::image(),
        MediaInputSupport::new(
            strings([
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/heic",
                "image/heif",
            ]),
            inline_binary,
        )
        .unwrap(),
    );
    media.insert(
        Modality::audio(),
        MediaInputSupport::new(
            strings([
                "audio/wav",
                "audio/mp3",
                "audio/aiff",
                "audio/aac",
                "audio/ogg",
                "audio/flac",
            ]),
            inline_binary,
        )
        .unwrap(),
    );
    media.insert(
        Modality::video(),
        MediaInputSupport::new(
            strings([
                "video/mp4",
                "video/mpeg",
                "video/mov",
                "video/avi",
                "video/x-flv",
                "video/mpg",
                "video/webm",
                "video/wmv",
                "video/3gpp",
            ]),
            inline_binary,
        )
        .unwrap(),
    );
    media.insert(
        Modality::text(),
        MediaInputSupport::new(
            strings([
                "text/plain",
                "text/markdown",
                "text/html",
                "text/css",
                "text/xml",
            ]),
            inline_document,
        )
        .unwrap(),
    );
    media.insert(
        Modality::pdf(),
        MediaInputSupport::new(strings(["application/pdf"]), inline_document).unwrap(),
    );
    let document = Modality::new("document").unwrap();
    media.insert(
        document.clone(),
        MediaInputSupport::new(
            strings(["application/json", "application/xml"]),
            inline_document,
        )
        .unwrap(),
    );
    ModelCapabilities {
        features: Capability::TOOL_CALLING
            | Capability::PARALLEL_TOOLS
            | Capability::REASONING
            | Capability::STRUCTURED_OUTPUT
            | Capability::TEMPERATURE
            | Capability::TOP_P
            | Capability::MAX_OUTPUT_TOKENS
            | Capability::PROMPT_CACHING
            | Capability::USAGE
            | Capability::PROVIDER_TOOLS
            | Capability::SOURCES,
        limits: ModelLimits::new(Some(1_048_576), None, Some(65_536)),
        modalities: Modalities::new(
            [
                Modality::text(),
                Modality::image(),
                Modality::audio(),
                Modality::video(),
                Modality::pdf(),
                document,
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

pub fn model(base_url: impl AsRef<str>, model_id: &str) -> GoogleModel {
    model_with_key(base_url, model_id, "secret")
}

pub fn model_with_key(base_url: impl AsRef<str>, model_id: &str, api_key: &str) -> GoogleModel {
    GoogleModel::new(config_with(
        base_url,
        model_id,
        &format!("models/{model_id}"),
        api_key,
        full_capabilities(),
        level_thinking(),
        default_tools(),
    ))
    .unwrap()
}

pub fn budget_model(base_url: impl AsRef<str>, model_id: &str) -> GoogleModel {
    model_with(
        base_url,
        model_id,
        &format!("models/{model_id}"),
        full_capabilities(),
        budget_thinking(),
        default_tools(),
    )
}

pub fn model_with(
    base_url: impl AsRef<str>,
    model_id: &str,
    model_resource: &str,
    capabilities: ModelCapabilities,
    thinking: GoogleThinkingSettings,
    tools: GoogleToolSettings,
) -> GoogleModel {
    GoogleModel::new(config_with(
        base_url,
        model_id,
        model_resource,
        "secret",
        capabilities,
        thinking,
        tools,
    ))
    .unwrap()
}

pub fn config_with(
    base_url: impl AsRef<str>,
    model_id: &str,
    model_resource: &str,
    api_key: &str,
    capabilities: ModelCapabilities,
    thinking: GoogleThinkingSettings,
    tools: GoogleToolSettings,
) -> ModelConfig<GoogleApiKeyAuth, GoogleGenerateContentSettings> {
    let provider_id = ProviderId::new(GOOGLE_PROVIDER_ID);
    let model_id = ModelId::new(model_id);
    let provider = ProviderConfig::new(
        provider_id,
        ApiEndpoint::parse(base_url).unwrap(),
        GoogleApiKeyAuth::new(api_key),
        HeaderConfig::empty(),
    )
    .unwrap();
    let declaration = ModelDeclaration::new(model_id, capabilities).unwrap();
    ModelConfig::new(
        provider,
        declaration,
        GoogleGenerateContentSettings {
            model_resource: model_resource.into(),
            timeouts: GoogleTimeouts::default(),
            thinking,
            tools,
        },
    )
}

pub fn level_thinking() -> GoogleThinkingSettings {
    GoogleThinkingSettings::Level {
        effort_levels: BTreeMap::from([
            ("low".into(), "LOW".into()),
            ("medium".into(), "MEDIUM".into()),
            ("high".into(), "HIGH".into()),
            ("future-effort".into(), "future-level".into()),
        ]),
    }
}

pub fn budget_thinking() -> GoogleThinkingSettings {
    GoogleThinkingSettings::Budget {
        effort_budgets: BTreeMap::from([
            ("low".into(), 128),
            ("medium".into(), 512),
            ("high".into(), 1_024),
            ("future-effort".into(), 2_048),
        ]),
    }
}

pub fn default_tools() -> GoogleToolSettings {
    GoogleToolSettings {
        strict_functions: true,
        mixed_client_and_provider_tools: true,
        current_turn_signature_sentinel: true,
    }
}

fn strings<const N: usize>(values: [&str; N]) -> Vec<String> {
    values.into_iter().map(str::to_owned).collect()
}
