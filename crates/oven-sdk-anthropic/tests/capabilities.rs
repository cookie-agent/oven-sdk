mod common;

use common::{
    Anthropic, anthropic_capabilities, anthropic_protocol, minimax_capabilities,
    try_anthropic_model, try_aws_model, try_minimax_model,
};
use oven_sdk::{
    CancellationCapability, Capability, CompactionCapability, LanguageModel, MediaInputSupport,
    MediaSourceSupport, Modality, ReplayCapability, ReplayPolicy, Request,
};
use oven_sdk_anthropic::{AnthropicRequestExt, AnthropicRequestOptions, AnthropicThinking};

#[test]
fn explicit_declaration_controls_limits_features_media_and_replay() {
    let model = Anthropic::builder()
        .capabilities(anthropic_capabilities(ReplayPolicy::IfValid))
        .protocol(anthropic_protocol())
        .build()
        .unwrap()
        .model("caller-deployment-2026-08");
    let capabilities = model.capabilities();

    assert_eq!(capabilities.limits.context, Some(200_000));
    assert_eq!(capabilities.limits.output, Some(64_000));
    assert!(capabilities.features.contains(Capability::TOOL_CALLING));
    assert!(capabilities.features.contains(Capability::REASONING));
    assert!(
        capabilities
            .media
            .input
            .contains_key(&oven_sdk::Modality::image())
    );
    assert_eq!(capabilities.replay.policy, ReplayPolicy::IfValid);
    assert_eq!(capabilities.replay.capability, ReplayCapability::Required);
    assert!(capabilities.replay.reasoning);
}

#[test]
fn model_names_do_not_select_behavior_and_future_ids_are_accepted() {
    let provider = Anthropic::builder().build().unwrap();
    let current = provider.model("claude-present-id");
    let future = provider.model("future-provider-id-2099");

    assert_eq!(current.capabilities(), future.capabilities());
    let adaptive = Request::new(Vec::new()).with_anthropic_options(AnthropicRequestOptions {
        thinking: Some(AnthropicThinking::Adaptive { display: None }),
        ..Default::default()
    });
    assert_eq!(
        current.validate_request(&adaptive).is_ok(),
        future.validate_request(&adaptive).is_ok()
    );
    assert_eq!(
        current.descriptor().identity.model_id.as_str(),
        "claude-present-id"
    );
    assert_eq!(
        future.descriptor().identity.model_id.as_str(),
        "future-provider-id-2099"
    );
}

#[test]
fn replay_policy_is_declared_in_capabilities_instead_of_inferred() {
    let never = Anthropic::builder()
        .replay_policy(ReplayPolicy::Never)
        .build()
        .unwrap()
        .model("any-model-id");
    assert_eq!(never.capabilities().replay.policy, ReplayPolicy::Never);
    assert_eq!(
        never.capabilities().replay.capability,
        ReplayCapability::Unsupported
    );
}

#[test]
fn protocol_feature_and_cancellation_ceilings_reject_expansion() {
    let endpoint = "https://example.test/v1";

    let mut direct = anthropic_capabilities(ReplayPolicy::IfValid);
    direct.features |= Capability::SOURCES;
    assert!(
        try_anthropic_model(endpoint, "future-id", direct, anthropic_protocol(), None).is_err()
    );

    let mut minimax = minimax_capabilities(ReplayPolicy::IfValid, true);
    minimax.features |= Capability::PARALLEL_TOOLS;
    assert!(
        try_minimax_model(
            endpoint,
            "future-id",
            minimax,
            oven_sdk_anthropic::MiniMaxProtocolSettings {
                thinking: true,
                thinking_disable_allowed: true,
            },
            None,
        )
        .is_err()
    );

    let mut aws = anthropic_capabilities(ReplayPolicy::IfValid);
    aws.cancellation = CancellationCapability::RemoteBestEffort;
    assert!(
        try_aws_model(
            endpoint,
            "future-id",
            aws,
            anthropic_protocol(),
            "us-west-2",
            "workspace",
            None,
        )
        .is_err()
    );
}

#[test]
fn protocol_media_ceilings_reject_wildcards_sources_and_modalities() {
    let endpoint = "https://example.test/v1";
    let mut direct = anthropic_capabilities(ReplayPolicy::IfValid);
    direct.media.input.insert(
        Modality::image(),
        MediaInputSupport::new(
            ["image/*".into()],
            MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::PROVIDER_REFERENCE,
        )
        .unwrap(),
    );
    assert!(
        try_anthropic_model(endpoint, "future-id", direct, anthropic_protocol(), None,).is_err()
    );

    let mut minimax = minimax_capabilities(ReplayPolicy::IfValid, true);
    minimax.modalities.input.insert(Modality::audio());
    minimax.media.input.insert(
        Modality::audio(),
        MediaInputSupport::new(["audio/*".into()], MediaSourceSupport::INLINE_BYTES).unwrap(),
    );
    assert!(
        try_minimax_model(
            endpoint,
            "future-id",
            minimax,
            oven_sdk_anthropic::MiniMaxProtocolSettings {
                thinking: true,
                thinking_disable_allowed: true,
            },
            None,
        )
        .is_err()
    );
}

#[test]
fn declarations_may_reduce_protocol_capabilities_and_media() {
    let mut reduced = anthropic_capabilities(ReplayPolicy::IfValid);
    reduced
        .features
        .remove(Capability::STRUCTURED_OUTPUT | Capability::PROMPT_CACHING);
    reduced
        .media
        .input
        .get_mut(&Modality::image())
        .unwrap()
        .media_types = vec!["image/png".into()];
    reduced
        .media
        .input
        .get_mut(&Modality::image())
        .unwrap()
        .sources = MediaSourceSupport::INLINE_BYTES;
    assert!(
        try_anthropic_model(
            "https://example.test/v1",
            "future-id",
            reduced,
            anthropic_protocol(),
            None,
        )
        .is_ok()
    );
}

#[test]
fn all_anthropic_protocols_reject_native_compaction_declarations() {
    let endpoint = "https://example.test/v1";

    let mut direct = anthropic_capabilities(ReplayPolicy::IfValid);
    direct.compaction = CompactionCapability::Native;
    assert!(
        try_anthropic_model(endpoint, "future-id", direct, anthropic_protocol(), None).is_err()
    );

    let mut minimax = minimax_capabilities(ReplayPolicy::IfValid, true);
    minimax.compaction = CompactionCapability::Native;
    assert!(
        try_minimax_model(
            endpoint,
            "future-id",
            minimax,
            oven_sdk_anthropic::MiniMaxProtocolSettings {
                thinking: true,
                thinking_disable_allowed: true,
            },
            None,
        )
        .is_err()
    );

    let mut aws = anthropic_capabilities(ReplayPolicy::IfValid);
    aws.compaction = CompactionCapability::Native;
    assert!(
        try_aws_model(
            endpoint,
            "future-id",
            aws,
            anthropic_protocol(),
            "us-west-2",
            "workspace",
            None,
        )
        .is_err()
    );
}

#[test]
fn query_bearing_base_endpoints_are_rejected() {
    let endpoint = "https://example.test/v1?route=messages";
    assert!(
        try_anthropic_model(
            endpoint,
            "future-id",
            anthropic_capabilities(ReplayPolicy::IfValid),
            anthropic_protocol(),
            None,
        )
        .is_err()
    );
    assert!(
        try_minimax_model(
            endpoint,
            "future-id",
            minimax_capabilities(ReplayPolicy::IfValid, true),
            oven_sdk_anthropic::MiniMaxProtocolSettings {
                thinking: true,
                thinking_disable_allowed: true,
            },
            None,
        )
        .is_err()
    );
    assert!(
        try_aws_model(
            endpoint,
            "future-id",
            anthropic_capabilities(ReplayPolicy::IfValid),
            anthropic_protocol(),
            "us-west-2",
            "workspace",
            None,
        )
        .is_err()
    );
}

#[test]
fn provider_source_contains_no_model_catalog_or_name_inference() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    assert!(!root.join("factory.rs").exists());
    assert!(!root.join("profile.rs").exists());

    let forbidden = [
        "official_model",
        "claude_model",
        "minimax_model",
        "model.starts_with(",
        "model.contains(",
        "match model_id.as_str()",
        "official_family",
        "catalog_model",
        "generation(model",
        "for_model(",
        "preset(",
        "Registry",
        ".alias(",
        "std::env::var",
        "claude-opus-",
        "claude-sonnet-",
        "claude-haiku-",
        "MiniMax-M3",
        "MiniMax-M2",
    ];
    fn rust_sources(directory: &std::path::Path, output: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_sources(&path, output);
            } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
                output.push(path);
            }
        }
    }

    let mut paths = Vec::new();
    rust_sources(&root, &mut paths);
    for path in paths {
        let source = std::fs::read_to_string(&path).unwrap();
        for needle in forbidden {
            assert!(
                !source.contains(needle),
                "forbidden model inference marker `{needle}` remains in {}",
                path.display()
            );
        }
    }
}
