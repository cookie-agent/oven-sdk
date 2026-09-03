use std::collections::{BTreeMap, BTreeSet};

use http::{HeaderMap, HeaderValue};
use oven_sdk::{
    AdapterId, ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    HeaderContext, HeaderOverrides, LanguageModelDescriptor, MediaCapabilities, MediaInputSupport,
    MediaSourceSupport, Modalities, Modality, ModelCapabilities, ModelConfig, ModelDeclaration,
    ModelIdentity, ModelLimits, ProviderConfig, ProviderId, ReplayCapability, ReplayDeclaration,
    ReplayPolicy, SecretString, Usage, UsageTotals,
};
use serde_json::json;
use serde_test::{Token, assert_tokens};

fn capabilities(features: Capability) -> ModelCapabilities {
    ModelCapabilities {
        features,
        limits: ModelLimits::new(Some(128_000), Some(120_000), Some(8_000)),
        modalities: Modalities::new([Modality::text(), Modality::image()], [Modality::text()]),
        media: MediaCapabilities {
            input: BTreeMap::from([(
                Modality::image(),
                MediaInputSupport::new(
                    ["image/png".to_owned(), "image/jpeg".to_owned()],
                    MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL,
                )
                .expect("media support"),
            )]),
        },
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        replay: ReplayDeclaration {
            policy: ReplayPolicy::IfValid,
            capability: ReplayCapability::Optional,
            reasoning: false,
        },
    }
}

#[test]
fn capability_serializes_as_snake_case_names() {
    let features = Capability::TOOL_CALLING
        | Capability::REASONING
        | Capability::TEMPERATURE
        | Capability::MAX_OUTPUT_TOKENS;
    assert_eq!(
        serde_json::to_value(features).expect("features serialize"),
        json!([
            "tool_calling",
            "reasoning",
            "temperature",
            "max_output_tokens"
        ])
    );
}

#[test]
fn capability_deserializes_lists_and_rejects_strings_and_unknown_names() {
    let expected = Capability::TOOL_CALLING | Capability::REASONING;
    let from_list: Capability =
        serde_json::from_value(json!(["tool_calling", "reasoning"])).expect("list");
    assert_eq!(from_list, expected);

    let error = serde_json::from_value::<Capability>(json!("tool_calling"))
        .expect_err("string form must fail")
        .to_string();
    assert!(error.contains("sequence of snake_case capability names"));

    let error = serde_json::from_value::<Capability>(json!(["not_a_capability"]))
        .expect_err("unknown name must fail")
        .to_string();
    assert!(error.contains("unknown capability `not_a_capability`"));
}

#[test]
fn capability_sequence_length_is_exact_for_non_human_serde() {
    assert_tokens(
        &(Capability::TOOL_CALLING | Capability::REASONING | Capability::TOP_P),
        &[
            Token::Seq { len: Some(3) },
            Token::Str("tool_calling"),
            Token::Str("reasoning"),
            Token::Str("top_p"),
            Token::SeqEnd,
        ],
    );
}

#[test]
fn modality_is_open_validated_and_round_trips() {
    let modalities = Modalities::new(
        [
            Modality::text(),
            Modality::new("chemical").expect("open modality"),
        ],
        [Modality::audio()],
    );
    let encoded = serde_json::to_string(&modalities).expect("serialize modalities");
    let decoded: Modalities = serde_json::from_str(&encoded).expect("deserialize modalities");
    assert_eq!(decoded, modalities);
    assert!(Modality::new("").is_err());
    assert!(serde_json::from_value::<Modality>(json!("\n")).is_err());
}

#[test]
fn media_sources_serialize_as_names_and_reject_unknown_values() {
    let sources = MediaSourceSupport::INLINE_BYTES | MediaSourceSupport::URL;
    assert_eq!(
        serde_json::to_value(sources).expect("serialize sources"),
        json!(["inline_bytes", "url"])
    );
    assert!(serde_json::from_value::<MediaSourceSupport>(json!(["ambient_path"])).is_err());
}

#[test]
fn model_capabilities_and_descriptor_round_trip_with_complete_metadata() {
    let descriptor = LanguageModelDescriptor::new(
        ModelIdentity::new(
            ProviderId::new("openai"),
            oven_sdk::ModelId::new("gpt-test"),
        )
        .expect("identity"),
        AdapterId::new("oven.openai.responses"),
        capabilities(
            Capability::TOOL_CALLING
                | Capability::STRUCTURED_OUTPUT
                | Capability::MAX_OUTPUT_TOKENS,
        ),
    )
    .expect("descriptor")
    .with_provider_metadata(BTreeMap::from([
        ("region".into(), json!("us-east-1")),
        ("surface".into(), json!("responses")),
    ]));

    let encoded = serde_json::to_string(&descriptor).expect("descriptor serializes");
    let decoded: LanguageModelDescriptor =
        serde_json::from_str(&encoded).expect("descriptor deserializes");
    assert_eq!(decoded, descriptor);
    assert_eq!(decoded.identity.provider_id, ProviderId::new("openai"));
    assert_eq!(decoded.capabilities.limits.output, Some(8_000));
    assert_eq!(decoded.provider_metadata.len(), 2);
}

#[test]
fn descriptor_serde_requires_identity_capabilities_and_metadata_fields() {
    let error = serde_json::from_value::<LanguageModelDescriptor>(json!({
        "identity": {"provider_id":"openai","model_id":"gpt-test"},
        "adapter_id":"oven.openai.responses"
    }))
    .expect_err("capabilities are explicit")
    .to_string();
    assert!(error.contains("missing field `capabilities`"));

    let mut value = serde_json::to_value(
        LanguageModelDescriptor::new(
            ModelIdentity::new(
                ProviderId::new("openai"),
                oven_sdk::ModelId::new("gpt-test"),
            )
            .expect("identity"),
            AdapterId::new("oven.openai.responses"),
            capabilities(Capability::empty()),
        )
        .expect("descriptor"),
    )
    .expect("value");
    value["identity"]["provider_id"] = json!("");
    assert!(serde_json::from_value::<LanguageModelDescriptor>(value).is_err());
}

#[test]
fn capability_dependency_validation_is_strict() {
    let mut declaration = capabilities(Capability::PARALLEL_TOOLS);
    assert!(declaration.validate().is_err());
    assert!(
        serde_json::from_value::<ModelCapabilities>(
            serde_json::to_value(&declaration).expect("serialize invalid declaration")
        )
        .is_err()
    );

    declaration = capabilities(Capability::TOOL_INPUT_DELTAS);
    assert!(declaration.validate().is_err());

    declaration = capabilities(Capability::empty());
    declaration.replay.reasoning = true;
    assert!(declaration.validate().is_err());

    declaration = capabilities(Capability::REASONING);
    declaration.replay.capability = ReplayCapability::Required;
    declaration.replay.policy = ReplayPolicy::Never;
    assert!(declaration.validate().is_err());

    declaration = capabilities(Capability::empty());
    declaration.modalities.output.clear();
    assert!(
        serde_json::from_value::<ModelCapabilities>(
            serde_json::to_value(&declaration).expect("serialize invalid modalities")
        )
        .is_err()
    );
}

#[test]
fn limits_validate_without_assuming_input_plus_output_relationship() {
    assert!(
        ModelLimits::new(Some(100), Some(80), Some(90))
            .validate()
            .is_ok()
    );
    assert!(
        ModelLimits::new(Some(100), Some(101), None)
            .validate()
            .is_err()
    );
    assert!(
        ModelLimits::new(Some(100), None, Some(101))
            .validate()
            .is_err()
    );
    assert!(
        ModelLimits::new(Some(0), Some(20), Some(30))
            .validate()
            .is_ok()
    );
}

#[test]
fn endpoint_validation_rejects_unsafe_or_unresolved_values() {
    let endpoint = ApiEndpoint::parse("https://api.example.com/v1").expect("safe endpoint");
    let encoded = serde_json::to_string(&endpoint).expect("endpoint serializes");
    let decoded: ApiEndpoint = serde_json::from_str(&encoded).expect("endpoint deserializes");
    assert_eq!(decoded, endpoint);
    assert!(!format!("{endpoint:?}").contains("api.example.com"));
    assert!(!format!("{endpoint:?}").contains("/v1"));
    for value in [
        "ftp://api.example.com",
        "https://user:password@api.example.com",
        "https://api.example.com/#fragment",
        "https://api.example.com/v1?api_key=super-secret",
        "https://${API_HOST}/v1",
        "https://",
    ] {
        assert!(ApiEndpoint::parse(value).is_err(), "accepted {value}");
        assert!(serde_json::from_value::<ApiEndpoint>(json!(value)).is_err());
    }
    let error = serde_json::from_value::<ApiEndpoint>(json!(
        "https://api.example.com/v1?api_key=super-secret"
    ))
    .expect_err("query endpoint must fail")
    .to_string();
    assert!(!error.contains("super-secret"));
}

#[test]
fn secret_and_header_debug_output_redacts_values() {
    let secret = SecretString::new("super-secret");
    assert!(!format!("{secret:?}").contains("super-secret"));
    assert_eq!(secret.to_string(), "<redacted>");

    let context = HeaderContext::new("session-secret").with_parent_session_id("parent-secret");
    let debug = format!("{context:?}");
    assert!(debug.contains("has_session_id: true"));
    assert!(debug.contains("has_parent_session_id: true"));
    assert!(!debug.contains("session-secret"));
    assert!(!debug.contains("parent-secret"));

    let mut headers = HeaderMap::new();
    headers.insert("x-api-key", HeaderValue::from_static("header-secret"));
    let overrides = HeaderOverrides::new(headers);
    let debug = format!("{overrides:?}");
    assert!(debug.contains("x-api-key"));
    assert!(!debug.contains("header-secret"));

    let provider = ProviderConfig::new(
        ProviderId::new("debug-provider"),
        ApiEndpoint::parse("https://private.example.com/secret-path").expect("endpoint"),
        SecretString::new("auth-secret"),
        HeaderConfig {
            static_headers: overrides,
            dynamic_headers: None,
        },
    )
    .expect("provider config");
    let model = ModelDeclaration::new(
        oven_sdk::ModelId::new("debug-model"),
        capabilities(Capability::empty()),
    )
    .expect("model declaration");
    let config = ModelConfig::new(provider, model, SecretString::new("settings-secret"));
    let debug = format!("{config:?}");
    for secret in [
        "private.example.com",
        "secret-path",
        "auth-secret",
        "header-secret",
        "settings-secret",
    ] {
        assert!(!debug.contains(secret), "debug exposed {secret}");
    }
}

#[test]
fn usage_totals_aggregate_like_fields_and_omit_raw_usage() {
    let first = Usage {
        input_tokens: Some(10),
        input_tokens_no_cache: Some(8),
        input_tokens_cache_write: Some(2),
        output_tokens: Some(4),
        output_tokens_text: Some(3),
        raw: Some(json!({"provider": "first"})),
        ..Usage::default()
    };
    let second = Usage {
        input_tokens: Some(5),
        input_tokens_cache_read: Some(1),
        input_tokens_cache_write: Some(3),
        output_tokens_text: Some(2),
        output_tokens_reasoning: Some(1),
        raw: Some(json!({"provider": "second"})),
        ..Usage::default()
    };
    let mut totals = UsageTotals::default();
    totals.record(&first);
    totals.record(&second);
    assert_eq!(totals.calls(), 2);
    assert_eq!(totals.usage().input_tokens, Some(15));
    assert_eq!(totals.usage().output_tokens_text, Some(5));
    assert_eq!(totals.usage().raw, None);
}

#[test]
fn modalities_use_sets_without_duplicate_serialized_entries() {
    let modalities = Modalities::new(
        [Modality::text(), Modality::text(), Modality::image()],
        [Modality::text()],
    );
    assert_eq!(
        modalities.input,
        BTreeSet::from([Modality::text(), Modality::image()])
    );
}
