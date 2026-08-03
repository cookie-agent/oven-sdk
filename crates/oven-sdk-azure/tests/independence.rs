use oven_sdk::{
    ApiEndpoint, CancellationCapability, HeaderConfig, LanguageModel, ModelCapabilities,
    ModelConfig, ModelDeclaration, ModelId, ProviderConfig, ProviderId, SecretString,
};
use oven_sdk_azure::{
    AZURE_OPENAI_PROVIDER_ID, AzureOpenAiAuth, AzureOpenAiChatConfig, AzureOpenAiChatModel,
    AzureOpenAiChatSettings, AzureOpenAiResponsesConfig, AzureOpenAiResponsesModel,
    AzureOpenAiResponsesSettings,
};

fn capabilities() -> ModelCapabilities {
    let mut capabilities = ModelCapabilities::conservative();
    capabilities.cancellation = CancellationCapability::LocalOnly;
    capabilities
}

fn provider() -> ProviderConfig<AzureOpenAiAuth> {
    ProviderConfig::new(
        ProviderId::new(AZURE_OPENAI_PROVIDER_ID),
        ApiEndpoint::parse("https://example.test").unwrap(),
        AzureOpenAiAuth::ApiKey(SecretString::new("secret")),
        HeaderConfig::empty(),
    )
    .unwrap()
}

#[test]
fn package_has_no_openai_adapter_or_registry_dependency() {
    let manifest = include_str!("../Cargo.toml");
    assert!(!manifest.contains("oven-sdk-openai"));
    assert!(!manifest.contains("models.dev"));
    assert!(!manifest.contains("models-dev"));
}

#[test]
fn concrete_models_construct_only_from_core_registry_free_config() {
    let chat: AzureOpenAiChatConfig = ModelConfig::new(
        provider(),
        ModelDeclaration::new(ModelId::new("chat-deployment"), capabilities()).unwrap(),
        AzureOpenAiChatSettings::default(),
    );
    let responses: AzureOpenAiResponsesConfig = ModelConfig::new(
        provider(),
        ModelDeclaration::new(ModelId::new("responses-deployment"), capabilities()).unwrap(),
        AzureOpenAiResponsesSettings::default(),
    );

    let chat = AzureOpenAiChatModel::new(chat).unwrap();
    let responses = AzureOpenAiResponsesModel::new(responses).unwrap();
    assert_eq!(chat.model_id().as_str(), "chat-deployment");
    assert_eq!(responses.model_id().as_str(), "responses-deployment");
}

#[test]
fn model_names_do_not_select_behavior() {
    let construct = |id: &str| {
        AzureOpenAiChatModel::new(ModelConfig::new(
            provider(),
            ModelDeclaration::new(ModelId::new(id), capabilities()).unwrap(),
            AzureOpenAiChatSettings::default(),
        ))
        .unwrap()
    };
    let named = construct("gpt-4o-special-preview");
    let arbitrary = construct("customer-deployment-17");
    assert_eq!(named.capabilities(), arbitrary.capabilities());
}
