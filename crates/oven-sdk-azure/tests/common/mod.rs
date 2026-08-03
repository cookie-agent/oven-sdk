// Each integration-test binary uses a different subset of these shared fixtures.
#![allow(dead_code)]
// Test helpers intentionally return the public `ModelError` API by value.
#![allow(clippy::result_large_err)]

use oven_sdk::{
    ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    HeaderOverrides, MediaInputSupport, MediaSourceSupport, Modalities, Modality,
    ModelCapabilities, ModelConfig, ModelDeclaration, ModelId, ModelLimits, ProviderConfig,
    ProviderId, ReplayCapability, ReplayDeclaration, ReplayPolicy, SecretString,
};
use oven_sdk_azure::{
    AzureApiRoute, AzureMaxTokensField, AzureOpenAiAuth, AzureOpenAiChatModel,
    AzureOpenAiChatSettings, AzureOpenAiCompletionsConfig, AzureOpenAiResponsesCompaction,
    AzureOpenAiResponsesModel, AzureOpenAiResponsesSettings, AzureOpenAiRevision,
    AzureStructuredOutputSupport, AzureSystemMessageRole,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

pub fn chat_document(text: &str) -> String {
    format!(
        concat!(
            "data: {{\"prompt_filter_results\":[{{\"prompt_index\":0,\"content_filter_results\":{{\"hate\":{{\"filtered\":false,\"severity\":\"safe\"}}}}}}],\"choices\":[]}}\n\n",
            "data: {{\"id\":\"chat_1\",\"model\":\"gpt-4o-mini\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{text:?}}},\"content_filter_results\":{{\"violence\":{{\"filtered\":false,\"severity\":\"safe\"}}}},\"finish_reason\":null}}]}}\n\n",
            "data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n",
            "data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":2,\"completion_tokens\":3,\"completion_tokens_details\":{{\"reasoning_tokens\":1}}}}}}\n\n",
            "data: [DONE]\n\n"
        ),
        text = text
    )
}

pub fn responses_document(text: &str) -> String {
    format!(
        concat!(
            "event: response.created\ndata: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp_1\",\"model\":\"gpt-5-mini\",\"content_filters\":[{{\"blocked\":false,\"source_type\":\"prompt\"}}]}}}}\n\n",
            "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}}}\n\n",
            "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":{text:?}}}\n\n",
            "event: response.output_item.done\ndata: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":{text:?}}}]}}}}\n\n",
            "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_1\",\"model\":\"gpt-5-mini\",\"status\":\"completed\",\"content_filters\":[{{\"blocked\":false,\"source_type\":\"completion\"}}],\"output\":[{{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":{text:?}}}]}}],\"usage\":{{\"input_tokens\":2,\"output_tokens\":3,\"output_tokens_details\":{{\"reasoning_tokens\":1}}}}}}}}\n\n"
        ),
        text = text
    )
}

pub async fn mount(server: &MockServer, endpoint: &str, body: String) {
    Mock::given(method("POST"))
        .and(path(endpoint))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "text/event-stream")
                .insert_header("apim-request-id", "req_azure_1"),
        )
        .mount(server)
        .await;
}

pub struct TestProvider {
    api: ApiEndpoint,
    route: AzureApiRoute,
    headers: reqwest::header::HeaderMap,
}

#[derive(Clone)]
pub struct ModelSetup {
    pub revision: Option<AzureOpenAiRevision>,
    pub capabilities: ModelCapabilities,
    pub completions: AzureOpenAiCompletionsConfig,
    pub compaction: AzureOpenAiResponsesCompaction,
}

impl TestProvider {
    pub fn with_header(mut self, name: &'static str, value: &'static str) -> Self {
        self.headers.insert(
            reqwest::header::HeaderName::from_static(name),
            reqwest::header::HeaderValue::from_static(value),
        );
        self
    }

    pub fn chat(
        &self,
        deployment: impl Into<String>,
        setup: ModelSetup,
    ) -> Result<AzureOpenAiChatModel, oven_sdk::ModelError> {
        let provider = self.provider_config()?;
        let model = ModelDeclaration::new(ModelId::new(deployment), setup.capabilities)?;
        AzureOpenAiChatModel::new(ModelConfig::new(
            provider,
            model,
            AzureOpenAiChatSettings {
                route: self.route.clone(),
                revision: setup.revision,
                timeouts: Default::default(),
                completions: setup.completions,
            },
        ))
    }

    pub fn responses(
        &self,
        deployment: impl Into<String>,
        setup: ModelSetup,
    ) -> Result<AzureOpenAiResponsesModel, oven_sdk::ModelError> {
        let provider = self.provider_config()?;
        let model = ModelDeclaration::new(ModelId::new(deployment), setup.capabilities)?;
        AzureOpenAiResponsesModel::new(ModelConfig::new(
            provider,
            model,
            AzureOpenAiResponsesSettings {
                route: self.route.clone(),
                revision: setup.revision,
                timeouts: Default::default(),
                compaction: setup.compaction,
            },
        ))
    }

    fn provider_config(&self) -> Result<ProviderConfig<AzureOpenAiAuth>, oven_sdk::ModelError> {
        ProviderConfig::new(
            ProviderId::new(oven_sdk_azure::AZURE_OPENAI_PROVIDER_ID),
            self.api.clone(),
            AzureOpenAiAuth::ApiKey(SecretString::new("secret")),
            HeaderConfig {
                static_headers: HeaderOverrides::new(self.headers.clone()),
                dynamic_headers: None,
            },
        )
    }
}

pub fn provider(server: &MockServer, route: AzureApiRoute) -> TestProvider {
    TestProvider {
        api: ApiEndpoint::parse(server.uri()).expect("wiremock URL"),
        route,
        headers: reqwest::header::HeaderMap::new(),
    }
}

pub fn chat_config(
    api: impl AsRef<str>,
    route: AzureApiRoute,
    deployment: impl Into<String>,
    setup: ModelSetup,
    auth: AzureOpenAiAuth,
) -> oven_sdk_azure::AzureOpenAiChatConfig {
    let provider = ProviderConfig::new(
        ProviderId::new(oven_sdk_azure::AZURE_OPENAI_PROVIDER_ID),
        ApiEndpoint::parse(api).expect("valid test API endpoint"),
        auth,
        HeaderConfig::empty(),
    )
    .expect("valid Azure provider configuration");
    let model = ModelDeclaration::new(ModelId::new(deployment), setup.capabilities)
        .expect("valid test model declaration");
    ModelConfig::new(
        provider,
        model,
        AzureOpenAiChatSettings {
            route,
            revision: setup.revision,
            timeouts: Default::default(),
            completions: setup.completions,
        },
    )
}

pub fn responses_config(
    api: impl AsRef<str>,
    route: AzureApiRoute,
    deployment: impl Into<String>,
    setup: ModelSetup,
    auth: AzureOpenAiAuth,
) -> oven_sdk_azure::AzureOpenAiResponsesConfig {
    let provider = ProviderConfig::new(
        ProviderId::new(oven_sdk_azure::AZURE_OPENAI_PROVIDER_ID),
        ApiEndpoint::parse(api).expect("valid test API endpoint"),
        auth,
        HeaderConfig::empty(),
    )
    .expect("valid Azure provider configuration");
    let model = ModelDeclaration::new(ModelId::new(deployment), setup.capabilities)
        .expect("valid test model declaration");
    ModelConfig::new(
        provider,
        model,
        AzureOpenAiResponsesSettings {
            route,
            revision: setup.revision,
            timeouts: Default::default(),
            compaction: setup.compaction,
        },
    )
}

pub fn conservative() -> ModelSetup {
    let mut capabilities = ModelCapabilities::conservative();
    capabilities.cancellation = CancellationCapability::LocalOnly;
    ModelSetup {
        revision: None,
        capabilities,
        completions: AzureOpenAiCompletionsConfig::default(),
        compaction: AzureOpenAiResponsesCompaction::Unsupported,
    }
}

pub fn gpt5_compaction() -> ModelSetup {
    let mut setup = gpt5();
    setup.capabilities.compaction = CompactionCapability::Native;
    setup.compaction = AzureOpenAiResponsesCompaction::V1 {
        routing_discriminator: "test-v1-route".into(),
    };
    setup
}

pub fn gpt4o() -> ModelSetup {
    let mut setup = conservative();
    setup.revision = Some(AzureOpenAiRevision {
        model: "gpt-4o-mini".into(),
        version: "2024-07-18".into(),
        deployment_type: "standard".into(),
    });
    setup.capabilities.features = Capability::USAGE
        | Capability::TOOL_CALLING
        | Capability::TOOL_INPUT_DELTAS
        | Capability::PARALLEL_TOOLS
        | Capability::STRUCTURED_OUTPUT;
    setup.capabilities.limits = ModelLimits::new(Some(128_000), None, Some(16_384));
    setup.capabilities.modalities =
        Modalities::new([Modality::text(), Modality::image()], [Modality::text()]);
    setup.capabilities.media.input.insert(
        Modality::image(),
        MediaInputSupport::new(
            ["image/png", "image/jpeg", "image/webp", "image/gif"].map(str::to_owned),
            MediaSourceSupport::INLINE_BYTES
                | MediaSourceSupport::INLINE_TEXT
                | MediaSourceSupport::URL,
        )
        .expect("valid image support"),
    );
    setup.capabilities.replay = ReplayDeclaration {
        policy: ReplayPolicy::IfValid,
        capability: ReplayCapability::Optional,
        reasoning: false,
    };
    setup.completions.stream_usage = true;
    setup.completions.structured_output = AzureStructuredOutputSupport::JsonSchema;
    setup
}

pub fn gpt5() -> ModelSetup {
    let mut setup = gpt4o();
    setup.revision = Some(AzureOpenAiRevision {
        model: "gpt-5-mini".into(),
        version: "2025-08-07".into(),
        deployment_type: "global_standard".into(),
    });
    setup.capabilities.features |= Capability::REASONING;
    setup.capabilities.limits = ModelLimits::new(Some(400_000), None, Some(128_000));
    setup.capabilities.modalities.input.insert(Modality::pdf());
    setup.capabilities.media.input.insert(
        Modality::pdf(),
        MediaInputSupport::new(
            ["application/pdf".to_owned()],
            MediaSourceSupport::INLINE_BYTES
                | MediaSourceSupport::INLINE_TEXT
                | MediaSourceSupport::URL,
        )
        .expect("valid PDF support"),
    );
    setup.capabilities.replay = ReplayDeclaration {
        policy: ReplayPolicy::Always,
        capability: ReplayCapability::Required,
        reasoning: true,
    };
    setup.completions.system_role = AzureSystemMessageRole::Developer;
    setup.completions.max_tokens_field = AzureMaxTokensField::MaxCompletionTokens;
    setup.completions.omit_reasoning_sampling = true;
    setup
}

pub fn gpt5_chat() -> ModelSetup {
    let mut setup = gpt5();
    setup.capabilities.modalities.input.remove(&Modality::pdf());
    setup.capabilities.media.input.remove(&Modality::pdf());
    setup.capabilities.replay = ReplayDeclaration {
        policy: ReplayPolicy::IfValid,
        capability: ReplayCapability::Optional,
        reasoning: false,
    };
    setup
}
