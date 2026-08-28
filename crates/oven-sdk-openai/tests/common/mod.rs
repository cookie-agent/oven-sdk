use std::collections::BTreeMap;

use oven_sdk::{
    ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    MediaCapabilities, MediaInputSupport, MediaSourceSupport, Modalities, Modality,
    ModelCapabilities, ModelConfig, ModelDeclaration, ModelId, ModelLimits, ProviderConfig,
    ProviderId, ReplayCapability, ReplayDeclaration, ReplayPolicy, SecretString,
};
use oven_sdk_openai::{
    MaxTokensField, OpenAiAuth, OpenAiChatModel, OpenAiChatSettings, OpenAiCompatibleAuth,
    OpenAiCompatibleChatModel, OpenAiCompatibleChatSettings, OpenAiResponsesCompaction,
    OpenAiResponsesModel, OpenAiResponsesSettings, ReasoningField, StructuredOutputSupport,
    SystemMessageRole,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

pub fn chat_document(text: &str) -> String {
    format!(
        concat!(
            "data: {{\"id\":\"chat_1\",\"model\":\"gpt-4o-mini\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{text:?}}},\"finish_reason\":null}}]}}\n\n",
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
            "event: response.created\ndata: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp_1\",\"model\":\"gpt-5-mini\"}}}}\n\n",
            "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}}}\n\n",
            "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":{text:?}}}\n\n",
            "event: response.output_item.done\ndata: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":{text:?}}}]}}}}\n\n",
            "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_1\",\"model\":\"gpt-5-mini\",\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":{text:?}}}]}}],\"usage\":{{\"input_tokens\":2,\"output_tokens\":3,\"output_tokens_details\":{{\"reasoning_tokens\":1}}}}}}}}\n\n"
        ),
        text = text
    )
}

pub fn compact_document() -> serde_json::Value {
    serde_json::json!({
        "id": "cmp_1",
        "created_at": 1_754_000_000_u64,
        "object": "response.compaction",
        "output": [
            {
                "type": "message",
                "id": "msg_retained_1",
                "role": "user",
                "status": "completed",
                "content": [
                    {
                        "type": "input_text",
                        "text": "retained",
                        "prompt_cache_breakpoint": {"mode": "explicit"}
                    },
                    {
                        "type": "input_image",
                        "image_url": "https://example.test/image.png",
                        "detail": "original",
                        "prompt_cache_breakpoint": {"mode": "explicit"}
                    },
                    {
                        "type": "input_file",
                        "file_url": "https://example.test/document.pdf",
                        "filename": "document.pdf",
                        "detail": "high",
                        "prompt_cache_breakpoint": {"mode": "explicit"}
                    }
                ],
            },
            {
                "type": "compaction",
                "id": "cmp_item_1",
                "encrypted_content": "opaque-compacted-state",
                "created_by": "openai"
            }
        ],
        "usage": {
            "input_tokens": 120,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens": 8,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 128
        }
    })
}

pub async fn mount(server: &MockServer, endpoint: &str, body: String) {
    Mock::given(method("POST"))
        .and(path(endpoint))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "text/event-stream")
                .insert_header("x-request-id", "req_1"),
        )
        .mount(server)
        .await;
}

pub async fn mount_compaction(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/responses/compact"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(compact_document())
                .insert_header("x-request-id", "req_compact_1"),
        )
        .mount(server)
        .await;
}

pub fn official_chat(server: &MockServer, model_id: &str) -> OpenAiChatModel {
    OpenAiChatModel::new(official_chat_config(server, model_id)).expect("official Chat model")
}

pub fn official_chat_config(
    server: &MockServer,
    model_id: &str,
) -> ModelConfig<OpenAiAuth, OpenAiChatSettings> {
    official_chat_config_at(&server.uri(), model_id, "secret")
}

pub fn official_chat_config_at(
    api: &str,
    model_id: &str,
    api_key: &str,
) -> ModelConfig<OpenAiAuth, OpenAiChatSettings> {
    let mut auth = OpenAiAuth::new(SecretString::new(api_key));
    auth.organization = Some("org".into());
    auth.project = Some("project".into());
    ModelConfig::new(
        ProviderConfig::new(
            ProviderId::new("openai"),
            ApiEndpoint::parse(api).unwrap(),
            auth,
            HeaderConfig::empty(),
        )
        .unwrap(),
        ModelDeclaration::new(ModelId::new(model_id), chat_capabilities()).unwrap(),
        OpenAiChatSettings::new(
            SystemMessageRole::System,
            MaxTokensField::MaxTokens,
            StructuredOutputSupport::JsonSchema,
        ),
    )
}

pub fn official_responses(server: &MockServer, model_id: &str) -> OpenAiResponsesModel {
    OpenAiResponsesModel::new(official_responses_config(server, model_id))
        .expect("official Responses model")
}

pub fn official_responses_config(
    server: &MockServer,
    model_id: &str,
) -> ModelConfig<OpenAiAuth, OpenAiResponsesSettings> {
    official_responses_config_at(&server.uri(), model_id, "secret")
}

pub fn official_responses_config_at(
    api: &str,
    model_id: &str,
    api_key: &str,
) -> ModelConfig<OpenAiAuth, OpenAiResponsesSettings> {
    let mut auth = OpenAiAuth::new(SecretString::new(api_key));
    auth.organization = Some("org".into());
    auth.project = Some("project".into());
    ModelConfig::new(
        ProviderConfig::new(
            ProviderId::new("openai"),
            ApiEndpoint::parse(api).unwrap(),
            auth,
            HeaderConfig::empty(),
        )
        .unwrap(),
        ModelDeclaration::new(ModelId::new(model_id), responses_capabilities()).unwrap(),
        OpenAiResponsesSettings::new(),
    )
}

pub fn official_responses_native(server: &MockServer, model_id: &str) -> OpenAiResponsesModel {
    OpenAiResponsesModel::new(official_responses_native_config_at(
        &server.uri(),
        model_id,
        "secret",
    ))
    .expect("official Responses native-compaction model")
}

pub fn official_responses_native_config_at(
    api: &str,
    model_id: &str,
    api_key: &str,
) -> ModelConfig<OpenAiAuth, OpenAiResponsesSettings> {
    let mut config = official_responses_config_at(api, model_id, api_key);
    config.model.capabilities.compaction = CompactionCapability::Native;
    config.settings.compaction = OpenAiResponsesCompaction::V1;
    config
}

pub fn compatible(server: &MockServer) -> OpenAiCompatibleChatModel {
    OpenAiCompatibleChatModel::new(compatible_config(server, "fixture-model"))
        .expect("compatible Chat model")
}

pub fn compatible_config(
    server: &MockServer,
    model_id: &str,
) -> ModelConfig<OpenAiCompatibleAuth, OpenAiCompatibleChatSettings> {
    compatible_config_at(&server.uri(), model_id, "secret")
}

pub fn compatible_config_at(
    api: &str,
    model_id: &str,
    api_key: &str,
) -> ModelConfig<OpenAiCompatibleAuth, OpenAiCompatibleChatSettings> {
    ModelConfig::new(
        ProviderConfig::new(
            ProviderId::new("fixture"),
            ApiEndpoint::parse(api).unwrap(),
            OpenAiCompatibleAuth::bearer(SecretString::new(api_key)),
            HeaderConfig::empty(),
        )
        .unwrap(),
        ModelDeclaration::new(ModelId::new(model_id), compatible_chat_capabilities()).unwrap(),
        {
            let mut settings = OpenAiCompatibleChatSettings::new(
                oven_sdk::AdapterId::new("fixture.chat"),
                SystemMessageRole::System,
                MaxTokensField::MaxTokens,
                StructuredOutputSupport::JsonSchema,
                ReasoningField::None,
            );
            settings.stream_usage = true;
            settings
        },
    )
}

pub fn chat_capabilities() -> ModelCapabilities {
    capabilities(false)
}

pub fn compatible_chat_capabilities() -> ModelCapabilities {
    let mut capabilities = capabilities(false);
    capabilities.modalities.input.insert(Modality::video());
    capabilities.media.input.insert(
        Modality::video(),
        MediaInputSupport::new(["video/mp4".to_owned()], MediaSourceSupport::INLINE_BYTES).unwrap(),
    );
    capabilities
}

pub fn responses_capabilities() -> ModelCapabilities {
    capabilities(true)
}

fn capabilities(reasoning_replay: bool) -> ModelCapabilities {
    let mut media = BTreeMap::new();
    media.insert(
        Modality::image(),
        MediaInputSupport::new(
            vec!["image/*".into()],
            MediaSourceSupport::INLINE_BYTES
                | MediaSourceSupport::INLINE_TEXT
                | MediaSourceSupport::URL,
        )
        .unwrap(),
    );
    media.insert(
        Modality::pdf(),
        MediaInputSupport::new(
            vec!["application/pdf".into()],
            MediaSourceSupport::INLINE_BYTES
                | MediaSourceSupport::INLINE_TEXT
                | if reasoning_replay {
                    MediaSourceSupport::URL
                } else {
                    MediaSourceSupport::empty()
                },
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
            | Capability::USAGE,
        limits: ModelLimits::new(Some(400_000), Some(272_000), Some(128_000)),
        modalities: Modalities::new(
            [Modality::text(), Modality::image(), Modality::pdf()],
            [Modality::text()],
        ),
        media: MediaCapabilities { input: media },
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        replay: ReplayDeclaration {
            policy: ReplayPolicy::IfValid,
            capability: if reasoning_replay {
                ReplayCapability::Required
            } else {
                ReplayCapability::Optional
            },
            reasoning: reasoning_replay,
        },
    }
}
