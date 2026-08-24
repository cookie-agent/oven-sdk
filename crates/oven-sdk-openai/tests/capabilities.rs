pub mod common;

use oven_sdk::{Capability, LanguageModel, ReplayCapability, ReplayPolicy};
use oven_sdk_conformance::assert_model_id_independence;
use wiremock::MockServer;

#[tokio::test]
async fn descriptors_use_only_caller_declared_capabilities_and_limits() {
    let server = MockServer::start().await;
    let model = common::official_chat(&server, "unpublished/future-model");
    let capabilities = model.capabilities();
    assert_eq!(capabilities, &common::chat_capabilities());
    assert_eq!(capabilities.limits.context, Some(400_000));
    assert_eq!(capabilities.limits.output, Some(128_000));
    assert!(capabilities.features.contains(Capability::TOOL_CALLING));
}

#[tokio::test]
async fn chat_and_responses_are_model_id_independent() {
    let server = MockServer::start().await;
    common::mount(&server, "/chat/completions", common::chat_document("ok")).await;
    common::mount(&server, "/responses", common::responses_document("ok")).await;
    let chat_known = common::official_chat(&server, "gpt-4o-mini");
    let chat_future = common::official_chat(&server, "vendor/future-chat-2099");
    assert_model_id_independence(&chat_known, &chat_future)
        .await
        .unwrap();

    let responses_known = common::official_responses(&server, "gpt-5-mini");
    let responses_future = common::official_responses(&server, "vendor/future-responses-2099");
    assert_model_id_independence(&responses_known, &responses_future)
        .await
        .unwrap();
}

#[tokio::test]
async fn replay_policy_is_explicit_declaration_data() {
    let server = MockServer::start().await;
    let mut config = common::official_chat_config(&server, "any-model");
    config.model.capabilities.replay.policy = ReplayPolicy::Never;
    config.model.capabilities.replay.capability = ReplayCapability::Unsupported;
    let model = oven_sdk_openai::OpenAiChatModel::new(config).unwrap();
    assert_eq!(
        model.capabilities().replay.capability,
        ReplayCapability::Unsupported
    );
}

#[tokio::test]
async fn compatible_descriptor_uses_explicit_provider_adapter_and_declaration() {
    let server = MockServer::start().await;
    let model = common::compatible(&server);
    let descriptor = model.descriptor();
    assert_eq!(descriptor.identity.provider_id.as_str(), "fixture");
    assert_eq!(descriptor.identity.model_id.as_str(), "fixture-model");
    assert_eq!(descriptor.adapter_id.as_str(), "fixture.chat");
    assert_eq!(descriptor.capabilities, common::chat_capabilities());
}
