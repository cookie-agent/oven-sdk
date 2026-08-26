#![warn(missing_docs)]
// `ModelError` carries the core contract's structured diagnostics by value.
#![allow(clippy::result_large_err)]
//! Independent Azure OpenAI Chat Completions and Responses adapters.

mod chat;
mod configuration;
mod error;
mod media;
mod options;
mod responses;
mod schema;
mod sse;
mod transport;
mod wire;

pub use chat::model::AzureOpenAiChatModel;
pub use configuration::{
    AzureMaxTokensField, AzureOpenAiAuth, AzureOpenAiChatConfig, AzureOpenAiChatSettings,
    AzureOpenAiCompletionsConfig, AzureOpenAiResponsesCompaction, AzureOpenAiResponsesConfig,
    AzureOpenAiResponsesSettings, AzureOpenAiRevision, AzureReasoningField,
    AzureStructuredOutputSupport, AzureSystemMessageRole, AzureTokenFuture, AzureTokenProvider,
};
pub use options::{
    AzureOpenAiChatOptions, AzureOpenAiChatRequestExt, AzureOpenAiCompactionOptions,
    AzureOpenAiCompactionRequestExt, AzureOpenAiOptions, AzureOpenAiPromptCacheRetention,
    AzureOpenAiResponsesOptions, AzureOpenAiResponsesRequestExt,
};
pub use responses::model::AzureOpenAiResponsesModel;
pub use transport::AzureOpenAiTimeouts;
pub use wire::{
    AZURE_OPENAI_CHAT_ADAPTER_ID, AZURE_OPENAI_PROVIDER_ID, AZURE_OPENAI_RESPONSES_ADAPTER_ID,
    AzureApiRoute, AzureApiVersion,
};
