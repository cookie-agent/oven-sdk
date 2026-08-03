#![warn(missing_docs)]
// `ModelError` carries the core contract's structured diagnostics by value.
#![allow(clippy::result_large_err)]
//! Official OpenAI Chat Completions and Responses adapters plus conservative
//! configurable OpenAI-compatible Chat support.

mod chat;
mod configuration;
mod error;
mod options;
mod responses;
mod sse;
mod transport;
mod wire;

pub use chat::model::{OpenAiChatModel, OpenAiCompatibleChatModel};
pub use configuration::{
    MaxTokensField, OpenAiAuth, OpenAiChatSettings, OpenAiCompatibleAuth,
    OpenAiCompatibleChatSettings, OpenAiResponsesCompaction, OpenAiResponsesSettings,
    ReasoningField, StructuredOutputSupport, SystemMessageRole,
};
pub use options::{
    CompatibleChatOptions, OpenAiChatOptions, OpenAiChatRequestExt, OpenAiOptions,
    OpenAiPromptCacheOptions, OpenAiResponsesCompactionOptions,
    OpenAiResponsesCompactionRequestExt, OpenAiResponsesOptions, OpenAiResponsesRequestExt,
};
pub use responses::model::OpenAiResponsesModel;
pub use transport::OpenAiTimeouts;
pub use wire::{OPENAI_CHAT_ADAPTER_ID, OPENAI_PROVIDER_ID, OPENAI_RESPONSES_ADAPTER_ID};
