//! OpenAI wire protocol constants.

pub(crate) mod chat;
pub(crate) mod responses;

/// Stable official OpenAI provider identity.
pub const OPENAI_PROVIDER_ID: &str = "openai";
/// Stable official Chat Completions adapter identity.
pub const OPENAI_CHAT_ADAPTER_ID: &str = "oven.openai.chat";
/// Stable official Responses adapter identity.
pub const OPENAI_RESPONSES_ADAPTER_ID: &str = "oven.openai.responses";
