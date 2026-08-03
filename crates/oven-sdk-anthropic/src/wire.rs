//! Anthropic-compatible Messages wire identifiers and constants.

use oven_sdk::AdapterId;

/// Stable direct Anthropic provider identity.
pub const ANTHROPIC_PROVIDER_ID: &str = "anthropic";
/// Stable MiniMax provider identity.
pub const MINIMAX_PROVIDER_ID: &str = "minimax";
/// Stable Claude Platform on AWS provider identity.
pub const ANTHROPIC_AWS_PROVIDER_ID: &str = "anthropic-aws";
/// Stable adapter identity for direct Anthropic Messages.
pub const ANTHROPIC_MESSAGES_ADAPTER_ID: &str = "oven.anthropic.messages";
/// Stable adapter identity for MiniMax Messages compatibility.
pub const MINIMAX_MESSAGES_ADAPTER_ID: &str = "oven.minimax.messages";
/// Stable adapter identity for Claude Platform on AWS Messages.
pub const ANTHROPIC_AWS_MESSAGES_ADAPTER_ID: &str = "oven.anthropic.aws.messages";

pub(crate) const VERSION: &str = "2023-06-01";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Protocol {
    Anthropic,
    MiniMax,
    AnthropicAws,
}

impl Protocol {
    pub(crate) fn adapter_id(self) -> AdapterId {
        AdapterId::new(match self {
            Self::Anthropic => ANTHROPIC_MESSAGES_ADAPTER_ID,
            Self::MiniMax => MINIMAX_MESSAGES_ADAPTER_ID,
            Self::AnthropicAws => ANTHROPIC_AWS_MESSAGES_ADAPTER_ID,
        })
    }

    pub(crate) fn replay_format(self) -> &'static str {
        match self {
            Self::Anthropic => "oven.anthropic.messages.assistant.v3",
            Self::MiniMax => "oven.minimax.messages.assistant.v3",
            Self::AnthropicAws => "oven.anthropic.aws.messages.assistant.v3",
        }
    }

    pub(crate) fn metadata_namespace(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::MiniMax => "minimax",
            Self::AnthropicAws => "anthropic_aws",
        }
    }

    pub(crate) fn display_name(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic",
            Self::MiniMax => "MiniMax",
            Self::AnthropicAws => "Anthropic AWS",
        }
    }

    pub(crate) fn is_first_party(self) -> bool {
        matches!(self, Self::Anthropic | Self::AnthropicAws)
    }
}
