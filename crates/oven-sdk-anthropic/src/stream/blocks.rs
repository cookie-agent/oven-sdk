//! Internal block state retained while normalizing Anthropic SSE events.

use oven_sdk::JsonValue;

pub(super) enum Block {
    Text {
        /// Normalized part ID, cached so each delta clones instead of formatting.
        id: String,
        text: String,
    },
    Thinking {
        /// Normalized `thinking:{index}` delta ID.
        id: String,
        redacted: bool,
        text: String,
        data: Option<JsonValue>,
        signature: Option<String>,
    },
    Tool {
        id: String,
        name: String,
        input: String,
    },
    Custom,
}
