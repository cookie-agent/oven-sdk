//! Internal block state retained while normalizing Anthropic SSE events.

use oven_sdk::JsonValue;

pub(super) enum Block {
    Text {
        text: String,
    },
    Thinking {
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
}
