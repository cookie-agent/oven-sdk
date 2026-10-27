//! Borrowed Messages request body.
//!
//! Every type here borrows from the caller's [`oven_sdk::Request`] (or the validated native
//! replay payload) and is serialized once, straight into the request byte buffer.

use std::borrow::Cow;

use base64::{display::Base64Display, engine::general_purpose::STANDARD};
use oven_sdk::JsonValue;
use serde::{Serialize, Serializer, ser::SerializeMap};

use crate::options::{AnthropicCacheControl, AnthropicCacheTtl};

/// Complete Messages request body.
#[derive(Serialize)]
pub(crate) struct MessagesBody<'a> {
    pub(crate) model: &'a str,
    pub(crate) messages: Vec<Message<'a>>,
    pub(crate) stream: bool,
    pub(crate) max_tokens: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) system: Vec<Block<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tools: Vec<Tool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<ToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thinking: Option<Thinking<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metadata: Option<Metadata<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) output_config: Option<OutputConfig<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) inference_geo: Option<&'a str>,
}

impl MessagesBody<'_> {
    /// Serializes into a buffer pre-sized from a cheap walk of the borrowed content.
    pub(crate) fn to_bytes(&self) -> serde_json::Result<Vec<u8>> {
        let hint = self.size_hint();
        let mut buffer = Vec::with_capacity(hint + hint / 8 + 512);
        serde_json::to_writer(&mut buffer, self)?;
        Ok(buffer)
    }

    fn size_hint(&self) -> usize {
        let messages = self
            .messages
            .iter()
            .map(|message| 32 + blocks_hint(&message.content))
            .sum::<usize>();
        let tools = self
            .tools
            .iter()
            .map(|tool| {
                64 + tool.name.len() + tool.description.len() + json_hint(tool.input_schema)
            })
            .sum::<usize>();
        let schema = self
            .output_config
            .as_ref()
            .and_then(|config| config.format.as_ref())
            .map_or(0, |format| json_hint(format.schema));
        self.model.len() + messages + blocks_hint(&self.system) + tools + schema
    }
}

#[derive(Serialize)]
pub(crate) struct Message<'a> {
    pub(crate) role: &'static str,
    pub(crate) content: Vec<Block<'a>>,
}

#[derive(Serialize)]
pub(crate) struct Tool<'a> {
    pub(crate) name: &'a str,
    pub(crate) description: &'a str,
    pub(crate) input_schema: &'a JsonValue,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) strict: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) defer_loading: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ToolChoice<'a> {
    Auto,
    Any,
    Tool { name: &'a str },
}

#[derive(Serialize)]
pub(crate) struct Thinking<'a> {
    #[serde(rename = "type")]
    pub(crate) mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) budget_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) display: Option<&'a str>,
}

#[derive(Serialize)]
pub(crate) struct Metadata<'a> {
    pub(crate) user_id: &'a str,
}

#[derive(Serialize)]
pub(crate) struct OutputConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) format: Option<OutputFormat<'a>>,
}

#[derive(Serialize)]
pub(crate) struct OutputFormat<'a> {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) schema: &'a JsonValue,
}

/// `{"type":"ephemeral","ttl":...}` cache marker.
#[derive(Clone, Copy, Serialize)]
pub(crate) struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
    ttl: &'static str,
}

impl From<&AnthropicCacheControl> for CacheControl {
    fn from(control: &AnthropicCacheControl) -> Self {
        Self {
            kind: "ephemeral",
            ttl: match control.ttl {
                AnthropicCacheTtl::FiveMinutes => "5m",
                AnthropicCacheTtl::OneHour => "1h",
            },
        }
    }
}

/// One content block: either encoded from normalized parts or a validated native replay block.
#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum Block<'a> {
    Typed(TypedBlock<'a>),
    Native(NativeBlock<'a>),
}

impl<'a> Block<'a> {
    pub(crate) fn block_type(&self) -> Option<&str> {
        match self {
            Self::Typed(block) => Some(block.block_type()),
            Self::Native(block) => block.value.get("type").and_then(JsonValue::as_str),
        }
    }

    pub(crate) fn set_cache(&mut self, cache: CacheControl) {
        match self {
            Self::Typed(block) => block.set_cache(cache),
            Self::Native(block) => block.cache = Some(cache),
        }
    }

    /// Trims trailing whitespace from a text block; non-string native text is left unchanged.
    pub(crate) fn trim_text_end(&mut self) {
        match self {
            Self::Typed(TypedBlock::Text {
                text: TextValue::Str(text),
                ..
            }) => *text = text.trim_end(),
            Self::Typed(_) => {}
            Self::Native(block) => {
                if let Some(text) = block.value.get("text").and_then(JsonValue::as_str) {
                    block.text = Some(text.trim_end());
                }
            }
        }
    }

    fn size_hint(&self) -> usize {
        match self {
            Self::Typed(block) => block.size_hint(),
            Self::Native(block) => 32 + json_map_hint(block.value),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TypedBlock<'a> {
    Text {
        text: TextValue<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: Source<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Document {
        source: Source<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Video {
        source: Source<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: ToolInput<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: ToolResultContent<'a>,
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Loads a deferred tool; only valid as the whole content of a tool result.
    ToolReference { tool_name: &'a str },
}

impl TypedBlock<'_> {
    fn block_type(&self) -> &'static str {
        match self {
            Self::Text { .. } => "text",
            Self::Image { .. } => "image",
            Self::Document { .. } => "document",
            Self::Video { .. } => "video",
            Self::ToolUse { .. } => "tool_use",
            Self::ToolResult { .. } => "tool_result",
            Self::ToolReference { .. } => "tool_reference",
        }
    }

    fn set_cache(&mut self, cache: CacheControl) {
        match self {
            Self::Text { cache_control, .. }
            | Self::Image { cache_control, .. }
            | Self::Document { cache_control, .. }
            | Self::Video { cache_control, .. }
            | Self::ToolUse { cache_control, .. }
            | Self::ToolResult { cache_control, .. } => *cache_control = Some(cache),
            // A reference never carries a marker; the result holding it does.
            Self::ToolReference { .. } => {}
        }
    }

    fn size_hint(&self) -> usize {
        32 + match self {
            Self::Text { text, .. } => text.size_hint(),
            Self::Image { source, .. }
            | Self::Document { source, .. }
            | Self::Video { source, .. } => source.size_hint(),
            Self::ToolUse {
                id, name, input, ..
            } => {
                id.len()
                    + name.len()
                    + match input {
                        ToolInput::Object(value) | ToolInput::Raw { raw: value } => {
                            json_hint(value)
                        }
                    }
            }
            Self::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                tool_use_id.len()
                    + match content {
                        ToolResultContent::Text(text) => text.size_hint(),
                        ToolResultContent::Blocks(blocks) => blocks_hint(blocks),
                    }
            }
            Self::ToolReference { tool_name } => tool_name.len(),
        }
    }
}

/// Text that is either borrowed, rendered from JSON while serializing, or owned.
pub(crate) enum TextValue<'a> {
    Str(&'a str),
    Json(&'a JsonValue),
    Owned(String),
}

impl Serialize for TextValue<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Str(text) => serializer.serialize_str(text),
            // `Display` for JSON values is the compact serialization (`to_string`).
            Self::Json(value) => serializer.collect_str(value),
            Self::Owned(text) => serializer.serialize_str(text),
        }
    }
}

impl TextValue<'_> {
    fn size_hint(&self) -> usize {
        match self {
            Self::Str(text) => text.len(),
            Self::Json(value) => json_hint(value),
            Self::Owned(text) => text.len(),
        }
    }
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum ToolInput<'a> {
    Object(&'a JsonValue),
    Raw {
        #[serde(rename = "rawInvalidInput")]
        raw: &'a JsonValue,
    },
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum ToolResultContent<'a> {
    Text(TextValue<'a>),
    Blocks(Vec<Block<'a>>),
}

/// Media source. MiniMax-style sources may carry extra media controls.
#[derive(Serialize)]
pub(crate) struct Source<'a> {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) media_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) data: Option<SourceData<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) url: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) detail: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) fps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_long_side_pixel: Option<u64>,
}

impl<'a> Source<'a> {
    pub(crate) fn base64(media_type: &'a str, bytes: &'a [u8]) -> Self {
        Self::new(
            "base64",
            Some(media_type),
            Some(SourceData::Base64(bytes)),
            None,
        )
    }

    pub(crate) fn text(media_type: &'a str, text: &'a str) -> Self {
        Self::new("text", Some(media_type), Some(SourceData::Text(text)), None)
    }

    pub(crate) fn url(url: Cow<'a, str>) -> Self {
        Self::new("url", None, None, Some(url))
    }

    fn new(
        kind: &'static str,
        media_type: Option<&'a str>,
        data: Option<SourceData<'a>>,
        url: Option<Cow<'a, str>>,
    ) -> Self {
        Self {
            kind,
            media_type,
            data,
            url,
            detail: None,
            fps: None,
            max_long_side_pixel: None,
        }
    }

    fn size_hint(&self) -> usize {
        let data = match &self.data {
            Some(SourceData::Base64(bytes)) => bytes.len().div_ceil(3) * 4,
            Some(SourceData::Text(text)) => text.len(),
            None => 0,
        };
        64 + data + self.url.as_ref().map_or(0, |url| url.len())
    }
}

/// Inline source data; base64 is encoded directly into the output buffer.
pub(crate) enum SourceData<'a> {
    Base64(&'a [u8]),
    Text(&'a str),
}

impl Serialize for SourceData<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Base64(bytes) => serializer.collect_str(&Base64Display::new(bytes, &STANDARD)),
            Self::Text(text) => serializer.serialize_str(text),
        }
    }
}

/// A validated native replay block, optionally with a cache marker or trimmed prefill text.
pub(crate) struct NativeBlock<'a> {
    pub(crate) value: &'a serde_json::Map<String, JsonValue>,
    pub(crate) cache: Option<CacheControl>,
    pub(crate) text: Option<&'a str>,
}

impl Serialize for NativeBlock<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for (key, value) in self.value {
            match (key.as_str(), self.text) {
                ("cache_control", _) if self.cache.is_some() => {}
                ("text", Some(text)) => map.serialize_entry(key, text)?,
                _ => map.serialize_entry(key, value)?,
            }
        }
        if let Some(cache) = &self.cache {
            map.serialize_entry("cache_control", cache)?;
        }
        map.end()
    }
}

fn blocks_hint(blocks: &[Block<'_>]) -> usize {
    blocks.iter().map(Block::size_hint).sum()
}

/// Approximate serialized size of a JSON value, without allocating.
fn json_hint(value: &JsonValue) -> usize {
    match value {
        JsonValue::Null | JsonValue::Bool(_) => 5,
        JsonValue::Number(_) => 8,
        JsonValue::String(text) => text.len() + 2,
        JsonValue::Array(values) => {
            2 + values
                .iter()
                .map(|value| json_hint(value) + 1)
                .sum::<usize>()
        }
        JsonValue::Object(map) => json_map_hint(map),
    }
}

fn json_map_hint(map: &serde_json::Map<String, JsonValue>) -> usize {
    2 + map
        .iter()
        .map(|(key, value)| key.len() + 4 + json_hint(value))
        .sum::<usize>()
}
