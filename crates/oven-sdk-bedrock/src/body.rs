//! Borrowed Converse request body.
//!
//! Every type here borrows from the caller's [`oven_sdk::Request`], its typed Bedrock options,
//! or the validated native replay payload, and is serialized once into the byte buffer that is
//! both signed and sent.

use std::collections::BTreeMap;

use base64::{display::Base64Display, engine::general_purpose::STANDARD};
use oven_sdk::JsonValue;
use serde::{Serialize, Serializer, ser::SerializeMap};

use crate::options::{BedrockCachePoint, BedrockCacheTtl};

/// Complete Converse / ConverseStream request body.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ConverseBody<'a> {
    pub(crate) messages: Vec<Message<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) system: Vec<Block<'a>>,
    #[serde(skip_serializing_if = "InferenceConfig::is_empty")]
    pub(crate) inference_config: InferenceConfig<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_config: Option<ToolConfig<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) output_config: Option<OutputConfig<'a>>,
    #[serde(skip_serializing_if = "AdditionalFields::is_empty")]
    pub(crate) additional_model_request_fields: AdditionalFields<'a>,
    #[serde(skip_serializing_if = "is_empty_slice")]
    pub(crate) additional_model_response_field_paths: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) service_tier: Option<ServiceTier<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) performance_config: Option<PerformanceConfig<'a>>,
    #[serde(skip_serializing_if = "is_empty_map")]
    pub(crate) request_metadata: &'a BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) guardrail_config: Option<GuardrailConfig<'a>>,
}

impl ConverseBody<'_> {
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
            .tool_config
            .as_ref()
            .map_or(0, |config| blocks_hint(&config.tools));
        let schema = self.output_config.as_ref().map_or(0, |config| {
            json_hint(config.text_format.structure.json_schema.schema.0)
        });
        let additional = self
            .additional_model_request_fields
            .fields
            .map_or(0, json_map_hint);
        messages + blocks_hint(&self.system) + tools + schema + additional
    }
}

/// Appends content to the previous message when the role repeats, as Converse requires.
pub(crate) fn push_message<'a>(
    messages: &mut Vec<Message<'a>>,
    role: &'static str,
    content: Vec<Block<'a>>,
) {
    if content.is_empty() {
        return;
    }
    match messages.last_mut() {
        Some(last) if last.role == role => last.content.extend(content),
        _ => messages.push(Message { role, content }),
    }
}

#[derive(Serialize)]
pub(crate) struct Message<'a> {
    pub(crate) role: &'static str,
    pub(crate) content: Vec<Block<'a>>,
}

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InferenceConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    pub(crate) top_p: Option<f64>,
    #[serde(skip_serializing_if = "is_empty_slice")]
    pub(crate) stop_sequences: &'a [String],
}

impl InferenceConfig<'_> {
    fn is_empty(&self) -> bool {
        self.max_tokens.is_none()
            && self.temperature.is_none()
            && self.top_p.is_none()
            && self.stop_sequences.is_empty()
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ToolConfig<'a> {
    pub(crate) tools: Vec<Block<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<ToolChoice<'a>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum ToolChoice<'a> {
    Auto {},
    Any {},
    Tool { name: &'a str },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OutputConfig<'a> {
    pub(crate) text_format: TextFormat<'a>,
}

impl<'a> OutputConfig<'a> {
    pub(crate) fn json_schema(schema: &'a JsonValue) -> Self {
        Self {
            text_format: TextFormat {
                kind: "json_schema",
                structure: Structure {
                    json_schema: JsonSchemaFormat {
                        schema: JsonString(schema),
                    },
                },
            },
        }
    }
}

#[derive(Serialize)]
pub(crate) struct TextFormat<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    structure: Structure<'a>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Structure<'a> {
    json_schema: JsonSchemaFormat<'a>,
}

#[derive(Serialize)]
pub(crate) struct JsonSchemaFormat<'a> {
    schema: JsonString<'a>,
}

/// A JSON value serialized as its compact JSON text (the same bytes as `to_string`).
pub(crate) struct JsonString<'a>(&'a JsonValue);

impl Serialize for JsonString<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self.0)
    }
}

/// Caller-provided additional model fields plus the typed reasoning controls.
///
/// Reserved reasoning keys are rejected in caller fields during validation, so the typed
/// entries never collide with caller entries.
#[derive(Default)]
pub(crate) struct AdditionalFields<'a> {
    pub(crate) fields: Option<&'a serde_json::Map<String, JsonValue>>,
    pub(crate) thinking: Option<Thinking<'a>>,
    pub(crate) output_config: Option<Effort<'a>>,
    pub(crate) reasoning_effort: Option<&'a str>,
    pub(crate) reasoning_config: Option<ReasoningConfig<'a>>,
}

impl AdditionalFields<'_> {
    fn is_empty(&self) -> bool {
        self.fields.is_none_or(serde_json::Map::is_empty)
            && self.thinking.is_none()
            && self.output_config.is_none()
            && self.reasoning_effort.is_none()
            && self.reasoning_config.is_none()
    }
}

impl Serialize for AdditionalFields<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for (key, value) in self.fields.into_iter().flatten() {
            map.serialize_entry(key, value)?;
        }
        if let Some(thinking) = &self.thinking {
            map.serialize_entry("thinking", thinking)?;
        }
        if let Some(effort) = &self.output_config {
            map.serialize_entry("output_config", effort)?;
        }
        if let Some(effort) = self.reasoning_effort {
            map.serialize_entry("reasoning_effort", effort)?;
        }
        if let Some(config) = &self.reasoning_config {
            map.serialize_entry("reasoningConfig", config)?;
        }
        map.end()
    }
}

/// Anthropic `thinking` field.
#[derive(Serialize)]
pub(crate) struct Thinking<'a> {
    #[serde(rename = "type")]
    pub(crate) kind: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) budget_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) display: Option<&'a str>,
}

/// Anthropic `output_config` effort field.
#[derive(Serialize)]
pub(crate) struct Effort<'a> {
    pub(crate) effort: &'a str,
}

/// Bedrock `reasoningConfig` field.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ReasoningConfig<'a> {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub(crate) kind: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) budget_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_reasoning_effort: Option<&'a str>,
}

#[derive(Serialize)]
pub(crate) struct ServiceTier<'a> {
    #[serde(rename = "type")]
    pub(crate) kind: &'a str,
}

#[derive(Serialize)]
pub(crate) struct PerformanceConfig<'a> {
    pub(crate) latency: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GuardrailConfig<'a> {
    pub(crate) guardrail_identifier: &'a str,
    pub(crate) guardrail_version: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) trace: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream_processing_mode: Option<&'a str>,
}

/// One Converse content block (a single-key object), or a validated native replay block.
pub(crate) enum Block<'a> {
    Text(&'a str),
    Json(&'a JsonValue),
    Image(Media<'a>),
    Video(Media<'a>),
    Document(Document<'a>),
    ToolUse(ToolUse<'a>),
    ToolResult(ToolResult<'a>),
    ToolSpec(ToolSpec<'a>),
    CachePoint(CachePoint),
    Native(&'a serde_json::Map<String, JsonValue>),
}

impl Serialize for Block<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Self::Native(block) = self {
            return block.serialize(serializer);
        }
        let mut map = serializer.serialize_map(Some(1))?;
        match self {
            Self::Text(text) => map.serialize_entry("text", text)?,
            Self::Json(value) => map.serialize_entry("json", value)?,
            Self::Image(media) => map.serialize_entry("image", media)?,
            Self::Video(media) => map.serialize_entry("video", media)?,
            Self::Document(document) => map.serialize_entry("document", document)?,
            Self::ToolUse(tool_use) => map.serialize_entry("toolUse", tool_use)?,
            Self::ToolResult(result) => map.serialize_entry("toolResult", result)?,
            Self::ToolSpec(spec) => map.serialize_entry("toolSpec", spec)?,
            Self::CachePoint(point) => map.serialize_entry("cachePoint", point)?,
            Self::Native(_) => unreachable!("native blocks are serialized above"),
        }
        map.end()
    }
}

impl Block<'_> {
    fn size_hint(&self) -> usize {
        16 + match self {
            Self::Text(text) => text.len(),
            Self::Json(value) => json_hint(value),
            Self::Image(media) | Self::Video(media) => media.source.size_hint(),
            Self::Document(document) => document.name.len() + document.source.size_hint(),
            Self::ToolUse(tool_use) => {
                tool_use.tool_use_id.len() + tool_use.name.len() + json_hint(tool_use.input)
            }
            Self::ToolResult(result) => {
                48 + result.tool_use_id.len() + blocks_hint(&result.content)
            }
            Self::ToolSpec(spec) => {
                48 + spec.name.len() + spec.description.len() + json_hint(spec.input_schema.json)
            }
            Self::CachePoint(_) => 32,
            Self::Native(block) => json_map_hint(block),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct Media<'a> {
    pub(crate) format: &'static str,
    pub(crate) source: Source<'a>,
}

#[derive(Serialize)]
pub(crate) struct Document<'a> {
    pub(crate) format: &'static str,
    pub(crate) name: String,
    pub(crate) source: Source<'a>,
}

/// Media source: inline bytes (base64-encoded into the buffer), inline text, or S3.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum Source<'a> {
    Bytes(Base64<'a>),
    Text(&'a str),
    S3Location(S3Location<'a>),
}

impl Source<'_> {
    fn size_hint(&self) -> usize {
        32 + match self {
            Self::Bytes(Base64(bytes)) => bytes.len().div_ceil(3) * 4,
            Self::Text(text) => text.len(),
            Self::S3Location(location) => location.uri.len(),
        }
    }
}

pub(crate) struct Base64<'a>(pub(crate) &'a [u8]);

impl Serialize for Base64<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&Base64Display::new(self.0, &STANDARD))
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct S3Location<'a> {
    pub(crate) uri: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bucket_owner: Option<&'a str>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ToolUse<'a> {
    pub(crate) tool_use_id: &'a str,
    pub(crate) name: &'a str,
    pub(crate) input: &'a JsonValue,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ToolResult<'a> {
    pub(crate) tool_use_id: &'a str,
    pub(crate) content: Vec<Block<'a>>,
    pub(crate) status: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ToolSpec<'a> {
    pub(crate) name: &'a str,
    pub(crate) description: &'a str,
    pub(crate) input_schema: InputSchema<'a>,
}

#[derive(Serialize)]
pub(crate) struct InputSchema<'a> {
    pub(crate) json: &'a JsonValue,
}

/// `{"type":"default","ttl":...}` cache point.
#[derive(Clone, Copy, Serialize)]
pub(crate) struct CachePoint {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<BedrockCacheTtl>,
}

impl From<&BedrockCachePoint> for CachePoint {
    fn from(point: &BedrockCachePoint) -> Self {
        Self {
            kind: "default",
            ttl: point.ttl,
        }
    }
}

fn is_empty_slice(value: &&[String]) -> bool {
    value.is_empty()
}

fn is_empty_map(value: &&BTreeMap<String, String>) -> bool {
    value.is_empty()
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
