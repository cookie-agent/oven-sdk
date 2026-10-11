//! Borrowed Gemini `generateContent` request body.
//!
//! Every type here borrows from the caller's [`oven_sdk::Request`] (or a validated native
//! replay payload) and is serialized once, straight into the request byte buffer.

use base64::{display::Base64Display, engine::general_purpose::STANDARD};
use oven_sdk::{ContentValue, JsonValue};
use serde::{Serialize, Serializer, ser::SerializeMap, ser::SerializeSeq};

use crate::options::{GoogleSafetySetting, GoogleThinkingConfig};

pub(crate) type JsonMap = serde_json::Map<String, JsonValue>;

/// Complete `generateContent` request body.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Body<'a> {
    pub(crate) contents: Vec<Content<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) system_instruction: Option<Parts<'a>>,
    #[serde(skip_serializing_if = "GenerationConfig::is_empty")]
    pub(crate) generation_config: GenerationConfig<'a>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tools: Vec<Tool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_config: Option<ToolConfig<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cached_content: Option<&'a str>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub(crate) safety_settings: &'a [GoogleSafetySetting],
}

impl Body<'_> {
    /// Serializes into a buffer pre-sized from a cheap walk of the borrowed content.
    pub(crate) fn to_bytes(&self) -> serde_json::Result<Vec<u8>> {
        let hint = self.size_hint();
        let mut buffer = Vec::with_capacity(hint + hint / 8 + 512);
        serde_json::to_writer(&mut buffer, self)?;
        Ok(buffer)
    }

    fn size_hint(&self) -> usize {
        let contents = self
            .contents
            .iter()
            .map(|content| 32 + parts_hint(&content.parts))
            .sum::<usize>();
        let system = self
            .system_instruction
            .as_ref()
            .map_or(0, |system| parts_hint(&system.parts));
        let tools = self
            .tools
            .iter()
            .map(|tool| match tool {
                Tool::FunctionDeclarations(declarations) => declarations
                    .iter()
                    .map(|declaration| {
                        64 + declaration.name.len()
                            + declaration.description.len()
                            + json_hint(declaration.parameters)
                    })
                    .sum::<usize>(),
                _ => 64,
            })
            .sum::<usize>();
        let schema = self
            .generation_config
            .response_format
            .as_ref()
            .and_then(|format| format.text.schema)
            .map_or(0, json_hint);
        contents + system + tools + schema
    }
}

#[derive(Serialize)]
pub(crate) struct Content<'a> {
    pub(crate) role: &'static str,
    pub(crate) parts: Vec<Part<'a>>,
}

#[derive(Serialize)]
pub(crate) struct Parts<'a> {
    pub(crate) parts: Vec<Part<'a>>,
}

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerationConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_k: Option<u32>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub(crate) stop_sequences: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) frequency_penalty: Option<f64>,
    /// Serialized with its own derive, so unset fields stay explicit nulls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thinking_config: Option<GoogleThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) response_format: Option<ResponseFormat<'a>>,
}

impl GenerationConfig<'_> {
    fn is_empty(&self) -> bool {
        self.max_output_tokens.is_none()
            && self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.stop_sequences.is_empty()
            && self.seed.is_none()
            && self.presence_penalty.is_none()
            && self.frequency_penalty.is_none()
            && self.thinking_config.is_none()
            && self.response_format.is_none()
    }
}

#[derive(Serialize)]
pub(crate) struct ResponseFormat<'a> {
    pub(crate) text: ResponseFormatText<'a>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ResponseFormatText<'a> {
    pub(crate) mime_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) schema: Option<&'a JsonValue>,
}

/// `{}` placeholder for provider tools without configuration.
#[derive(Serialize)]
pub(crate) struct Empty {}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum Tool<'a> {
    GoogleSearch(Empty),
    UrlContext(Empty),
    CodeExecution(Empty),
    FileSearch {
        #[serde(rename = "fileSearchStoreNames")]
        stores: &'a [String],
    },
    GoogleMaps(Empty),
    FunctionDeclarations(Vec<FunctionDeclaration<'a>>),
}

#[derive(Serialize)]
pub(crate) struct FunctionDeclaration<'a> {
    pub(crate) name: &'a str,
    pub(crate) description: &'a str,
    pub(crate) parameters: &'a JsonValue,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ToolConfig<'a> {
    pub(crate) function_calling_config: FunctionCallingConfig<'a>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) include_server_side_tool_invocations: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FunctionCallingConfig<'a> {
    pub(crate) mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) allowed_function_names: Option<[&'a str; 1]>,
}

/// One content part: encoded from normalized history or a validated native replay part.
pub(crate) enum Part<'a> {
    Text(&'a str),
    InlineData {
        mime_type: &'a str,
        data: &'a [u8],
    },
    FileData {
        mime_type: &'a str,
        file_uri: &'a str,
    },
    FunctionCall {
        id: &'a str,
        name: &'a str,
        args: &'a JsonValue,
        thought_signature: Option<&'static str>,
    },
    FunctionResponse {
        id: Option<&'a str>,
        name: &'a str,
        is_error: bool,
        output: ToolOutput<'a>,
    },
    /// A validated native part; `thoughtSignature` is dropped when replay is ineligible.
    Native {
        value: &'a JsonMap,
        strip_signature: bool,
    },
}

impl<'a> Part<'a> {
    /// `Some` for a function-call part, carrying its wire `functionCall.id` when it is a string.
    pub(crate) fn function_call_id(&self) -> Option<Option<&'a str>> {
        match *self {
            Self::FunctionCall { id, .. } => Some(Some(id)),
            Self::Native { value, .. } => value
                .get("functionCall")
                .map(|call| call.get("id").and_then(JsonValue::as_str)),
            _ => None,
        }
    }

    fn size_hint(&self) -> usize {
        32 + match self {
            Self::Text(text) => text.len(),
            Self::InlineData { mime_type, data } => mime_type.len() + data.len().div_ceil(3) * 4,
            Self::FileData {
                mime_type,
                file_uri,
            } => mime_type.len() + file_uri.len(),
            Self::FunctionCall { id, name, args, .. } => {
                64 + id.len() + name.len() + json_hint(args)
            }
            Self::FunctionResponse {
                id, name, output, ..
            } => 64 + id.map_or(0, str::len) + name.len() + output.size_hint(),
            Self::Native { value, .. } => json_map_hint(value),
        }
    }
}

impl Serialize for Part<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Text(text) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("text", text)?;
                map.end()
            }
            Self::InlineData { mime_type, data } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(
                    "inlineData",
                    &InlineData {
                        mime_type,
                        data: Base64(data),
                    },
                )?;
                map.end()
            }
            Self::FileData {
                mime_type,
                file_uri,
            } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(
                    "fileData",
                    &FileData {
                        mime_type,
                        file_uri,
                    },
                )?;
                map.end()
            }
            Self::FunctionCall {
                id,
                name,
                args,
                thought_signature,
            } => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("functionCall", &FunctionCall { id, name, args })?;
                if let Some(signature) = thought_signature {
                    map.serialize_entry("thoughtSignature", signature)?;
                }
                map.end()
            }
            Self::FunctionResponse {
                id,
                name,
                is_error,
                output,
            } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(
                    "functionResponse",
                    &FunctionResponse {
                        name,
                        response: FunctionResponseBody {
                            is_error: *is_error,
                            output,
                        },
                        id: *id,
                    },
                )?;
                map.end()
            }
            Self::Native {
                value,
                strip_signature,
            } => {
                let mut map = serializer.serialize_map(None)?;
                for (key, value) in *value {
                    if !(*strip_signature && key == "thoughtSignature") {
                        map.serialize_entry(key, value)?;
                    }
                }
                map.end()
            }
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InlineData<'a> {
    mime_type: &'a str,
    data: Base64<'a>,
}

/// Inline bytes, base64-encoded directly into the output buffer.
struct Base64<'a>(&'a [u8]);

impl Serialize for Base64<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&Base64Display::new(self.0, &STANDARD))
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileData<'a> {
    mime_type: &'a str,
    file_uri: &'a str,
}

#[derive(Serialize)]
struct FunctionCall<'a> {
    id: &'a str,
    name: &'a str,
    args: &'a JsonValue,
}

#[derive(Serialize)]
struct FunctionResponse<'a> {
    name: &'a str,
    response: FunctionResponseBody<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
}

/// `{"output":...}` or `{"error":...}`.
struct FunctionResponseBody<'a> {
    is_error: bool,
    output: &'a ToolOutput<'a>,
}

impl Serialize for FunctionResponseBody<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(if self.is_error { "error" } else { "output" }, self.output)?;
        map.end()
    }
}

/// Function result content, borrowed from the normalized tool result.
pub(crate) enum ToolOutput<'a> {
    Str(&'a str),
    Json(&'a JsonValue),
    /// Text and JSON values only; files are rejected while building the body.
    Mixed(&'a [ContentValue]),
}

impl ToolOutput<'_> {
    fn size_hint(&self) -> usize {
        match self {
            Self::Str(text) => text.len(),
            Self::Json(value) => json_hint(value),
            Self::Mixed(values) => values
                .iter()
                .map(|value| match value {
                    ContentValue::Text(text) => text.len() + 3,
                    ContentValue::Json(value) => json_hint(value) + 1,
                    ContentValue::File(_) => 0,
                })
                .sum(),
        }
    }
}

impl Serialize for ToolOutput<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Str(text) => serializer.serialize_str(text),
            Self::Json(value) => value.serialize(serializer),
            Self::Mixed(values) => {
                let mut seq = serializer.serialize_seq(Some(values.len()))?;
                for value in *values {
                    match value {
                        ContentValue::Text(text) => seq.serialize_element(text)?,
                        ContentValue::Json(value) => seq.serialize_element(value)?,
                        ContentValue::File(_) => {
                            return Err(serde::ser::Error::custom(
                                "multimodal Gemini function results are not enabled",
                            ));
                        }
                    }
                }
                seq.end()
            }
        }
    }
}

fn parts_hint(parts: &[Part<'_>]) -> usize {
    parts.iter().map(Part::size_hint).sum()
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

fn json_map_hint(map: &JsonMap) -> usize {
    2 + map
        .iter()
        .map(|(key, value)| key.len() + 4 + json_hint(value))
        .sum::<usize>()
}
