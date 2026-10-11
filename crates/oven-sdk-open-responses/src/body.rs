//! Borrowed Open Responses request-body wire structs.
//!
//! The body borrows the caller's history and is serialized once, straight
//! into the outgoing byte buffer.

use std::{collections::BTreeMap, fmt};

use base64::{display::Base64Display, engine::general_purpose::STANDARD};
use oven_sdk::JsonValue;
use serde::{Serialize, Serializer};

/// `POST /responses` body.
#[derive(Serialize)]
pub(crate) struct Body<'a> {
    pub(crate) model: &'a str,
    pub(crate) input: Vec<Item<'a>>,
    pub(crate) stream: bool,
    pub(crate) store: bool,
    pub(crate) include: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<Reasoning<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metadata: Option<&'a BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) truncation: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tool_calls: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) safety_identifier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<FunctionTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<ToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<TextConfig<'a>>,
}

#[derive(Serialize)]
pub(crate) struct Reasoning<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) summary: Option<&'a str>,
}

#[derive(Serialize)]
pub(crate) struct FunctionTool<'a> {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) name: &'a str,
    pub(crate) description: &'a str,
    pub(crate) parameters: &'a JsonValue,
    pub(crate) strict: bool,
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum ToolChoice<'a> {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        name: &'a str,
    },
}

#[derive(Serialize)]
pub(crate) struct TextConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) format: Option<TextFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) verbosity: Option<&'a str>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TextFormat<'a> {
    JsonObject,
    JsonSchema {
        name: &'static str,
        strict: bool,
        schema: &'a JsonValue,
    },
}

/// One `input` entry: a typed item built from normalized history, or a
/// validated provider item borrowed from replay state or custom history.
#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum Item<'a> {
    Typed(TypedItem<'a>),
    Native(&'a JsonValue),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TypedItem<'a> {
    Message {
        role: &'static str,
        content: Vec<Content<'a>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        phase: Option<&'a str>,
    },
    FunctionCall {
        call_id: &'a str,
        name: &'a str,
        arguments: Text<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<&'a str>,
        status: &'static str,
    },
    FunctionCallOutput {
        call_id: &'a str,
        output: Output<'a>,
        status: &'static str,
    },
    Reasoning {
        summary: Vec<TypedText<'a>>,
        /// `None` serializes as `null`.
        content: Option<Vec<TypedText<'a>>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<Null>,
        status: &'static str,
    },
}

/// Serializes as JSON `null`.
#[derive(Serialize)]
pub(crate) struct Null;

#[derive(Serialize)]
pub(crate) struct TypedText<'a> {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) text: &'a str,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Content<'a> {
    InputText {
        text: Text<'a>,
    },
    InputImage {
        image_url: FileRef<'a>,
        detail: &'static str,
    },
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<FileRef<'a>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<&'a str>,
    },
    OutputText {
        text: &'a str,
        annotations: Vec<&'a BTreeMap<String, JsonValue>>,
    },
    Refusal {
        refusal: &'a str,
    },
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum Output<'a> {
    Text(Text<'a>),
    Parts(Vec<Content<'a>>),
}

/// A JSON string whose contents are borrowed, owned, or rendered from a JSON
/// value (`value.to_string()` without the intermediate allocation).
pub(crate) enum Text<'a> {
    Borrowed(&'a str),
    Json(&'a JsonValue),
}

impl Serialize for Text<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Borrowed(value) => serializer.serialize_str(value),
            Self::Json(value) => serializer.collect_str(value),
        }
    }
}

impl Text<'_> {
    fn size_hint(&self) -> usize {
        match self {
            Self::Borrowed(value) => value.len() + 2,
            Self::Json(value) => json_size_hint(value) * 5 / 4,
        }
    }
}

/// A caller URL or inline bytes rendered as a base64 `data:` URL while the
/// body is serialized.
pub(crate) enum FileRef<'a> {
    Url(&'a str),
    Data { media_type: &'a str, data: &'a [u8] },
}

impl FileRef<'_> {
    fn size_hint(&self) -> usize {
        match self {
            Self::Url(url) => url.len() + 2,
            Self::Data { media_type, data } => media_type.len() + 16 + data.len().div_ceil(3) * 4,
        }
    }
}

impl Serialize for FileRef<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct DataUrl<'a>(&'a str, &'a [u8]);
        impl fmt::Display for DataUrl<'_> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "data:{};base64,{}",
                    self.0,
                    Base64Display::new(self.1, &STANDARD)
                )
            }
        }
        match self {
            Self::Url(url) => serializer.serialize_str(url),
            Self::Data { media_type, data } => serializer.collect_str(&DataUrl(media_type, data)),
        }
    }
}

impl Item<'_> {
    pub(crate) fn size_hint(&self) -> usize {
        match self {
            Self::Native(value) => json_size_hint(value),
            Self::Typed(TypedItem::Message { content, .. }) => {
                64 + content.iter().map(Content::size_hint).sum::<usize>()
            }
            Self::Typed(TypedItem::FunctionCall {
                call_id,
                name,
                arguments,
                id,
                ..
            }) => {
                96 + call_id.len()
                    + name.len()
                    + arguments.size_hint()
                    + id.map_or(0, |id| id.len() + 8)
            }
            Self::Typed(TypedItem::FunctionCallOutput {
                call_id, output, ..
            }) => {
                80 + call_id.len()
                    + match output {
                        Output::Text(text) => text.size_hint(),
                        Output::Parts(parts) => parts.iter().map(Content::size_hint).sum(),
                    }
            }
            Self::Typed(TypedItem::Reasoning {
                summary, content, ..
            }) => {
                96 + summary
                    .iter()
                    .chain(content.iter().flatten())
                    .map(|text| text.text.len() + 40)
                    .sum::<usize>()
            }
        }
    }
}

impl Content<'_> {
    fn size_hint(&self) -> usize {
        40 + match self {
            Self::InputText { text } => text.size_hint(),
            Self::InputImage { image_url, .. } => image_url.size_hint(),
            Self::InputFile {
                filename,
                file_data,
                file_url,
            } => {
                filename.map_or(0, str::len)
                    + file_data.as_ref().map_or(0, FileRef::size_hint)
                    + file_url.map_or(0, str::len)
            }
            Self::OutputText { text, annotations } => text.len() + annotations.len() * 128,
            Self::Refusal { refusal } => refusal.len(),
        }
    }
}

/// Rough serialized size of a JSON value, used only to pre-size buffers.
pub(crate) fn json_size_hint(value: &JsonValue) -> usize {
    match value {
        JsonValue::Null | JsonValue::Bool(_) => 5,
        JsonValue::Number(_) => 8,
        JsonValue::String(value) => value.len() + 2,
        JsonValue::Array(values) => {
            values
                .iter()
                .map(|value| json_size_hint(value) + 1)
                .sum::<usize>()
                + 2
        }
        JsonValue::Object(object) => {
            object
                .iter()
                .map(|(key, value)| key.len() + 4 + json_size_hint(value))
                .sum::<usize>()
                + 2
        }
    }
}
