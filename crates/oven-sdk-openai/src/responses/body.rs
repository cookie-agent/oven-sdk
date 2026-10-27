//! Borrowed Responses request-body wire structs.

use std::borrow::Cow;

use oven_sdk::JsonValue;
use serde::Serialize;

use crate::{
    options::{OpenAiPromptCacheOptions, OpenAiPromptCacheRetention},
    wire::encode::{Breakpoint, FileRef, Text, json_size_hint},
};

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
    pub(crate) user: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) truncation: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_retention: Option<OpenAiPromptCacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_options: Option<&'a OpenAiPromptCacheOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<FunctionTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<ToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<TextConfig<'a>>,
}

/// `POST /responses/compact` body.
#[derive(Serialize)]
pub(crate) struct CompactionBody<'a> {
    pub(crate) model: &'a str,
    pub(crate) input: Vec<Item<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_options: Option<&'a OpenAiPromptCacheOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_retention: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) service_tier: Option<&'a str>,
}

#[derive(Serialize)]
pub(crate) struct Reasoning<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) summary: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mode: Option<&'a str>,
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

impl FunctionTool<'_> {
    pub(crate) fn size_hint(&self) -> usize {
        self.name.len() + self.description.len() + json_size_hint(self.parameters) + 64
    }
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
/// validated provider-native item borrowed from replay or compaction state.
#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum Item<'a> {
    Typed(TypedItem<'a>),
    Native(&'a JsonValue),
    /// A provider-native item adjusted for this request.
    NativeOwned(JsonValue),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TypedItem<'a> {
    Message {
        role: &'static str,
        content: Vec<Content<'a>>,
    },
    FunctionCall {
        call_id: &'a str,
        name: &'a str,
        arguments: Text<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<&'a str>,
        /// The namespace a tool loaded by `additional_tools` lives in.
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<&'a str>,
    },
    FunctionCallOutput {
        call_id: &'a str,
        output: Output<'a>,
    },
    /// Makes deferred tools callable from this point of the input on.
    AdditionalTools {
        role: &'static str,
        tools: Vec<FunctionTool<'a>>,
    },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Content<'a> {
    InputText {
        text: Text<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<Breakpoint>,
    },
    InputImage {
        image_url: FileRef<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<Breakpoint>,
    },
    InputFile {
        filename: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<FileRef<'a>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<Breakpoint>,
    },
    OutputText {
        text: Cow<'a, str>,
    },
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum Output<'a> {
    Text(Text<'a>),
    Parts(Vec<Content<'a>>),
}

impl Item<'_> {
    pub(crate) fn message<'a>(role: &'static str, content: Vec<Content<'a>>) -> Item<'a> {
        Item::Typed(TypedItem::Message { role, content })
    }

    pub(crate) fn size_hint(&self) -> usize {
        match self {
            Self::Native(value) => json_size_hint(value),
            Self::NativeOwned(value) => json_size_hint(value),
            Self::Typed(TypedItem::Message { content, .. }) => {
                48 + content.iter().map(Content::size_hint).sum::<usize>()
            }
            Self::Typed(TypedItem::FunctionCall {
                call_id,
                name,
                arguments,
                id,
                namespace,
            }) => {
                80 + call_id.len()
                    + name.len()
                    + arguments.size_hint()
                    + id.map_or(0, |id| id.len() + 8)
                    + namespace.map_or(0, |namespace| namespace.len() + 16)
            }
            Self::Typed(TypedItem::AdditionalTools { tools, .. }) => {
                64 + tools.iter().map(FunctionTool::size_hint).sum::<usize>()
            }
            Self::Typed(TypedItem::FunctionCallOutput { call_id, output }) => {
                64 + call_id.len()
                    + match output {
                        Output::Text(text) => text.size_hint(),
                        Output::Parts(parts) => parts.iter().map(Content::size_hint).sum(),
                    }
            }
        }
    }
}

impl Content<'_> {
    fn size_hint(&self) -> usize {
        40 + match self {
            Self::InputText { text, .. } => text.size_hint(),
            Self::InputImage { image_url, .. } => image_url.size_hint(),
            Self::InputFile {
                filename,
                file_data,
                file_url,
                ..
            } => {
                filename.len()
                    + file_data.as_ref().map_or(0, FileRef::size_hint)
                    + file_url.map_or(0, str::len)
            }
            Self::OutputText { text } => text.len(),
        }
    }
}
