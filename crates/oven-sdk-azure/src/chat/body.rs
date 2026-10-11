//! Borrowed Chat Completions request-body wire structs.

use std::borrow::Cow;

use oven_sdk::{JsonValue, replay::ChatReplayMessage};
use serde::Serialize;

use crate::{
    encode::{Breakpoint, FileRef, Text, json_size_hint},
    options::{AzureOpenAiPromptCacheOptions, AzureOpenAiPromptCacheRetention},
};

/// `POST /chat/completions` body.
#[derive(Serialize)]
pub(crate) struct Body<'a> {
    pub(crate) model: &'a str,
    pub(crate) messages: Vec<Message<'a>>,
    pub(crate) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) user: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) verbosity: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_retention: Option<AzureOpenAiPromptCacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_options: Option<&'a AzureOpenAiPromptCacheOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<Tool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<ToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) response_format: Option<ResponseFormat<'a>>,
}

#[derive(Serialize)]
pub(crate) struct StreamOptions {
    pub(crate) include_usage: bool,
}

#[derive(Serialize)]
pub(crate) struct Tool<'a> {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) function: Function<'a>,
}

#[derive(Serialize)]
pub(crate) struct Function<'a> {
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
        function: Named<'a>,
    },
}

#[derive(Serialize)]
pub(crate) struct Named<'a> {
    pub(crate) name: &'a str,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponseFormat<'a> {
    JsonObject,
    JsonSchema { json_schema: JsonSchemaFormat<'a> },
}

#[derive(Serialize)]
pub(crate) struct JsonSchemaFormat<'a> {
    pub(crate) name: &'static str,
    pub(crate) strict: bool,
    pub(crate) schema: &'a JsonValue,
}

/// One `messages` entry: built from normalized history, or a validated
/// replayed assistant message borrowed from its artifact.
#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum Message<'a> {
    Typed(TypedMessage<'a>),
    Replayed(ChatReplayMessage<'a>),
}

#[derive(Serialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub(crate) enum TypedMessage<'a> {
    System {
        content: Content<'a>,
    },
    Developer {
        content: Content<'a>,
    },
    User {
        content: Content<'a>,
    },
    Assistant {
        content: Option<Cow<'a, str>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall<'a>>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<Cow<'a, str>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<Cow<'a, str>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<Cow<'a, str>>,
    },
    Tool {
        tool_call_id: &'a str,
        content: Text<'a>,
    },
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum Content<'a> {
    Text(Cow<'a, str>),
    Parts(Vec<Part<'a>>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Part<'a> {
    Text {
        text: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<Breakpoint>,
    },
    ImageUrl {
        image_url: Url<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<Breakpoint>,
    },
    File {
        file: File<'a>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<Breakpoint>,
    },
}

#[derive(Serialize)]
pub(crate) struct Url<'a> {
    pub(crate) url: FileRef<'a>,
}

#[derive(Serialize)]
pub(crate) struct File<'a> {
    pub(crate) filename: &'a str,
    pub(crate) file_data: FileRef<'a>,
}

#[derive(Serialize)]
pub(crate) struct ToolCall<'a> {
    pub(crate) id: &'a str,
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) function: CallFunction<'a>,
}

#[derive(Serialize)]
pub(crate) struct CallFunction<'a> {
    pub(crate) name: &'a str,
    pub(crate) arguments: Text<'a>,
}

impl Message<'_> {
    pub(crate) fn size_hint(&self) -> usize {
        match self {
            // Sized by the encoder from the normalized parts it matched.
            Self::Replayed(_) => 0,
            Self::Typed(
                TypedMessage::System { content }
                | TypedMessage::Developer { content }
                | TypedMessage::User { content },
            ) => 32 + content.size_hint(),
            Self::Typed(TypedMessage::Assistant {
                content,
                tool_calls,
                reasoning_content,
                reasoning,
                refusal,
            }) => {
                let text =
                    |value: &Option<Cow<'_, str>>| value.as_ref().map_or(0, |v| v.len() + 24);
                48 + text(content)
                    + text(reasoning_content)
                    + text(reasoning)
                    + text(refusal)
                    + tool_calls.as_ref().map_or(0, |calls| {
                        calls
                            .iter()
                            .map(|call| {
                                80 + call.id.len()
                                    + call.function.name.len()
                                    + call.function.arguments.size_hint()
                            })
                            .sum()
                    })
            }
            Self::Typed(TypedMessage::Tool {
                tool_call_id,
                content,
            }) => 48 + tool_call_id.len() + content.size_hint(),
        }
    }
}

impl Content<'_> {
    fn size_hint(&self) -> usize {
        match self {
            Self::Text(text) => text.len() + 2,
            Self::Parts(parts) => parts
                .iter()
                .map(|part| {
                    40 + match part {
                        Part::Text { text, .. } => text.len(),
                        Part::ImageUrl { image_url: url, .. } => url.url.size_hint(),
                        Part::File { file, .. } => file.filename.len() + file.file_data.size_hint(),
                    }
                })
                .sum(),
        }
    }
}

/// Rough serialized size of declared tools, used only to pre-size buffers.
pub(crate) fn tools_size_hint(tools: Option<&Vec<Tool<'_>>>) -> usize {
    tools.map_or(0, |tools| {
        tools
            .iter()
            .map(|tool| {
                tool.function.name.len()
                    + tool.function.description.len()
                    + json_size_hint(tool.function.parameters)
                    + 80
            })
            .sum()
    })
}
