//! Borrowed Cohere v2 Chat request body and in-place native replay comparison.
//!
//! Every type here borrows from the caller's [`oven_sdk::Request`], the model settings, or the
//! validated native replay payload, and is serialized once into the request byte buffer.

use std::{collections::BTreeMap, fmt};

use base64::{display::Base64Display, engine::general_purpose::STANDARD};
use oven_sdk::{AssistantPart, ContentValue, InputPart, JsonValue, SystemPart, ToolCallPart};
use serde::{Serialize, Serializer};

use crate::CohereDocument;

/// Complete v2 Chat request body.
#[derive(Serialize)]
pub(crate) struct ChatBody<'a> {
    pub(crate) model: &'a str,
    pub(crate) messages: Vec<Message<'a>>,
    pub(crate) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "is_empty_slice")]
    pub(crate) stop_sequences: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) priority: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) safety_mode: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "is_empty_slice")]
    pub(crate) documents: &'a [CohereDocument],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) citation_options: Option<CitationOptions<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<Tool<'a>>>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) strict_tools: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) response_format: Option<ResponseFormat<'a>>,
}

impl ChatBody<'_> {
    /// Serializes into a buffer pre-sized from a cheap walk of the borrowed content.
    pub(crate) fn to_bytes(&self) -> serde_json::Result<Vec<u8>> {
        let hint = self.size_hint();
        let mut buffer = Vec::with_capacity(hint + hint / 8 + 512);
        serde_json::to_writer(&mut buffer, self)?;
        Ok(buffer)
    }

    fn size_hint(&self) -> usize {
        let messages = self.messages.iter().map(Message::size_hint).sum::<usize>();
        let tools = self
            .tools
            .iter()
            .flatten()
            .map(Tool::size_hint)
            .sum::<usize>();
        let documents = self
            .documents
            .iter()
            .map(|document| {
                32 + document.id.as_ref().map_or(0, String::len) + json_map_hint(&document.data)
            })
            .sum::<usize>();
        let schema = self
            .response_format
            .as_ref()
            .and_then(|format| format.json_schema)
            .map_or(0, json_hint);
        self.model.len() + messages + tools + documents + schema
    }
}

#[derive(Serialize)]
pub(crate) struct Thinking {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) token_budget: Option<u64>,
}

#[derive(Serialize)]
pub(crate) struct CitationOptions<'a> {
    pub(crate) mode: &'a str,
}

#[derive(Serialize)]
pub(crate) struct Tool<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: FunctionSpec<'a>,
}

impl<'a> Tool<'a> {
    pub(crate) fn function(name: &'a str, description: &'a str, parameters: &'a JsonValue) -> Self {
        Self {
            kind: "function",
            function: FunctionSpec {
                name,
                description,
                parameters,
            },
        }
    }

    fn size_hint(&self) -> usize {
        64 + self.function.name.len()
            + self.function.description.len()
            + json_hint(self.function.parameters)
    }
}

#[derive(Serialize)]
struct FunctionSpec<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a JsonValue,
}

#[derive(Serialize)]
pub(crate) struct ResponseFormat<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<&'a JsonValue>,
}

impl<'a> ResponseFormat<'a> {
    pub(crate) fn json_object(json_schema: Option<&'a JsonValue>) -> Self {
        Self {
            kind: "json_object",
            json_schema,
        }
    }
}

/// One chat message.
#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum Message<'a> {
    System(TextMessage<SystemText<'a>>),
    UserText(TextMessage<UserText<'a>>),
    UserParts(PartsMessage<'a>),
    Tool(ToolMessage<'a>),
    Assistant(AssistantMessage<'a>),
    /// A validated native replay message, sent verbatim.
    Native(&'a JsonValue),
}

impl<'a> Message<'a> {
    /// A system message, or `None` when the joined system text is empty.
    pub(crate) fn system(parts: &'a [SystemPart]) -> Option<Self> {
        let content = SystemText(parts);
        (content.len() != 0).then_some(Self::System(TextMessage {
            role: "system",
            content,
        }))
    }

    pub(crate) fn user_text(parts: &'a [InputPart]) -> Self {
        Self::UserText(TextMessage {
            role: "user",
            content: UserText(parts),
        })
    }

    pub(crate) fn user_parts(content: Vec<UserPart<'a>>) -> Self {
        Self::UserParts(PartsMessage {
            role: "user",
            content,
        })
    }

    fn size_hint(&self) -> usize {
        32 + match self {
            Self::System(message) => message.content.len(),
            Self::UserText(message) => message.content.len(),
            Self::UserParts(message) => message.content.iter().map(UserPart::size_hint).sum(),
            Self::Tool(message) => message.tool_call_id.len() + message.content.size_hint(),
            Self::Assistant(message) => message.size_hint(),
            Self::Native(value) => json_hint(value),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct TextMessage<T> {
    role: &'static str,
    content: T,
}

/// System text parts joined with newlines, written straight into the buffer.
pub(crate) struct SystemText<'a>(&'a [SystemPart]);

impl<'a> SystemText<'a> {
    fn texts(&self) -> impl Iterator<Item = &'a str> + use<'a> {
        self.0.iter().filter_map(|part| match part {
            SystemPart::Text(text) => Some(text.text.as_str()),
            SystemPart::Custom(_) => None,
        })
    }

    /// Byte length of the joined text.
    pub(crate) fn len(&self) -> usize {
        let (count, bytes) = self
            .texts()
            .fold((0_usize, 0_usize), |(count, bytes), text| {
                (count + 1, bytes + text.len())
            });
        bytes + count.saturating_sub(1)
    }
}

impl fmt::Display for SystemText<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, text) in self.texts().enumerate() {
            if index > 0 {
                formatter.write_str("\n")?;
            }
            formatter.write_str(text)?;
        }
        Ok(())
    }
}

impl Serialize for SystemText<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// User text parts concatenated without a separator, written straight into the buffer.
pub(crate) struct UserText<'a>(&'a [InputPart]);

impl<'a> UserText<'a> {
    fn texts(&self) -> impl Iterator<Item = &'a str> + use<'a> {
        self.0.iter().filter_map(|part| match part {
            InputPart::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
    }

    fn len(&self) -> usize {
        self.texts().map(str::len).sum()
    }
}

impl fmt::Display for UserText<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.texts().try_for_each(|text| formatter.write_str(text))
    }
}

impl Serialize for UserText<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

#[derive(Serialize)]
pub(crate) struct PartsMessage<'a> {
    role: &'static str,
    content: Vec<UserPart<'a>>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum UserPart<'a> {
    Text { text: &'a str },
    ImageUrl { image_url: ImageUrl<'a> },
}

impl UserPart<'_> {
    fn size_hint(&self) -> usize {
        32 + match self {
            Self::Text { text } => text.len(),
            Self::ImageUrl { image_url } => match &image_url.url {
                ImageSource::Data { media_type, bytes } => {
                    16 + media_type.len() + bytes.len().div_ceil(3) * 4
                }
                ImageSource::Url(url) => url.len(),
            },
        }
    }
}

#[derive(Serialize)]
pub(crate) struct ImageUrl<'a> {
    pub(crate) url: ImageSource<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) detail: Option<&'a str>,
}

/// An image URL, or a base64 data URL encoded straight into the buffer.
pub(crate) enum ImageSource<'a> {
    Data {
        media_type: &'a str,
        bytes: &'a [u8],
    },
    Url(&'a str),
}

impl Serialize for ImageSource<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Data { media_type, bytes } => serializer.collect_str(&format_args!(
                "data:{media_type};base64,{}",
                Base64Display::new(bytes, &STANDARD)
            )),
            Self::Url(url) => serializer.serialize_str(url),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct ToolMessage<'a> {
    role: &'static str,
    tool_call_id: &'a str,
    content: ToolResultContent<'a>,
}

impl<'a> ToolMessage<'a> {
    pub(crate) fn new(tool_call_id: &'a str, content: ToolResultContent<'a>) -> Self {
        Self {
            role: "tool",
            tool_call_id,
            content,
        }
    }
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum ToolResultContent<'a> {
    Text(TextValue<'a>),
    Parts(Vec<ToolResultPart<'a>>),
}

impl ToolResultContent<'_> {
    fn size_hint(&self) -> usize {
        match self {
            Self::Text(text) => text.size_hint(),
            Self::Parts(parts) => parts.iter().map(|part| 32 + part.text.size_hint()).sum(),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct ToolResultPart<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: TextValue<'a>,
}

impl<'a> ToolResultPart<'a> {
    pub(crate) fn from_value(value: &'a ContentValue) -> Option<Self> {
        let text = match value {
            ContentValue::Text(value) => TextValue::Str(value),
            ContentValue::Json(value) => TextValue::Json(value),
            ContentValue::File(_) => return None,
        };
        Some(Self { kind: "text", text })
    }
}

/// Text that is borrowed or rendered from JSON (compact, as `to_string`) while serializing.
pub(crate) enum TextValue<'a> {
    Str(&'a str),
    Json(&'a JsonValue),
}

impl Serialize for TextValue<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Str(text) => serializer.serialize_str(text),
            Self::Json(value) => serializer.collect_str(value),
        }
    }
}

impl TextValue<'_> {
    fn size_hint(&self) -> usize {
        match self {
            Self::Str(text) => text.len(),
            Self::Json(value) => json_hint(value),
        }
    }
}

/// A normalized assistant message.
#[derive(Default, Serialize)]
pub(crate) struct AssistantMessage<'a> {
    role: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) content: Vec<AssistantContent<'a>>,
    #[serde(skip_serializing_if = "ToolPlan::is_empty")]
    pub(crate) tool_plan: ToolPlan<'a>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tool_calls: Vec<ToolCall<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) citations: Vec<&'a BTreeMap<String, JsonValue>>,
}

impl AssistantMessage<'_> {
    pub(crate) fn new() -> Self {
        Self {
            role: "assistant",
            ..Self::default()
        }
    }

    /// Whether any field besides the role would be serialized.
    pub(crate) fn has_content(&self) -> bool {
        !self.content.is_empty()
            || !self.tool_plan.is_empty()
            || !self.tool_calls.is_empty()
            || !self.citations.is_empty()
    }

    fn size_hint(&self) -> usize {
        let content = self
            .content
            .iter()
            .map(|item| {
                32 + match item {
                    AssistantContent::Text { text } => text.len(),
                    AssistantContent::Thinking { thinking } => thinking.len(),
                }
            })
            .sum::<usize>();
        let plan = self
            .tool_plan
            .0
            .iter()
            .map(|piece| piece.len())
            .sum::<usize>();
        let calls = self
            .tool_calls
            .iter()
            .map(|call| {
                64 + call.id.len()
                    + call.function.name.map_or(0, str::len)
                    + match call.function.arguments {
                        Arguments::Raw(raw) => raw.len(),
                        Arguments::Json(value) => json_hint(value),
                    }
            })
            .sum::<usize>();
        let citations = self
            .citations
            .iter()
            .map(|citation| {
                citation
                    .iter()
                    .map(|(key, value)| key.len() + 4 + json_hint(value))
                    .sum::<usize>()
            })
            .sum::<usize>();
        content + plan + calls + citations
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AssistantContent<'a> {
    Text { text: &'a str },
    Thinking { thinking: &'a str },
}

/// Tool-plan reasoning pieces, concatenated while serializing.
#[derive(Default)]
pub(crate) struct ToolPlan<'a>(pub(crate) Vec<&'a str>);

impl ToolPlan<'_> {
    fn is_empty(&self) -> bool {
        self.0.iter().all(|piece| piece.is_empty())
    }
}

impl Serialize for ToolPlan<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct Concat<'b>(&'b [&'b str]);
        impl fmt::Display for Concat<'_> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0
                    .iter()
                    .try_for_each(|piece| formatter.write_str(piece))
            }
        }
        serializer.collect_str(&Concat(&self.0))
    }
}

#[derive(Serialize)]
pub(crate) struct ToolCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    function: FunctionCall<'a>,
}

impl<'a> ToolCall<'a> {
    pub(crate) fn from_part(call: &'a ToolCallPart) -> Self {
        Self {
            id: &call.id,
            kind: "function",
            function: FunctionCall {
                arguments: match &call.raw_input {
                    Some(raw) => Arguments::Raw(raw),
                    None => Arguments::Json(&call.input),
                },
                name: tool_name_present(call).then_some(call.name.as_str()),
            },
        }
    }
}

#[derive(Serialize)]
struct FunctionCall<'a> {
    arguments: Arguments<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
}

enum Arguments<'a> {
    Raw(&'a str),
    Json(&'a JsonValue),
}

impl Serialize for Arguments<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Raw(raw) => serializer.serialize_str(raw),
            Self::Json(value) => serializer.collect_str(value),
        }
    }
}

/// Whether the normalized call carries its tool name on the wire.
pub(crate) fn tool_name_present(call: &ToolCallPart) -> bool {
    call.metadata
        .as_ref()
        .and_then(|metadata| metadata.get("cohere.name_missing"))
        != Some(&JsonValue::Bool(true))
}

/// Whether a reasoning part is a Cohere tool plan rather than thinking.
pub(crate) fn is_tool_plan(reasoning: &oven_sdk::ReasoningPart) -> bool {
    reasoning
        .metadata
        .as_ref()
        .and_then(|value| value.get("cohere.kind"))
        .and_then(JsonValue::as_str)
        == Some("tool_plan")
}

/// Compares a stored native message with the single assistant message that normalized
/// reconstruction would produce from `parts`, without building it.
///
/// The caller must already have rejected parts that normalized reconstruction cannot encode.
pub(crate) fn assistant_message_matches(
    message: &JsonValue,
    parts: &[AssistantPart],
    include_thinking: bool,
) -> bool {
    if parts
        .iter()
        .any(|part| matches!(part, AssistantPart::ToolResult(_)))
    {
        return false;
    }
    let Some(object) = message.as_object() else {
        return false;
    };
    if object.get("role").and_then(JsonValue::as_str) != Some("assistant") {
        return false;
    }
    let mut keys = 1;

    let mut content = parts.iter().filter_map(|part| match part {
        AssistantPart::Text(text) => Some(("text", text.text.as_str())),
        AssistantPart::Reasoning(reasoning) if !is_tool_plan(reasoning) && include_thinking => {
            Some(("thinking", reasoning.text.as_str()))
        }
        _ => None,
    });
    match object.get("content") {
        Some(JsonValue::Array(items)) => {
            keys += 1;
            if items.is_empty() {
                return false;
            }
            for item in items {
                let Some((kind, text)) = content.next() else {
                    return false;
                };
                let Some(item) = item.as_object() else {
                    return false;
                };
                if item.len() != 2
                    || item.get("type").and_then(JsonValue::as_str) != Some(kind)
                    || item.get(kind).and_then(JsonValue::as_str) != Some(text)
                {
                    return false;
                }
            }
            if content.next().is_some() {
                return false;
            }
        }
        Some(_) => return false,
        None => {
            if content.next().is_some() {
                return false;
            }
        }
    }

    let mut plan = parts.iter().filter_map(|part| match part {
        AssistantPart::Reasoning(reasoning) if is_tool_plan(reasoning) => {
            Some(reasoning.text.as_str())
        }
        _ => None,
    });
    match object.get("tool_plan") {
        Some(value) => {
            keys += 1;
            let Some(mut rest) = value.as_str() else {
                return false;
            };
            if rest.is_empty() {
                return false;
            }
            for piece in plan {
                match rest.strip_prefix(piece) {
                    Some(remaining) => rest = remaining,
                    None => return false,
                }
            }
            if !rest.is_empty() {
                return false;
            }
        }
        None => {
            if plan.any(|piece| !piece.is_empty()) {
                return false;
            }
        }
    }

    let mut calls = parts.iter().filter_map(|part| match part {
        AssistantPart::ToolCall(call) => Some(call),
        _ => None,
    });
    match object.get("tool_calls") {
        Some(JsonValue::Array(items)) => {
            keys += 1;
            if items.is_empty() {
                return false;
            }
            for item in items {
                if !calls
                    .next()
                    .is_some_and(|call| tool_call_matches(item, call))
                {
                    return false;
                }
            }
            if calls.next().is_some() {
                return false;
            }
        }
        Some(_) => return false,
        None => {
            if calls.next().is_some() {
                return false;
            }
        }
    }

    let mut citations = parts.iter().filter_map(|part| match part {
        AssistantPart::Source(source) => source.metadata.as_ref(),
        _ => None,
    });
    match object.get("citations") {
        Some(JsonValue::Array(items)) => {
            keys += 1;
            if items.is_empty() {
                return false;
            }
            for item in items {
                let Some(expected) = citations.next() else {
                    return false;
                };
                let Some(item) = item.as_object() else {
                    return false;
                };
                if item.len() != expected.len()
                    || !expected
                        .iter()
                        .all(|(key, value)| item.get(key) == Some(value))
                {
                    return false;
                }
            }
            if citations.next().is_some() {
                return false;
            }
        }
        Some(_) => return false,
        None => {
            if citations.next().is_some() {
                return false;
            }
        }
    }
    object.len() == keys
}

fn tool_call_matches(item: &JsonValue, call: &ToolCallPart) -> bool {
    let Some(item) = item.as_object() else {
        return false;
    };
    if item.len() != 3
        || item.get("id").and_then(JsonValue::as_str) != Some(call.id.as_str())
        || item.get("type").and_then(JsonValue::as_str) != Some("function")
    {
        return false;
    }
    let Some(function) = item.get("function").and_then(JsonValue::as_object) else {
        return false;
    };
    let named = tool_name_present(call);
    if function.len() != 1 + usize::from(named)
        || named && function.get("name").and_then(JsonValue::as_str) != Some(call.name.as_str())
    {
        return false;
    }
    let Some(arguments) = function.get("arguments").and_then(JsonValue::as_str) else {
        return false;
    };
    match &call.raw_input {
        Some(raw) => arguments == raw,
        None => json_text_equals(&call.input, arguments),
    }
}

/// Whether `text` is exactly the compact JSON serialization of `value`, without allocating it.
pub(crate) fn json_text_equals(value: &JsonValue, text: &str) -> bool {
    struct Prefix<'b>(&'b [u8]);
    impl std::io::Write for Prefix<'_> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            match self.0.strip_prefix(bytes) {
                Some(rest) => {
                    self.0 = rest;
                    Ok(bytes.len())
                }
                None => Err(std::io::ErrorKind::InvalidData.into()),
            }
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut prefix = Prefix(text.as_bytes());
    serde_json::to_writer(&mut prefix, value).is_ok() && prefix.0.is_empty()
}

/// SHA-256 sink so replay fingerprints hash the exact serialized bytes without buffering them.
pub(crate) struct HashWriter(pub(crate) sha2::Sha256);

impl std::io::Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        sha2::Digest::update(&mut self.0, bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn is_empty_slice<T>(value: &&[T]) -> bool {
    value.is_empty()
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
