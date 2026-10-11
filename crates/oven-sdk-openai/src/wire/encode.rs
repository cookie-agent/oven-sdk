//! Borrowed request-body building blocks shared by Chat and Responses.
//!
//! Request bodies are assembled from structs that borrow the caller's
//! history and serialized once, straight into the outgoing byte buffer.

use std::{borrow::Cow, fmt};

use base64::{display::Base64Display, engine::general_purpose::STANDARD};
use oven_sdk::{ErrorStage, FilePart, FileSource, JsonValue, ModelError, PartMetadata};
use serde::{Serialize, Serializer};

use crate::options::prompt_cache_breakpoint;

/// The `{"mode":"explicit"}` prompt-cache breakpoint marker.
#[derive(Clone, Copy, Serialize)]
pub(crate) struct Breakpoint {
    mode: &'static str,
}

/// Returns the wire marker for a part flagged as an explicit cache breakpoint.
pub(crate) fn breakpoint(metadata: &PartMetadata) -> Result<Option<Breakpoint>, ModelError> {
    Ok(prompt_cache_breakpoint(metadata)?.then_some(Breakpoint { mode: "explicit" }))
}

/// A JSON string whose contents are borrowed, owned, or rendered from a JSON
/// value (`value.to_string()` without the intermediate allocation).
pub(crate) enum Text<'a> {
    Borrowed(&'a str),
    Owned(String),
    Json(&'a JsonValue),
}

impl Serialize for Text<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Borrowed(value) => serializer.serialize_str(value),
            Self::Owned(value) => serializer.serialize_str(value),
            Self::Json(value) => serializer.collect_str(value),
        }
    }
}

impl Text<'_> {
    pub(crate) fn size_hint(&self) -> usize {
        match self {
            Self::Borrowed(value) => value.len() + 2,
            Self::Owned(value) => value.len() + 2,
            Self::Json(value) => json_size_hint(value) * 5 / 4,
        }
    }
}

/// Tool-call arguments exactly as the history recorded them, or the compact
/// rendering of the parsed input when no raw text was kept.
pub(crate) fn arguments(call: &oven_sdk::ToolCallPart) -> Text<'_> {
    match &call.raw_input {
        Some(raw) => Text::Borrowed(raw),
        None => Text::Json(&call.input),
    }
}

/// Appends one text fragment, borrowing until a second fragment arrives.
pub(crate) fn push_text<'a>(run: &mut Cow<'a, str>, text: &'a str) {
    if text.is_empty() {
        return;
    }
    if run.is_empty() {
        *run = Cow::Borrowed(text);
    } else {
        run.to_mut().push_str(text);
    }
}

/// Joins fragments with `separator`, borrowing a lone fragment.
pub(crate) fn join<'a>(parts: impl Iterator<Item = &'a str>, separator: &str) -> Cow<'a, str> {
    let parts = parts.collect::<Vec<_>>();
    match parts.as_slice() {
        [] => Cow::Borrowed(""),
        [only] => Cow::Borrowed(only),
        _ => Cow::Owned(parts.join(separator)),
    }
}

/// A caller URL or inline bytes rendered as a base64 `data:` URL while the
/// body is serialized.
pub(crate) enum FileRef<'a> {
    Url(&'a str),
    Data { media_type: &'a str, data: &'a [u8] },
}

impl<'a> FileRef<'a> {
    /// Inline sources become `data:` URLs; caller URLs pass through.
    pub(crate) fn of(file: &'a FilePart) -> Option<Self> {
        match &file.source {
            FileSource::Bytes(bytes) => Some(Self::data(file, bytes)),
            FileSource::Text(text) => Some(Self::data(file, text.as_bytes())),
            FileSource::Url(url) => Some(Self::Url(url.as_str())),
            FileSource::ProviderReference { .. } => None,
        }
    }

    fn data(file: &'a FilePart, data: &'a [u8]) -> Self {
        Self::Data {
            media_type: &file.media_type,
            data,
        }
    }

    pub(crate) fn size_hint(&self) -> usize {
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

/// Serializes a request body into one pre-sized buffer.
pub(crate) fn to_body(
    value: &impl Serialize,
    size_hint: usize,
    error: impl FnOnce() -> ModelError,
) -> Result<Vec<u8>, ModelError> {
    // The hint ignores JSON escaping; leave headroom so the buffer is not
    // reallocated (and copied) near the end of serialization.
    let mut body = Vec::with_capacity(size_hint + size_hint / 8 + 256);
    serde_json::to_writer(&mut body, value).map_err(|_| error())?;
    Ok(body)
}

/// The generic encoding failure for a request body.
pub(crate) fn encoding_error(message: &'static str) -> impl FnOnce() -> ModelError {
    move || ModelError::invalid_request(message).with_stage(ErrorStage::RequestEncoding)
}
