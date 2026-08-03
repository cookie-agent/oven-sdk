//! Incremental SSE framing for the Anthropic event stream.

use oven_sdk::{ErrorStage, ModelError};

/// Decoded SSE event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Event {
    /// Event name.
    pub name: String,
    /// Joined data lines.
    pub data: String,
}
/// Incremental SSE parser accepting arbitrary byte boundaries and line endings.
#[derive(Default)]
pub struct Parser {
    bytes: Vec<u8>,
    name: String,
    data: Vec<String>,
    bom: bool,
}
impl Parser {
    /// Feeds raw bytes.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<Event>, ModelError> {
        self.bytes.extend_from_slice(chunk);
        let mut out = Vec::new();
        let mut start = 0;
        let mut i = 0;
        while i < self.bytes.len() {
            if self.bytes[i] == b'\n' || self.bytes[i] == b'\r' {
                let end = i;
                if self.bytes[i] == b'\r' && i + 1 == self.bytes.len() {
                    break;
                }
                if self.bytes[i] == b'\r' && self.bytes.get(i + 1) == Some(&b'\n') {
                    i += 1;
                }
                let line = self.bytes[start..end].to_vec();
                self.line(&line, &mut out)?;
                start = i + 1;
            }
            i += 1;
        }
        self.bytes.drain(..start);
        Ok(out)
    }
    /// Flushes a complete final line/event.
    pub fn finish(&mut self) -> Result<Vec<Event>, ModelError> {
        let mut out = Vec::new();
        if self.bytes.last() == Some(&b'\r') {
            self.bytes.pop();
        }
        if !self.bytes.is_empty() {
            let line = std::mem::take(&mut self.bytes);
            self.line(&line, &mut out)?;
        }
        if !self.data.is_empty() {
            out.push(Event {
                name: std::mem::take(&mut self.name),
                data: self.data.join("\n"),
            });
            self.data.clear();
        }
        Ok(out)
    }
    fn line(&mut self, raw: &[u8], out: &mut Vec<Event>) -> Result<(), ModelError> {
        let raw = if !self.bom {
            self.bom = true;
            raw.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(raw)
        } else {
            raw
        };
        let line = std::str::from_utf8(raw).map_err(|_| {
            ModelError::invalid_response("SSE contains invalid UTF-8")
                .with_stage(ErrorStage::StreamDecode)
        })?;
        if line.is_empty() {
            if !self.data.is_empty() {
                out.push(Event {
                    name: std::mem::take(&mut self.name),
                    data: self.data.join("\n"),
                });
                self.data.clear();
            }
            return Ok(());
        }
        if line.starts_with(':') {
            return Ok(());
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => self.name = value.into(),
            "data" => self.data.push(value.into()),
            "id" | "retry" => {}
            _ => {}
        }
        Ok(())
    }
}
