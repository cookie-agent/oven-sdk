//! Incremental SSE parser accepting arbitrary byte boundaries.

use oven_sdk::{ErrorStage, ModelError};

/// One decoded server-sent event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Event {
    /// Optional event name.
    pub name: String,
    /// Joined `data:` lines.
    pub data: String,
}

/// Incremental SSE byte parser supporting LF, CRLF, CR, comments, multiline
/// data, a UTF-8 BOM, and split multi-byte characters.
#[derive(Default)]
pub(crate) struct Parser {
    bytes: Vec<u8>,
    name: String,
    data: Vec<String>,
    saw_first_line: bool,
}

impl Parser {
    /// Feeds one arbitrary byte chunk.
    pub(crate) fn feed(&mut self, chunk: &[u8]) -> Result<Vec<Event>, ModelError> {
        self.bytes.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut start = 0;
        let mut index = 0;
        while index < self.bytes.len() {
            if self.bytes[index] == b'\n' || self.bytes[index] == b'\r' {
                if self.bytes[index] == b'\r' && index + 1 == self.bytes.len() {
                    break;
                }
                let end = index;
                if self.bytes[index] == b'\r' && self.bytes.get(index + 1) == Some(&b'\n') {
                    index += 1;
                }
                let line = self.bytes[start..end].to_vec();
                self.line(&line, &mut events)?;
                start = index + 1;
            }
            index += 1;
        }
        self.bytes.drain(..start);
        Ok(events)
    }

    /// Flushes a final unterminated line and pending event.
    pub(crate) fn finish(&mut self) -> Result<Vec<Event>, ModelError> {
        let mut events = Vec::new();
        if self.bytes.last() == Some(&b'\r') {
            self.bytes.pop();
        }
        if !self.bytes.is_empty() {
            let line = std::mem::take(&mut self.bytes);
            self.line(&line, &mut events)?;
        }
        self.dispatch(&mut events);
        Ok(events)
    }

    fn line(&mut self, raw: &[u8], events: &mut Vec<Event>) -> Result<(), ModelError> {
        let raw = if self.saw_first_line {
            raw
        } else {
            self.saw_first_line = true;
            raw.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(raw)
        };
        let line = std::str::from_utf8(raw).map_err(|_| {
            ModelError::invalid_response("Azure OpenAI SSE contains invalid UTF-8")
                .with_stage(ErrorStage::StreamDecode)
        })?;
        if line.is_empty() {
            self.dispatch(events);
            return Ok(());
        }
        if line.starts_with(':') {
            return Ok(());
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => self.name = value.to_owned(),
            "data" => self.data.push(value.to_owned()),
            "id" | "retry" => {}
            _ => {}
        }
        Ok(())
    }

    fn dispatch(&mut self, events: &mut Vec<Event>) {
        if !self.data.is_empty() {
            events.push(Event {
                name: std::mem::take(&mut self.name),
                data: self.data.join("\n"),
            });
            self.data.clear();
        } else {
            self.name.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use oven_sdk_conformance::sse::{ChunkPattern, LineEnding, SseEvent, chunk_bytes, encode_sse};

    use super::Parser;

    #[test]
    fn handles_one_byte_utf8_crlf_comments_multiline_and_bom() {
        let document = encode_sse(
            &[
                SseEvent::comment("keepalive"),
                SseEvent::named("message", "{\"text\":\"hé\"").with_data_line("}"),
            ],
            LineEnding::Crlf,
        );
        let mut bytes = vec![0xef, 0xbb, 0xbf];
        bytes.extend(document);
        let mut parser = Parser::default();
        let mut events = Vec::new();
        for chunk in chunk_bytes(&bytes, &ChunkPattern::OneByte) {
            events.extend(parser.feed(&chunk).unwrap());
        }
        events.extend(parser.finish().unwrap());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "message");
        assert_eq!(events[0].data, "{\"text\":\"hé\"\n}");
    }
}
