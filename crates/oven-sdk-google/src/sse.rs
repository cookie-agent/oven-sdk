//! Incremental server-sent-event framing.

use oven_sdk::{ErrorStage, ModelError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Event {
    pub(crate) name: String,
    pub(crate) data: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_byte_utf8_crlf_multiline_and_comments_are_incremental() {
        let input = ": ping\r\ndata: {\"text\":\"hé\"}\r\ndata: second\r\n\r\n";
        let mut parser = Parser::default();
        let mut events = Vec::new();
        for byte in input.as_bytes() {
            events.extend(parser.feed_events(std::slice::from_ref(byte)).unwrap());
        }
        events.extend(parser.finish_events().unwrap());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"text\":\"hé\"}\nsecond");
    }
}

/// Incremental UTF-8 SSE parser accepting arbitrary byte boundaries.
#[derive(Default)]
pub struct Parser {
    bytes: Vec<u8>,
    name: String,
    data: Vec<String>,
    bom_seen: bool,
}

impl Parser {
    /// Feeds one arbitrary network chunk.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<(String, String)>, ModelError> {
        self.feed_events(chunk).map(|events| {
            events
                .into_iter()
                .map(|event| (event.name, event.data))
                .collect()
        })
    }

    pub(crate) fn feed_events(&mut self, chunk: &[u8]) -> Result<Vec<Event>, ModelError> {
        self.bytes.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut start = 0;
        let mut index = 0;
        while index < self.bytes.len() {
            if self.bytes[index] == b'\n' || self.bytes[index] == b'\r' {
                let end = index;
                if self.bytes[index] == b'\r' && index + 1 == self.bytes.len() {
                    break;
                }
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

    pub(crate) fn finish_events(&mut self) -> Result<Vec<Event>, ModelError> {
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
        let raw = if self.bom_seen {
            raw
        } else {
            self.bom_seen = true;
            raw.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(raw)
        };
        let line = std::str::from_utf8(raw).map_err(|_| {
            ModelError::invalid_response("Gemini SSE contains invalid UTF-8")
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
            _ => {}
        }
        Ok(())
    }

    fn dispatch(&mut self, events: &mut Vec<Event>) {
        if self.data.is_empty() {
            self.name.clear();
            return;
        }
        events.push(Event {
            name: std::mem::take(&mut self.name),
            data: self.data.join("\n"),
        });
        self.data.clear();
    }
}
