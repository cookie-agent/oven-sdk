//! Incremental server-sent-event framing.

use oven_sdk::{ModelError, provider_support::SseParser};

pub(crate) use oven_sdk::provider_support::SseEvent as Event;

pub(crate) struct Parser {
    inner: SseParser,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            inner: SseParser::new("Gemini SSE contains invalid UTF-8").clear_name_on_empty_event(),
        }
    }
}

impl Parser {
    pub(crate) fn feed_events(&mut self, chunk: &[u8]) -> Result<Vec<Event>, ModelError> {
        self.inner.feed(chunk)
    }

    pub(crate) fn finish_events(&mut self) -> Result<Vec<Event>, ModelError> {
        self.inner.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::Parser;

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
