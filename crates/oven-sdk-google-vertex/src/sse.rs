//! Incremental server-sent-event framing.

use oven_sdk::{ModelError, provider_support::SseParser};
use std::collections::VecDeque;

pub(crate) use oven_sdk::provider_support::SseEvent as Event;

/// Incremental UTF-8 SSE parser accepting arbitrary byte boundaries.
pub struct Parser {
    inner: SseParser,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            inner: SseParser::new("Vertex SSE contains invalid UTF-8").clear_name_on_empty_event(),
        }
    }
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
        self.inner.feed(chunk)
    }

    pub(crate) fn feed_events_into(
        &mut self,
        chunk: &[u8],
        events: &mut VecDeque<Event>,
    ) -> Result<(), ModelError> {
        self.inner.feed_into(chunk, events)
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

    #[test]
    fn public_feed_returns_name_data_tuples() {
        let mut parser = Parser::default();
        assert_eq!(
            parser.feed(b"event: message\ndata: x\n\n").unwrap(),
            vec![("message".into(), "x".into())]
        );
    }
}
