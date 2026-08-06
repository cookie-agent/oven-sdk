//! Shared incremental SSE framing for the Anthropic event stream.

pub use oven_sdk::provider_support::{SseEvent as Event, SseParser as Parser};

#[cfg(test)]
mod tests {
    use super::Parser;

    #[test]
    fn data_less_event_keeps_name_for_following_data() {
        let mut parser = Parser::default();
        let events = parser.feed(b"event: foo\n\ndata: x\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "foo");
        assert_eq!(events[0].data, "x");
    }
}
