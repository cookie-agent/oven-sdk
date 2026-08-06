//! Shared incremental SSE framing.

pub(crate) use oven_sdk::provider_support::{SseEvent as Event, SseParser as Parser};

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
        let mut parser =
            Parser::new("OpenAI SSE contains invalid UTF-8").clear_name_on_empty_event();
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
