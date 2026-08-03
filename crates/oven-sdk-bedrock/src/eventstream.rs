//! Incremental AWS EventStream decoder with strict length, header, and CRC checks.

use std::collections::BTreeMap;

use bytes::{Buf, BytesMut};
use oven_sdk::{ErrorStage, ModelError};

/// A decoded AWS EventStream header value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeaderValue {
    /// Boolean value.
    Bool(bool),
    /// Signed byte value.
    Byte(i8),
    /// Signed 16-bit value.
    Int16(i16),
    /// Signed 32-bit value.
    Int32(i32),
    /// Signed 64-bit value.
    Int64(i64),
    /// Binary byte array.
    Bytes(Vec<u8>),
    /// UTF-8 string.
    String(String),
    /// Millisecond timestamp.
    Timestamp(i64),
    /// UUID bytes.
    Uuid([u8; 16]),
}

/// One fully validated AWS EventStream message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    /// Decoded unique headers.
    pub headers: BTreeMap<String, HeaderValue>,
    /// Exact payload bytes.
    pub payload: Vec<u8>,
}

impl Message {
    /// Returns one string header.
    #[must_use]
    pub fn string_header(&self, name: &str) -> Option<&str> {
        match self.headers.get(name) {
            Some(HeaderValue::String(value)) => Some(value),
            _ => None,
        }
    }
}

/// Incremental decoder independent of HTTP chunk boundaries.
///
/// Input slices are consumed frame-by-frame. The decoder retains at most one
/// incomplete frame whose logical size is capped by `max_message_bytes`.
#[derive(Debug)]
pub struct Decoder {
    buffer: BytesMut,
    max_message_bytes: usize,
}

impl Decoder {
    /// Creates a decoder with a strict frame-size cap.
    #[must_use]
    pub fn new(max_message_bytes: usize) -> Self {
        Self {
            buffer: BytesMut::new(),
            max_message_bytes,
        }
    }

    /// Feeds arbitrary bytes and returns every complete validated message.
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<Message>, ModelError> {
        if self.max_message_bytes < 16 && !bytes.is_empty() {
            return Err(decode_error(
                "AWS EventStream maximum frame size is smaller than the minimum frame",
            ));
        }
        let mut messages = Vec::new();
        let mut offset = 0_usize;
        while offset < bytes.len() {
            if self.buffer.len() < 12 {
                let needed = 12 - self.buffer.len();
                let take = needed.min(bytes.len() - offset);
                self.buffer.extend_from_slice(&bytes[offset..offset + take]);
                offset = offset
                    .checked_add(take)
                    .ok_or_else(|| decode_error("AWS EventStream input offset overflowed"))?;
                if self.buffer.len() < 12 {
                    break;
                }
            }
            let (total, headers) = decode_prelude(&self.buffer[..12], self.max_message_bytes)?;
            if self.buffer.len() < total {
                let needed = total - self.buffer.len();
                let take = needed.min(bytes.len() - offset);
                self.buffer.extend_from_slice(&bytes[offset..offset + take]);
                offset = offset
                    .checked_add(take)
                    .ok_or_else(|| decode_error("AWS EventStream input offset overflowed"))?;
                if self.buffer.len() < total {
                    break;
                }
            }
            messages.push(decode_frame(&self.buffer[..total], headers)?);
            self.buffer.clear();
        }
        Ok(messages)
    }

    /// Finishes decoding, rejecting a truncated trailing frame.
    pub fn finish(&mut self) -> Result<Vec<Message>, ModelError> {
        let messages = self.feed(&[])?;
        if !self.buffer.is_empty() {
            return Err(
                ModelError::unexpected_eof("AWS EventStream ended with a truncated frame")
                    .with_stage(ErrorStage::StreamDecode),
            );
        }
        Ok(messages)
    }
}

fn decode_prelude(prelude: &[u8], maximum: usize) -> Result<(usize, usize), ModelError> {
    let total = usize::try_from(u32::from_be_bytes(
        prelude[0..4]
            .try_into()
            .map_err(|_| decode_error("AWS EventStream prelude is truncated"))?,
    ))
    .map_err(|_| decode_error("AWS EventStream total length overflowed"))?;
    let headers = usize::try_from(u32::from_be_bytes(
        prelude[4..8]
            .try_into()
            .map_err(|_| decode_error("AWS EventStream prelude is truncated"))?,
    ))
    .map_err(|_| decode_error("AWS EventStream header length overflowed"))?;
    if total < 16 || total > maximum || headers > total - 16 {
        return Err(decode_error("AWS EventStream frame lengths are invalid"));
    }
    let expected = u32::from_be_bytes(
        prelude[8..12]
            .try_into()
            .map_err(|_| decode_error("AWS EventStream prelude is truncated"))?,
    );
    if crc32fast::hash(&prelude[..8]) != expected {
        return Err(decode_error("AWS EventStream prelude CRC mismatch"));
    }
    Ok((total, headers))
}

fn decode_frame(frame: &[u8], headers: usize) -> Result<Message, ModelError> {
    let expected_message = u32::from_be_bytes(
        frame[frame.len() - 4..]
            .try_into()
            .map_err(|_| decode_error("AWS EventStream message CRC is truncated"))?,
    );
    if crc32fast::hash(&frame[..frame.len() - 4]) != expected_message {
        return Err(decode_error("AWS EventStream message CRC mismatch"));
    }
    let header_end = 12_usize
        .checked_add(headers)
        .ok_or_else(|| decode_error("AWS EventStream header boundary overflowed"))?;
    let decoded_headers = decode_headers(&frame[12..header_end])?;
    let payload = frame[header_end..frame.len() - 4].to_vec();
    Ok(Message {
        headers: decoded_headers,
        payload,
    })
}

fn decode_headers(mut bytes: &[u8]) -> Result<BTreeMap<String, HeaderValue>, ModelError> {
    let mut headers = BTreeMap::new();
    while !bytes.is_empty() {
        let name_len = take_u8(&mut bytes)? as usize;
        if name_len == 0 || bytes.len() < name_len + 1 {
            return Err(decode_error("AWS EventStream header name is truncated"));
        }
        let name = std::str::from_utf8(&bytes[..name_len])
            .map_err(|_| decode_error("AWS EventStream header name is invalid UTF-8"))?
            .to_owned();
        bytes.advance(name_len);
        let value_type = take_u8(&mut bytes)?;
        let value =
            match value_type {
                0 => HeaderValue::Bool(true),
                1 => HeaderValue::Bool(false),
                2 => HeaderValue::Byte(take_u8(&mut bytes)? as i8),
                3 => HeaderValue::Int16(i16::from_be_bytes(take::<2>(&mut bytes)?)),
                4 => HeaderValue::Int32(i32::from_be_bytes(take::<4>(&mut bytes)?)),
                5 => HeaderValue::Int64(i64::from_be_bytes(take::<8>(&mut bytes)?)),
                6 => {
                    let len = u16::from_be_bytes(take::<2>(&mut bytes)?) as usize;
                    HeaderValue::Bytes(take_vec(&mut bytes, len)?)
                }
                7 => {
                    let len = u16::from_be_bytes(take::<2>(&mut bytes)?) as usize;
                    HeaderValue::String(String::from_utf8(take_vec(&mut bytes, len)?).map_err(
                        |_| decode_error("AWS EventStream string header is invalid UTF-8"),
                    )?)
                }
                8 => HeaderValue::Timestamp(i64::from_be_bytes(take::<8>(&mut bytes)?)),
                9 => HeaderValue::Uuid(take::<16>(&mut bytes)?),
                _ => return Err(decode_error("AWS EventStream header type is invalid")),
            };
        if headers.insert(name, value).is_some() {
            return Err(decode_error("AWS EventStream contains a duplicate header"));
        }
    }
    Ok(headers)
}

fn take_u8(bytes: &mut &[u8]) -> Result<u8, ModelError> {
    if bytes.is_empty() {
        return Err(decode_error("AWS EventStream header is truncated"));
    }
    let value = bytes[0];
    bytes.advance(1);
    Ok(value)
}

fn take<const N: usize>(bytes: &mut &[u8]) -> Result<[u8; N], ModelError> {
    if bytes.len() < N {
        return Err(decode_error("AWS EventStream header value is truncated"));
    }
    let value = bytes[..N].try_into().expect("length checked");
    bytes.advance(N);
    Ok(value)
}

fn take_vec(bytes: &mut &[u8], len: usize) -> Result<Vec<u8>, ModelError> {
    if bytes.len() < len {
        return Err(decode_error("AWS EventStream header value is truncated"));
    }
    let value = bytes[..len].to_vec();
    bytes.advance(len);
    Ok(value)
}

fn decode_error(message: &str) -> ModelError {
    ModelError::invalid_response(message).with_stage(ErrorStage::StreamDecode)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let mut encoded_headers = Vec::new();
        for (name, value) in headers {
            encoded_headers.push(name.len() as u8);
            encoded_headers.extend_from_slice(name.as_bytes());
            encoded_headers.push(7);
            encoded_headers.extend_from_slice(&(value.len() as u16).to_be_bytes());
            encoded_headers.extend_from_slice(value.as_bytes());
        }
        let total = 16 + encoded_headers.len() + payload.len();
        let mut frame = Vec::new();
        frame.extend_from_slice(&(total as u32).to_be_bytes());
        frame.extend_from_slice(&(encoded_headers.len() as u32).to_be_bytes());
        frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());
        frame.extend_from_slice(&encoded_headers);
        frame.extend_from_slice(payload);
        frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());
        frame
    }

    fn encode_raw(encoded_headers: &[u8], payload: &[u8]) -> Vec<u8> {
        let total = 16 + encoded_headers.len() + payload.len();
        let mut frame = Vec::new();
        frame.extend_from_slice(&(total as u32).to_be_bytes());
        frame.extend_from_slice(&(encoded_headers.len() as u32).to_be_bytes());
        frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());
        frame.extend_from_slice(encoded_headers);
        frame.extend_from_slice(payload);
        frame.extend_from_slice(&crc32fast::hash(&frame).to_be_bytes());
        frame
    }

    fn prelude(total: u32, headers: u32) -> Vec<u8> {
        let mut value = Vec::new();
        value.extend_from_slice(&total.to_be_bytes());
        value.extend_from_slice(&headers.to_be_bytes());
        value.extend_from_slice(&crc32fast::hash(&value).to_be_bytes());
        value
    }

    #[test]
    fn every_byte_split_and_multiple_frames_decode() {
        let first = encode(
            &[(":message-type", "event"), (":event-type", "messageStart")],
            b"{}",
        );
        let second = encode(
            &[(":message-type", "event"), (":event-type", "messageStop")],
            b"{}",
        );
        let mut decoder = Decoder::new(1024);
        let mut messages = Vec::new();
        for byte in first.iter().chain(&second) {
            messages.extend(decoder.feed(&[*byte]).unwrap());
        }
        decoder.finish().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0].string_header(":event-type"),
            Some("messageStart")
        );
    }

    #[test]
    fn crc_lengths_types_duplicates_and_truncation_fail_closed() {
        let frame = encode(&[(":message-type", "event")], b"{}");
        for mutation in [0_usize, 8, frame.len() - 1] {
            let mut bad = frame.clone();
            bad[mutation] ^= 1;
            assert!(Decoder::new(1024).feed(&bad).is_err());
        }
        let mut decoder = Decoder::new(1024);
        decoder.feed(&frame[..frame.len() - 1]).unwrap();
        assert!(decoder.finish().is_err());
        assert!(Decoder::new(15).feed(&frame).is_err());
    }

    #[test]
    fn every_header_type_decodes_and_invalid_or_duplicate_headers_fail() {
        let mut headers = Vec::new();
        let mut push = |name: &str, kind: u8, value: &[u8]| {
            headers.push(name.len() as u8);
            headers.extend_from_slice(name.as_bytes());
            headers.push(kind);
            headers.extend_from_slice(value);
        };
        push("t", 0, &[]);
        push("f", 1, &[]);
        push("b", 2, &[0xFE]);
        push("s", 3, &(-2_i16).to_be_bytes());
        push("i", 4, &(-3_i32).to_be_bytes());
        push("l", 5, &(-4_i64).to_be_bytes());
        push("x", 6, &[0, 2, 1, 2]);
        push("v", 7, &[0, 2, b'o', b'k']);
        push("d", 8, &5_i64.to_be_bytes());
        push("u", 9, &[7; 16]);
        let message = Decoder::new(1024)
            .feed(&encode_raw(&headers, b"payload"))
            .unwrap()
            .remove(0);
        assert_eq!(message.headers["t"], HeaderValue::Bool(true));
        assert_eq!(message.headers["f"], HeaderValue::Bool(false));
        assert_eq!(message.headers["b"], HeaderValue::Byte(-2));
        assert_eq!(message.headers["s"], HeaderValue::Int16(-2));
        assert_eq!(message.headers["i"], HeaderValue::Int32(-3));
        assert_eq!(message.headers["l"], HeaderValue::Int64(-4));
        assert_eq!(message.headers["x"], HeaderValue::Bytes(vec![1, 2]));
        assert_eq!(message.headers["v"], HeaderValue::String("ok".into()));
        assert_eq!(message.headers["d"], HeaderValue::Timestamp(5));
        assert_eq!(message.headers["u"], HeaderValue::Uuid([7; 16]));

        let duplicate = [1, b'x', 0, 1, b'x', 1];
        assert!(
            Decoder::new(1024)
                .feed(&encode_raw(&duplicate, b""))
                .is_err()
        );
        let invalid = [1, b'x', 10];
        assert!(Decoder::new(1024).feed(&encode_raw(&invalid, b"")).is_err());
    }

    #[test]
    fn huge_single_chunks_never_expand_the_incomplete_frame_past_the_cap() {
        let mut huge = prelude(4096, 0);
        huge.resize(16 * 1024 * 1024, 0);
        let mut decoder = Decoder::new(1024);
        assert!(decoder.feed(&huge).is_err());
        assert!(decoder.buffer.len() <= 12);

        let mut too_small = Decoder::new(15);
        assert!(too_small.feed(&vec![0; 1024 * 1024]).is_err());
        assert!(too_small.buffer.is_empty());
    }

    #[test]
    fn huge_multi_frame_chunks_are_consumed_frame_by_frame() {
        let frame = encode(
            &[(":message-type", "event"), (":event-type", "metadata")],
            b"{}",
        );
        let mut chunk = Vec::new();
        for _ in 0..10_000 {
            chunk.extend_from_slice(&frame);
        }
        let mut decoder = Decoder::new(frame.len());
        let messages = decoder.feed(&chunk).unwrap();
        assert_eq!(messages.len(), 10_000);
        assert!(decoder.buffer.is_empty());
        assert!(decoder.buffer.capacity() <= frame.len());
    }

    #[test]
    fn partial_preludes_and_partial_frames_retain_only_one_bounded_frame() {
        let first = encode(
            &[(":message-type", "event"), (":event-type", "messageStart")],
            b"{}",
        );
        let second = encode(
            &[(":message-type", "event"), (":event-type", "messageStop")],
            b"{}",
        );
        let maximum = first.len().max(second.len());
        let mut decoder = Decoder::new(maximum);
        assert!(decoder.feed(&first[..7]).unwrap().is_empty());
        assert_eq!(decoder.buffer.len(), 7);
        let mut remainder = first[7..].to_vec();
        remainder.extend_from_slice(&second);
        assert_eq!(decoder.feed(&remainder).unwrap().len(), 2);
        assert!(decoder.buffer.is_empty());

        assert!(decoder.feed(&first[..first.len() - 1]).unwrap().is_empty());
        assert_eq!(decoder.buffer.len(), first.len() - 1);
        assert!(decoder.buffer.len() <= maximum);
        assert!(decoder.finish().is_err());
    }

    #[test]
    fn declared_length_overflow_shapes_fail_before_payload_buffering() {
        let mut decoder = Decoder::new(usize::MAX);
        let invalid = prelude(u32::MAX, u32::MAX);
        assert!(decoder.feed(&invalid).is_err());
        assert_eq!(decoder.buffer.len(), 12);
    }
}
