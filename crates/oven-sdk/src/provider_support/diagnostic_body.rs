//! Best-effort credential scrubbing at the provider response boundary.

use crate::{ErrorStage, JsonValue, SanitizedBody};

const REDACTED: &str = "[REDACTED]";
const MAX_STRING_LAYERS: usize = 8;

/// Retains a useful, credential-scrubbed provider error body, capped at 64 KiB.
///
/// JSON keeps its original fields and structure (credential values become
/// `[REDACTED]`); other bodies retain scrubbed text. Input is capped before parsing,
/// and output is capped at a UTF-8 boundary. `bytes_received` detects truncation by
/// the transport even when the retained input fits, only at `ResponseBody` stage;
/// stream counts cover multiple events. A truncated body may not be JSON.
///
/// Scrubs nested credential keys, header name/value pairs, common credential
/// prefixes, authorization schemes, and secret assignments/query parameters.
/// Controls are replaced with spaces, including in JSON strings. This is not a
/// general PII or arbitrary-secret detector: unlabelled, encoded, or unfamiliar
/// credentials and echoed prompts may remain. Request credentials are not available
/// at this classifier boundary; callers must not treat this text as public data.
#[must_use]
pub fn sanitize_error_body(
    raw: &[u8],
    bytes_received: u64,
    stage: ErrorStage,
) -> Option<SanitizedBody> {
    sanitize_error_body_with_secrets(raw, bytes_received, stage, &[])
}

/// Also scrubs explicitly supplied credential values from text and decoded JSON
/// strings, including complete JSON strings in prefixed or malformed bodies.
///
/// Supply only credentials already available at the response boundary. Empty
/// secrets are ignored; overlapping matches prefer the longest value. This does
/// not discover unknown secrets or decode arbitrary encodings. The input/output
/// bounds and other limitations of [`sanitize_error_body`] still apply.
/// Sanitize the provider body before adding presentation prefixes or markers.
#[must_use]
pub fn sanitize_error_body_with_secrets(
    raw: &[u8],
    bytes_received: u64,
    stage: ErrorStage,
    known_secrets: &[&str],
) -> Option<SanitizedBody> {
    if raw.is_empty() {
        return None;
    }
    let end = raw.len().min(SanitizedBody::MAX_BYTES);
    let input = String::from_utf8_lossy(&raw[..end]);
    let text = match serde_json::from_str::<JsonValue>(&input) {
        Ok(mut value) => {
            scrub_json(&mut value, known_secrets);
            value.to_string()
        }
        Err(_) => scrub_text(&input, known_secrets),
    };
    let mut body = SanitizedBody::new(text);
    body.truncated |=
        raw.len() > end || (stage == ErrorStage::ResponseBody && bytes_received > end as u64);
    Some(body)
}

fn normalized_key(key: &str) -> String {
    // Query names can percent-encode separators or letters.
    let decoded = url::form_urlencoded::parse(key.as_bytes())
        .next()
        .map(|(key, _)| key.into_owned())
        .unwrap_or_default();
    decoded
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect()
}

fn credential_key(key: &str) -> bool {
    let key = normalized_key(key);
    matches!(
        key.as_str(),
        "key"
            | "auth"
            | "authentication"
            | "authorization"
            | "proxyauthorization"
            | "cookie"
            | "setcookie"
            | "password"
            | "passwd"
            | "pwd"
            | "secret"
            | "credentials"
            | "credential"
            | "token"
            | "session"
            | "sessionid"
            | "sessionauth"
            | "signature"
            | "sig"
            | "xamzcredential"
            | "xamzsignature"
            | "xamzsecuritytoken"
            | "xgoogsignature"
            | "xgoogcredential"
    ) || key.ends_with("apikey")
        || key.ends_with("token")
        || key.ends_with("sessionauth")
        || key.ends_with("secretkey")
        || key.ends_with("accesskey")
        || key.ends_with("sessionkey")
        || key.ends_with("subscriptionkey")
        || key.ends_with("clientsecret")
        || key.ends_with("secretaccesskey")
        || key.ends_with("accesskeyid")
        || key.ends_with("privatekey")
}

fn scrub_json(value: &mut JsonValue, known_secrets: &[&str]) {
    match value {
        JsonValue::Object(object) => {
            // Also cover arrays of headers such as {"name":"Authorization","value":...}.
            let credential_pair = object.iter().any(|(key, value)| {
                matches!(normalized_key(key).as_str(), "name" | "key")
                    && value.as_str().is_some_and(credential_key)
            });
            let original = std::mem::take(object);
            for (key, mut value) in original {
                if credential_key(&key) || (credential_pair && normalized_key(&key) == "value") {
                    value = REDACTED.into();
                } else {
                    scrub_json(&mut value, known_secrets);
                }
                object.insert(scrub_text(&key, known_secrets), value);
            }
        }
        JsonValue::Array(values) => values
            .iter_mut()
            .for_each(|value| scrub_json(value, known_secrets)),
        JsonValue::String(text) => *text = scrub_text(text, known_secrets),
        _ => {}
    }
}

fn word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"_-%.".contains(&byte)
}

fn credential_prefix(word: &str) -> bool {
    [
        "sk-",
        "sk_",
        "AIza",
        "AKIA",
        "ASIA",
        "ghp_",
        "gho_",
        "github_pat_",
        "xoxb-",
        "xoxp-",
    ]
    .iter()
    .any(|prefix| word.starts_with(prefix))
        || (word.starts_with("eyJ") && word.matches('.').count() >= 2)
}

fn whitespace_end(bytes: &[u8], mut index: usize) -> usize {
    while bytes.get(index).is_some_and(u8::is_ascii_whitespace) {
        index += 1;
    }
    index
}

// Consume a complete quoted or composite value, or the retained prefix if cut off.
fn value_end(bytes: &[u8], start: usize, header: bool) -> usize {
    let Some(&first) = bytes.get(start) else {
        return start;
    };
    if matches!(first, b'\'' | b'"') {
        let mut index = start + 1;
        while index < bytes.len() {
            if bytes[index] == b'\\' {
                index = (index + 2).min(bytes.len());
            } else if bytes[index] == first {
                return index + 1;
            } else {
                index += 1;
            }
        }
        return index;
    }
    if matches!(first, b'{' | b'[') {
        let mut depth = 0usize;
        let mut index = start;
        while index < bytes.len() {
            match bytes[index] {
                b'"' | b'\'' => {
                    index = value_end(bytes, index, false);
                    continue;
                }
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return index + 1;
                    }
                }
                _ => {}
            }
            index += 1;
        }
        return index;
    }
    let mut index = start;
    while let Some(&byte) = bytes.get(index) {
        if b"\r\n\"'<>".contains(&byte)
            || (!header && (byte.is_ascii_whitespace() || b"&,;}]".contains(&byte)))
        {
            break;
        }
        index += 1;
    }
    index
}

fn scrub_text(input: &str, known_secrets: &[&str]) -> String {
    scrub_text_layer(input, known_secrets, 0)
}

// Index spans once, rather than repeatedly parsing every nested JSON suffix.
// The explicit stacks and span tables are bounded by the retained input length.
fn paired_header_values(input: &str) -> Vec<usize> {
    let bytes = input.as_bytes();
    let mut ends = vec![bytes.len(); bytes.len() + 1];
    let mut stack = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                ends[index] = value_end(bytes, index, false);
                index = ends[index];
                continue;
            }
            b'{' | b'[' => stack.push(index),
            b'}' | b']' => {
                if let Some(start) = stack.pop() {
                    if matches!((bytes[start], bytes[index]), (b'{', b'}') | (b'[', b']')) {
                        ends[start] = index + 1;
                    } else {
                        // Mismatched composites keep the conservative EOF end.
                        stack.clear();
                    }
                }
            }
            _ => {}
        }
        index += 1;
    }

    struct Object {
        kind: u8,
        credential: bool,
        values: Vec<usize>,
    }
    let mut objects: Vec<Object> = Vec::new();
    let mut redactions = vec![0; bytes.len() + 1];
    index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'{' | b'[' => objects.push(Object {
                kind: bytes[index],
                credential: false,
                values: Vec::new(),
            }),
            b'}' | b']' => {
                if let Some(object) = objects.pop().filter(|object| object.credential) {
                    for start in object.values {
                        redactions[start] = ends[start];
                    }
                }
            }
            b'"' => {
                let end = ends[index];
                let separator = whitespace_end(bytes, end);
                if bytes.get(separator) == Some(&b':')
                    && let Some(object) = objects.last_mut().filter(|object| object.kind == b'{')
                    && let Ok(key) = serde_json::from_str::<String>(&input[index..end])
                {
                    let start = whitespace_end(bytes, separator + 1);
                    match normalized_key(&key).as_str() {
                        "name" | "key" if bytes.get(start) == Some(&b'"') => {
                            object.credential |=
                                serde_json::from_str::<String>(&input[start..ends[start]])
                                    .is_ok_and(|name| credential_key(&name));
                        }
                        "value" => {
                            if !matches!(bytes.get(start), Some(b'"' | b'{' | b'[')) {
                                ends[start] = value_end(bytes, start, false);
                            }
                            object.values.push(start);
                        }
                        _ => {}
                    }
                }
                index = end;
                continue;
            }
            _ => {}
        }
        index += 1;
    }
    // An incomplete outer envelope must not erase an already observed association.
    for object in objects.into_iter().filter(|object| object.credential) {
        for start in object.values {
            redactions[start] = ends[start];
        }
    }
    redactions
}

fn decoded_string(raw: &str) -> Option<String> {
    if let Ok(decoded) = serde_json::from_str(raw) {
        return Some(decoded);
    }
    // Repair an unterminated quoted string's retained prefix. At most one JSON
    // escape/surrogate pair can straddle the input cap; attempts are constant.
    for trim in 0..=12.min(raw.len().saturating_sub(1)) {
        let end = raw.len() - trim;
        if raw.is_char_boundary(end) {
            let mut repaired = raw[..end].to_owned();
            repaired.push('"');
            if let Ok(decoded) = serde_json::from_str(&repaired) {
                return Some(decoded);
            }
        }
    }
    None
}

fn scrub_text_layer(input: &str, known_secrets: &[&str], layer: usize) -> String {
    let bytes = input.as_bytes();
    let paired_values = paired_header_values(input);
    let mut output = String::with_capacity(input.len());
    let mut index = 0;
    while index < bytes.len() {
        if paired_values[index] > index {
            output.push_str("\"[REDACTED]\"");
            index = paired_values[index];
            continue;
        }
        // Decode complete JSON string tokens even in a truncated/malformed envelope.
        // Otherwise escaped key names or escaped token characters evade text scanning.
        if bytes[index] == b'"' {
            let end = value_end(bytes, index, false);
            if let Some(decoded) = decoded_string(&input[index..end]) {
                let separator = whitespace_end(bytes, end);
                if bytes.get(separator) == Some(&b':') && credential_key(&decoded) {
                    output
                        .push_str(&serde_json::to_string(&decoded).expect("string serialization"));
                    output.push(':');
                    output
                        .push_str(&serde_json::to_string(REDACTED).expect("string serialization"));
                    index = value_end(bytes, whitespace_end(bytes, separator + 1), false);
                    continue;
                }
                // Decode embedded JSON strings with the same object-aware scanner.
                // Excessive encoding layers redact only this quoted value.
                if input[index..end].contains('\\') {
                    let scrubbed = if layer < MAX_STRING_LAYERS {
                        scrub_text_layer(&decoded, known_secrets, layer + 1)
                    } else {
                        REDACTED.to_owned()
                    };
                    output
                        .push_str(&serde_json::to_string(&scrubbed).expect("string serialization"));
                    index = end;
                    continue;
                }
            }
        }
        if word_byte(bytes[index]) {
            let start = index;
            while bytes.get(index).is_some_and(|byte| word_byte(*byte)) {
                index += 1;
            }
            let word = &input[start..index];
            if (word.eq_ignore_ascii_case("http") || word.eq_ignore_ascii_case("https"))
                && input[index..].starts_with("://")
            {
                let authority_start = index + 3;
                // RFC 3986 userinfo permits all sub-delims: !$&'()*+,;=.
                // In particular, an apostrophe is not an authority boundary.
                let authority_end = input[authority_start..]
                    .find(|c: char| c.is_whitespace() || c.is_control() || "/?#\\\"<>".contains(c))
                    .map_or(input.len(), |end| authority_start + end);
                if let Some(at) = input[authority_start..authority_end].rfind('@') {
                    output.push_str(word);
                    output.push_str("://[REDACTED]@");
                    index = authority_start + at + 1;
                    continue;
                }
            }
            if credential_prefix(word) {
                // Include punctuation used by token formats, stopping at text delimiters.
                index = value_end(bytes, start, false);
                output.push_str(REDACTED);
                continue;
            }
            let mut separator = index;
            if matches!(bytes.get(separator), Some(b'"' | b'\'')) {
                separator += 1;
            }
            separator = whitespace_end(bytes, separator);
            if credential_key(word) && matches!(bytes.get(separator), Some(b':' | b'=')) {
                let start = whitespace_end(bytes, separator + 1);
                output.push_str(&input[index - word.len()..start]);
                let header = matches!(
                    normalized_key(word).as_str(),
                    "authorization" | "proxyauthorization" | "cookie" | "setcookie"
                );
                index = value_end(bytes, start, header);
                // Keep quotes in incomplete JSON and text assignments.
                if matches!(bytes.get(start), Some(b'"' | b'\'')) {
                    output.push(bytes[start] as char);
                    output.push_str(REDACTED);
                    output.push(bytes[start] as char);
                } else {
                    output.push_str(REDACTED);
                }
                continue;
            }
            if (word.eq_ignore_ascii_case("bearer") || word.eq_ignore_ascii_case("basic"))
                && bytes.get(index).is_some_and(u8::is_ascii_whitespace)
            {
                output.push_str(word);
                output.push(' ');
                index = value_end(bytes, whitespace_end(bytes, index), false);
                output.push_str(REDACTED);
                continue;
            }
            output.push_str(word);
        } else {
            let character = input[index..].chars().next().expect("character boundary");
            output.push(character);
            index += character.len_utf8();
        }
    }
    // Match decoded credential values before controls are normalized or the
    // retained output is truncated. Structured JSON is serialized after scrubbing
    // its decoded string values.
    redact_known(&output, known_secrets)
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}') {
                ' '
            } else {
                c
            }
        })
        .collect()
}

fn redact_known(input: &str, known_secrets: &[&str]) -> String {
    let mut output = String::with_capacity(input.len());
    let mut index = 0;
    while index < input.len() {
        if let Some(secret) = known_secrets
            .iter()
            .filter(|secret| !secret.is_empty() && input[index..].starts_with(**secret))
            .max_by_key(|secret| secret.len())
        {
            output.push_str(REDACTED);
            index += secret.len();
        } else {
            let character = input[index..].chars().next().expect("character boundary");
            output.push(character);
            index += character.len_utf8();
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sanitize_error_body(raw: &[u8], bytes: u64) -> Option<SanitizedBody> {
        super::sanitize_error_body(raw, bytes, ErrorStage::ResponseBody)
    }

    #[test]
    fn stream_byte_count_does_not_imply_body_truncation() {
        let body = super::sanitize_error_body(b"reason", 100_000, ErrorStage::StreamEvent).unwrap();
        assert!(!body.truncated());
    }

    #[test]
    fn nested_credentials_preserve_reason_shape_and_unknown_fields() {
        let raw = br#"{"error":{"message":"unsupported dimension; Bearer bearer-secret","future":{"reason":"use 1024","count":12},"details":[{"API_Key":"key-secret","session-auth":{"value":"session-secret"}},{"name":"Authorization","value":"header-secret"}],"url":"https://host/?access_token=query-secret&mode=fast"}}"#;
        let body = sanitize_error_body(raw, raw.len() as u64).unwrap();
        let json: JsonValue = serde_json::from_str(body.text()).unwrap();
        assert_eq!(json["error"]["future"]["reason"], "use 1024");
        assert_eq!(json["error"]["future"]["count"], 12);
        assert_eq!(json["error"]["details"][0]["API_Key"], REDACTED);
        assert!(body.text().contains("unsupported dimension"));
        assert!(body.text().contains("mode=fast"));
        for secret in [
            "bearer-secret",
            "key-secret",
            "session-secret",
            "header-secret",
            "query-secret",
        ] {
            assert!(!serde_json::to_string(&body).unwrap().contains(secret));
        }
        assert!(!format!("{body:?}").contains("unsupported dimension"));
    }

    #[test]
    fn credential_key_variants_preserve_noncredential_usage_fields() {
        for key in [
            "aUtHeNtIcAtIoN",
            "API_KEY",
            "X-Api-Key",
            "Proxy-Authorization",
            "Set-Cookie",
            "password",
            "credentials",
            "clientSecret",
            "secret_key",
            "access_key",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "sessionId",
            "session-auth",
            "Ocp-Apim-Subscription-Key",
            "private_key",
            "X-Amz-Signature",
        ] {
            let raw = serde_json::json!({"message":"unsupported dimension", "nested":[{key:"hidden-credential", "input_tokens":42, "token_count":12}]}).to_string();
            let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
            let value: JsonValue = serde_json::from_str(body.text()).unwrap();
            assert_eq!(value["nested"][0][key], REDACTED, "{key}");
            assert_eq!(value["nested"][0]["input_tokens"], 42);
            assert_eq!(value["nested"][0]["token_count"], 12);
            assert!(body.text().contains("unsupported dimension"));
            assert!(!body.text().contains("hidden-credential"));
        }
    }

    #[test]
    fn text_headers_queries_prefixes_and_controls() {
        let raw = b"<html>invalid dimension\nAuthorization: Basic basic-secret\r\nCookie: sid=cookie-secret; other=hidden\nuse 1024 token='token-secret' https://host/?api%5Fkey=query-secret&mode=fast sk-test-secret\x1b\0</html>";
        let body = sanitize_error_body(raw, raw.len() as u64).unwrap();
        assert!(body.text().contains("invalid dimension"));
        assert!(body.text().contains("use 1024"));
        assert!(body.text().contains("mode=fast"));
        for secret in [
            "basic-secret",
            "cookie-secret",
            "hidden",
            "token-secret",
            "query-secret",
            "sk-test-secret",
        ] {
            assert!(!body.text().contains(secret), "{secret}");
        }
        assert!(!body.text().chars().any(char::is_control));
    }

    #[test]
    fn bounded_input_output_and_transport_truncation() {
        for raw in [
            format!("reason token={}", "é".repeat(SanitizedBody::MAX_BYTES)),
            format!("reason {}", "é".repeat(SanitizedBody::MAX_BYTES)),
            format!(
                r#"{{"message":"reason","secret":{{"nested":"{}"#,
                "x".repeat(SanitizedBody::MAX_BYTES)
            ),
        ] {
            let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
            assert!(body.truncated());
            assert!(body.len_bytes() <= SanitizedBody::MAX_BYTES);
            assert!(body.text().contains("reason"));
            if raw.contains("token=") || raw.contains("secret") {
                assert!(!body.text().contains("é"));
                assert!(!body.text().contains("xxxx"));
            }
        }
        assert!(sanitize_error_body(b"reason", 100_000).unwrap().truncated());
        let raw = "x".repeat(SanitizedBody::MAX_BYTES);
        assert!(
            !sanitize_error_body(raw.as_bytes(), raw.len() as u64)
                .unwrap()
                .truncated()
        );
        assert!(sanitize_error_body(b"", 0).is_none());
        // Redaction expansion must also respect the cap.
        let raw = "token=x ".repeat(8000);
        let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
        assert!(body.truncated());
        assert!(body.len_bytes() <= SanitizedBody::MAX_BYTES);
    }

    #[test]
    fn json_escapes_controls_and_serde_round_trip() {
        let raw = br#"{"message":"bad\u001b\nfield sk-escaped\u002dsecret","a\u0070i_key":"hidden","unknown":"use another field"}"#;
        let body = sanitize_error_body(raw, raw.len() as u64).unwrap();
        let json: JsonValue = serde_json::from_str(body.text()).unwrap();
        assert_eq!(json["api_key"], REDACTED);
        assert_eq!(json["unknown"], "use another field");
        assert!(
            !json["message"]
                .as_str()
                .unwrap()
                .chars()
                .any(char::is_control)
        );
        let wire = serde_json::to_string(&body).unwrap();
        assert!(!wire.contains("hidden"));
        assert!(!wire.contains("escaped"));
        assert_eq!(serde_json::from_str::<SanitizedBody>(&wire).unwrap(), body);
    }

    #[test]
    fn malformed_json_still_scrubs_escaped_credentials() {
        for raw in [
            br#"{"message":"unsupported dimension","api\u005fkey":"escaped-secret","tail":""#.as_slice(),
            br#"{"message":"unsupported dimension; sk-escaped\u002dsecret","tail":""#.as_slice(),
            br#"{"message":"unsupported dimension","AuThOrIzAtIoN":{"nested":"escaped-secret"},"tail":""#.as_slice(),
        ] {
            let body = sanitize_error_body(raw, 100_000).unwrap();
            assert!(body.text().contains("unsupported dimension"));
            assert!(!body.text().contains("escaped"));
            assert!(body.truncated());
        }
    }

    #[test]
    fn supplied_secrets_are_scrubbed_after_json_decoding() {
        for prefix in ["", "provider response: "] {
            let raw = format!(
                r#"{prefix}{{"message":"review\u002dsecret","reason":"unsupported dimension","details":["review-secret", "review\"secret-tail", "review\nsecret"]}}"#
            );
            let body = sanitize_error_body_with_secrets(
                raw.as_bytes(),
                raw.len() as u64,
                ErrorStage::ResponseBody,
                &[
                    "",
                    "review-secret",
                    "review\"secret-tail",
                    "review\nsecret",
                    "review",
                ],
            )
            .unwrap();
            assert!(body.text().contains("unsupported dimension"));
            assert!(!body.text().contains("review"));
            assert!(!body.text().contains("secret-tail"));
            assert!(!body.text().contains("-secret"));
            assert!(!body.truncated());
            let wire = serde_json::to_string(&body).unwrap();
            assert!(!wire.contains("review"));
            assert!(!wire.contains("secret-tail"));
            let value: JsonValue =
                serde_json::from_str(body.text().strip_prefix(prefix).unwrap()).unwrap();
            assert_eq!(value["message"], REDACTED);
            assert_eq!(
                value["details"],
                serde_json::json!([REDACTED, REDACTED, REDACTED])
            );
        }
    }

    #[test]
    fn prefixed_password_values_are_escape_aware_and_preserve_reason() {
        for value in [
            r#""review\"secret-tail""#,
            r#""review\\\"secret-tail""#,
            r#""review\"secret-tail"#,
        ] {
            let raw = format!(
                r#"provider response: {{"reason":"unsupported dimension","password":{value}}}"#
            );
            let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
            assert!(body.text().contains("unsupported dimension"));
            assert!(body.text().contains(r#""password":"[REDACTED]""#));
            assert!(!body.text().contains("review"));
            assert!(!body.text().contains("secret-tail"));
            assert!(
                !serde_json::to_string(&body)
                    .unwrap()
                    .contains("secret-tail")
            );
        }
    }

    #[test]
    fn prefixed_credential_objects_redact_complete_values_or_retained_prefixes() {
        for value in [
            r#"{"nested":[{"value":"review\"secret-tail"},"second-secret"]}"#,
            r#"{"nested":[{"value":"review\"secret-tail"},"second-secret""#,
            r#"[{"nested":"review\"secret-tail"},"second-secret"]"#,
            r#"[{"nested":"review\"secret-tail"},"second-secret""#,
        ] {
            let raw = format!(
                r#"provider response: {{"reason":"unsupported dimension","credentials":{value}"#
            );
            let body = sanitize_error_body(raw.as_bytes(), 100_000).unwrap();
            assert!(body.text().contains("unsupported dimension"));
            assert!(body.text().contains(r#""credentials":"[REDACTED]""#));
            assert!(body.truncated());
            for secret in ["review", "secret-tail", "second-secret"] {
                assert!(!serde_json::to_string(&body).unwrap().contains(secret));
            }
        }
        let raw = format!(
            r#"provider response: {{"reason":"unsupported dimension","credentials":{{"nested":"review\"secret-tail{}"#,
            "second-secret".repeat(SanitizedBody::MAX_BYTES / 4)
        );
        let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
        assert!(body.truncated());
        assert!(body.len_bytes() <= SanitizedBody::MAX_BYTES);
        assert!(body.text().contains("unsupported dimension"));
        assert!(!body.text().contains("review"));
        assert!(!body.text().contains("secret"));
    }

    #[test]
    fn scrubbed_body_precedes_presentation_prefix_and_output_truncation() {
        let raw = format!(
            r#"{{"message":"review\u002dsecret","reason":"unsupported dimension {}"}}"#,
            "token=x ".repeat(7000)
        );
        assert!(raw.len() < SanitizedBody::MAX_BYTES);
        let body = sanitize_error_body_with_secrets(
            raw.as_bytes(),
            raw.len() as u64,
            ErrorStage::ResponseBody,
            &["review-secret"],
        )
        .unwrap();
        assert!(body.truncated());
        assert!(body.len_bytes() <= SanitizedBody::MAX_BYTES);
        assert!(body.text().contains("unsupported dimension"));
        let display = format!("provider response: {} [truncated]", body.text());
        assert!(!display.contains("review"));
        assert!(!serde_json::to_string(&body).unwrap().contains("review"));
    }

    #[test]
    fn header_pairs_survive_prefixed_truncated_and_encoded_envelopes() {
        for name in [
            "X-Api-Key",
            "aUtHoRiZaTiOn",
            "Proxy-Authorization",
            "x-api-token",
        ] {
            for label in ["name", "KEY"] {
                for pair in [
                    format!(r#"{{"{label}":"{name}","VaLuE":"opaque-review-credential"}}"#),
                    format!(r#"{{"VaLuE":"opaque-review-credential","{label}":"{name}"}}"#),
                ] {
                    let envelope = format!(
                        r#"{{"reason":"unsupported dimension","headers":[{pair},{{"name":"X-Request-Id","value":"harmless-review-value"}}]}}"#
                    );
                    for raw in [
                        envelope.clone(),
                        format!("provider response: {envelope}"),
                        envelope[..envelope.len() - 1].to_owned(),
                    ] {
                        let mut encoded = raw;
                        for _ in 0..3 {
                            let body =
                                sanitize_error_body(encoded.as_bytes(), encoded.len() as u64)
                                    .unwrap();
                            assert!(body.text().contains("unsupported dimension"));
                            assert!(body.text().contains("harmless-review-value"));
                            assert!(
                                !body.text().contains("opaque-review-credential"),
                                "{name}/{label}"
                            );
                            assert!(
                                !serde_json::to_string(&body)
                                    .unwrap()
                                    .contains("opaque-review-credential")
                            );
                            assert!(!body.truncated());
                            encoded = serde_json::to_string(&encoded).unwrap();
                        }
                    }
                }
            }
        }
        let raw = br#"provider response: {"reason":"unsupported dimension","headers":[{"na\u006de":"X-Api-\u004bey","va\u006cue":"opaque-review-credential"}]"#;
        let body = sanitize_error_body(raw, raw.len() as u64).unwrap();
        assert!(body.text().contains("unsupported dimension"));
        assert!(!body.text().contains("opaque-review-credential"));
    }

    #[test]
    fn incomplete_header_pair_values_are_redacted_through_the_input_cap() {
        for value in [
            r#""opaque-review-credential"#,
            r#"{"nested":["opaque-review-credential"#,
            r#"["opaque-review-credential"#,
        ] {
            let prefix = format!(
                r#"provider response: {{"reason":"unsupported dimension","headers":[{{"name":"X-Api-Key","value":{value}"#
            );
            for raw in [
                prefix.clone(),
                format!("{prefix}{}", "opaque-review-credential".repeat(4000)),
            ] {
                let encoded = serde_json::to_string(&raw).unwrap();
                for input in [&raw, &encoded, &encoded[..encoded.len() - 1]] {
                    let body = sanitize_error_body(input.as_bytes(), input.len() as u64).unwrap();
                    assert!(body.text().contains("unsupported dimension"));
                    assert!(!body.text().contains("opaque-review"));
                    assert!(body.len_bytes() <= SanitizedBody::MAX_BYTES);
                    assert_eq!(body.truncated(), input.len() > SanitizedBody::MAX_BYTES);
                    assert!(
                        !serde_json::to_string(&body)
                            .unwrap()
                            .contains("opaque-review")
                    );
                }
            }
        }
    }

    #[test]
    fn header_pair_associations_stay_within_their_object() {
        for raw in [
            r#"provider: [{"name":"X-Api-Key"},{"name":"X-Request-Id","value":"harmless-review-value"}]"#,
            r#"provider: {"name":"Authorization","details":{"value":"harmless-review-value"}}"#,
            r#"provider: {"reason":"unsupported dimension","value":"harmless-review-value"}"#,
        ] {
            let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
            assert!(body.text().contains("harmless-review-value"));
        }
    }

    #[test]
    fn http_url_userinfo_is_redacted_in_json_text_and_encoded_strings() {
        for url in [
            "https://review-user:review-password@example.test/path?mode=fast",
            "https://review-user:review'password@example.test/path?mode=fast",
            "HTTP://review-user:review-password@example.test/path?mode=fast",
            "https://review%2Duser:review%2Dpassword%40extra@example.test/path?mode=fast",
            "https://review-user@example.test/path?mode=fast",
        ] {
            let message = format!("gateway rejected {url}; unsupported dimension");
            let envelope = serde_json::json!({"message": message}).to_string();
            for raw in [
                message,
                envelope.clone(),
                format!("provider response: {envelope}"),
                serde_json::to_string(&envelope).unwrap(),
            ] {
                let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
                assert!(body.text().contains("gateway rejected"));
                assert!(body.text().contains("unsupported dimension"));
                assert!(
                    body.text()
                        .contains("[REDACTED]@example.test/path?mode=fast")
                );
                assert!(!serde_json::to_string(&body).unwrap().contains("review"));
                assert!(!body.truncated());
            }
        }
        for prefix in ["", "provider response: "] {
            let raw = format!(
                r#"{prefix}{{"message":"gateway rejected https:\/\/review\u002duser:review\u002dpassword@example.test/path?mode=fast"}}"#
            );
            let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
            assert!(
                body.text()
                    .contains("[REDACTED]@example.test/path?mode=fast")
            );
            assert!(!serde_json::to_string(&body).unwrap().contains("review"));
        }
        let raw = b"gateway rejected https://example.test/path/user@example.test?mode=fast";
        assert_eq!(
            sanitize_error_body(raw, raw.len() as u64).unwrap().text(),
            std::str::from_utf8(raw).unwrap()
        );
    }

    #[test]
    fn http_url_userinfo_accepts_rfc3986_sub_delimiters() {
        for delimiter in "!$&'()*+,;=".chars() {
            let url = format!(
                "https://review{delimiter}user:review{delimiter}password@example.test/path?mode=fast"
            );
            let message = format!("gateway rejected '{url}'; unsupported dimension");
            let envelope = serde_json::json!({"message":message}).to_string();
            for raw in [
                message,
                envelope.clone(),
                format!("provider response: {envelope}"),
            ] {
                let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
                assert!(body.text().contains("gateway rejected"));
                assert!(body.text().contains("unsupported dimension"));
                assert!(
                    body.text()
                        .contains("https://[REDACTED]@example.test/path?mode=fast")
                );
                assert!(
                    !body.text().contains("review"),
                    "userinfo delimiter {delimiter}"
                );
                assert!(!serde_json::to_string(&body).unwrap().contains("review"));
                assert!(!body.truncated());
            }
        }
    }

    #[test]
    fn http_authority_scanning_stops_before_unrelated_prose_at_signs() {
        for message in [
            "gateway rejected 'https://example.test' contact ops@example.test",
            "gateway rejected https://example.test/path/ops@example.test",
            "gateway rejected https://example.test?contact=ops@example.test",
            "gateway rejected https://example.test#ops@example.test",
            "gateway rejected https://example.test\"ops@example.test",
            "gateway rejected https://example.test<ops@example.test",
            "gateway rejected https://example.test>ops@example.test",
            "gateway rejected https://example.test\\ops@example.test",
            "gateway rejected https://example.test\nops@example.test",
            "gateway rejected https://example.test\0ops@example.test",
        ] {
            let expected = message
                .chars()
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect::<String>();
            let body = sanitize_error_body(message.as_bytes(), message.len() as u64).unwrap();
            assert_eq!(body.text(), expected);
            let envelope = serde_json::json!({"message":message}).to_string();
            for prefix in ["", "provider response: "] {
                let raw = format!("{prefix}{envelope}");
                let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
                let value: JsonValue =
                    serde_json::from_str(body.text().strip_prefix(prefix).unwrap()).unwrap();
                assert_eq!(value["message"], expected);
                assert!(!body.text().contains(REDACTED));
            }
        }
    }

    #[test]
    fn fragment_scanning_and_string_decoding_have_bounded_depth() {
        let pair = r#"{"name":"X-Api-Key","value":"opaque-review-credential"}"#;
        // More nesting than serde's recursion limit, without recursive fragment parsing.
        let raw = format!(
            "unsupported dimension {}{}{}",
            "[".repeat(256),
            pair,
            "]".repeat(256)
        );
        let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
        assert!(body.text().contains("unsupported dimension"));
        assert!(!body.text().contains("opaque-review-credential"));
        let mut encoded = pair.to_owned();
        for _ in 0..MAX_STRING_LAYERS + 2 {
            encoded = serde_json::to_string(&encoded).unwrap();
        }
        let raw = format!("unsupported dimension {encoded}");
        assert!(raw.len() < SanitizedBody::MAX_BYTES);
        let body = sanitize_error_body(raw.as_bytes(), raw.len() as u64).unwrap();
        assert!(body.text().contains("unsupported dimension"));
        assert!(!body.text().contains("opaque-review-credential"));
    }
}
