//! Exact-byte AWS Signature Version 4 signing for Bedrock Runtime.

use std::{collections::BTreeSet, time::SystemTime};

use hmac::{Hmac, Mac};
use oven_sdk::{ErrorStage, ModelError};
use reqwest::header::{AUTHORIZATION, HOST, HeaderMap, HeaderName, HeaderValue};
use sha2::{Digest, Sha256};
use time::{OffsetDateTime, macros::format_description};
use url::Url;

use crate::AwsCredentials;

const SERVICE: &str = "bedrock";

/// Signs a Bedrock request at the current system time.
pub fn sign(
    method: &str,
    url: &Url,
    body: &[u8],
    headers: &mut HeaderMap,
    region: &str,
    credentials: &AwsCredentials,
) -> Result<(), ModelError> {
    sign_at(
        method,
        url,
        body,
        headers,
        region,
        credentials,
        SystemTime::now(),
    )
}

/// Signs a Bedrock request at an explicit time for deterministic tests.
pub fn sign_at(
    method: &str,
    url: &Url,
    body: &[u8],
    headers: &mut HeaderMap,
    region: &str,
    credentials: &AwsCredentials,
    now: SystemTime,
) -> Result<(), ModelError> {
    if credentials.access_key_id.is_empty() || credentials.secret_access_key.is_empty() {
        return Err(signing_error("AWS credentials must not be empty"));
    }
    let timestamp = OffsetDateTime::from(now);
    let amz_date = timestamp
        .format(format_description!(
            "[year][month][day]T[hour][minute][second]Z"
        ))
        .map_err(|_| signing_error("could not format AWS signing timestamp"))?;
    let short_date = timestamp
        .format(format_description!("[year][month][day]"))
        .map_err(|_| signing_error("could not format AWS signing date"))?;
    let host = url
        .host_str()
        .ok_or_else(|| signing_error("Bedrock URL has no host"))?;
    let host = match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_owned(),
    };
    let payload_hash = hex_sha256(body);
    headers.insert(HOST, header_value(&host, "invalid Bedrock endpoint host")?);
    headers.insert(
        HeaderName::from_static("x-amz-date"),
        header_value(&amz_date, "invalid AWS signing timestamp")?,
    );
    headers.insert(
        HeaderName::from_static("x-amz-content-sha256"),
        header_value(&payload_hash, "invalid AWS payload hash")?,
    );
    if let Some(token) = &credentials.session_token {
        headers.insert(
            HeaderName::from_static("x-amz-security-token"),
            header_value(token, "invalid AWS session token")?,
        );
    }
    headers.remove(AUTHORIZATION);

    let canonical_uri = canonical_uri(url);
    let canonical_query = canonical_query(url);
    let (canonical_headers, signed_headers) = canonical_headers(headers)?;
    let canonical_request = format!(
        "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );
    let scope = format!("{short_date}/{region}/{SERVICE}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex_sha256(canonical_request.as_bytes())
    );
    let date_key = hmac(
        format!("AWS4{}", credentials.secret_access_key).as_bytes(),
        short_date.as_bytes(),
    )?;
    let region_key = hmac(&date_key, region.as_bytes())?;
    let service_key = hmac(&region_key, SERVICE.as_bytes())?;
    let signing_key = hmac(&service_key, b"aws4_request")?;
    let signature = hex::encode(hmac(&signing_key, string_to_sign.as_bytes())?);
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
        credentials.access_key_id
    );
    headers.insert(
        AUTHORIZATION,
        header_value(&authorization, "invalid AWS authorization header")?,
    );
    Ok(())
}

fn canonical_query(url: &Url) -> String {
    let mut pairs = url
        .query_pairs()
        .map(|(key, value)| {
            (
                percent_encode(key.as_bytes()),
                percent_encode(value.as_bytes()),
            )
        })
        .collect::<Vec<_>>();
    pairs.sort();
    pairs
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

fn canonical_uri(url: &Url) -> String {
    let path = if url.path().is_empty() {
        "/"
    } else {
        url.path()
    };
    let mut output = String::with_capacity(path.len());
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'/') {
            output.push(char::from(byte));
        } else {
            use std::fmt::Write as _;
            write!(&mut output, "%{byte:02X}").expect("writing to String cannot fail");
        }
    }
    output
}

fn percent_encode(value: &[u8]) -> String {
    let mut output = String::new();
    for byte in value {
        if byte.is_ascii_alphanumeric() || matches!(*byte, b'-' | b'_' | b'.' | b'~') {
            output.push(char::from(*byte));
        } else {
            use std::fmt::Write as _;
            write!(&mut output, "%{byte:02X}").expect("writing to String cannot fail");
        }
    }
    output
}

fn canonical_headers(headers: &HeaderMap) -> Result<(String, String), ModelError> {
    const VOLATILE: &[&str] = &[
        "connection",
        "user-agent",
        "x-amzn-trace-id",
        "transfer-encoding",
        "content-length",
    ];
    let names = headers
        .keys()
        .filter(|name| *name != AUTHORIZATION && !VOLATILE.contains(&name.as_str()))
        .map(|name| name.as_str().to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    let mut canonical = String::new();
    for name in &names {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| signing_error("AWS signed header name is invalid"))?;
        let values = headers
            .get_all(header_name)
            .iter()
            .map(|value| {
                value
                    .to_str()
                    .map(|value| value.split_whitespace().collect::<Vec<_>>().join(" "))
                    .map_err(|_| signing_error("AWS signed header is not valid text"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        canonical.push_str(name);
        canonical.push(':');
        canonical.push_str(&values.join(","));
        canonical.push('\n');
    }
    Ok((canonical, names.into_iter().collect::<Vec<_>>().join(";")))
}

fn hex_sha256(value: &[u8]) -> String {
    hex::encode(Sha256::digest(value))
}

fn hmac(key: &[u8], value: &[u8]) -> Result<Vec<u8>, ModelError> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key)
        .map_err(|_| signing_error("could not initialize AWS signer"))?;
    mac.update(value);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn header_value(value: &str, message: &str) -> Result<HeaderValue, ModelError> {
    HeaderValue::from_str(value).map_err(|_| signing_error(message))
}

fn signing_error(message: &str) -> ModelError {
    ModelError::invalid_request(message).with_stage(ErrorStage::RequestEncoding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_body_session_token_and_bedrock_scope_are_signed() {
        let url = Url::parse("https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123%3Ainference-profile%2Fx/converse-stream").unwrap();
        let credentials = AwsCredentials {
            access_key_id: "AKIDEXAMPLE".into(),
            secret_access_key: "secret".into(),
            session_token: Some("session".into()),
        };
        let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_767_225_600);
        let mut first = HeaderMap::new();
        first.insert("content-type", HeaderValue::from_static("application/json"));
        let mut second = first.clone();
        sign_at(
            "POST",
            &url,
            br#"{"a":1}"#,
            &mut first,
            "us-east-1",
            &credentials,
            now,
        )
        .unwrap();
        sign_at(
            "POST",
            &url,
            br#"{"a":2}"#,
            &mut second,
            "us-east-1",
            &credentials,
            now,
        )
        .unwrap();
        assert_ne!(first[AUTHORIZATION], second[AUTHORIZATION]);
        let auth = first[AUTHORIZATION].to_str().unwrap();
        assert!(auth.contains("/us-east-1/bedrock/aws4_request"));
        assert!(auth.contains("x-amz-content-sha256"));
        assert_eq!(first["x-amz-security-token"], "session");
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260101/us-east-1/bedrock/aws4_request, SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token, Signature=e4d1d528bb908d27164cac5f614627b78f0e61f0123f42fcace9b4b41d4f9e1b"
        );
    }

    #[test]
    fn canonical_query_and_headers_are_stable() {
        let url = Url::parse("https://example.com/?z=a%20b&a=%2F").unwrap();
        assert_eq!(canonical_query(&url), "a=%2F&z=a%20b");
        let mut headers = HeaderMap::new();
        headers.append("x-test", HeaderValue::from_static("a   b"));
        headers.append("x-test", HeaderValue::from_static("c\t d"));
        headers.insert("content-length", HeaderValue::from_static("1"));
        let (canonical, signed) = canonical_headers(&headers).unwrap();
        assert_eq!(canonical, "x-test:a b,c d\n");
        assert_eq!(signed, "x-test");
    }

    #[test]
    fn aws_sdk_non_s3_double_encoding_vectors_match() {
        // AWS SDK for Rust signing-suite vectors:
        // smithy-rs/aws/rust-runtime/aws-sigv4/aws-signing-test-suite/v4/
        // double-encode-path and double-url-encode.
        let connections = Url::parse(
            "https://tj9n5r0m12.execute-api.us-east-1.amazonaws.com/test/@connections/JBDvjfGEIAMCERw%3D",
        )
        .unwrap();
        assert_eq!(
            canonical_uri(&connections),
            "/test/%40connections/JBDvjfGEIAMCERw%253D"
        );
        assert_eq!(connections.path(), "/test/@connections/JBDvjfGEIAMCERw%3D");

        let lambda = Url::parse("https://lambda.us-east-2.amazonaws.com/2015-03-31/functions/arn%3Aaws%3Alambda%3Aus-west-2%3A892717189312%3Afunction%3Amy-rusty-fun/invocations").unwrap();
        assert_eq!(
            canonical_uri(&lambda),
            "/2015-03-31/functions/arn%253Aaws%253Alambda%253Aus-west-2%253A892717189312%253Afunction%253Amy-rusty-fun/invocations"
        );
        assert!(lambda.path().contains("arn%3Aaws%3Alambda"));
    }

    #[test]
    fn bedrock_model_and_resource_arns_are_double_encoded_only_for_signing() {
        for (url, expected) in [
            (
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
                "/model/anthropic.claude-haiku-4-5-20251001-v1%253A0/converse",
            ),
            (
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Fexample/converse-stream",
                "/model/arn%253Aaws%253Abedrock%253Aus-east-1%253A123456789012%253Ainference-profile%252Fexample/converse-stream",
            ),
        ] {
            let url = Url::parse(url).unwrap();
            assert_eq!(canonical_uri(&url), expected);
            assert!(!url.path().contains("%25"));
        }
    }
}
