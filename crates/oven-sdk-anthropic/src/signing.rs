//! AWS Signature Version 4 signing for Claude Platform on AWS.

use std::{collections::BTreeSet, time::SystemTime};

use hmac::{Hmac, Mac};
use oven_sdk::{ErrorStage, ModelError};
use reqwest::header::{AUTHORIZATION, HOST, HeaderMap, HeaderName, HeaderValue};
use sha2::{Digest, Sha256};
use time::{OffsetDateTime, macros::format_description};
use url::Url;

use crate::config::AnthropicAwsCredentials;

const SERVICE: &str = "aws-external-anthropic";

pub(crate) fn sign(
    method: &str,
    url: &Url,
    body: &[u8],
    headers: &mut HeaderMap,
    region: &str,
    credentials: &AnthropicAwsCredentials,
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

fn sign_at(
    method: &str,
    url: &Url,
    body: &[u8],
    headers: &mut HeaderMap,
    region: &str,
    credentials: &AnthropicAwsCredentials,
    now: SystemTime,
) -> Result<(), ModelError> {
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
        .ok_or_else(|| signing_error("Claude Platform on AWS URL has no host"))?;
    let host = match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_owned(),
    };

    headers.insert(
        HOST,
        HeaderValue::from_str(&host).map_err(|_| signing_error("invalid AWS endpoint host"))?,
    );
    headers.insert(
        HeaderName::from_static("x-amz-date"),
        HeaderValue::from_str(&amz_date)
            .map_err(|_| signing_error("invalid AWS signing timestamp"))?,
    );
    if let Some(token) = &credentials.session_token {
        headers.insert(
            HeaderName::from_static("x-amz-security-token"),
            HeaderValue::from_str(token.expose_secret())
                .map_err(|_| signing_error("invalid AWS session token header"))?,
        );
    }
    headers.remove(AUTHORIZATION);

    let payload_hash = hex_sha256(body);
    let canonical_uri = if url.path().is_empty() {
        "/"
    } else {
        url.path()
    };
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
        format!("AWS4{}", credentials.secret_access_key.expose_secret()).as_bytes(),
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
        HeaderValue::from_str(&authorization)
            .map_err(|_| signing_error("invalid AWS authorization header"))?,
    );
    Ok(())
}

fn canonical_query(url: &Url) -> String {
    let mut pairs = url
        .query_pairs()
        .map(|(key, value)| (percent_encode(&key), percent_encode(&value)))
        .collect::<Vec<_>>();
    pairs.sort();
    pairs
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

fn percent_encode(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes())
        .collect::<String>()
        .replace('+', "%20")
}

fn canonical_headers(headers: &HeaderMap) -> Result<(String, String), ModelError> {
    let names = headers
        .keys()
        .filter(|name| *name != AUTHORIZATION)
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
            .collect::<Result<Vec<_>, ModelError>>()?;
        canonical.push_str(name);
        canonical.push(':');
        canonical.push_str(&values.join(","));
        canonical.push('\n');
    }
    let signed_headers = names.into_iter().collect::<Vec<_>>().join(";");
    Ok((canonical, signed_headers))
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

fn signing_error(message: &str) -> ModelError {
    ModelError::invalid_request(message).with_stage(ErrorStage::RequestEncoding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signing_is_deterministic_and_body_sensitive() {
        let url =
            Url::parse("https://aws-external-anthropic.us-west-2.api.aws/v1/messages").unwrap();
        let credentials = AnthropicAwsCredentials {
            access_key_id: "AKIDEXAMPLE".into(),
            secret_access_key: oven_sdk::SecretString::new("secret"),
            session_token: Some(oven_sdk::SecretString::new("session")),
        };
        let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_767_225_600);
        let mut first = HeaderMap::from_iter([
            (
                "content-type".parse().unwrap(),
                "application/json".parse().unwrap(),
            ),
            (
                "anthropic-version".parse().unwrap(),
                "2023-06-01".parse().unwrap(),
            ),
            (
                "anthropic-workspace-id".parse().unwrap(),
                "wrkspc_test".parse().unwrap(),
            ),
        ]);
        first.append("x-custom", HeaderValue::from_static("alpha   beta"));
        first.append("x-custom", HeaderValue::from_static("gamma\t delta"));
        let mut second = first.clone();
        sign_at(
            "POST",
            &url,
            br#"{"a":1}"#,
            &mut first,
            "us-west-2",
            &credentials,
            now,
        )
        .unwrap();
        sign_at(
            "POST",
            &url,
            br#"{"a":2}"#,
            &mut second,
            "us-west-2",
            &credentials,
            now,
        )
        .unwrap();
        assert_ne!(first[AUTHORIZATION], second[AUTHORIZATION]);
        assert!(
            first[AUTHORIZATION]
                .to_str()
                .unwrap()
                .contains("/us-west-2/aws-external-anthropic/aws4_request")
        );
        assert_eq!(first["x-amz-security-token"], "session");
        assert_eq!(
            first[AUTHORIZATION],
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260101/us-west-2/aws-external-anthropic/aws4_request, SignedHeaders=anthropic-version;anthropic-workspace-id;content-type;host;x-amz-date;x-amz-security-token;x-custom, Signature=ceb19fa49d0f01fa39f80ec76c59d2e33bb75fd68018817ef51841200fe49d99"
        );
    }

    #[test]
    fn timestamp_format_is_utc_basic_iso() {
        let instant = OffsetDateTime::UNIX_EPOCH;
        assert_eq!(
            instant
                .format(format_description!(
                    "[year][month][day]T[hour][minute][second]Z"
                ))
                .unwrap(),
            "19700101T000000Z"
        );
    }

    #[test]
    fn repeated_headers_are_joined_once_in_insertion_order() {
        let mut headers = HeaderMap::new();
        headers.append("x-custom", HeaderValue::from_static("alpha   beta"));
        headers.append("x-custom", HeaderValue::from_static("gamma\t delta"));
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        let (canonical, signed) = canonical_headers(&headers).unwrap();
        assert_eq!(
            canonical,
            "content-type:application/json\nx-custom:alpha beta,gamma delta\n"
        );
        assert_eq!(signed, "content-type;x-custom");
    }
}
