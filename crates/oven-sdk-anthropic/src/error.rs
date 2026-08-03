//! Anthropic error-envelope classification.

use std::time::{Duration, SystemTime};

use oven_sdk::{ErrorStage, JsonValue, ModelError, ModelErrorKind};
use reqwest::header::HeaderMap;

use crate::wire::Protocol;

/// Parses and classifies a provider error envelope.
pub fn classify_error(
    status: u16,
    body: &[u8],
    request_id: Option<String>,
    stage: ErrorStage,
    bytes: u64,
    headers: &HeaderMap,
) -> ModelError {
    classify_error_for(
        Protocol::Anthropic,
        status,
        body,
        request_id,
        stage,
        bytes,
        headers,
    )
}

pub(crate) fn classify_error_for(
    protocol: Protocol,
    status: u16,
    body: &[u8],
    request_id: Option<String>,
    stage: ErrorStage,
    bytes: u64,
    headers: &HeaderMap,
) -> ModelError {
    let value: JsonValue = serde_json::from_slice(body).unwrap_or(JsonValue::Null);
    let code = value
        .pointer("/error/type")
        .or_else(|| value.get("type"))
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    let provider_message = value
        .pointer("/error/message")
        .and_then(JsonValue::as_str)
        .unwrap_or("Messages request failed");
    let message = format!("{} request failed", protocol.display_name());
    let lower = format!("{code} {provider_message}").to_lowercase();
    let request_id = request_id
        .or_else(|| {
            value
                .get("request_id")
                .and_then(JsonValue::as_str)
                .map(str::to_owned)
        })
        .and_then(|value| safe_identifier(&value));
    let mut error = if status == 401
        || lower.contains("authentication")
        || (protocol == Protocol::MiniMax && matches!(code, "1004" | "2049"))
    {
        ModelError::new(ModelErrorKind::Auth, message)
    } else if status == 403 || lower.contains("permission") {
        ModelError::new(ModelErrorKind::PermissionDenied, message)
    } else if has_model_code(&value)
        || (protocol == Protocol::MiniMax && status == 404 && code == "not_found_error")
    {
        ModelError::new(ModelErrorKind::ModelNotFound, message)
    } else if protocol.is_first_party()
        && [
            "context length",
            "context window",
            "maximum context",
            "maximum number of tokens",
            "context_length_exceeded",
            "too many tokens",
            "prompt is too long",
        ]
        .iter()
        .any(|m| lower.contains(m))
    {
        ModelError::new(ModelErrorKind::ContextLength, message)
    } else if lower.contains("insufficient_quota")
        || (protocol == Protocol::MiniMax && matches!(code, "1008" | "2056"))
    {
        ModelError::new(ModelErrorKind::Quota, message)
    } else if status == 429 || lower.contains("rate_limit") {
        ModelError::rate_limited(message)
    } else if status == 529 || lower.contains("overload") {
        ModelError::new(ModelErrorKind::Overload, message).with_retryable(true)
    } else if status == 408
        || status == 504
        || lower.contains("timeout")
        || (protocol == Protocol::MiniMax && code == "1001")
    {
        ModelError::timeout(message)
    } else if status >= 500 {
        ModelError::provider(message).with_retryable(true)
    } else {
        ModelError::invalid_request(message)
    };
    error = error
        .with_http_status(status)
        .with_stage(stage)
        .with_bytes_received(bytes);
    if let Some(body) = positive_sanitized_body(&value) {
        error = error.with_sanitized_body(body);
    }
    if let Some(code) = safe_identifier(code) {
        error = error.with_vendor_code(code);
    }
    if let Some(id) = request_id {
        error = error.with_request_id(id);
    }
    if let Some(delay) = retry_after(headers) {
        error = error.with_retry_after(delay);
    }
    error
}

fn safe_identifier(value: &str) -> Option<String> {
    (!value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.:".contains(&byte)))
    .then(|| value.to_owned())
}

fn positive_sanitized_body(value: &JsonValue) -> Option<oven_sdk::SanitizedBody> {
    let code = value
        .pointer("/error/type")
        .or_else(|| value.get("type"))
        .and_then(JsonValue::as_str)
        .and_then(safe_identifier)?;
    let body = serde_json::json!({"type":"error","error":{"type":code}}).to_string();
    Some(oven_sdk::SanitizedBody::new(body))
}
fn has_model_code(value: &JsonValue) -> bool {
    match value {
        JsonValue::String(value) => [
            "model_not_found",
            "invalid_model",
            "model_does_not_exist",
            "model_doesnt_exist",
            "model_not_exist",
        ]
        .contains(&value.as_str()),
        JsonValue::Array(values) => values.iter().any(has_model_code),
        JsonValue::Object(values) => values.values().any(has_model_code),
        _ => false,
    }
}
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    if let Some(value) = headers
        .get("retry-after-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
    {
        return Some(Duration::from_millis(value));
    }
    let value = headers.get("retry-after")?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    httpdate::parse_http_date(value)
        .ok()?
        .duration_since(SystemTime::now())
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    #[test]
    fn error_classification_prefers_model_not_found_over_500() {
        let error = classify_error_for(
            Protocol::Anthropic,
            500,
            br#"{"type":"error","error":{"type":"model_not_found","message":"missing"}}"#,
            Some("req_1".into()),
            ErrorStage::ResponseBody,
            7,
            &HeaderMap::new(),
        );
        assert_eq!(error.kind, ModelErrorKind::ModelNotFound);
        assert_eq!(error.diagnostics.request_id.as_deref(), Some("req_1"));
    }

    #[test]
    fn error_classification_records_retry_after_milliseconds() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after-ms", HeaderValue::from_static("25"));
        let error = classify_error_for(
            Protocol::Anthropic,
            429,
            br#"{"error":{"type":"rate_limit_error","message":"slow down"}}"#,
            None,
            ErrorStage::ResponseBody,
            0,
            &headers,
        );
        assert_eq!(error.kind, ModelErrorKind::RateLimited);
        assert_eq!(
            error.diagnostics.retry_after,
            Some(Duration::from_millis(25))
        );
    }
}
