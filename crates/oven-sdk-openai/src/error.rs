//! OpenAI error-envelope parsing and classification.

use std::time::{Duration, SystemTime};

use oven_sdk::{ErrorStage, JsonValue, ModelError, ModelErrorKind, SanitizedBody};
use reqwest::header::HeaderMap;

/// Classifies an OpenAI-shaped error envelope into the core taxonomy.
pub(crate) fn classify_error(
    status: u16,
    body: &[u8],
    request_id: Option<String>,
    stage: ErrorStage,
    bytes: u64,
    headers: &HeaderMap,
) -> ModelError {
    let text = String::from_utf8_lossy(body).into_owned();
    let value: JsonValue = serde_json::from_slice(body).unwrap_or(JsonValue::Null);
    let code_value = value
        .pointer("/error/code")
        .or_else(|| value.pointer("/code"));
    let code = code_value.and_then(|value| match value {
        JsonValue::String(value) => Some(value.clone()),
        JsonValue::Number(value) => Some(value.to_string()),
        _ => None,
    });
    let kind_text = value
        .pointer("/error/type")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    let provider_message = value
        .pointer("/error/message")
        .or_else(|| value.get("message"))
        .and_then(JsonValue::as_str)
        .unwrap_or("OpenAI request failed");
    let lower = format!(
        "{} {kind_text} {provider_message}",
        code.as_deref().unwrap_or("")
    )
    .to_lowercase();
    let mut error = if has_model_code(&value) {
        ModelError::new(ModelErrorKind::ModelNotFound, "OpenAI request failed")
    } else if status == 401 || lower.contains("auth") || lower.contains("api key") {
        ModelError::new(ModelErrorKind::Auth, "OpenAI request failed")
    } else if status == 403 || lower.contains("permission") {
        ModelError::new(ModelErrorKind::PermissionDenied, "OpenAI request failed")
    } else if context_error(&lower) {
        ModelError::new(ModelErrorKind::ContextLength, "OpenAI request failed")
    } else if lower.contains("insufficient_quota") {
        ModelError::new(ModelErrorKind::Quota, "OpenAI request failed")
    } else if status == 429 || lower.contains("rate_limit") {
        ModelError::rate_limited("OpenAI request failed")
    } else if lower.contains("overload") {
        ModelError::new(ModelErrorKind::Overload, "OpenAI request failed").with_retryable(true)
    } else if status == 408 || status == 504 || lower.contains("timeout") {
        ModelError::timeout("OpenAI request failed")
    } else if status >= 500 {
        ModelError::provider("OpenAI request failed").with_retryable(true)
    } else if status == 404 {
        ModelError::new(ModelErrorKind::ModelNotFound, "OpenAI request failed")
    } else {
        ModelError::invalid_request("OpenAI request failed")
    };
    error = error
        .with_http_status(status)
        .with_stage(stage)
        .with_bytes_received(bytes)
        .with_sanitized_body(SanitizedBody::new(text));
    if let Some(code) = code {
        error = error.with_vendor_code(code);
    }
    if let Some(request_id) = request_id {
        error = error.with_request_id(request_id);
    }
    if let Some(delay) = retry_after(headers) {
        error = error.with_retry_after(delay);
    }
    error
}

fn context_error(lower: &str) -> bool {
    [
        "context_length_exceeded",
        "context length",
        "context window",
        "maximum context",
        "too many tokens",
        "prompt is too long",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
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
    if let Some(milliseconds) = headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
    {
        return Some(Duration::from_millis(milliseconds));
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
    fn model_not_found_code_overrides_server_status() {
        let error = classify_error(
            500,
            br#"{"error":{"message":"missing","code":"model_not_found"}}"#,
            Some("req_1".into()),
            ErrorStage::ResponseBody,
            42,
            &HeaderMap::new(),
        );
        assert_eq!(error.kind, ModelErrorKind::ModelNotFound);
        assert!(!error.retryable);
        assert_eq!(error.diagnostics.request_id.as_deref(), Some("req_1"));
        assert_eq!(error.diagnostics.bytes_received, 42);
    }

    #[test]
    fn context_quota_rate_limit_and_retry_after_are_typed() {
        let context = classify_error(
            400,
            br#"{"error":{"message":"maximum context length exceeded"}}"#,
            None,
            ErrorStage::ResponseBody,
            0,
            &HeaderMap::new(),
        );
        assert_eq!(context.kind, ModelErrorKind::ContextLength);
        let quota = classify_error(
            429,
            br#"{"error":{"message":"quota","code":"insufficient_quota"}}"#,
            None,
            ErrorStage::ResponseBody,
            0,
            &HeaderMap::new(),
        );
        assert_eq!(quota.kind, ModelErrorKind::Quota);
        assert!(!quota.retryable);
        let mut headers = HeaderMap::new();
        headers.insert("retry-after-ms", HeaderValue::from_static("25"));
        let rate = classify_error(
            429,
            br#"{"error":{"message":"slow","type":"rate_limit_error"}}"#,
            None,
            ErrorStage::ResponseBody,
            0,
            &headers,
        );
        assert_eq!(rate.kind, ModelErrorKind::RateLimited);
        assert!(rate.retryable);
        assert_eq!(
            rate.diagnostics.retry_after,
            Some(Duration::from_millis(25))
        );
    }

    #[test]
    fn retry_after_http_date_is_parsed() {
        let mut headers = HeaderMap::new();
        let future = SystemTime::now() + Duration::from_secs(5);
        headers.insert(
            "retry-after",
            HeaderValue::from_str(&httpdate::fmt_http_date(future)).unwrap(),
        );
        let error = classify_error(
            429,
            br#"{"error":{"message":"slow","type":"rate_limit_error"}}"#,
            None,
            ErrorStage::ResponseBody,
            0,
            &headers,
        );
        assert!(
            error
                .diagnostics
                .retry_after
                .is_some_and(|delay| delay <= Duration::from_secs(5))
        );
    }
}
