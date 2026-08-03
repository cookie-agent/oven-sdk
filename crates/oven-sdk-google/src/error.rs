//! Google error-envelope classification.

use std::time::{Duration, SystemTime};

use oven_sdk::{ErrorStage, JsonValue, ModelError, ModelErrorKind};
use reqwest::header::HeaderMap;

/// Parses and classifies a Google JSON error envelope.
pub fn classify_error(
    status: u16,
    body: &[u8],
    request_id: Option<String>,
    stage: ErrorStage,
    bytes: u64,
    headers: &HeaderMap,
) -> ModelError {
    let value: JsonValue = serde_json::from_slice(body).unwrap_or(JsonValue::Null);
    let code = value
        .pointer("/error/status")
        .or_else(|| value.pointer("/error/code"))
        .and_then(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .or_else(|| value.as_u64().map(|v| v.to_string()))
        })
        .unwrap_or_default();
    let provider_message = value
        .pointer("/error/message")
        .and_then(JsonValue::as_str)
        .unwrap_or("Google Gemini request failed");
    let lower = format!("{code} {provider_message}").to_lowercase();
    let mut error = if status == 401 || lower.contains("unauthenticated") {
        ModelError::new(ModelErrorKind::Auth, "Google Gemini authentication failed")
    } else if status == 403 || lower.contains("permission_denied") {
        ModelError::new(
            ModelErrorKind::PermissionDenied,
            "Google Gemini permission denied",
        )
    } else if status == 404 && (lower.contains("model") || lower.contains("not_found")) {
        ModelError::new(
            ModelErrorKind::ModelNotFound,
            "Google Gemini model was not found",
        )
    } else if [
        "context length",
        "token limit",
        "too many tokens",
        "prompt is too long",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
    {
        ModelError::new(
            ModelErrorKind::ContextLength,
            "Google Gemini context limit exceeded",
        )
    } else if status == 429 && (lower.contains("quota") || lower.contains("resource_exhausted")) {
        ModelError::new(ModelErrorKind::Quota, "Google Gemini quota exhausted")
    } else if status == 429 {
        ModelError::rate_limited("Google Gemini rate limited the request")
    } else if status == 408 || status == 504 || lower.contains("deadline_exceeded") {
        ModelError::timeout("Google Gemini request timed out")
    } else if status == 503 || lower.contains("unavailable") || lower.contains("overload") {
        ModelError::new(ModelErrorKind::Overload, "Google Gemini is unavailable")
            .with_retryable(true)
    } else if status >= 500 {
        ModelError::provider("Google Gemini provider request failed").with_retryable(true)
    } else {
        ModelError::invalid_request("Google Gemini rejected the request")
    };
    error = error
        .with_http_status(status)
        .with_stage(stage)
        .with_bytes_received(bytes);
    if let Some(code) = safe_identifier(&code) {
        error = error.with_vendor_code(code);
    }
    if let Some(request_id) = request_id.and_then(|value| safe_identifier(&value)) {
        error = error.with_request_id(request_id);
    }
    if let Some(delay) = retry_after(headers) {
        error = error.with_retry_after(delay);
    }
    if !code.is_empty() {
        error = error.with_sanitized_body(oven_sdk::SanitizedBody::new(
            serde_json::json!({"error":{"status":code}}).to_string(),
        ));
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

fn retry_after(headers: &HeaderMap) -> Option<Duration> {
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
    fn auth_model_context_and_quota_errors_are_typed() {
        let headers = HeaderMap::new();
        for (status, body, expected) in [
            (
                401,
                br#"{"error":{"status":"UNAUTHENTICATED"}}"#.as_slice(),
                ModelErrorKind::Auth,
            ),
            (
                404,
                br#"{"error":{"status":"NOT_FOUND","message":"model missing"}}"#.as_slice(),
                ModelErrorKind::ModelNotFound,
            ),
            (
                400,
                br#"{"error":{"message":"prompt is too long"}}"#.as_slice(),
                ModelErrorKind::ContextLength,
            ),
            (
                429,
                br#"{"error":{"status":"RESOURCE_EXHAUSTED","message":"quota"}}"#.as_slice(),
                ModelErrorKind::Quota,
            ),
        ] {
            assert_eq!(
                classify_error(
                    status,
                    body,
                    None,
                    ErrorStage::ResponseBody,
                    body.len() as u64,
                    &headers
                )
                .kind,
                expected
            );
        }
    }

    #[test]
    fn retry_after_and_safe_diagnostics_are_preserved() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("7"));
        let error = classify_error(
            429,
            br#"{"error":{"status":"RATE_LIMITED","message":"secret detail"}}"#,
            Some("request-1".into()),
            ErrorStage::ResponseBody,
            64,
            &headers,
        );
        assert_eq!(error.kind, ModelErrorKind::RateLimited);
        assert_eq!(error.diagnostics.retry_after, Some(Duration::from_secs(7)));
        assert_eq!(error.diagnostics.request_id.as_deref(), Some("request-1"));
        assert_eq!(
            error.diagnostics.vendor_code.as_deref(),
            Some("RATE_LIMITED")
        );
        assert!(!error.to_string().contains("secret detail"));
    }
}
