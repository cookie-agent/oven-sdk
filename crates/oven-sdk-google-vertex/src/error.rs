//! Vertex Google error-envelope classification.

use std::time::{Duration, SystemTime};

use oven_sdk::{ErrorStage, JsonValue, ModelError, ModelErrorKind};
use reqwest::header::HeaderMap;

/// Parses and classifies a Google Cloud JSON error envelope.
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
        .unwrap_or("Vertex Gemini request failed");
    let lower = format!("{code} {provider_message}").to_lowercase();
    let mut error = if status == 401 || lower.contains("unauthenticated") {
        ModelError::new(ModelErrorKind::Auth, "Vertex OAuth authentication failed")
    } else if status == 403 || lower.contains("permission_denied") {
        ModelError::new(ModelErrorKind::PermissionDenied, "Vertex permission denied")
    } else if status == 404 && (lower.contains("model") || lower.contains("not_found")) {
        ModelError::new(
            ModelErrorKind::ModelNotFound,
            "Vertex Gemini resource was not found",
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
            "Vertex Gemini context limit exceeded",
        )
    } else if status == 429 && has_quota_evidence(&value, provider_message) {
        ModelError::new(ModelErrorKind::Quota, "Vertex quota exhausted")
    } else if status == 429
        && [
            "capacity",
            "overload",
            "overloaded",
            "temporarily unavailable",
        ]
        .iter()
        .any(|needle| lower.contains(needle))
    {
        ModelError::new(
            ModelErrorKind::Overload,
            "Vertex Gemini capacity is unavailable",
        )
        .with_retryable(true)
    } else if status == 429 {
        ModelError::rate_limited("Vertex rate limited the request")
    } else if status == 408 || status == 504 || lower.contains("deadline_exceeded") {
        ModelError::timeout("Vertex Gemini request timed out")
    } else if status == 503 || lower.contains("unavailable") || lower.contains("overload") {
        ModelError::new(ModelErrorKind::Overload, "Vertex Gemini is unavailable")
            .with_retryable(true)
    } else if status >= 500 {
        ModelError::provider("Vertex Gemini provider request failed").with_retryable(true)
    } else {
        ModelError::invalid_request("Vertex Gemini rejected the request")
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

fn has_quota_evidence(value: &JsonValue, message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    let textual = message.contains("quota")
        && ["exceed", "exhaust", "limit", "insufficient"]
            .iter()
            .any(|needle| message.contains(needle));
    let structured = value
        .pointer("/error/details")
        .and_then(JsonValue::as_array)
        .is_some_and(|details| {
            details.iter().any(|detail| {
                detail
                    .get("@type")
                    .and_then(JsonValue::as_str)
                    .is_some_and(|value| value.ends_with("/google.rpc.QuotaFailure"))
                    || detail
                        .get("reason")
                        .and_then(JsonValue::as_str)
                        .is_some_and(|value| value.to_ascii_uppercase().contains("QUOTA"))
                    || detail.as_object().is_some_and(|object| {
                        object.keys().any(|key| quota_key(key))
                            || object
                                .get("metadata")
                                .and_then(JsonValue::as_object)
                                .is_some_and(|metadata| metadata.keys().any(|key| quota_key(key)))
                    })
            })
        });
    textual || structured
}

fn quota_key(key: &str) -> bool {
    matches!(
        key,
        "quotaInfo"
            | "quotaMetric"
            | "quotaLimit"
            | "quotaLimitValue"
            | "quota_info"
            | "quota_metric"
            | "quota_limit"
            | "quota_limit_value"
    )
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
    fn auth_model_and_context_errors_are_typed() {
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
    fn resource_exhausted_requires_positive_quota_evidence() {
        let headers = HeaderMap::new();
        for (body, expected) in [
            (
                br#"{"error":{"status":"RESOURCE_EXHAUSTED","message":"request rate exceeded"}}"#.as_slice(),
                ModelErrorKind::RateLimited,
            ),
            (
                br#"{"error":{"status":"RESOURCE_EXHAUSTED","message":"temporary model capacity exhausted"}}"#.as_slice(),
                ModelErrorKind::Overload,
            ),
            (
                br#"{"error":{"status":"RESOURCE_EXHAUSTED","message":"quota limit exceeded"}}"#.as_slice(),
                ModelErrorKind::Quota,
            ),
            (
                br#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure"}]}}"#.as_slice(),
                ModelErrorKind::Quota,
            ),
            (
                br#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","metadata":{"quota_limit":"requests-per-minute"}}]}}"#.as_slice(),
                ModelErrorKind::Quota,
            ),
        ] {
            let error = classify_error(
                429,
                body,
                None,
                ErrorStage::ResponseBody,
                body.len() as u64,
                &headers,
            );
            assert_eq!(error.kind, expected);
            assert!(error.retryable || expected == ModelErrorKind::Quota);
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
