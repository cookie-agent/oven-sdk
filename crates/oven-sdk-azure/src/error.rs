//! Azure OpenAI error-envelope parsing and classification.

use std::time::{Duration, SystemTime};

use oven_sdk::{ErrorStage, JsonValue, ModelError, ModelErrorKind, SanitizedBody};
use reqwest::header::HeaderMap;

/// Classifies an Azure OpenAI error envelope into the core taxonomy.
pub(crate) fn classify_error(
    status: u16,
    body: &[u8],
    request_id: Option<String>,
    stage: ErrorStage,
    bytes: u64,
    headers: &HeaderMap,
) -> ModelError {
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
        .unwrap_or("Azure OpenAI request failed");
    let lower = format!(
        "{} {kind_text} {provider_message}",
        code.as_deref().unwrap_or("")
    )
    .to_lowercase();
    let mut error = if has_model_code(&value) {
        ModelError::new(ModelErrorKind::ModelNotFound, "Azure OpenAI request failed")
    } else if status == 401 || lower.contains("auth") || lower.contains("api key") {
        ModelError::new(ModelErrorKind::Auth, "Azure OpenAI request failed")
    } else if status == 403 || lower.contains("permission") {
        ModelError::new(
            ModelErrorKind::PermissionDenied,
            "Azure OpenAI request failed",
        )
    } else if lower.contains("content_filter") || lower.contains("responsibleaipolicyviolation") {
        ModelError::new(ModelErrorKind::ContentFilter, "Azure OpenAI request failed")
    } else if context_error(&lower) {
        ModelError::new(ModelErrorKind::ContextLength, "Azure OpenAI request failed")
    } else if lower.contains("insufficient_quota")
        || lower.contains("quota_exceeded")
        || lower.contains("outofquota")
    {
        ModelError::new(ModelErrorKind::Quota, "Azure OpenAI request failed")
    } else if lower.contains("no_capacity") || lower.contains("capacity") {
        ModelError::new(ModelErrorKind::Overload, "Azure OpenAI request failed")
            .with_retryable(true)
    } else if status == 429 || lower.contains("rate_limit") {
        ModelError::rate_limited("Azure OpenAI request failed")
    } else if lower.contains("overload") {
        ModelError::new(ModelErrorKind::Overload, "Azure OpenAI request failed")
            .with_retryable(true)
    } else if status == 408 || status == 504 || lower.contains("timeout") {
        ModelError::timeout("Azure OpenAI request failed")
    } else if status >= 500 {
        ModelError::provider("Azure OpenAI request failed").with_retryable(true)
    } else if status == 404 {
        ModelError::new(ModelErrorKind::ModelNotFound, "Azure OpenAI request failed")
    } else {
        ModelError::invalid_request("Azure OpenAI request failed")
    };
    error = error
        .with_http_status(status)
        .with_stage(stage)
        .with_bytes_received(bytes);
    if let Some(safe_body) = safe_diagnostic_body(&value) {
        error = error.with_sanitized_body(SanitizedBody::new(safe_body));
    }
    if let Some(code) = code.as_deref().and_then(safe_code) {
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

fn safe_diagnostic_body(value: &JsonValue) -> Option<String> {
    let codes = [
        value.pointer("/error/code"),
        value.pointer("/error/type"),
        value.pointer("/error/inner_error/code"),
        value.pointer("/code"),
        value.pointer("/type"),
    ]
    .into_iter()
    .flatten()
    .filter_map(JsonValue::as_str)
    .filter_map(safe_code)
    .collect::<Vec<_>>();
    (!codes.is_empty()).then(|| serde_json::json!({"codes":codes}).to_string())
}

fn safe_code(value: &str) -> Option<String> {
    let lower = value.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "model_not_found"
            | "invalid_model"
            | "model_does_not_exist"
            | "model_doesnt_exist"
            | "model_not_exist"
            | "authentication_error"
            | "invalid_api_key"
            | "permission_denied"
            | "content_filter"
            | "responsibleaipolicyviolation"
            | "context_length_exceeded"
            | "insufficient_quota"
            | "quota_exceeded"
            | "outofquota"
            | "no_capacity"
            | "rate_limit_error"
            | "rate_limit_exceeded"
            | "overloaded_error"
            | "timeout"
            | "invalid_request_error"
    )
    .then_some(lower)
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
        .get("x-ms-retry-after-ms")
        .or_else(|| headers.get("retry-after-ms"))
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

    #[test]
    fn arbitrary_error_text_and_codes_are_never_retained() {
        let secret = "sk-secret-value";
        let error = classify_error(
            400,
            format!(
                r#"{{"error":{{"code":"{secret}","message":"token {secret}","details":{{"raw":"{secret}"}}}}}}"#
            )
            .as_bytes(),
            None,
            ErrorStage::ResponseBody,
            0,
            &HeaderMap::new(),
        );
        for rendered in [
            format!("{error}"),
            format!("{error:?}"),
            serde_json::to_string(&error).unwrap(),
        ] {
            assert!(!rendered.contains(secret));
        }
        assert!(error.diagnostics.vendor_code.is_none());
        assert!(error.diagnostics.sanitized_body.is_none());
    }

    #[test]
    fn diagnostics_retain_only_allow_listed_codes() {
        let error = classify_error(
            429,
            br#"{"error":{"code":"insufficient_quota","type":"rate_limit_error","message":"secret"}}"#,
            None,
            ErrorStage::ResponseBody,
            0,
            &HeaderMap::new(),
        );
        let rendered = serde_json::to_string(&error).unwrap();
        assert!(rendered.contains("insufficient_quota"));
        assert!(rendered.contains("rate_limit_error"));
        assert!(!rendered.contains("secret"));
    }
}
