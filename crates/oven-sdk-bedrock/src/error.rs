//! Bedrock error-envelope classification.

use oven_sdk::{
    ErrorStage, JsonValue, ModelError, ModelErrorKind,
    provider_support::{parse_retry_after, sanitize_error_body},
};
use reqwest::header::HeaderMap;

/// Parses and classifies a Bedrock JSON error envelope.
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
        .get("__type")
        .or_else(|| value.get("code"))
        .or_else(|| value.get("type"))
        .and_then(JsonValue::as_str)
        .unwrap_or("")
        .rsplit('#')
        .next()
        .unwrap_or("");
    let message = value
        .get("message")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    let lower = format!("{code} {message}").to_ascii_lowercase();
    let mut error = if status == 401 || lower.contains("unrecognizedclient") {
        ModelError::new(ModelErrorKind::Auth, "Bedrock authentication failed")
    } else if status == 403 || lower.contains("accessdenied") {
        ModelError::new(
            ModelErrorKind::PermissionDenied,
            "Bedrock permission denied",
        )
    } else if status == 404 || lower.contains("resourcenotfound") {
        ModelError::new(
            ModelErrorKind::ModelNotFound,
            "Bedrock model resource was not found",
        )
    } else if lower.contains("context") && (lower.contains("window") || lower.contains("token")) {
        ModelError::new(
            ModelErrorKind::ContextLength,
            "Bedrock context limit exceeded",
        )
    } else if status == 408 || status == 504 || lower.contains("modeltimeout") {
        ModelError::timeout("Bedrock model request timed out")
    } else if lower.contains("modelnotready")
        || status == 503
        || lower.contains("serviceunavailable")
    {
        ModelError::new(ModelErrorKind::Overload, "Bedrock model is unavailable")
            .with_retryable(true)
    } else if status == 429 || lower.contains("throttling") {
        ModelError::rate_limited("Bedrock rate limited the request")
    } else if status >= 500 || lower.contains("internalserver") || lower.contains("modelerror") {
        ModelError::provider("Bedrock provider request failed").with_retryable(true)
    } else {
        ModelError::invalid_request("Bedrock rejected the request")
    };
    error = error
        .with_http_status(status)
        .with_stage(stage)
        .with_bytes_received(bytes);
    if let Some(code) = safe_identifier(code) {
        error = error.with_vendor_code(code);
    }
    if let Some(id) = request_id.and_then(|value| safe_identifier(&value)) {
        error = error.with_request_id(id);
    }
    if let Some(delay) = parse_retry_after(headers, &[]) {
        error = error.with_retry_after(delay);
    }
    if let Some(body) = sanitize_error_body(body, bytes, stage) {
        error = error.with_sanitized_body(body);
    }
    error
}

pub(crate) fn classify_stream_exception(
    event_type: &str,
    payload: &JsonValue,
    request_id: Option<String>,
    bytes: u64,
) -> ModelError {
    let status = match event_type {
        "validationException" => 400,
        "throttlingException" | "modelNotReadyException" => 429,
        "serviceUnavailableException" => 503,
        "modelStreamErrorException" => 424,
        _ => 500,
    };
    let body = serde_json::to_vec(payload).unwrap_or_default();
    classify_error(
        status,
        &body,
        request_id,
        ErrorStage::StreamEvent,
        bytes,
        &HeaderMap::new(),
    )
}

pub(crate) fn classify_stream_error(
    code: &str,
    message: &str,
    request_id: Option<String>,
    bytes: u64,
) -> ModelError {
    let lower = code.to_ascii_lowercase();
    let status = if lower.contains("validation") {
        400
    } else if lower.contains("throttl") || lower.contains("notready") {
        429
    } else if lower.contains("unavailable") {
        503
    } else {
        500
    };
    let body = serde_json::json!({"code":code,"message":message});
    classify_error(
        status,
        &serde_json::to_vec(&body).unwrap_or_default(),
        request_id,
        ErrorStage::StreamEvent,
        bytes,
        &HeaderMap::new(),
    )
}

fn safe_identifier(value: &str) -> Option<String> {
    (!value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.:#".contains(&byte)))
    .then(|| value.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_body_diagnostics() {
        oven_sdk_conformance::assert_error_body_diagnostics(|status, body, stage, bytes| {
            classify_error(
                status,
                body,
                Some("req-1".into()),
                stage,
                bytes,
                &HeaderMap::new(),
            )
        });
    }

    #[test]
    fn stream_exceptions_preserve_scrubbed_messages_and_details() {
        let payload = serde_json::json!({"message":"unsupported dimension; Bearer stream-secret", "detail":{"reason":"use 1024", "sessionToken":"nested-secret"}});
        for error in [
            classify_stream_exception(
                "validationException",
                &payload,
                Some("req-1".into()),
                100_000,
            ),
            classify_stream_error(
                "ValidationException",
                "unsupported dimension; Bearer stream-secret",
                Some("req-1".into()),
                100_000,
            ),
        ] {
            assert_eq!(error.kind, ModelErrorKind::InvalidRequest);
            assert_eq!(error.diagnostics.stage, ErrorStage::StreamEvent);
            let body = error.diagnostics.sanitized_body.as_ref().unwrap();
            assert!(body.text().contains("unsupported dimension"));
            assert!(!body.truncated());
            let wire = serde_json::to_string(&error).unwrap();
            assert!(!wire.contains("stream-secret"));
            assert!(!wire.contains("nested-secret"));
        }
        let payload = serde_json::json!({"message":format!("unsupported dimension token={}", "x".repeat(70_000))});
        let error = classify_stream_exception("validationException", &payload, None, 100_000);
        let body = error.diagnostics.sanitized_body.unwrap();
        assert!(body.truncated());
        assert!(body.len_bytes() <= oven_sdk::SanitizedBody::MAX_BYTES);
        assert!(body.text().contains("unsupported dimension"));
        assert!(!body.text().contains("xxxx"));
    }

    #[test]
    fn common_aws_errors_are_typed_and_redacted() {
        for (status, code, expected) in [
            (
                403,
                "AccessDeniedException",
                ModelErrorKind::PermissionDenied,
            ),
            (
                404,
                "ResourceNotFoundException",
                ModelErrorKind::ModelNotFound,
            ),
            (429, "ThrottlingException", ModelErrorKind::RateLimited),
            (503, "ServiceUnavailableException", ModelErrorKind::Overload),
        ] {
            let body =
                serde_json::to_vec(&serde_json::json!({"__type":code,"message":"secret"})).unwrap();
            let error = classify_error(
                status,
                &body,
                Some("req-1".into()),
                ErrorStage::ResponseBody,
                body.len() as u64,
                &HeaderMap::new(),
            );
            assert_eq!(error.kind, expected);
            assert!(!error.to_string().contains("secret"));
            assert_eq!(error.diagnostics.request_id.as_deref(), Some("req-1"));
        }
    }
}
