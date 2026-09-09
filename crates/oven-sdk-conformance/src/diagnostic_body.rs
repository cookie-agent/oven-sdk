use oven_sdk::{ErrorStage, ModelError, ModelErrorKind, SanitizedBody};

/// Checks provider error retention, credential scrubbing, bounds, and metadata.
/// The classifier callback should supply request ID `req-1`.
pub fn assert_error_body_diagnostics(classify: impl Fn(u16, &[u8], ErrorStage, u64) -> ModelError) {
    assert_fragment_and_url_diagnostics(&classify);
    let json = br#"{"message":"unsupported dimension; Bearer bearer-secret","error":{"message":"unsupported dimension; token=message-secret","future":{"reason":"use 1024","Api_Key":"nested-secret"}},"headers":{"Cookie":"session=cookie-secret"},"url":"https://host/?session_token=query-secret&mode=fast"}"#;
    let text = b"<html>unsupported dimension; use 1024 token=message-secret Bearer bearer-secret https://host/?key=query-secret&mode=fast</html>";
    for (status, kind) in [
        (400, ModelErrorKind::InvalidRequest),
        (403, ModelErrorKind::PermissionDenied),
        (429, ModelErrorKind::RateLimited),
    ] {
        for raw in [json.as_slice(), text.as_slice()] {
            for stage in [ErrorStage::ResponseBody, ErrorStage::StreamEvent] {
                let bytes = raw.len() as u64
                    + if stage == ErrorStage::StreamEvent {
                        1000
                    } else {
                        0
                    };
                let error = classify(status, raw, stage, bytes);
                assert_eq!(error.kind, kind);
                assert_eq!(error.retryable, status == 429);
                assert_eq!(error.diagnostics.http_status, Some(status));
                assert_eq!(error.diagnostics.stage, stage);
                assert_eq!(error.diagnostics.bytes_received, bytes);
                assert_eq!(error.diagnostics.request_id.as_deref(), Some("req-1"));
                let body = error
                    .diagnostics
                    .sanitized_body
                    .as_ref()
                    .expect("retained body");
                assert!(!body.truncated());
                assert!(body.text().contains("unsupported dimension"));
                assert!(body.text().contains("use 1024"));
                assert!(body.text().contains("mode=fast"));
                if raw == json {
                    let value: serde_json::Value = serde_json::from_str(body.text()).unwrap();
                    assert_eq!(value["error"]["future"]["reason"], "use 1024");
                    assert_eq!(value["error"]["future"]["Api_Key"], "[REDACTED]");
                }
                let wire = serde_json::to_string(&error).unwrap();
                assert!(wire.contains("unsupported dimension"));
                for secret in [
                    "bearer-secret",
                    "message-secret",
                    "nested-secret",
                    "cookie-secret",
                    "query-secret",
                ] {
                    assert!(!wire.contains(secret), "credential leaked: {secret}");
                    assert!(!format!("{error:?}").contains(secret));
                }
                assert!(!format!("{error:?}").contains("unsupported dimension"));
            }
        }
    }
    for stage in [ErrorStage::ResponseBody, ErrorStage::StreamEvent] {
        for raw in [
            format!(
                "unsupported dimension token={}",
                "é".repeat(SanitizedBody::MAX_BYTES)
            ),
            format!(
                "unsupported dimension {}",
                "é".repeat(SanitizedBody::MAX_BYTES)
            ),
        ] {
            let error = classify(400, raw.as_bytes(), stage, raw.len() as u64);
            let body = error.diagnostics.sanitized_body.unwrap();
            assert!(body.truncated());
            assert!(body.len_bytes() <= SanitizedBody::MAX_BYTES);
            assert!(body.text().contains("unsupported dimension"));
            if raw.contains("token=") {
                assert!(!body.text().contains('é'));
            }
        }
    }
}

fn assert_fragment_and_url_diagnostics(
    classify: &impl Fn(u16, &[u8], ErrorStage, u64) -> ModelError,
) {
    let mut cases = Vec::new();
    for name in ["X-Api-Key", "aUtHoRiZaTiOn", "Proxy-Authorization"] {
        let envelope = format!(
            r#"{{"message":"unsupported dimension","headers":[{{"VaLuE":"opaque-review-credential","NAME":"{name}"}},{{"name":"X-Request-Id","value":"harmless-review-value"}}]}}"#
        );
        let encoded = serde_json::to_string(&envelope).unwrap();
        for raw in [
            envelope.clone(),
            format!("provider response: {envelope}"),
            envelope[..envelope.len() - 1].to_owned(),
            encoded.clone(),
            serde_json::to_string(&encoded).unwrap(),
            serde_json::json!({"message":"unsupported dimension", "details":envelope}).to_string(),
            format!(
                r#"provider response: {{"message":"unsupported dimension","value":"harmless-review-value","headers":[{{"key":"{name}","value":{{"nested":"opaque-review-credential"#
            ),
        ] {
            cases.push((raw, "harmless-review-value", false));
        }
    }
    let capped = format!(
        r#"provider response: {{"message":"unsupported dimension","value":"harmless-review-value","headers":[{{"name":"X-Api-Key","value":"{}"#,
        "opaque-review-credential".repeat(4000)
    );
    cases.push((
        serde_json::to_string(&capped).unwrap(),
        "harmless-review-value",
        true,
    ));
    cases.push((capped, "harmless-review-value", true));
    for url in [
        "https://review-user:review-password@example.test/path?mode=fast",
        "https://review-user:review'password@example.test/path?mode=fast",
        "https://review!$&'()*+,;=user:review!$&'()*+,;=password@example.test/path?mode=fast",
        "HTTP://review%2Duser:review%2Dpassword%40extra@example.test/path?mode=fast",
    ] {
        let message = format!("unsupported dimension; gateway rejected {url}");
        let envelope = serde_json::json!({"message":message}).to_string();
        for raw in [
            message,
            envelope.clone(),
            format!("provider response: {envelope}"),
            serde_json::to_string(&envelope).unwrap(),
        ] {
            cases.push((raw, "[REDACTED]@example.test/path?mode=fast", false));
        }
    }
    for message in [
        "unsupported dimension; gateway rejected 'https://example.test' contact ops@example.test",
        "unsupported dimension; gateway rejected https://example.test/path?contact=ops@example.test",
    ] {
        let envelope = serde_json::json!({"message":message}).to_string();
        for raw in [
            message.to_owned(),
            envelope.clone(),
            format!("provider response: {envelope}"),
        ] {
            cases.push((raw, message, false));
        }
    }
    cases.push((r#"provider response: {"message":"unsupported dimension; https:\/\/review\u002duser:review\u002dpassword@example.test/path?mode=fast"}"#.into(), "[REDACTED]@example.test/path?mode=fast", false));
    for (raw, retained, truncated) in cases {
        for status in [400, 403, 429] {
            for stage in [ErrorStage::ResponseBody, ErrorStage::StreamEvent] {
                let bytes = raw.len() as u64;
                let error = classify(status, raw.as_bytes(), stage, bytes);
                assert_eq!(error.diagnostics.http_status, Some(status));
                assert_eq!(error.diagnostics.stage, stage);
                assert_eq!(error.diagnostics.bytes_received, bytes);
                let body = error
                    .diagnostics
                    .sanitized_body
                    .as_ref()
                    .expect("retained body");
                assert!(body.text().contains("unsupported dimension"));
                assert!(body.text().contains(retained));
                assert_eq!(body.truncated(), truncated);
                assert!(body.len_bytes() <= SanitizedBody::MAX_BYTES);
                let wire = serde_json::to_string(&error).unwrap();
                for secret in [
                    "opaque-review",
                    "review-user",
                    "review-password",
                    "review'password",
                    "review!$&",
                    "review%2D",
                    "review\\u002d",
                ] {
                    assert!(!body.text().contains(secret), "credential leaked: {secret}");
                    assert!(
                        !wire.contains(secret),
                        "serialized credential leaked: {secret}"
                    );
                }
                assert!(!format!("{error:?}").contains("unsupported dimension"));
            }
        }
    }
}
