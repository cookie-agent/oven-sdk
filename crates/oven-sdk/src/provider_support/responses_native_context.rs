//! The wire names a Responses compacted window travels under, shared by
//! the OpenAI and Azure OpenAI adapters, and the error check built on them.
//!
//! A compacted window ends in one item of type [`RESPONSES_COMPACTION_ITEM_TYPE`]
//! whose opaque state is under [`RESPONSES_ENCRYPTED_CONTENT_FIELD`]; the
//! adapters validate and send exactly those names. A provider that cannot
//! use the window, or a gateway that does not take it, names one of them
//! when it refuses the request, as in "unknown item type `compaction`" or
//! OpenAI's `invalid_encrypted_content`.

use crate::ModelError;

/// The `type` of the item that carries a compacted window's state.
pub const RESPONSES_COMPACTION_ITEM_TYPE: &str = "compaction";

/// The field of the compaction item that holds its opaque state.
pub const RESPONSES_ENCRYPTED_CONTENT_FIELD: &str = "encrypted_content";

/// Every wire name a compacted window travels under.
pub const RESPONSES_NATIVE_CONTEXT_WIRE_NAMES: [&str; 2] = [
    RESPONSES_COMPACTION_ITEM_TYPE,
    RESPONSES_ENCRYPTED_CONTENT_FIELD,
];

/// Whether `error` is a provider's refusal (HTTP 400 or 422) whose vendor
/// code, message, or sanitized body names one of
/// [`RESPONSES_NATIVE_CONTEXT_WIRE_NAMES`], ignoring case.
///
/// This says the refusal concerns the compacted window or the compaction
/// call; it does not say why. Callers decide what a match means for a
/// request that carried a window or for a compaction call.
#[must_use]
pub fn error_names_responses_native_context(error: &ModelError) -> bool {
    if !matches!(error.diagnostics.http_status, Some(400 | 422)) {
        return false;
    }
    let texts = [
        error.diagnostics.vendor_code.as_deref(),
        Some(error.message.as_str()),
        error
            .diagnostics
            .sanitized_body
            .as_ref()
            .map(|body| body.text()),
    ];
    texts.into_iter().flatten().any(|text| {
        let text = text.to_ascii_lowercase();
        RESPONSES_NATIVE_CONTEXT_WIRE_NAMES
            .iter()
            .any(|name| text.contains(name))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelErrorKind, SanitizedBody};

    fn refusal(status: u16, body: &str) -> ModelError {
        ModelError::new(ModelErrorKind::InvalidRequest, "request failed")
            .with_http_status(status)
            .with_sanitized_body(SanitizedBody::new(body))
    }

    #[test]
    fn a_gateway_refusal_naming_a_window_wire_name_matches() {
        for body in [
            r#"{"error":{"message":"unknown item type `compaction` in input[0]"}}"#,
            r#"{"error":{"message":"content keyed `encrypted_content` isn't supported"}}"#,
            r#"{"detail":"Input item TYPE COMPACTION is not supported"}"#,
        ] {
            for status in [400, 422] {
                assert!(
                    error_names_responses_native_context(&refusal(status, body)),
                    "{status} {body}"
                );
            }
        }
    }

    #[test]
    fn openais_encrypted_content_code_matches() {
        let error = ModelError::new(ModelErrorKind::InvalidRequest, "OpenAI request failed")
            .with_http_status(400)
            .with_vendor_code("invalid_encrypted_content");
        assert!(error_names_responses_native_context(&error));
    }

    #[test]
    fn a_refusal_without_the_names_or_another_status_does_not_match() {
        assert!(!error_names_responses_native_context(&refusal(
            400,
            r#"{"error":{"message":"unknown parameter: top_k"}}"#
        )));
        for status in [401, 404, 429, 500, 501] {
            assert!(!error_names_responses_native_context(&refusal(
                status,
                r#"{"error":{"message":"compaction is unavailable"}}"#
            )));
        }
        let local = ModelError::new(ModelErrorKind::InvalidRequest, "compaction request invalid");
        assert!(!error_names_responses_native_context(&local));
    }
}
