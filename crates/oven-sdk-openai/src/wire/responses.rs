//! Responses API wire constants.

/// Relative Responses endpoint.
pub(crate) const PATH: &str = "responses";
/// Relative standalone Responses compaction endpoint.
pub(crate) const COMPACT_PATH: &str = "responses/compact";
/// Current private replay format.
pub(crate) const REPLAY_FORMAT: &str = "oven.openai.responses.output.v1";
/// Current private provider-native compacted-context format.
pub(crate) const NATIVE_CONTEXT_FORMAT: &str = "oven.openai.responses.compaction.v1";

/// Reads `input_tokens_details.cache_write_tokens_details` from Responses usage.
///
/// Every `ttl_<label>_tokens` entry becomes `<label>` (`ttl_12h_tokens` is
/// `12h`), so a lifetime OpenAI adds later is kept without a crate release.
/// `None` means a present entry is not a non-negative integer.
pub(crate) fn cache_write_by_ttl(
    usage: &serde_json::Value,
) -> Option<std::collections::BTreeMap<String, u64>> {
    let mut by_ttl = std::collections::BTreeMap::new();
    let Some(details) = usage.pointer("/input_tokens_details/cache_write_tokens_details") else {
        return Some(by_ttl);
    };
    if details.is_null() {
        return Some(by_ttl);
    }
    for (key, value) in details.as_object()? {
        let Some(label) = key
            .strip_prefix("ttl_")
            .and_then(|key| key.strip_suffix("_tokens"))
            .filter(|label| !label.is_empty())
        else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        by_ttl.insert(label.to_owned(), value.as_u64()?);
    }
    Some(by_ttl)
}
