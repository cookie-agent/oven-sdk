//! Responses API wire constants.

/// Relative Responses endpoint.
pub(crate) const PATH: &str = "responses";
/// Relative standalone Responses compaction endpoint.
pub(crate) const COMPACT_PATH: &str = "responses/compact";
/// Current private replay format.
pub(crate) const REPLAY_FORMAT: &str = "oven.openai.responses.output.v1";
/// Current private provider-native compacted-context format.
pub(crate) const NATIVE_CONTEXT_FORMAT: &str = "oven.openai.responses.compaction.v1";
