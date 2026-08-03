# oven-sdk

Runtime-neutral language-model contract and normalized streaming types for the
oven-sdk workspace. See the workspace README and `oven-sdk-conformance` for
adapter implementation and conformance guidance.

## Installation

```bash
cargo add oven-sdk@0.4.0
```

Version 0.4.0 adds provider-native compaction, bounded opaque native context,
scope-aware native replay, open modalities with exact media rules, and no
model-name inference. `StreamPart::ApprovalRequested` is still collected as
`AssistantPart::ToolApproval`; approval policy and execution remain harness
responsibilities.

`ModelCapabilities::compaction` is required. `Request::native_context` accepts
only a bounded `NativeContextWindow`, and `CompactionRequest`/`CompactionResult`
drive the object-safe `LanguageModel::compact` operation. `ReplayScope` was
renamed outright to `NativeContextScope`; there is no alias or old decoder.

`ApiEndpoint` rejects credentials, queries, fragments, and unresolved templates;
its debug output and the debug output of provider/model configuration wrappers
are redacted. Media validation iterates explicit open-modality rules rather than
inferring modality from MIME type. Valid replay declarations are exactly
`Never/Unsupported`, `IfValid/Optional`, `IfValid/Required`, and
`Always/Required`.
