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

## Native replay policy

The current source tree evaluates validated native content by block format and
target support, not by equality of origin provider, adapter identity, endpoint,
headers, local model alias, or deployment fingerprint. Standard supported
blocks can cross those boundaries; differing provenance is not itself a reason
to reconstruct them. This is not a promise of arbitrary protocol conversion:
unknown custom codecs and blocks need explicit target support.

Encrypted reasoning requires equal, known source and target **effective wire
model IDs**, plus support for that format. This includes Responses
`encrypted_content`, redacted Messages/Converse blocks, and Gemini/Vertex
`thoughtSignature`. Signed visible Messages/Converse thinking is distinct from
encrypted state. Safe filtering preserves portable siblings in a mixed artifact
when the format permits separation. Integrity and normalized-content validation
still apply before replay eligibility is evaluated.

Ordinary text/tool history can reconstruct when native data is unavailable.
Required opaque continuation state cannot silently disappear: its absence,
corruption, or ineligibility fails closed. Evidence of a required continuation
is not an ordinary integrity fingerprint; a fingerprint or tool call alone does
not make a turn require encrypted state. Legacy data without required-state
evidence remains best-effort, not recoverable by inventing that evidence or
native payloads. Missing source wire identity does not establish model equality.

SDK capture records the actual request model ID. Applications constructing or
restoring artifacts must preserve known source wire identity rather than
substituting a local alias. The target receives eligible native state and can
reject signatures or ciphertext even for the same wire model ID, especially
across services with different keys. There is no hidden retry that strips
reasoning after an HTTP 400, and no live acceptance or cache-hit guarantee.

`NativeContextScope` still records provenance. Native compaction windows retain
their own exact scope checks; cache resource/policy constraints also remain
separate from ordinary block replay. These source-tree changes do not describe
the historical 0.4.0 release's replay policy.
