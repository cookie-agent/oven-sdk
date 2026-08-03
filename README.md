# oven-sdk

<p align="center"><img src="assets/logo.png" alt="oven-sdk logo" width="256"></p>

oven-sdk is a runtime-neutral Rust SDK for language-model providers. Core 0.4.0
uses explicit registry-free model declarations: callers supply provider/model
identity, API endpoint, resolved authentication, headers, limits, capabilities,
modalities, exact media rules, replay/cancellation/compaction declarations, and
provider-specific structural settings. Model names never infer behavior.

The SDK owns normalized contracts, typed streaming, strict completion,
structured errors, cancellation, opaque scope-aware replay, and bounded
provider-native context compaction. The calling
harness owns persistence, routing, retries, fallback, environment lookup,
permissions, approvals, tool execution, and the agent loop.

## Release matrix

All 10 releases were published on crates.io on 2026-08-02 UTC in the exact
order below and are active (not yanked).

| Order | Crate | Version | Published (UTC) | Status | Required `oven-sdk` | Dev `oven-sdk-conformance` |
|---:|---|---:|---|---|---:|---:|
| 1 | `oven-sdk` | 0.4.0 | 2026-08-02 03:40:10.181057 | Published; not yanked | — | — |
| 2 | `oven-sdk-conformance` | 0.4.0 | 2026-08-02 03:40:15.943506 | Published; not yanked | =0.4.0 | — |
| 3 | `oven-sdk-anthropic` | 0.5.0 | 2026-08-02 03:41:07.901606 | Published; not yanked | =0.4.0 | =0.4.0 |
| 4 | `oven-sdk-openai` | 0.4.0 | 2026-08-02 03:41:13.042064 | Published; not yanked | =0.4.0 | =0.4.0 |
| 5 | `oven-sdk-google` | 0.4.0 | 2026-08-02 03:41:17.791992 | Published; not yanked | =0.4.0 | =0.4.0 |
| 6 | `oven-sdk-google-vertex` | 0.4.0 | 2026-08-02 03:41:22.500094 | Published; not yanked | =0.4.0 | =0.4.0 |
| 7 | `oven-sdk-bedrock` | 0.3.0 | 2026-08-02 03:41:27.131484 | Published; not yanked | =0.4.0 | =0.4.0 |
| 8 | `oven-sdk-azure` | 0.3.0 | 2026-08-02 03:41:32.474860 | Published; not yanked | =0.4.0 | =0.4.0 |
| 9 | `oven-sdk-cohere` | 0.2.0 | 2026-08-02 03:41:37.000521 | Published; not yanked | =0.4.0 | =0.4.0 |
| 10 | `oven-sdk-open-responses` | 0.2.0 | 2026-08-02 03:41:41.629016 | Published; not yanked | =0.4.0 | =0.4.0 |

Core, conformance, and all workspace provider crates are migrated to the 0.4.0
contract and compile together. OpenAI Responses and Azure OpenAI Responses V1
implement provider-native compaction; other provider surfaces explicitly
declare native compaction unsupported and reject it before provider I/O.

## Architecture

- One configured `LanguageModel` represents one exact provider offering.
- `LanguageModelDescriptor` contains `ModelIdentity`, `AdapterId`, complete
  `ModelCapabilities`, and safe provider metadata.
- `ModelCapabilities` explicitly declares features, limits, open modalities,
  MIME/source media rules, cancellation, compaction, and replay semantics.
- `Request::validate_for` checks tool, sampling, output, reasoning, structured
  output, history, media, and capability dependencies before network I/O.
- `NativeReplayArtifact` and 32 MiB `NativeContextWindow` values are bounded,
  payload-redacted, and tied to an exact `NativeContextScope` containing
  provider, model, and resource identity.
- `LanguageModel::compact` is object-safe; default validation rejects
  unsupported compaction before I/O, and conformance covers cancellation and
  native-context round trips.
- There is no model registry, alias resolution, provider preset, environment
  lookup, automatic catalog access, or model-name inference.
- `oven-sdk-cohere` implements native v2 Chat tools, parallel calls, tool plans,
  thinking, citations, exact image rules, structured output, strict tools, SSE,
  usage/errors, cancellation, and scoped replay.
- `oven-sdk-open-responses` implements only the standardized item/event
  protocol over a caller-supplied bearer endpoint; its Hugging Face profile does
  not add a catalog, probes, model rewriting, or OpenAI-only hosted tools.
- `StreamPart::ApprovalRequested` remains assistant output and is collected as
  `AssistantPart::ToolApproval`; the harness owns approval decisions.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the normative contract and provider
construction matrix.
