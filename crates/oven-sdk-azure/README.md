# oven-sdk-azure

Independent Tokio/reqwest Azure OpenAI Chat Completions and Responses adapters
for `oven-sdk` 0.4. The crate version is `0.3.0`.

## Installation

```bash
cargo add oven-sdk@0.4.0 oven-sdk-azure@0.3.0
```

## Registry-free construction

The crate contains no Azure model registry and never infers behavior from a
deployment or model name. Callers construct `AzureOpenAiChatModel` or
`AzureOpenAiResponsesModel` directly from core `ModelConfig`, `ProviderConfig`,
`ModelDeclaration`, and `ModelCapabilities` values.

```rust,no_run
use oven_sdk::{
    ApiEndpoint, CancellationCapability, HeaderConfig, ModelCapabilities,
    ModelConfig, ModelDeclaration, ModelId, ProviderConfig, ProviderId,
    SecretString,
};
use oven_sdk_azure::{
    AZURE_OPENAI_PROVIDER_ID, AzureApiRoute, AzureOpenAiAuth,
    AzureOpenAiResponsesModel, AzureOpenAiResponsesSettings,
};

let provider = ProviderConfig::new(
    ProviderId::new(AZURE_OPENAI_PROVIDER_ID),
    ApiEndpoint::parse("https://my-resource.openai.azure.com")?,
    AzureOpenAiAuth::ApiKey(SecretString::new("caller-resolved-secret")),
    HeaderConfig::empty(),
)?;
let mut capabilities = ModelCapabilities::conservative();
capabilities.cancellation = CancellationCapability::LocalOnly;
let model = ModelDeclaration::new(ModelId::new("production-deployment"), capabilities)?;
let settings = AzureOpenAiResponsesSettings {
    route: AzureApiRoute::V1,
    ..Default::default()
};
let model = AzureOpenAiResponsesModel::new(ModelConfig::new(provider, model, settings))?;
# let _ = model;
# Ok::<(), oven_sdk::ModelError>(())
```

The fixed Provider ID is `azure.openai`; the Model ID is the exact Azure
deployment name. Endpoint, authentication, headers, capabilities, route,
revision, compaction, and wire settings are explicit caller data.

## models.dev terminology mapping

Compared against models.dev commit
`c3057690bbb8bd41cafdefadcd2a7b958e2a4642`. This is terminology mapping only:
the crate never fetches, synchronizes with, or depends on models.dev and ships
no model/version/SKU catalog.

| models.dev concept | oven-sdk 0.4 representation |
| --- | --- |
| provider `api` | `ProviderConfig::api` |
| provider `headers` | `ProviderConfig::headers` |
| model ID | `ModelDeclaration::id` |
| model capabilities | `ModelCapabilities::features` |
| `limit.context/input/output` | `ModelCapabilities::limits` |
| `modalities.input/output` | `ModelCapabilities::modalities` |
| media MIME/source support | `ModelCapabilities::media` |
| provider shape | concrete Chat or Responses model type |

## Azure Responses V1 native compaction

The adapter implements the standalone Azure endpoint:

```text
POST /openai/v1/responses/compact
```

Native compaction is available only when all of these are explicit:

- the concrete model is `AzureOpenAiResponsesModel`;
- `ModelCapabilities::compaction` is `CompactionCapability::Native`;
- `AzureOpenAiResponsesSettings::route` is exactly `AzureApiRoute::V1`;
- settings select `AzureOpenAiResponsesCompaction::V1` with a non-empty,
  non-secret `routing_discriminator`;
- a complete model/version/deployment-type `AzureOpenAiRevision` is present.

Chat, V1 preview, dated routes, missing settings, unsupported declarations, and
empty/control-character routing discriminators fail during construction. Model
IDs never enable compaction.

```rust,no_run
use oven_sdk::CompactionCapability;
use oven_sdk_azure::{
    AzureOpenAiResponsesCompaction, AzureOpenAiResponsesSettings,
};
# let mut capabilities = oven_sdk::ModelCapabilities::conservative();
# capabilities.cancellation = oven_sdk::CancellationCapability::LocalOnly;
capabilities.compaction = CompactionCapability::Native;
let settings = AzureOpenAiResponsesSettings {
    route: oven_sdk_azure::AzureApiRoute::V1,
    revision: Some(oven_sdk_azure::AzureOpenAiRevision {
        model: "caller-known-model".into(),
        version: "caller-known-version".into(),
        deployment_type: "global_standard".into(),
    }),
    compaction: AzureOpenAiResponsesCompaction::V1 {
        routing_discriminator: "production-v1".into(),
    },
    ..Default::default()
};
# let _ = (capabilities, settings);
```

The approved architecture is standalone stateless compaction over caller-held
local context. The normalized request must encode at least one input item;
instructions or options alone are not context. Zero-item input is rejected
before credential resolution or HTTP dispatch, and the adapter never sends
`input: []`.

`AzureOpenAiCompactionOptions` contains only documented Azure fields supported
by this adapter: instructions, prompt-cache key, prompt-cache retention, and
service tier. Options are attached through `AzureOpenAiCompactionRequestExt`;
malformed or oversized values fail before dispatch. `previous_response_id` and
stored-response chaining are intentionally absent. Azure
`prompt_cache_options` is unsupported and rejected as an unknown field.

`service_tier` is intentionally a forward-open `String`, following the
provider-label rule used across owner adapters. Non-empty length/control checks
are enforced, but otherwise the label is passed through unchanged so future
Azure values do not require an SDK release.

The compact response must be the canonical provider window: zero or more
bounded user message items followed by exactly one encrypted compaction item.
The complete output is retained and passed to the next `/responses` request
unchanged and before newly normalized input. It is never summarized, pruned, or
interpreted locally. Root fields, item/content forms, IDs, token accounting,
item counts, body sizes, and the SHA-256 output fingerprint are checked fail
closed. Native context uses the core 32 MiB bound. Usage normalization checks
input/output totals and separately preserves cache-read and cache-write token
counts while deriving uncached input tokens with checked arithmetic.

Compaction performs one provider call without hidden retry or local fallback.
Cancellation covers pre-dispatch, Entra resolution, response headers, and
bounded body reads. HTTP errors use the same structured Azure classification;
native encoding and decoding failures use `NativeContextEncode` and
`NativeContextDecode` stages.

## Scope and replay

Core 0.4 `NativeContextScope` is shared by native replay and compaction. For V1
native compaction the versioned resource fingerprint binds the provider,
deployment, canonical endpoint, exact V1 route and Responses surface, revision,
capabilities, static routing headers, compaction settings, and the explicit
routing discriminator. Resolved credentials and dynamic header values are not
serialized or fingerprinted; the caller-supplied discriminator represents any
behavior-affecting dynamic route.

Private replay formats are current positive allow lists only:

- `oven.azure.openai.chat.assistant.v4`
- `oven.azure.openai.responses.output.v4`

The native compact-window format is
`oven.azure.openai.responses.compaction.v1`. No legacy, scope-less, v3 replay,
or alternate compact-window shape is decoded. Foreign adapters/scopes, malformed
payloads, unknown items/fields, duplicate IDs, semantic mismatches, and forged
fingerprints fail closed or reconstruct only normalized replay history.

## Validation, media, and lifecycle

`ModelCapabilities` remains authoritative. Both adapters accept text input and
text-only output, with exact declared image support; Responses additionally
supports exact PDF input. Audio, video, provider references, undeclared media,
and unsupported source forms fail before dispatch. URLs are never downloaded.

Chat accepts at most ten images at 20 MiB each. Responses accepts at most 50
images at 20 MiB each and less than 50 MiB combined inline image data. Each
inline PDF and all inline PDFs combined must remain below 50 MiB. PNG, JPEG,
WebP, non-animated GIF, and exact `application/pdf` rules remain fail closed.

Responses indices use checked conversion and arithmetic before mutation. Output
is contiguous and capped at 128 items; content and summary vectors are capped at
128 slots with at most a 16-slot extension gap. Terminal events require a
compatible status, present non-empty bounded output, supported items, and a
documented incomplete reason. Strict schemas, arbitrary-boundary SSE, mandatory
terminal `Finish`, structured errors, request IDs, retry hints, and byte
diagnostics remain enforced.

## Live tests

```text
cargo test -p oven-sdk-azure --test live -- --ignored
```

The library performs no environment lookup. Live-test credentials,
deployments, revisions, and routing labels are caller supplied.

## Differences from the Vercel AI SDK

Compared against Vercel AI SDK commit
[`e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0`](https://github.com/vercel/ai/commit/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0),
especially its Azure provider and OpenAI Chat/Responses conversion and stream
lifecycle sources.

**Coverage gaps.** This crate excludes completions, embeddings, images, speech,
transcription, hosted Azure tools, stored/background Responses,
`previous_response_id` compaction, Azure `prompt_cache_options`, automatic
server-side `context_management` compaction, realtime, automatic environment
credentials, and model catalogs. Standalone compaction accepts only explicit
caller-held local context.

**Intentional divergences.** The crate is independent of `oven-sdk-openai`,
uses concrete registry-free models, protects auth headers, performs one call
without hidden retries, requires terminal `Finish`, uses bounded scope-aware
native context, never downloads URLs, and reports core structured capabilities
and errors. Standalone compaction returns an opaque canonical context window;
there is no local-summary fallback.

**Normalization differences.** Azure filter extensions become provider events.
Chat tool arguments finalize only at a valid terminal lifecycle. Responses
refusal and full-item content are checked against terminal authoritative state;
call IDs remain separate from provider item IDs, and terminal usage/metadata is
authoritative.
