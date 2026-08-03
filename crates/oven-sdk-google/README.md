# oven-sdk-google

Google Gemini `generateContent` adapter version 0.4.0 for `oven-sdk` 0.4.0.
Construction is registry-free: callers provide the endpoint, model declaration,
capabilities, limits, media rules, thinking behavior, tool behavior, and replay
declaration explicitly. Replay resource scope is derived internally. Model IDs
are identity only and never select adapter behavior.

## Installation

```bash
cargo add oven-sdk@0.4.0 oven-sdk-google@0.4.0
```

## Explicit construction

```rust,no_run
use oven_sdk::{
    ApiEndpoint, CancellationCapability, Capability, CompactionCapability, HeaderConfig,
    MediaCapabilities, Modalities, Modality, ModelCapabilities, ModelConfig, ModelDeclaration,
    ModelId, ModelLimits, ProviderConfig, ProviderId, ReplayCapability, ReplayDeclaration,
    ReplayPolicy,
};
use oven_sdk_google::{
    GoogleApiKeyAuth, GoogleGenerateContentSettings, GoogleModel, GoogleThinkingSettings,
    GoogleTimeouts, GoogleToolSettings,
};

# fn model() -> Result<GoogleModel, oven_sdk::ModelError> {
let provider_id = ProviderId::new("google");
let model_id = ModelId::new("caller-selected-model-id");
let capabilities = ModelCapabilities {
    features: Capability::TEMPERATURE
        | Capability::TOP_P
        | Capability::MAX_OUTPUT_TOKENS
        | Capability::USAGE,
    limits: ModelLimits::new(Some(1_048_576), None, Some(65_536)),
    modalities: Modalities::new([Modality::text()], [Modality::text()]),
    media: MediaCapabilities::default(),
    cancellation: CancellationCapability::LocalOnly,
    compaction: CompactionCapability::Unsupported,
    replay: ReplayDeclaration {
        policy: ReplayPolicy::IfValid,
        capability: ReplayCapability::Optional,
        reasoning: false,
    },
};
let provider = ProviderConfig::new(
    provider_id,
    ApiEndpoint::parse("https://generativelanguage.googleapis.com/v1beta")?,
    GoogleApiKeyAuth::new(std::env::var("GEMINI_API_KEY").unwrap()),
    HeaderConfig::empty(),
)?;
let declaration = ModelDeclaration::new(model_id, capabilities)?;
GoogleModel::new(ModelConfig::new(
    provider,
    declaration,
    GoogleGenerateContentSettings {
        model_resource: "models/caller-selected-resource".into(),
        timeouts: GoogleTimeouts::default(),
        thinking: GoogleThinkingSettings::Unsupported,
        tools: GoogleToolSettings {
            strict_functions: false,
            mixed_client_and_provider_tools: false,
            current_turn_signature_sentinel: false,
        },
    },
))
# }
```

The adapter never reads credentials or model metadata from the environment. It
does not contain a Google model registry, exact-name catalog, prefix matcher, or
generation inference. Changing only `ModelDeclaration.id` changes identity and
replay compatibility, not capabilities, request validation, thinking mode, tool
mode, media support, limits, or endpoint routing. `model_resource` explicitly
controls the REST resource used on the wire.

## Mapping models.dev data

Applications may load models.dev themselves and map one selected record into
`ModelDeclaration` and `ModelCapabilities`. The reference mapping was reviewed
against models.dev commit
[`c3057690bbb8bd41cafdefadcd2a7b958e2a4642`](https://github.com/anomalyco/models.dev/commit/c3057690bbb8bd41cafdefadcd2a7b958e2a4642):

- map the external provider/model keys to `ProviderId("google")` and `ModelId`;
- map context/input/output limits to `ModelLimits`;
- map input/output modality labels to `Modalities`;
- map exact MIME and source support to `MediaCapabilities`;
- map normalized feature facts to `Capability` flags;
- choose `ReplayDeclaration`, `GoogleThinkingSettings`, and
  `GoogleToolSettings` explicitly from trusted application configuration.

`oven-sdk-google` does not fetch, cache, bundle, or interpret models.dev records.
Known-looking, alias, preview, and future model names follow the same configured
behavior.

## Current adapter behavior

- Supports streaming and non-streaming Google AI Studio `generateContent`.
- Preserves current `generationConfig.responseFormat.text`, schema-less JSON,
  cached-content usage, client tools, provider tools, mixed tools, current media
  encoding, the 20 MiB serialized request cap, 3,600-image cap, and 10-video cap.
- Media MIME/source support is taken from explicit `ModelCapabilities`; Google
  HTTPS and provider-reference safety checks still run before dispatch.
- Thinking budget versus level acceptance, strict/mixed tools, and current-turn
  signature-sentinel use are explicit settings, never model-name decisions.
- Normalized `reasoning_effort` labels use caller-supplied exact budget or level
  maps in `GoogleThinkingSettings`; unmapped or unsupported labels fail before
  dispatch and are never silently discarded.
- Provider server tools normalize to safe `CustomPart` values and grounding
  normalizes to `SourcePart`; opaque thought signatures remain private.
- Native replay artifacts contain the exact Google model `Content` payload
  directly in the current core 0.4 format. Their `NativeContextScope` contains a
  versioned SHA-256 `ResourceId` derived internally from the canonical endpoint,
  `generateContent` surface, and `model_resource`; raw scope inputs and credentials
  are never exposed or trusted from caller data. There is no legacy replay decoder
  or private format-version fallback.
- Provider-native compaction is unsupported. A model declaration claiming
  `CompactionCapability::Native` is rejected during construction.
- Every successful stream ends in exactly one `Finish`; EOF is not success.

## Differences from the Vercel AI SDK

Compared against Vercel AI SDK commit
[`e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0`](https://github.com/vercel/ai/commit/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0), especially:

- [`packages/google/src/google-language-model.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/google/src/google-language-model.ts)
- [`packages/google/src/convert-to-google-messages.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/google/src/convert-to-google-messages.ts)
- [`packages/google/src/google-prepare-tools.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/google/src/google-prepare-tools.ts)

### Coverage gaps

- No Interactions, Files upload management, embedding, image generation, speech,
  video generation, realtime, or Live API surface.
- No automatic tool execution, retries, fallbacks, URL downloads, implicit
  credentials, or model registry.
- Provider tools are limited to verified Google Search, URL Context, Code
  Execution, File Search, and Google Maps shapes. Multimodal function results are
  rejected.

### Intentional divergences

- Configuration is caller-owned and registry-free rather than selected by model
  name.
- Every successful stream requires one terminal `Finish`.
- Errors use structured `ModelError`; scoped native replay is private and
  fail-closed.
- Cancellation is local, with separate connect/header/idle timeouts and no total
  stream timeout.
- URLs are never fetched by the adapter.
- Current `responseFormat.text` is used instead of the pinned Vercel source's
  older `responseMimeType`/`responseSchema` pair.
- The documented signature sentinel is available only when explicitly enabled
  and only for eligible normalized client function calls.

### Normalization differences

- Provider-executed tools become sanitized `CustomPart` values and web, Maps,
  image, retrieved-context, and File Search grounding become `SourcePart`, never
  executable client calls.
- Function calls use oven's start/delta/end/finalized lifecycle; malformed final
  arguments are fatal.
- Visible thoughts become reasoning while opaque thought signatures remain only
  in scoped native replay.

Protocol references: Gemini v1beta discovery revision 20260731 and current Google
generation, function-calling, thought-signature, caching, media, and grounding
documentation as of 2026-08-01.
