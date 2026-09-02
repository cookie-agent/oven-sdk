# oven-sdk-bedrock

Amazon Bedrock Runtime `Converse` and `ConverseStream` for `oven-sdk`.

This independent `0.3.0` crate targets `oven-sdk` 0.4.0 and uses
`oven-sdk-conformance` 0.4.0 for development. It signs direct Bedrock Runtime
requests with AWS Signature Version 4 and incrementally decodes AWS EventStream
responses. It does **not** use Anthropic's AWS Messages API, a vendor AWS SDK,
ambient credential discovery, hidden retries, or provider fallback.

## Installation

```bash
cargo add oven-sdk@0.4.0 oven-sdk-bedrock@0.3.0
```

## Registry-free configuration

`BedrockModel::new(ModelConfig<BedrockAuth, BedrockConverseSettings>)` is the
only model construction path.

```rust,no_run
use oven_sdk::{
    AbortSignal, ApiEndpoint, CancellationCapability, Capability, CompactionCapability,
    HeaderConfig, HistoryTurn, InputPart, LanguageModel, MediaCapabilities, Modalities, Modality,
    ModelCapabilities, ModelConfig, ModelDeclaration, ModelId, ModelLimits, ProviderConfig,
    ProviderId, ReplayCapability, ReplayDeclaration, ReplayPolicy, Request, TextPart, UserMessage,
};
use oven_sdk_bedrock::{
    AwsCredentials, BedrockAuth, BedrockConverseSettings, BedrockEventStreamLimits, BedrockModel,
    BedrockReasoningWireFormat, BedrockStructuredOutput, BEDROCK_PROVIDER_ID,
};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let provider = ProviderConfig::new(
    ProviderId::new(BEDROCK_PROVIDER_ID),
    ApiEndpoint::parse("https://bedrock-runtime.us-east-1.amazonaws.com")?,
    BedrockAuth::Static(AwsCredentials {
        access_key_id: "...".into(),
        secret_access_key: "...".into(),
        session_token: None,
    }),
    HeaderConfig::empty(),
)?;
let declaration = ModelDeclaration::new(
    ModelId::new("caller-selected-bedrock-resource"),
    ModelCapabilities {
        features: Capability::USAGE,
        limits: ModelLimits::new(None, None, None),
        modalities: Modalities::new([Modality::text()], [Modality::text()]),
        media: MediaCapabilities::default(),
        cancellation: CancellationCapability::LocalOnly,
        compaction: CompactionCapability::Unsupported,
        replay: ReplayDeclaration {
            policy: ReplayPolicy::IfValid,
            capability: ReplayCapability::Optional,
            reasoning: false,
        },
    },
)?;
let settings = BedrockConverseSettings::new(
    "us-east-1",
    BedrockReasoningWireFormat::Unsupported,
    false,
    BedrockStructuredOutput::Unsupported,
    BedrockEventStreamLimits::new(16 * 1024 * 1024),
);
let model = BedrockModel::new(ModelConfig::new(provider, declaration, settings))?;
let request = Request::new(vec![HistoryTurn::user(UserMessage::new(vec![
    InputPart::Text(TextPart::new("Hello")),
]))]);
let result = model.complete(request, AbortSignal::default()).await?;
println!("{}", result.turn.text());
# Ok(())
# }
```

`BedrockAuth::Provider` accepts a caller-managed asynchronous credential
provider. Credentials are resolved once per request and participate in the
credential timeout and abort signal. `BedrockAuth::Static` supports fixed and
temporary credentials. Credentials, authorization, signed/redacted reasoning,
and replay payloads are never exposed by `Debug` or errors.

The endpoint and signing region are always caller-supplied. Model IDs, inference
profile IDs, and resource ARNs are opaque and encoded as one URI path segment
under `/model/{modelId}/converse` or `/model/{modelId}/converse-stream`. The
transmitted URL is singly encoded; the non-S3 SigV4 canonical URI applies AWS's
required second percent-encoding pass.

## models.dev terminology mapping

The public names follow useful parts of models.dev commit
`c3057690bbb8bd41cafdefadcd2a7b958e2a4642`; the crate does not fetch, embed, or
depend on models.dev at runtime or build time.

| models.dev concept | oven-sdk 0.4 declaration |
|---|---|
| provider `id` | `ProviderConfig::id` (`amazon.bedrock`) |
| provider model `id` | `ModelDeclaration::id` |
| provider/model `api` | explicit `ProviderConfig::api` |
| `limit.context/input/output` | `ModelCapabilities::limits` |
| `modalities.input/output` | `ModelCapabilities::modalities` |
| tool/reasoning/structured-output | `ModelCapabilities::features` |
| attachment support | exact `MediaCapabilities` MIME/source declarations |

There is no model registry, alias table, preset, exact catalog, model family,
or model-name inference. The model ID is used only for the Bedrock wire URL,
descriptor identity, and native-context scope. An ID containing `anthropic`, `amazon`,
`openai`, or another substring never changes validation or wire encoding.

Bedrock-specific behavior is explicit in `BedrockConverseSettings`: signing
region, reasoning wire format, signed/redacted reasoning requirement, native
structured-output shape, EventStream message limit, timeouts, and optional HTTP
client. Construction rejects contradictory capability and wire declarations.
It also enforces immutable adapter ceilings: declarations may remove supported
features or media, but cannot add prompt caching, provider tools, audio,
non-text output, unsupported modalities, MIME types such as BMP, or unsupported
source forms.

Public provider-defined scalar labels—including service tier, performance
latency, reasoning type/display/effort, guardrail trace, and stream processing
mode—remain open strings and are forwarded unchanged.

## Media, tools, and structured output

Callers declare exact modalities, MIME types, source forms, capabilities, and
limits through `ModelDeclaration`. Version 0.3.0 advertises only combinations
that the generic core declaration can represent honestly for every request:

- Images: PNG, JPEG, GIF, and WebP from inline bytes.
- Video: MKV, MOV, MP4, WebM, FLV, MPEG/MPG, WMV, and 3GP from inline bytes.
- Inline content must be non-empty. Images are limited to 3.75 MiB and inline
  video must remain below the documented 25 MiB encoded payload limit. At most
  20 images and one video are accepted per message and request.
- Generic `URL` declarations are rejected because Bedrock accepts only `s3://`
  media while core cannot express a URL-scheme restriction.
- PDF/document declarations are rejected because Bedrock requires companion
  text in the same message while core media declarations cannot express that
  contextual requirement.
- Audio, inline-text media, provider references, BMP, and other unsupported
  MIME/source combinations are rejected during model construction.
- Tool input must finalize as a JSON object. Text, JSON, denied, and mixed tool
  results are encoded without inventing successful output.
- Structured output uses `outputConfig.textFormat` only when both the core
  capability and `BedrockStructuredOutput::JsonSchema` are declared.
- Prompt caching is intentionally not advertised because normalized
  `cachePoint` request/history representation and encoding are not implemented.
- Guardrail `streamProcessingMode` is serialized only for ConverseStream.

Reasoning controls are validated against the selected explicit wire format
before network dispatch. Adaptive Anthropic thinking cannot carry a token
budget; display is adaptive-only; disabled reasoning cannot carry active
budget/effort controls; generic `reasoningConfig` rejects adaptive/display; and
OpenAI reasoning accepts effort only. Normalized and Bedrock-specific effort
controls cannot both be set. Valid provider labels remain open strings. Wire
objects omit absent `display`, budget, effort, and inference `maxTokens` fields
instead of serializing JSON nulls.
`additional_model_request_fields` cannot contain `thinking`, `output_config`,
`reasoning_effort`, `reasoningConfig`, or casing/separator aliases; typed
reasoning and output controls are the only path for those structural fields.

## Streaming and replay guarantees

The AWS EventStream decoder is independent of HTTP chunk boundaries, validates
prelude/message CRC32 values and every header type, rejects duplicate or invalid
headers and truncated tails, and consumes arbitrarily large public feed chunks
frame-by-frame. It retains at most one incomplete frame bounded by the explicit
`BedrockEventStreamLimits::max_message_bytes` value.

Converse event order, content indices, block lifecycle, tool JSON, configured
reasoning requirements, `messageStop`, and terminal `metadata` are validated
before success. Terminal parts are released only after clean EventStream and
HTTP EOF. Provider exceptions and `:message-type=error` frames emit `Error`,
then `Finish(Error)`, then EOF. Transport, CRC, JSON, event-order, cancellation,
and truncation failures never fabricate success.

Native replay uses only the current `oven.bedrock.converse.assistant.v2` private
format. The artifact's `NativeContextScope` binds provider identity, exact
model ID, and a safe fingerprint of endpoint,
region, reasoning wire format, signed-reasoning setting, and structured-output
setting. The payload contains only strictly decoded assistant text, tool-use,
and reasoning blocks. Extra, malformed, or ambiguous unions are rejected.
Foreign adapter/scope and invalid payloads reconstruct normalized text/tools,
omit authoritative reasoning, and record complete replay decisions. Earlier
scope-less and `assistant.scoped.v1` replay formats are not decoded.
Assistant-contained tool results are encoded independently of replay selection,
so successful native replay preserves the following ordered user-role tool
results instead of dropping them.

Default phase timeouts are 60 seconds connect, 300 seconds response headers,
30 seconds credentials, and 60 seconds stream/error-body idle. There is no
default total streaming deadline. Cancellation is local-only. Provider-native
compaction and native context windows are explicitly unsupported; declarations
requesting `CompactionCapability::Native` are rejected during construction.

## Testing

Coverage includes exact-byte SigV4, ARN/model URL encoding, AWS SDK
double-encoding vectors, temporary credentials, every-byte EventStream framing,
CRC/length/header/truncation failures, huge decoder feeds, bounded partial
frames, request translation, media boundaries, structured output,
text/tool/reasoning streams, terminal draining, usage/request IDs, error
taxonomy, strict native-context-scoped replay, safe reconstruction,
cancellation, conformance,
and model-name independence including anthropic-looking opaque IDs.

Credential-gated live tests use:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN          # optional
AWS_REGION                 # optional; defaults to us-east-1
BEDROCK_MODEL_ID           # optional
BEDROCK_ENDPOINT           # optional explicit endpoint override
```

Run them with:

```bash
cargo test -p oven-sdk-bedrock --test live -- --ignored --nocapture
```

## Differences from the Vercel AI SDK

Compared against Vercel AI SDK commit
[`e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0`](https://github.com/vercel/ai/tree/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0),
specifically:

- [`amazon-bedrock-chat-language-model.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/amazon-bedrock/src/amazon-bedrock-chat-language-model.ts)
- [`convert-to-amazon-bedrock-chat-messages.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/amazon-bedrock/src/convert-to-amazon-bedrock-chat-messages.ts)
- [`amazon-bedrock-event-stream-decoder.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/amazon-bedrock/src/amazon-bedrock-event-stream-decoder.ts)
- [`map-amazon-bedrock-finish-reason.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/amazon-bedrock/src/map-amazon-bedrock-finish-reason.ts)

Coverage gaps:

- Bedrock prompt resources, provider-executed tools, cache-point builders,
  guard-content/search-result/audio blocks, and image/file output are not yet
  normalized.
- Callers must declare capabilities and wire settings. The crate contains no
  Bedrock model catalog or provider presets.

Intentional divergences:

- oven-sdk requires one final typed `Finish`; corrupted, truncated, or
  post-terminal EventStream data is an error.
- oven-sdk uses typed `ModelError` stages, byte counts, request IDs,
  retry-after, bounded sanitized bodies, explicit declarations,
  native-context-scoped replay,
  phase timeouts, and local cancellation semantics.
- oven-sdk performs no retry or fallback, including no automatic retry for
  `ModelNotReadyException`.
- oven-sdk never auto-downloads URLs and rejects non-S3 media URLs locally.
- Native structured output is explicit and never injects JSON instructions or
  creates a synthetic `json` tool.
- Signed/redacted reasoning is replay-only authoritative state bound to the
  exact provider/model/resource scope.

Normalization differences:

- Text, reasoning, and tool-use blocks have strict start/delta/end IDs and a
  finalized `ToolCall`; malformed/non-object input fails instead of using an
  invalid-input wrapper or empty-object fallback.
- `malformed_model_output`, `malformed_tool_use`, and
  `model_context_window_exceeded` become typed in-band errors. Unknown stop
  labels remain `FinishReason::Other(String)`.
- Inclusive input usage contains base, cache-read, and cache-write tokens while
  each component remains separately available.
- Citation deltas become normalized `SourcePart` values with the complete safe
  citation object in metadata; unknown well-formed events become
  `ProviderEvent` rather than being silently discarded.
