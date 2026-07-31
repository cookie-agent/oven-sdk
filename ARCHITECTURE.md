# oven-sdk Redesign Proposal
**Status:** design for review before implementation  
**Repository:** `github.com/cookie-agent/oven-sdk`  
**Primary crate:** `oven-sdk` (`oven_sdk` in Rust paths)  
**Initial release target:** `0.1.0`
## Executive recommendation
Build `oven-sdk` as a Cargo workspace with a runtime-neutral core, separately published adapters, and a public conformance-test crate.
Initial packages:
- `oven-sdk`: core model contract, normalized types, stream lifecycle, errors, cancellation, replay envelope, registry, middleware, builders, and helpers.
- `oven-sdk-anthropic`: Anthropic Messages.
- `oven-sdk-openai`: OpenAI Responses, Chat Completions, and configurable OpenAI-compatible Chat profiles.
- `oven-sdk-conformance`: public dev/test support for adapter authors.
The SDK owns normalization, translation, adapter transport, typed stream parts, replay codecs, capabilities, structured errors, cancellation, factories, registry, and middleware.
The calling harness owns persistence, fallback chains, same-entry retries, sticky run state, meaningful-output policy, permissions, approvals, tool execution, and the agent loop.
This preserves the best current property—durable same-protocol replay in `ARCHITECTURE.md` section 6.2—while removing the duplicated orchestration in `crates/providers/src/lib.rs:389-575` and `crates/engine/src/lib.rs:3001-3209`.
## Design principles
1. Rust-first; use enums, newtypes, builders, trait objects, `Future`, and `futures_core::Stream`.
2. A `LanguageModel` object represents one configured model and wire profile.
3. Streaming is authoritative; `generate` can collect streaming output.
4. Exactly one typed `Finish` proves semantic completion.
5. Adapters report facts and hints; harnesses decide retry/fallback/run failure.
6. Official adapters perform one provider call and never hide retries.
7. Native replay is correctness state, not debug metadata.
8. Foreign replay is explicitly discarded and normalized reconstruction is reported.
9. Core is runtime-neutral; first-party HTTP adapters may be Tokio-backed.
10. Telemetry is metadata-only by default.

## Approved core-contract amendment

The initial core implementation uses the following approved amendment, which
supersedes any conflicting historical contract-version, replay-schema, or
replay-envelope text below.

- Contract-version identifiers, replay schema fields, and versioned replay
  envelopes are not part of the contract. A native replay artifact contains
  only a stable `AdapterId` and an opaque JSON payload.
- A provider-native payload that an adapter cannot decode is handled exactly as
  a provider swap: the adapter discards it, reconstructs from normalized
  content, and reports `ReplayDisposition::DiscardedInvalidPayload` with a
  sanitized reason. Foreign adapter artifacts are likewise discarded and
  reported.
- Assistant history is represented by `HistoryTurn::Assistant(CompletedTurn)`;
  therefore a completed turn's `Finish.native_replay` reaches the next
  `LanguageModel::stream` call. Ordered tool-result history uses
  `HistoryTurn::Tool`.
- History uses role-specific system, user, assistant, and tool message unions.
  Invalid role/content combinations are rejected by both constructors and
  serde, rather than trusting a generic role string.
- `NativeReplayArtifact` is bounded to 2 MiB on every construction and serde
  deserialization path. Its fields are private, access is read-only, and its
  custom `Debug` implementation redacts the payload.
- The default `LanguageModel::complete` drain enforces stream block lifecycle
  by ID, preserves block start order across interleaving, requires one first
  `StreamStart`, requires each finalized tool-call ID exactly once, requires
  finalized calls after ended tool-call blocks, and requires the documented
  `Error`, `Finish(Error)`, EOF sequence for in-band errors.
- Descriptors expose typed provider, model, and adapter identities. Capability
  declarations include cancellation and replay semantics in addition to the
  feature bitset and token limits.
- `Request::validate_for` centralizes non-empty unique tool-name validation;
  every `ToolMessage` must immediately follow an assistant turn with a
  non-empty tool-call set. Assistant-contained `ToolResult` values must pair
  with that assistant's calls and resolve those IDs; a following `ToolMessage`
  is required only for remaining unresolved IDs, and result IDs are unique
  across both locations. It also validates finite sampling,
  object-or-boolean JSON Schema validation, and capability validation for
  structured output.
# A. Crate layout
## Decision: workspace, not feature-flagged adapters
```text
oven-sdk/
  Cargo.toml
  crates/oven-sdk/
  crates/oven-sdk-anthropic/
  crates/oven-sdk-openai/
  crates/oven-sdk-conformance/
  README.md
  CHANGELOG.md
  LICENSE
```
Separate adapter crates are preferable because Cargo features are additive. A downstream graph can otherwise unify all adapters and unexpectedly pull every HTTP, TLS, SSE, and runtime dependency. Separate publishing also lets adapter fixes move without forcing a core release.
Keep generic OpenAI-compatible Chat inside `oven-sdk-openai`; it shares the Chat translator, parser, transport, and capability profile. Splitting it would mostly duplicate code.
Do not publish an `oven-sdk-http` crate initially. Extract shared transport support only after at least three adapters demonstrate a stable common API. Small private duplication is cheaper than stabilizing a premature abstraction.
Future protocol crates are `oven-sdk-google`, `oven-sdk-google-vertex`, `oven-sdk-bedrock`, `oven-sdk-azure`, and `oven-sdk-cohere`. `anthropic-aws` is not a separate crate: Anthropic models served through AWS use the Bedrock adapter plus an Anthropic model/profile because the wire transport is Bedrock Converse.
Core dependencies should remain close to:
```toml
bitflags = { version = "2", features = ["serde"] }
bytes = { version = "1", features = ["serde"] }
event-listener = "5"
futures-core = "0.3"
futures-util = { version = "0.3", default-features = false, features = ["alloc"] }
http = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
url = { version = "2", features = ["serde"] }
tracing = { version = "0.1", optional = true }
```
Core must not depend on Tokio, reqwest, async-trait, OpenSSL, or a vendor SDK. Adapter crates may use reqwest with default features disabled, Rustls, incremental SSE parsing, and Tokio timers.
# Provider coverage roadmap
Owner decision: `oven-sdk` targets full Vercel AI SDK **LLM-provider** coverage, organized by wire protocol rather than by vendor count. The tiers below are release scope and quality gates, not marketing labels.

## Tier 0.1 — initial release
Tier 0.1 is unchanged:

- `oven-sdk-anthropic`: Anthropic Messages.
- `oven-sdk-openai`: OpenAI Chat Completions, OpenAI Responses, and configurable OpenAI-compatible Chat.
- `oven-sdk-conformance`: the shared gate for first-party and community adapters/presets.

## Tier 2 — distinct protocol adapters
Each distinct protocol/auth family gets its own adapter crate and must implement the complete applicable conformance matrix, including request translation, streaming lifecycle, structured errors, cancellation, usage, capability claims, and native replay where the protocol requires it.

| Proposed order | Crate | Coverage |
|---|---|---|
| 1 | `oven-sdk-google` | Google Gemini `generateContent` / streaming generateContent for Google AI Studio, including the first full Tier 2 audio- and video-input encoding/validation coverage. |
| 1 | `oven-sdk-google-vertex` | Vertex-hosted Gemini with Vertex resource endpoints and Google Cloud authentication, published separately despite shared Gemini semantics. |
| 2 | `oven-sdk-bedrock` | Amazon Bedrock Converse/ConverseStream, including SigV4, AWS EventStream, Bedrock model IDs, and protocol-specific errors/replay. |
| 3 | `oven-sdk-azure` | Azure OpenAI using OpenAI Chat/Responses wire shapes plus Azure endpoint, deployment/API-version, `api-key`, and supported Entra-auth configuration. |
| 4 | `oven-sdk-cohere` | Cohere v2 Chat and its native message/tool/stream/error model. |

**Proposed implementation priority, pending owner confirmation:** Google family first (`google`, then `google-vertex`), then Bedrock, Azure, and Cohere. No Tier 2 crate is considered shipped until its full applicable conformance tier passes.

`anthropic-aws` maps to `oven-sdk-bedrock` plus an Anthropic model/profile; it is not another adapter crate. The SDK's unit of implementation is the actual wire protocol and authentication/transport family.

## Tier 3 — OpenAI-compatible preset catalog
Tier 3 expands `oven-sdk-openai` through shipped `CompatibilityProfile` presets, not one crate per vendor. Initial catalog:

- `groq`, `xai`, `deepseek`, `mistral`, `fireworks`, `deepinfra`, `cerebras`;
- `baseten`, `togetherai`, `perplexity`, `alibaba` (DashScope compatible mode);
- `bytedance` (Ark), `openrouter`, and `ollama`.

Each preset specifies its base URL template, authentication header shape, conservative capability defaults, request-field differences, error mappings, and known quirks. Examples include DeepSeek reasoning-field replay/omission rules and Groq tool-call/stream behavior. The preset API should remain data-driven, for example `CompatibilityProfile::preset(CompatibleProvider::DeepSeek)`, with typed overrides for endpoint or account-specific differences.

Preset additions are batchable: one release or pull request may add several presets, but each preset is independently gated by at least the OpenAI-compatible conformance baseline and one environment-gated ignored live test. A batch fails if any included preset fails its required tier.

Tier 3 presets are the intended community-contribution path. `oven-sdk-conformance` must make a preset contribution substantially smaller and safer than implementing a new protocol adapter while preventing unsupported capability claims.

## Coverage quality bar
No adapter crate or compatibility preset ships merely because a request succeeds once. Its declared conformance tier must pass first. Capability claims, replay behavior, error mapping, and stream finalization are part of coverage; a provider logo or base URL alone is not.

# B. Core trait
## Model-level and object-safe
Use explicit boxed futures rather than `async_trait`; allocation, lifetimes, dyn compatibility, and `Send` are then visible.
```rust
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
pub type ModelStream = Pin<Box<dyn futures_core::Stream<Item = Result<StreamPart, ModelError>> + Send + 'static>>;
pub trait LanguageModel: Send + Sync {
    fn identity(&self) -> &ModelIdentity;
    fn capabilities(&self) -> &ModelCapabilities;
    fn url_support(&self, media_type: &str, url: &url::Url) -> UrlSupport;
    fn generate<'a>(&'a self, request: ModelRequest, context: CallContext)
        -> BoxFuture<'a, Result<GenerateResult, ModelError>>
    { Box::pin(async move { self.stream(request, context).await?.collect().await }) }
    fn stream<'a>(&'a self, request: ModelRequest, context: CallContext)
        -> BoxFuture<'a, Result<StreamResponse, ModelError>>;
}
```
`stream` is required. Initial official adapters should implement `generate` by collecting `stream`, retaining one translator/parser/replay path. A native non-streaming override is allowed only when conformance proves equivalent content, finish, usage, warnings, errors, and replay.
## Identity and semver
```rust
pub struct ModelIdentity {
    pub provider_id: ProviderId, pub model_id: ModelId, pub adapter_id: AdapterId,
}
```
Crate semver is the sole public contract-era mechanism. In a statically linked Rust program, core and adapter API compatibility is resolved at build time; the separate runtime contract marker used by Vercel's independently deployed npm packages would duplicate semver without adding a distinct recovery path. Breaking normalized-contract changes therefore use ordinary crate semver.

Replay likewise has no SDK-level format number or envelope-era field. The removed trait marker, payload integer, and envelope-era marker did not encode distinct policies: adapter mismatch, declared format mismatch, and payload decode failure all have the same behavior—discard, reconstruct, and report. Successful adapter-private decoding is therefore the compatibility check, and serde errors provide more useful diagnostics than an integer.
`AdapterId` is a stable string newtype, not a closed enum. Official values:
- `oven.anthropic.messages`
- `oven.openai.chat`
- `oven.openai.responses`
A compatible endpoint receives a stable caller-selected ID such as `openrouter.chat` or `company.internal-vllm.chat`. Registry aliases never determine replay compatibility.
## Capabilities
Keep an explicit bitset plus limits and semantics:
```rust
bitflags! {
    pub struct CapabilitySet: u128 {
        const TOOL_CALLING = 1 << 0; const PARALLEL_TOOLS = 1 << 1;
        const TOOL_INPUT_DELTAS = 1 << 2; const REASONING = 1 << 3;
        const REASONING_REPLAY = 1 << 4; const IMAGE_INPUT = 1 << 5;
        const DOCUMENT_INPUT = 1 << 6; const AUDIO_INPUT = 1 << 7;
        const VIDEO_INPUT = 1 << 8; const STRUCTURED_OUTPUT = 1 << 9;
        const PROMPT_CACHING = 1 << 10; const USAGE = 1 << 11;
        const PROVIDER_TOOLS = 1 << 12; const SOURCES = 1 << 13;
        const FILE_OUTPUT = 1 << 14; const NATIVE_REPLAY = 1 << 15;
    }
}
pub struct ModelCapabilities {
    pub features: CapabilitySet, pub context_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>, pub cancellation: CancellationCapability,
    pub replay: ReplayCapability,
}
pub enum ReplayCapability { None, Optional, Required }
pub enum CancellationCapability { LocalOnly, RemoteBestEffort, Unsupported }
```
Capabilities belong to the configured model/profile, not a protocol or vendor globally. This is especially important for `AUDIO_INPUT` and `VIDEO_INPUT`: support varies by model and profile even when the wire format can represent the media. Unknown model IDs use conservative defaults.
Do not probe each call. Compatible adapters may expose an explicit probe that returns a serializable `CompatibilityProfile`; callers decide when to run and cache it. An unprobed endpoint claims only baseline Chat text, tool echo/result pairing, basic SSE, and ordinary 429/5xx handling.
## Supported URLs equivalent
```rust
pub enum UrlSupport { Native, DownloadRequired, UploadRequired, Unsupported }
```
The SDK reports whether a provider can consume a URL directly. It does not automatically fetch arbitrary URLs; the host owns SSRF, MIME, credential, and size policy.
# C. Content and message model
Use a coding-agent-first union that is richer than today's text/image/PDF model but smaller than Vercel's full cross-product.
```rust
pub enum HistoryTurn {
    System(SystemMessage), User(UserMessage), Assistant(CompletedTurn), Tool(ToolMessage),
}
pub struct SystemMessage { pub content: Vec<SystemPart> }
pub struct UserMessage { pub content: Vec<InputPart> }
pub struct AssistantMessage { pub content: Vec<AssistantPart> }
pub struct ToolMessage { pub results: Vec<ToolResult> }
pub enum AssistantPart {
    Text(TextPart), Reasoning(ReasoningPart), ToolCall(ToolCall), ToolResult(ToolResult),
    File(FilePart), Source(SourcePart), Custom(CustomPart),
}
```
Role-specific unions prevent obviously invalid combinations: system supports text/custom, user supports text/file/custom, and tool messages carry results.
`ReasoningPart` is visible normalized reasoning/summary. Signed Anthropic thinking, OpenAI encrypted reasoning, redacted blocks, and unknown continuation fields remain authoritative only in native replay.
Keep one generic MIME-typed file abstraction for images, documents, audio, video, and future media. Do not add `AudioPart` or `VideoPart` variants:
```rust
pub struct FilePart {
    pub media_type: String, pub filename: Option<String>,
    pub source: FileSource, pub metadata: PartMetadata,
}
pub enum FileSource {
    Bytes(bytes::Bytes), Text(String), Url(url::Url),
    ProviderReference { provider: ProviderId, id: String },
}
impl FilePart {
    pub fn new(media_type: impl Into<String>, source: FileSource) -> Self { Self { media_type: media_type.into(), filename: None, source, metadata: Default::default() } }
    pub fn audio(media_type: impl Into<String>, source: FileSource) -> Self { Self::new(media_type, source) }
    pub fn video(media_type: impl Into<String>, source: FileSource) -> Self { Self::new(media_type, source) }
    pub fn image(media_type: impl Into<String>, source: FileSource) -> Self { Self::new(media_type, source) }
    pub fn document(media_type: impl Into<String>, source: FileSource) -> Self { Self::new(media_type, source) }
}
```
These are ergonomic constructors, not semantic union variants; all four produce the same `FilePart`. A generic file is the simpler forward-compatible choice: even with `#[non_exhaustive]` outer enums, adding one variant per modality proliferates downstream branches and predicts future media poorly. MIME type plus `FileSource` lets new modalities flow through existing input unions without growing them; reserve new union variants for genuinely different semantics, not media subtypes.

Each adapter maintains media-type/source support tables per wire profile and model profile. Request encoding validates every `FilePart` against both the model capability flags and that table before network I/O. Unsupported MIME types, source forms, or model/media combinations return `ModelError { category: ErrorCategory::Unsupported, stage: ErrorStage::RequestEncoding, .. }`; adapters never silently drop media and never delegate predictable validation to a provider 400.

Current adapter-tier guidance, not core policy: Gemini profiles lead with audio and video support; OpenAI profiles advertise audio only for models/endpoints that actually accept it; Anthropic profiles advertise neither audio nor video. The tables, not protocol-wide assumptions, are authoritative.

`SourcePart` is output citation/provenance: optional ID, URL, title, media type, excerpt, and metadata. It is not a retriever abstraction.
`CustomPart { kind, data, metadata }` uses a namespaced kind such as `openai.refusal`. It is an inspectable normalized extension, not a replay substitute or unbounded body dump.
Tool IDs must remain distinct:
```rust
pub struct ToolCall {
    pub id: ToolCallId, pub provider_item_id: Option<String>, pub name: String,
    pub input: Value, pub raw_input: Option<String>,
    pub execution: ToolExecution, pub metadata: PartMetadata,
}
```
Adapters concatenate raw argument fragments by native block/item index, parse once at completion, and emit a finalized `ToolCall`. Malformed final JSON is a model/protocol failure. `raw_input` preserves the exact assembled argument string when needed.
`ToolResultOutput` supports text, JSON, mixed text/files, and denied execution. Keep `is_error` explicit. The SDK represents denials for model visibility but never makes permission decisions.
Use `ProviderOptions = BTreeMap<String, Value>` keyed by adapter/provider namespace at request, message, and part level through `PartMetadata`. Official adapter crates should add typed option builders over this escape hatch.
```rust
pub struct ModelRequest {
    pub history: Vec<HistoryTurn>, pub tools: Vec<ToolDefinition>, pub tool_choice: ToolChoice,
    pub response_format: ResponseFormat, pub inference: InferenceOptions,
    pub provider_options: ProviderOptions,
}
```
The request has no model ID because the model object owns it.
# D. StreamPart and lifecycle
```rust
#[non_exhaustive]
pub enum StreamPart {
    StreamStart { warnings: Vec<Warning> }, ResponseMetadata(ResponseMetadata),
    TextStart { id: BlockId, metadata: PartMetadata },
    TextDelta { id: BlockId, delta: String, metadata: PartMetadata },
    TextEnd { id: BlockId, metadata: PartMetadata },
    ReasoningStart { id: BlockId, metadata: PartMetadata },
    ReasoningDelta { id: BlockId, delta: String, metadata: PartMetadata },
    ReasoningEnd { id: BlockId, metadata: PartMetadata },
    ToolInputStart { block_id: BlockId, call_id: ToolCallId,
        provider_item_id: Option<String>, tool_name: String,
        execution: ToolExecution, metadata: PartMetadata },
    ToolInputDelta { block_id: BlockId, delta: String },
    ToolInputEnd { block_id: BlockId },
    ToolCall { block_id: BlockId, call: ToolCall }, ToolResult { result: ToolResult },
    File { file: FilePart }, Source { source: SourcePart }, Custom { part: CustomPart },
    Usage { usage: Usage, kind: UsageUpdateKind }, Error { error: ModelError },
    Abort { origin: AbortOrigin, reason: Option<String> }, Finish(FinishPart),
    Raw { value: serde_json::Value },
}
```
Text, reasoning, and tool-input IDs are mandatory. If the provider has no stable ID, the adapter generates a deterministic per-call ID. `provider_item_id` remains distinct from semantic `call_id`.
`ToolInputStart/Delta/End` expose argument streaming; `ToolCall` is the finalized parsed call. This moves argument assembly out of cookie-agent's engine.
Each stream accepts a tool-result ID at most once. A result referencing no
normalized in-stream `ToolCall` is retained because provider-executed hosted
tools may not expose a caller-visible call; collection records a non-fatal
warning on the resulting `CompletedTurn` instead of rejecting it.
`Raw` is opt-in and never automatically copied into replay.
```rust
pub struct FinishPart {
    pub reason: FinishReason, pub usage: Usage, pub response: ResponseMetadata,
    pub provider_metadata: ProviderMetadata, pub native_replay: Option<NativeReplayArtifact>,
}
pub enum FinishReason {
    Stop, ToolCalls, Length, ContentFilter, Cancelled, Error, Other { raw: Option<String> },
}
```
Lifecycle rules are normative:
1. A semantically completed stream emits exactly one `Finish` as its final successful item.
2. EOF before `Finish` is `UnexpectedEof`, never success.
3. No part follows `Finish`.
4. An in-band provider error after HTTP 200 emits `Error`, then `Finish(Error)`, then EOF.
5. A provider-originated abort emits `Abort`, then `Finish(Cancelled)`, then EOF.
6. Fatal transport, decode, parser, or invariant failures are stream `Err(ModelError)` and end without synthetic `Finish`.
7. Dropping a stream emits nothing; no consumer remains.
8. `collect()` rejects missing/duplicate finish, unclosed blocks, malformed calls, and terminal in-band errors.
This intentionally combines error-as-data and `Result`: in-band errors preserve provider semantics; stream `Err` means trustworthy normalized continuation is impossible; `Finish` proves semantic terminal state.
```rust
pub struct StreamResponse { pub stream: ModelStream, pub request: RequestMetadata, pub response: ResponseHead }
pub struct GenerateResult { pub turn: CompletedTurn, pub warnings: Vec<Warning>, pub request: RequestMetadata, pub response: ResponseMetadata }
pub struct CompletedTurn {
    pub message: AssistantMessage, pub finish_reason: FinishReason, pub usage: Usage,
    pub response: ResponseMetadata, pub native_replay: Option<NativeReplayArtifact>,
}
```
Usage contains optional input total/non-cache/cache-read/cache-write and output total/text/reasoning fields plus optional raw usage. Never add component subsets to inclusive totals. Final `Finish.usage` is authoritative; interim usage declares partial versus cumulative.
# E. ModelError and classification seam
Replace orchestration classes with structured facts:
```rust
#[derive(Debug, thiserror::Error)]
#[error("{category:?} error from {provider_id}/{model_id}: {message}")]
pub struct ModelError {
    pub category: ErrorCategory, pub message: String,
    pub provider_id: ProviderId, pub model_id: ModelId, pub adapter_id: AdapterId,
    pub vendor_code: Option<String>, pub http_status: Option<u16>,
    pub retry_after: Option<Duration>, pub request_id: Option<String>,
    pub sanitized_body: Option<SanitizedBody>, pub stage: ErrorStage,
    pub diagnostics: ErrorDiagnostics, pub retry_hint: RetryHint,
}
pub struct ErrorDiagnostics {
    pub bytes_received: u64, pub stream_parts_emitted: u64, pub elapsed: Option<Duration>,
}
```
`ErrorStage`: request validation/encoding, connect, response headers, stream read/decode/event/finalize, replay encode/decode, and middleware.
`ErrorCategory`: authentication, authorization, invalid request, model not found, context length, rate limit, quota, overload, timeout, transport, protocol, invalid tool input, content filter, cancelled, unsupported, replay, provider, and unknown.
`RetryHint`: `Retryable`, `NotRetryable`, `After(Duration)`, or `Unknown`. It is advisory and never decides chain behavior.
Adapters classify from `(status, vendor code/type, body text, in-stream event, retry-after headers)`, not status alone. Sanitized bodies are capped at 64 KiB and marked when truncated. Display/tracing omit credentials, request bodies, replay payloads, and unsanitized headers.
`request_id`, `stage`, and `bytes_received` are required diagnostic targets. A long Responses failure must identify read/decode phase and byte count instead of only saying `error decoding response body`.
Cookie-agent retains:
```rust
pub enum AttemptDisposition { RetrySameEntry, AdvanceEntry, FailRun }
pub trait ModelErrorClassifier { fn classify(&self, error: &ModelError) -> AttemptDisposition; }
```
Preserve policy ordering:
1. Cancellation and context overflow -> fail run.
2. Known `model_not_found`, `invalid_model`, and model-does-not-exist codes -> advance, even with 5xx.
3. Model-not-found/auth/invalid-request/quota -> advance.
4. Rate-limit/overload/timeout/transport/ordinary 5xx -> retry same entry.
5. Otherwise consult the hint and configured conservative default.
The engine separately applies its meaningful-output guard; retryable failures after text, reasoning, tool input, or a finalized call do not retry the same entry.
# F. Cancellation and deadlines
```rust
#[derive(Clone, Default)]
pub struct CancellationToken { /* Arc<State>: AtomicBool + event_listener */ }
impl CancellationToken {
    pub fn cancel(&self) { /* wake listeners */ }
    pub fn is_cancelled(&self) -> bool { unimplemented!() }
    pub async fn cancelled(&self) { unimplemented!() }
}
pub struct CallContext {
    pub cancellation: CancellationToken, pub deadline: Option<Instant>,
    pub telemetry: Option<Arc<dyn TelemetrySink>>, pub tags: BTreeMap<String, String>,
}
```
Honest guarantees:
- Cancellation stops local request initiation/reading as scheduling permits and drops the local response future/body.
- It does not prove the provider stopped computation or billing.
- `RemoteBestEffort` means the adapter attempts a provider cancellation operation, not that it succeeded.
- Dropping `ModelStream` is local abandonment only.
- The harness independently stops local tool scheduling/execution.
Separate connect, response-header, stream-idle, and caller deadlines. Recommended defaults: 10 seconds connect, 30 seconds headers, 60 seconds idle, and no total streaming timeout. Valid pings/activity reset idle time. This directly addresses phase-6 failures near the current two-minute overall timeout.
# G. Native replay extension
```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeReplayArtifact {
    pub adapter_id: AdapterId, pub payload: serde_json::Value,
}
```
The harness treats `payload` as opaque. It contains replay-critical provider-native assistant blocks/items and continuation fields only.
Every request encoding reports each artifact decision through `RequestMetadata`:
```rust
pub struct ReplayReport { pub decisions: Vec<ReplayDecision> }
pub struct ReplayDecision { pub history_index: usize, pub disposition: ReplayDisposition }
pub enum ReplayDisposition {
    Replayed, NoArtifact,
    DiscardedForeignAdapter { found: AdapterId, expected: AdapterId },
    DiscardedInvalidPayload { reason: String }, ReconstructedNormalized,
}
```
Provider-swap equivalence is normative: a foreign `AdapterId`, an artifact that cannot be loaded, and a payload that the selected adapter cannot decode all produce the same operational result. The artifact is discarded for that request, the turn is reconstructed from normalized content, and `ReplayReport` records the discard; serde decode errors are preserved in `DiscardedInvalidPayload { reason }`. These cases never fail the turn.

Adapters may embed a private format key inside their opaque JSON payload when they need to distinguish intentional format eras that otherwise look compatible. That key is adapter-private: core never interprets it, and an unrecognized value or failed decode still follows provider-swap semantics.
Artifact policy:
- Capture a positive allow-list of replay-required fields.
- Exclude complete SSE chunks, HTTP bodies, headers, and requests.
- Do not automatically redact/truncate signed or encrypted payloads.
- Never include payloads in `Debug`, tracing, or errors.
- Check serialized size before `Finish`.
- Default maximum: 2 MiB per assistant turn, configurable.
- A newly captured artifact exceeding that cap is the only replay-policy condition that fails the turn; fail-closed oversize handling remains required.
- Existing-artifact load/decode failures never fail the turn, including for `ReplayCapability::Required`; `Required` means successful new turns should capture replay state, not that old opaque bytes must decode forever.
Raw transport capture is a separate opt-in capped/redacted diagnostic channel and never becomes replay state.
`CompletedTurn` is also the assistant history shape; ordered `HistoryTurn::Tool` results follow it. The SDK does not persist either type.
Cookie-agent consumption:
1. Assemble committed journal events into `HistoryTurn`.
2. Attach stored artifacts to assistant `CompletedTurn`s.
3. Pass history to the selected model.
4. Persist/process normalized parts as desired.
5. Persist `Finish.native_replay` before committing the assistant turn.
6. Keep artifacts after fallback for possible later same-adapter replay.
This preserves section 6.2 without letting the SDK own durability.
## Migration from current `TurnOpaque`
| Legacy protocol | New adapter ID |
|---|---|
| `AnthropicMessages` | `oven.anthropic.messages` |
| `OpenAiChatCompletions` | `oven.openai.chat` |
| `OpenAiResponses` | `oven.openai.responses` |
| `OpenAiCompatible` | configured ID or `legacy.openai-compatible` |
The official Anthropic and OpenAI adapters ship tolerant decoders that accept today's `TurnOpaque` payload shapes. An old payload that no longer decodes is not a migration failure: it is reported as `DiscardedInvalidPayload`, reconstructed from normalized content, and otherwise treated exactly like a provider swap.

Do not rewrite old logs. During transition, the event may remain named `TurnOpaque`; the engine maps its legacy protocol tag to `AdapterId` and passes the payload unchanged at the SDK boundary. Same-adapter history uses native tool IDs when decoding succeeds, while foreign or undecodable replay uses canonical engine IDs, preserving `crates/engine/src/lib.rs:4180-4543`.
Any implementation change to this durable model requires a same-commit `ARCHITECTURE.md` update.
# H. Registry, factories, middleware, and custom providers
```rust
pub trait ProviderFactory: Send + Sync {
    fn provider_id(&self) -> &ProviderId;
    fn model(&self, model_id: &ModelId) -> Result<Arc<dyn LanguageModel>, ModelError>;
}
```
A configured factory owns credentials, base URL, HTTP client, headers, endpoint/profile, and artifact policy. Model creation is local; network probing remains explicit.
Prefer an immutable registry:
```rust
let registry = Registry::builder()
    .factory(openai_factory)
    .factory(anthropic_factory)
    .model("test:scripted", scripted_model)
    .alias("fast", "openai:gpt-5-mini")
    .alias("reasoning", "anthropic:claude-sonnet-4-6")
    .layer(tracing_layer)
    .build()?;
```
Aliases resolve to one model locator. They never encode chains, weights, retries, or run state. Reject duplicates and alias cycles during build.
Use decorator middleware:
```rust
pub trait LanguageModelMiddleware: Send + Sync {
    fn wrap(&self, inner: Arc<dyn LanguageModel>) -> Arc<dyn LanguageModel>;
}
```
A wrapper implements `LanguageModel` and may transform requests, wrap generate/stream, map parts, add telemetry, or enforce defaults. It must preserve finish rules, cancellation, transport errors, and adapter identity. Do not ship fallback/retry middleware in 0.1.
Custom providers implement `LanguageModel` directly or supply a factory. OpenAI-compatible configuration should be conservative:
```rust
let factory = OpenAiCompatible::builder()
    .provider_id("openrouter")
    .adapter_id("openrouter.chat")
    .base_url("https://openrouter.ai/api/v1")?
    .api_key(secret)
    .profile(CompatibilityProfile::baseline_chat()
        .with_parallel_tools(true)
        .with_stream_usage(true)
        .with_reasoning_field(ReasoningField::ReasoningContent))
    .build()?;
```
Only explicit profiles or saved probes enable advanced capabilities. Allow injected clients and custom auth/header functions, while always redacting credential headers.
# I. Fallback/orchestration extraction
Remove from the SDK:
- `FallbackExecutor`, `FallbackRunState`, and `ModelFallback`.
- `ProviderErrorClass`.
- retry counts/backoff timers.
- meaningful-output decisions.
- chain traversal and sticky state.
Cookie-agent retains:
- policy-snapshot model chains and per-run sticky index;
- per-model capability-aware request assembly;
- engine error classifier and code precedence;
- same-entry retries/backoff and meaningful-output guard;
- `AttemptAbandoned`/`ModelFallback` durability;
- partial-output abandonment and usage attribution;
- sessions, tools, permissions, approvals, and cancellation.
Exact seam:
```text
resolve chain entry -> Arc<dyn LanguageModel>
read capabilities -> assemble ModelRequest
call stream(request, CallContext)
persist/process StreamPart values
track meaningful output
persist Finish.native_replay before commit
classify ModelError in engine
retry/advance/fail in engine
```
An adapter performs one translated provider call. It never advances a model chain.
# J. Ergonomics and telemetry
```rust
let request = ModelRequest::builder()
    .system("You are a concise coding assistant.")
    .user_text("Explain this borrow checker error.")
    .tool(tool_definition)
    .max_output_tokens(2_000)
    .reasoning(ReasoningEffort::Medium)
    .build()?;
let result = model.generate(request.clone(), CallContext::default()).await?;
let response = model.stream(request, context).await?;
```
Provide:
- `CompletedTurn::text()`.
- `StreamResponse::collect()` with strict lifecycle validation.
- `StreamResponse::text_stream()` that still surfaces errors.
- `collect_text()` that rejects tool calls unless explicitly allowed.
- `ModelRequest::validate_for(capabilities)`.
- typed adapter option builders over generic provider options.
Document that `text_stream()` intentionally drops tools, files, sources, reasoning, and custom parts.
Core telemetry records only metadata by default: identity, operation, timing, status, request ID, byte/part counts, finish, usage, error category/stage/hint, replay disposition, and artifact size.
Do not record prompts, output text, tool data, provider options, bodies, arbitrary headers, or artifact payloads by default. Content logging is explicit opt-in; credentials are always redacted. Offer an optional `tracing` layer, not an OpenTelemetry dependency in core.
# K. Testing and conformance
Official adapters need tests for:
- request headers/bodies with wiremock;
- arbitrary SSE boundaries, including one-byte chunks;
- LF/CRLF, multiline data, comments, pings, empty and usage-only events;
- parallel/fragmented and full-call-only tool delivery;
- malformed finalized arguments;
- in-band errors after HTTP 200;
- truncated body and EOF before finish;
- cancellation before headers and mid-stream;
- request ID, retry-after, stage, byte count, and body sanitation;
- per-model/profile image, document, audio, and video capability-table cases;
- encoding cases for supported modality/MIME/`FileSource` combinations, including Gemini audio and video;
- negative media cases that return `Unsupported` during request encoding without sending an HTTP request;
- constructor equivalence proving `audio`, `video`, `image`, and `document` all remain ordinary `FilePart` values;
- same-adapter replay and foreign fallback;
- tolerant legacy-payload decode, garbage-payload discard/reconstruction, and artifact size policy;
- generate/stream equivalence.
Publish `oven-sdk-conformance`; it may depend on Tokio and wiremock because consumers use it as a dev-dependency.
```rust
assert_stream_contract(stream).await?;
assert_generate_stream_equivalent(model, request).await?;
assert_replay_round_trip(codec, turn).await?;
assert_foreign_replay_is_reported(model, turn).await?;
assert_invalid_replay_reconstructs(model, garbage_turn).await?;
```
Replay conformance also requires decode gracefulness: garbage or obsolete opaque payloads must produce `DiscardedInvalidPayload`, reconstruct from normalized content, and continue without a model-call failure. Capability claims are valid only when corresponding conformance cases pass. `IMAGE_INPUT`, `DOCUMENT_INPUT`, `AUDIO_INPUT`, and `VIDEO_INPUT` each have independent positive and negative conformance cases because media support is model/profile-specific. This public kit is a differentiator for third-party Rust adapters.
Live tests remain ignored and environment-gated. Cover Anthropic, Chat, Responses, compatible baseline, replay, cancellation, and a stream exceeding two minutes or a configured byte threshold. Failure output includes request ID, stage, elapsed time, and bytes, never credentials or full bodies.
Transport conformance requires incremental UTF-8/SSE parsing, byte counting before decode, no network-chunk/event assumption, idle reset on pings, no default total stream timeout, and no raw chunks in replay.
# L. Versioning, MSRV, dependencies, and publishing
Start all crates at 0.1.0. Coordinate minor versions during 0.x; adapter patches may move independently.
Compatibility/versioning principle:
- crate semver is the sole public contract-era mechanism for Rust APIs and normalized behavior;
- replay compatibility is `AdapterId` identity plus successful adapter-private decode; a decode error is the compatibility check and triggers report-and-reconstruct behavior.
Changing finish, error, or replay invariants is a semver-breaking core change. Optional metadata or new capabilities are minor during 0.x.

Use edition 2024 and MSRV **1.88** initially, matching cookie-agent. CI: Rust 1.88 minimal features, current stable, beta allowed-to-fail initially, and core on Linux/macOS/Windows. Raise MSRV only in a documented minor release during 0.x.

Dependency policy:
- no cookie-agent dependency;
- no Tokio/reqwest/vendor SDK in core;
- large dependency default features disabled;
- Rustls default;
- no global mutable registry/client;
- no published path dependencies;
- committed workspace lockfile;
- `cargo deny`, `cargo audit`, license checks, and `cargo semver-checks` in CI.
## License decision

**Plain MIT.** The repository ships a standard MIT `LICENSE` with `Copyright (c) 2026 Yunhao Cao`, and every published crate uses `license = "MIT"` (workspace-inherited). cookie_agent is licensed identically.

### Cargo, crates.io, and downstream implications

- Standard SPDX `MIT` metadata; no `license-file`, no custom identifiers, no downstream surprises for SPDX-allow-listing organizations.
- Verify `cargo package --list` for every crate includes the `LICENSE` file.

Publishing checklist:
1. Reserve all crate names and apply identical MIT license metadata/content to core, adapters, conformance, examples, and generated docs.
2. Complete repository/docs.rs/readme/rust-version metadata for each crate.
3. Document key, telemetry, body, and replay implications; ship direct-call, registry, compatible, cancellation, and persistence-handoff examples.
4. Run fmt, clippy, tests, doc tests, MSRV, audit/deny, and semver checks.
5. Inspect packaged contents; exclude secrets/logs/large recordings and confirm `LICENSE` inclusion.
6. Run `cargo publish --dry-run` from a clean checkout.
7. Publish core, adapters, then conformance using crates.io trusted publishing/provenance.
9. Test crates.io artifacts from a separate empty project, including common `cargo-deny`/SBOM behavior, before announcing the release.
# M. Migration plan
## Stage 0: approve the contract
Settle lifecycle, replay identity, and decode/discard policy before implementation commits.
## Stage 1: core plus compatibility bridge
Add temporary mappings:
- `ProviderRequest` -> `ModelRequest`.
- `PersistedTurn` -> `HistoryTurn`.
- `NormalizedEvent` -> `StreamPart`.
- `AssistantTurnOpaque` -> `NativeReplayArtifact { adapter_id, payload }`.
- `ProviderError` -> `ModelError`, while engine policy remains unchanged.
Do not alter durable journal behavior yet.
## Stage 2: extract adapters
Move Anthropic, Chat, Responses, and compatible code with existing fixtures and tolerant decoders for today's `TurnOpaque` payloads. Add typed finish and transport diagnostics before deleting old code.

Cookie-agent first uses pinned git revisions, never a moving branch:
```toml
oven-sdk = { git = "https://github.com/cookie-agent/oven-sdk", rev = "<sha>" }
oven-sdk-anthropic = { git = "...", rev = "<same-sha>" }
oven-sdk-openai = { git = "...", rev = "<same-sha>" }
```
## Stage 3: switch the engine seam
Resolve model objects from registry/factories. Require `Finish` for success, persist replay before commit, track meaningful output from rich parts, classify errors in engine, and treat missing finish as transport failure subject to the guard.
## Stage 4: migrate durable replay compatibly
Persist `AdapterId` plus opaque payload for new events while mapping legacy protocol tags at read time; add no SDK-level format fields. Keep artifacts after fallback, and degrade undecodable legacy payloads through provider-swap semantics. Update `ARCHITECTURE.md` in the same commit.
## Stage 5: delete duplicate orchestration
Remove `FallbackExecutor` and provider fallback types. Retain/expand engine tests for code precedence, pre-output retry, sticky fallback, attempt abandonment, foreign replay retention, cancellation, and persistence failures.
## Stage 6: compatibility release and crates.io
Optionally keep `cookie_agent_providers` as a deprecated shim for one release. After pinned-git integration is stable, publish 0.1.0, switch to crates.io ranges, and later remove the shim in a planned breaking cookie-agent release.

Expected source breaks:
- `Provider` becomes `LanguageModel`.
- Model ID moves from request to model object.
- Messages/persisted turns become typed history.
- `Stop` becomes mandatory `Finish`.
- SDK finalizes tool arguments.
- Opaque state becomes `AdapterId`-tagged replay with tolerant adapter-private decoding.
- SDK error classes disappear.
- Registry/factories replace provider maps.
- Implementations return boxed futures.
- OpenAI endpoints produce distinct model identities.
Section 6.2 must remain true throughout: compatible native replay wins, normalized reconstruction handles foreign/synthetic turns, artifacts remain durable, and tool IDs preserve same-adapter pairing.
# N. Explicit non-goals
`oven-sdk` 0.1 will not include:
- an automatic tool-loop or agent API;
- fallback, weighted routing, retries, budgets, or sticky run state;
- session/event-log persistence;
- permission, approval, sandbox, or tool execution policy;
- TypeScript or Python bindings;
- a server-side gateway, proxy, or router;
- model catalog, pricing service, or credential store;
- non-LLM Vercel-provider packages and their modality contracts during the LLM roadmap: `elevenlabs`, `cartesia`, `lmnt`, `assemblyai`, `deepgram`, `gladia`, `revai`, `fal`, `black-forest-labs`, `luma`, `replicate` (image), and `hume`;
- embedding, image, speech/audio, transcription, video, realtime, and reranking APIs during 0.x; embeddings, image, and speech contracts may be considered as separate post-1.0 extensions;
- automatic URL downloads or provider file uploads;
- implicit capability probing during ordinary calls.
Provider-executed tool parts may be represented, but the SDK neither approves nor executes them.
# Risks and open questions
1. **Compatible replay identity:** require caller-supplied stable IDs; do not derive them from aliases or raw base URLs. Decide whether common profiles use reserved `oven.*` IDs before publication.
2. **Artifact size:** 2 MiB fail-closed can reject unusually large turns. Keep configurable, measure real sizes, and never silently truncate.
3. **Middleware invariants:** decorators can drop finish or alter replay. Mitigate with strict collectors and conformance tests, not a complex type system.
4. **Generate equivalence:** stream collection is the default; native generate must earn duplicate implementation.
5. **Runtime boundary:** runtime-neutral core plus Tokio-backed first-party HTTP adapters is acceptable but must be explicit.
6. **Error plus finish:** in-band errors are `Error + Finish(Error)`; transport/parser failures are stream `Err` without finish. Harnesses treat both as failed attempts.
7. **Untyped provider options:** keep JSON for extensibility, with typed builders in official adapters.
8. **Unknown hosted items:** normalize categories needed for behavior; preserve every continuation item in replay.
9. **Replay decoder tolerance:** official adapters must retain practical support for current `TurnOpaque` shapes, but no opaque payload is guaranteed to decode forever; report-and-reconstruct is the stable compatibility behavior.
10. **Sanitation:** adapters need positive-field extraction and tests; regex redaction alone is insufficient.
# Review acceptance criteria
Implementation begins only after agreement that:

1. The workspace and dependency boundaries are acceptable.
2. `LanguageModel` is model-level, dyn-compatible, and runtime-neutral in core.
3. Exactly one final `Finish` is required for success.
4. In-band error, transport error, abort, drop, and EOF are unambiguous.
5. Error facts are separate from cookie-agent policy.
6. Replay is `AdapterId`-identified, decode-tolerant, bounded, and persistence-agnostic.
7. Foreign replay is reported and normalized reconstruction is preserved.
8. Registry aliases/middleware do not hide orchestration.
9. `FallbackExecutor` has a concrete deletion path.
10. Old logs preserve section 6.2 semantics.
11. The public conformance crate ships initially.
12. The provider coverage roadmap is accepted, the proposed Tier 2 order is owner-confirmed, and no adapter or preset can ship without its required conformance tier and live-test gate.
13. Agent loops, bindings, gateways, and non-LLM modality contracts remain out of the 0.x scope.
# Final recommendation
Proceed with this workspace design.
The decisive choices are mandatory `Finish`, structured mid-stream transport diagnostics, `AdapterId`-tagged decode-tolerant native replay, model-level objects, and complete extraction of fallback policy into the harness. They retain cookie-agent's strongest correctness properties while making `oven-sdk` a credible standalone Rust provider SDK rather than a renamed internal crate.
