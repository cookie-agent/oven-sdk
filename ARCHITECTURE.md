# oven-sdk Architecture

**Status:** core, conformance, and all workspace provider crates are migrated to
the breaking 0.4.0 contract and compile together. OpenAI Responses and Azure
OpenAI Responses V1 implement provider-native compaction; provider surfaces
without a native compaction endpoint explicitly declare it unsupported and
reject compaction/native-context requests before provider I/O. The exact
10-crate release matrix is published on crates.io.

**Current breaking core release:** `oven-sdk` 0.4.0.

`ARCHITECTURE.md` is the normative design record. The implementation and this
document change together. There are no compatibility aliases, old replay
decoders, dual capability types, migration shims, or deprecated construction
paths.

## 1. Scope and ownership

`oven-sdk` is a runtime-neutral normalized language-model contract. It owns:

- role-safe request/history and completion content;
- strict normalized streaming and terminal lifecycle;
- explicit provider/model declarations;
- capability, limit, modality, and media validation;
- structured errors and cancellation primitives;
- bounded, opaque, scope-aware native replay and provider-native context;
- public conformance helpers.

Provider crates own transport, authentication, protocol encoding/decoding,
protocol invariants, and one provider call. The calling harness owns
persistence, retries, fallback, routing, credentials discovery, environment
lookup, approvals, tool execution, permissions, and the agent loop.

## 2. Registry-free construction decision

Every provider model has one direct construction path:

```rust
XxxModel::new(ModelConfig<Auth, Settings>) -> Result<XxxModel, ModelError>
```

The selected concrete Rust type determines a structural API surface such as
Chat, Responses, Gemini `generateContent`, Bedrock Converse, or Azure Responses.
The model ID is opaque data used only for the provider request, descriptor
identity, structurally required resource construction, and replay scope.

The SDK has no:

- global or local model registry;
- provider/model aliases;
- built-in model families or capability tables;
- provider presets;
- automatic endpoint or credential environment lookup;
- runtime catalog download or dependency;
- model-name, prefix, suffix, substring, or case-folding inference;
- automatic probing during ordinary calls.

Applications own maps of configured `Arc<dyn LanguageModel>` values when they
need names, routing, or aliases.

## 3. models.dev terminology reference

Naming follows useful parts of the public models.dev schema at commit
`c3057690bbb8bd41cafdefadcd2a7b958e2a4642` without creating a dependency or
runtime integration.

| models.dev term | Oven term | Rule |
|---|---|---|
| Provider `id` | `ProviderId` | Serving provider identity |
| Provider model `id` | `ModelId` | Exact ID/deployment/resource sent to the provider |
| Provider + model | `ModelIdentity` | Runtime identity |
| Provider/model `api` | `ApiEndpoint` | Resolved and explicit |
| `limit.context/input/output` | `ModelLimits` | Direct numeric declaration |
| `modalities.input/output` | `Modalities` | Open string values |
| `tool_call` | `Capability::TOOL_CALLING` | Does not imply parallel calls or deltas |
| `reasoning` | `Capability::REASONING` | Wire controls remain explicit settings |
| `structured_output` | `Capability::STRUCTURED_OUTPUT` | Only when the selected adapter supports it |
| `temperature` | `Capability::TEMPERATURE` | Missing means unsupported |
| Provider headers | `HeaderOverrides` | Protected-header checks remain adapter-owned |
| Provider shape | Concrete model type | Chat and Responses are different types |
| Provider body | Typed adapter settings/options | Never blindly merged |
| Environment keys | Application concern | Core never reads process environment |

Intentional differences:

- Oven represents one selected provider offering, not a provider containing many
  models.
- Runtime identity is the selected serving provider plus exact serving model ID;
  canonical cross-provider IDs are application metadata.
- `ApiEndpoint` is explicit even when another SDK package would know a default.
- Oven has no npm/package selector, environment-key field, pricing, benchmarks,
  weights, lifecycle dates, or inheritance.
- Attachment booleans are too coarse; Oven declares modality, MIME type, and
  source form separately.
- Provider body data maps through typed settings or namespaced request options.

## 4. Core configuration contract

### 4.1 Identity, endpoint, secrets, and headers

```rust
pub struct ModelIdentity {
    pub provider_id: ProviderId,
    pub model_id: ModelId,
}

pub struct ApiEndpoint { /* validated Url */ }

impl ApiEndpoint {
    pub fn parse(value: impl AsRef<str>) -> Result<Self, ModelError>;
    pub fn as_url(&self) -> &url::Url;
}

pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self;
    pub fn expose_secret(&self) -> &str;
    pub fn is_empty(&self) -> bool;
}

pub struct HeaderOverrides { /* http::HeaderMap */ }

pub trait HeaderProvider: Send + Sync {
    fn headers(&self) -> Result<HeaderOverrides, ModelError>;
}

pub struct HeaderConfig {
    pub static_headers: HeaderOverrides,
    pub dynamic_headers: Option<Arc<dyn HeaderProvider>>,
}

pub struct ProviderConfig<A> {
    pub id: ProviderId,
    pub api: ApiEndpoint,
    pub auth: A,
    pub headers: HeaderConfig,
}
```

Identity deserialization rejects empty/control-character values. Endpoint
validation requires HTTP(S), a host, no userinfo, no query, no fragment, and no
unresolved `${...}` template. Query-bearing endpoints are rejected outright;
typed provider/request settings own any protocol-required query parameters.
Provider adapters may restrict HTTP further to loopback tests.

`SecretString` is cloneable but not serializable. Its `Debug` and `Display` are
redacted. `ApiEndpoint` debug output is fully redacted. `ProviderConfig` and
`ModelConfig` debug output redact the endpoint, authentication, and typed settings
values; header debug output contains names only and never values. Header wrappers
are not serializable. Dynamic headers are caller-managed and resolved without
core environment access.

### 4.2 Limits, modalities, and media

```rust
pub struct ModelLimits {
    pub context: Option<u64>,
    pub input: Option<u64>,
    pub output: Option<u64>,
}

#[serde(transparent)]
pub struct Modality(String);

pub struct Modalities {
    pub input: BTreeSet<Modality>,
    pub output: BTreeSet<Modality>,
}

bitflags! {
    pub struct MediaSourceSupport: u8 {
        const INLINE_BYTES;
        const INLINE_TEXT;
        const URL;
        const PROVIDER_REFERENCE;
    }
}

pub struct MediaInputSupport {
    pub media_types: Vec<String>,
    pub sources: MediaSourceSupport,
}

pub struct MediaCapabilities {
    pub input: BTreeMap<Modality, MediaInputSupport>,
}
```

`Modality` is open. Standard constructors are `text`, `image`, `audio`, `video`,
and `pdf`; future strings do not require a core release. A model declaration has
at least one input and output modality. Media rules use exact MIME values or a
single trailing wildcard such as `image/*`, and each rule declares at least one
source form.

The caller declaration can narrow protocol support but cannot expand immutable
adapter facts such as accepted MIME types, byte/count limits, URL restrictions,
signing rules, or schema subsets.

### 4.3 Features, replay, and complete declarations

```rust
bitflags! {
    pub struct Capability: u128 {
        const TOOL_CALLING;
        const PARALLEL_TOOLS;
        const TOOL_INPUT_DELTAS;
        const REASONING;
        const STRUCTURED_OUTPUT;
        const TEMPERATURE;
        const TOP_P;
        const MAX_OUTPUT_TOKENS;
        const PROMPT_CACHING;
        const USAGE;
        const PROVIDER_TOOLS;
        const SOURCES;
    }
}

pub struct ReplayDeclaration {
    pub policy: ReplayPolicy,
    pub capability: ReplayCapability,
    pub reasoning: bool,
}

pub struct ModelCapabilities {
    pub features: Capability,
    pub limits: ModelLimits,
    pub modalities: Modalities,
    pub media: MediaCapabilities,
    pub cancellation: CancellationCapability,
    pub compaction: CompactionCapability,
    pub replay: ReplayDeclaration,
}

pub struct ModelDeclaration {
    pub id: ModelId,
    pub capabilities: ModelCapabilities,
}

pub struct ModelConfig<A, S> {
    pub provider: ProviderConfig<A>,
    pub model: ModelDeclaration,
    pub settings: S,
}
```

There is no `ProviderCapabilities`. Model support belongs to one configured
provider offering. `ModelCapabilities::conservative()` is explicit text-only,
unknown-limit, no-feature, no-compaction, no-replay data; configuration types do not use
implicit model-name defaults.

Dependency validation includes:

- parallel tools and tool-input deltas require tool calling;
- reasoning replay requires reasoning and native replay support;
- media rules require their input modality;
- known nonzero input/output limits cannot exceed a known context limit.

The valid `ReplayPolicy` × `ReplayCapability` combinations are exactly:

- `Never` × `Unsupported`;
- `IfValid` × `Optional`;
- `IfValid` × `Required`;
- `Always` × `Required`.

Every other combination is invalid. `Never` × `Unsupported` performs no replay
inspection and captures no terminal artifact. `Optional` permits no terminal
artifact, but any artifact that is present must match the exact adapter and
`NativeContextScope`. `Required` requires a terminal artifact with the exact adapter and
scope.

## 5. Descriptor and model trait

```rust
pub struct LanguageModelDescriptor {
    pub identity: ModelIdentity,
    pub adapter_id: AdapterId,
    pub capabilities: ModelCapabilities,
    pub provider_metadata: ProviderMetadata,
}

pub trait LanguageModel: Send + Sync {
    fn descriptor(&self) -> LanguageModelDescriptor;

    fn capabilities(&self) -> ModelCapabilities {
        self.descriptor().capabilities
    }

    fn validate_request(&self, request: &Request) -> Result<(), ModelError>;
    fn supports_request(&self, request: &Request) -> bool;
    fn validate_compaction(&self, request: &CompactionRequest) -> Result<(), ModelError>;
    fn supports_compaction(&self, request: &CompactionRequest) -> bool;
    fn stream<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<StreamResponse, ModelError>>;
    fn complete<'a>(
        &'a self,
        request: Request,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<CompleteResult, ModelError>>;
    fn compact<'a>(
        &'a self,
        request: CompactionRequest,
        abort: AbortSignal,
    ) -> BoxFuture<'a, Result<CompactionResult, ModelError>>;
}
```

Validation and support methods have object-safe core defaults. `compact` has an
object-safe default that rejects unsupported declarations before provider I/O;
an adapter claiming native compaction must override it with one provider call.

The removed `supported_urls` method is represented by per-modality source rules.
Provider metadata is safe, complete descriptor metadata only; it never contains
secrets, arbitrary headers, request bodies, or replay payloads.

## 6. Request validation

`Request::validate_for(&ModelCapabilities)` is the common pre-network gate. It
validates:

- non-empty unique tool names and named/required tool-choice dependencies;
- complete assistant/tool-result pairing and unique IDs;
- tool requests and tool history against `TOOL_CALLING`;
- structured output against `STRUCTURED_OUTPUT`;
- finite/ranged temperature and top-p plus their explicit feature declarations;
- positive requested output tokens, feature support, and declared output limit;
- reasoning controls/history against `REASONING`;
- declared text output for the normalized LLM request contract;
- every file in user, assistant, and tool-result content;
- modality, MIME pattern, and source-form support;
- model capability/replay dependency consistency.
- native context only when `CompactionCapability::Native` is declared;
- compaction only when native compaction is declared, followed by the complete
  ordinary request validation matrix.

Adapters call this validation and then enforce their typed settings and immutable
protocol restrictions. No validation branch may inspect the model ID to select
behavior.

## 7. Role-safe content and tool approval

History is role-specific:

```rust
pub enum HistoryTurn {
    System(SystemMessage),
    User(UserMessage),
    Assistant(Box<CompletedTurn>),
    Tool(ToolMessage),
}
```

The generic `Role`, `Turn`, and `ContentPart` bridge types are deleted. Serde and
constructors cannot create invalid role/content combinations.

`StreamPart::ApprovalRequested` remains assistant completion output. The strict
collector preserves it as `AssistantPart::ToolApproval`. The calling harness
still owns approval policy and tool execution.

## 8. Stream and error lifecycle

Streaming remains authoritative. A successful semantic stream:

1. starts with exactly one `StreamStart`;
2. maintains ID-scoped text/reasoning/tool-call block lifecycle;
3. emits finalized tool calls exactly once after ended streamed blocks;
4. emits exactly one final `Finish`;
5. emits nothing after `Finish`.

EOF before `Finish` is `UnexpectedEof`. An in-band provider error is `Error`,
then `Finish(Error)`, then EOF. Fatal transport/parser/invariant errors are
stream `Err(ModelError)` without a synthetic finish. `LanguageModel::complete`
is the strict collector and preserves ordered content, warnings, request
metadata, response head, usage, and replay artifact.

`ModelError` remains structured factual data. Retry/fallback classification is
harness policy. `AbortSignal` remains runtime-neutral and proves only local
cancellation unless a declaration says `RemoteBestEffort`.

## 9. Scope-aware native replay and compaction

```rust
pub struct ResourceId(String);

pub struct NativeContextScope {
    pub provider_id: ProviderId,
    pub model_id: ModelId,
    pub resource_id: ResourceId,
}

pub struct NativeReplayArtifact {
    adapter_id: AdapterId,
    scope: NativeContextScope,
    payload: serde_json::Value,
}

pub struct NativeContextWindow {
    adapter_id: AdapterId,
    scope: NativeContextScope,
    payload: serde_json::Value,
}

pub struct CompactionRequest {
    pub request: Request,
}

pub struct CompactionResult {
    pub native_context: NativeContextWindow,
    pub usage: Usage,
    pub request: RequestMetadata,
    pub response: ResponseHead,
}

pub enum ReplayDisposition {
    Replayed,
    NoArtifact,
    DiscardedForeignAdapter { found: AdapterId, expected: AdapterId },
    DiscardedForeignScope { found: NativeContextScope, expected: NativeContextScope },
    DiscardedInvalidPayload { reason: String },
    ReconstructedNormalized,
}
```

Replay artifacts are capped at 2 MiB. Native context windows are capped at 32
MiB. Both bounds apply on construction and deserialization; fields are private,
access is read-only, and debug output redacts payloads. `NativeContextWindow`
accepts only its current `adapter_id`/`scope`/`payload` serde shape. Scope-less
or legacy shapes are invalid, and `ReplayScope` no longer exists.

`Request::native_context` carries an optional window into ordinary generation.
`CompactionRequest` wraps the complete normalized request so tools, structured
content, media, provider options, replay artifacts, and `ToolApproval` parts are
preserved. `CompactionResult` returns the opaque window plus usage and request/
response metadata. The normalized core never summarizes or interprets a native
window. Adapter validation must reject foreign adapter/scope windows as
`ModelErrorKind::NativeContext`; encoding and decoding use
`ErrorStage::NativeContextEncode` and `NativeContextDecode`.

Replay order for each assistant history entry is complete:

- valid same adapter/scope: `Replayed`;
- no artifact: `NoArtifact`, then `ReconstructedNormalized`;
- foreign adapter: `DiscardedForeignAdapter`, then reconstruction;
- foreign scope: `DiscardedForeignScope`, then reconstruction;
- invalid payload: `DiscardedInvalidPayload`, then reconstruction;
- `ReplayPolicy::Never`: reconstruction without artifact inspection.

Each provider derives a stable safe `ResourceId` from all behavior-affecting
resource data. Sensitive endpoint/header data must be represented by a safe
fingerprint rather than copied into metadata.

Current 0.4 scope inputs:

| Adapter | Resource scope inputs |
|---|---|
| Anthropic Messages | endpoint and Messages surface |
| MiniMax Messages | endpoint and MiniMax Messages surface |
| Claude Platform on AWS | endpoint, region, workspace |
| OpenAI Chat | endpoint and Chat surface |
| OpenAI Responses | endpoint and Responses surface |
| Compatible Chat | endpoint, caller adapter ID, replay-affecting settings |
| Google Gemini API | endpoint and `generateContent` surface |
| Vertex Gemini | endpoint, project, location, typed resource |
| Bedrock Converse | endpoint, region, full model/profile/ARN ID |
| Azure OpenAI | endpoint, route, deployment, surface, revision identity |
| Cohere v2 Chat | full endpoint, v2 Chat surface, capabilities/settings, resolved auth and caller headers |
| Open Responses | full endpoint, standardized HTTP/SSE surface, transport profile/routing, capabilities/settings, resolved auth and caller headers |

## 10. Provider 0.4 migration and compaction status

The 0.4.0 provider migration is complete across the workspace: provider source
uses `NativeContextScope`, supplies the required compaction declaration, handles
`Request::native_context`, and conforms to the object-safe validation/support/
compaction trait surface without compatibility aliases or migration defaults.
OpenAI Responses and Azure OpenAI Responses V1 override `compact` with one
provider-native call and exact scope validation. Chat surfaces and providers
without a standalone native compaction endpoint declare
`CompactionCapability::Unsupported` and reject native compaction before I/O.

| Provider | Sole direct constructor | Removed inference behavior |
|---|---|---|
| Anthropic | `AnthropicModel::new(ModelConfig<AnthropicAuth, AnthropicSettings>)` | Claude name/rule tables |
| MiniMax | `MiniMaxModel::new(ModelConfig<MiniMaxAuth, MiniMaxSettings>)` | M2/M3 profile lookup |
| Claude AWS | `AnthropicAwsModel::new(ModelConfig<AnthropicAwsAuth, AnthropicAwsSettings>)` | name rules and endpoint inference |
| OpenAI Chat | `OpenAiChatModel::new(ModelConfig<OpenAiAuth, OpenAiChatSettings>)` | family/system-role/token-field inference |
| OpenAI Responses | `OpenAiResponsesModel::new(...)` | default-surface and reasoning-name inference |
| Compatible Chat | `OpenAiCompatibleChatModel::new(...)` | compatibility profiles and provider presets |
| Google | `GoogleModel::new(...)` | Gemini generation lookup |
| Vertex | `GoogleVertexModel::new(...)` | generation and hostname/model-resource inference |
| Bedrock | `BedrockModel::new(...)` | built-in IDs and substring reasoning routing |
| Azure Chat | `AzureOpenAiChatModel::new(...)` | runtime surface wrapper and deployment lookup |
| Azure Responses | `AzureOpenAiResponsesModel::new(...)` | runtime surface wrapper and deployment lookup |

New adapters that did not have a pre-0.3 source migration use these sole
construction paths:

| Crate | Sole concrete constructor | Protocol boundary |
|---|---|---|
| `oven-sdk-cohere` | `CohereModel::new(ModelConfig<CohereAuth, CohereSettings>)` | Cohere native `/v2/chat` only |
| `oven-sdk-open-responses` | `OpenResponsesModel::new(ModelConfig<OpenResponsesAuth, OpenResponsesSettings>)` | standardized Open Responses HTTP/SSE only |

`oven-sdk-open-responses` has generic and Hugging Face transport declarations.
Both use an explicit full endpoint and bearer token. The Hugging Face routing
label binds the caller's already-exact model ID into metadata and replay scope;
the adapter never appends provider/policy suffixes, consults a model catalog, or
probes support. Provider-defined option labels remain validated open strings.

Cohere streaming terminates only at `message-end`. Open Responses streaming
requires matching SSE `event`/payload `type`, contiguous sequence numbers,
strict response/item/content lifecycle, a terminal response event, and the
literal `[DONE]`. Both adapters emit the core mandatory `Finish`, classify
in-band versus fatal failures, and capture only their current private bounded
replay format.

Structural protocol choices remain typed settings. Examples include Chat versus
Responses, system role, token field, structured-output shape, reasoning wire
format, Gemini thinking control, Vertex resource kind, Bedrock reasoning format,
and Azure route family.

## 11. Conformance 0.4.0

The public conformance crate provides:

- strict lifecycle and complete/stream equivalence checks;
- explicit capability probes for tools, structured output, sampling, output
  tokens, and reasoning;
- media probes generated from exact modality/MIME/source declarations;
- complete descriptor declaration validation;
- model-ID independence comparison;
- adapter-, scope-, and invalid-payload replay assertions;
- native compaction declaration, cancellation, round-trip, exact-context,
  current-serde-shape, and unsupported-before-I/O assertions;
- mock compaction result queues and captured compaction requests;
- arbitrary SSE byte-chunk fixtures.

Relevant public helpers include:

```rust
assert_declaration_honesty(model)?;
assert_capability_honesty(model)?;
assert_media_honesty(model)?;
assert_model_id_independence(first, second).await?;
assert_replay_artifact(&model.descriptor(), expected_scope, &turn)?;
assert_replay_round_trip(model, expected_scope, round_trip_request).await?;
assert_foreign_replay_is_reported(model, expected_scope, foreign_adapter_request).await?;
assert_foreign_replay_scope_is_reported(model, expected_scope, foreign_scope_request).await?;
assert_invalid_replay_reconstructs(model, expected_scope, invalid_payload_request).await?;
assert_native_context_window(&model.descriptor(), expected_scope, &window)?;
assert_native_compaction(model, expected_scope, compaction_request).await?;
assert_compaction_cancellation(model, compaction_request).await?;
assert_compaction_round_trip(model, expected_scope, compaction_request, continuation).await?;
assert_compaction_unsupported_before_io(model, compaction_request).await?;
```

## 12. Forbidden-inference review

The completed provider migration used the following forbidden-inference search,
which remains a required review check for future provider changes:

```text
model.starts_with(
model.contains(
match model_id.as_str()
official_family
catalog_model
generation(model
for_model(
preset(
Registry
.alias(
std::env::var
```

Matches are allowed only in tests proving names do not affect behavior, or in
application-owned examples that resolve credentials before configuration.
Protocol-required use of the exact model ID in request/resource paths is not
inference.

## 13. Versions, MSRV, and publishing

The exact 10-crate release matrix was published on crates.io on 2026-08-02 UTC
in this order; every listed release is active (not yanked):

| Order | Crate | Release | Published (UTC) | Status | Core dependency | Conformance dev dependency |
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

Edition 2024 and MSRV 1.88 remain normative. Core remains free of Tokio,
reqwest, vendor SDKs, environment lookup, and ambient clients.

Publication completed in dependency order: core 0.4.0, conformance 0.4.0 after
the exact core dependency became visible, then provider crates after both exact
dependencies became visible. Published manifests keep exact version
requirements.

## 14. Non-goals

The SDK does not provide routing, fallback, retries, weighted selection,
sessions, persistence, an agent loop, tool execution, permissions, credential
discovery, pricing, automatic URL downloads, automatic file upload, or non-LLM
generation APIs. Middleware must not reintroduce hidden orchestration or model
selection.
