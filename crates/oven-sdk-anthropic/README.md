# oven-sdk-anthropic

Tokio/reqwest Messages adapters for direct Anthropic, caller-selected
Anthropic-compatible providers, MiniMax, and Claude Platform on AWS.

Crate version **0.5.0** targets `oven-sdk` **0.4.0** and uses its registry-free
`ModelConfig` API. The crate has no provider factory, model registry, model
catalog, environment lookup, or model-name inference.

## Installation

```bash
cargo add oven-sdk@0.4.0 oven-sdk-anthropic@0.5.0 reqwest@0.12
```

## Explicit model construction

Each concrete type has exactly one constructor:

- `AnthropicModel::new(ModelConfig<AnthropicAuth, AnthropicSettings>)`
- `AnthropicCompatibleModel::new(ModelConfig<AnthropicCompatibleAuth, AnthropicCompatibleSettings>)`
- `MiniMaxModel::new(ModelConfig<MiniMaxAuth, MiniMaxSettings>)`
- `AnthropicAwsModel::new(ModelConfig<AnthropicAwsAuth, AnthropicAwsSettings>)`

The caller supplies every behavior-affecting value:

- provider ID and validated endpoint;
- resolved authentication and static/dynamic headers;
- exact provider model, deployment, or resource ID;
- feature flags, token limits, modalities, media MIME/source rules,
  cancellation, compaction, and replay declarations;
- HTTP client, phase timeouts, native-context identity, and protocol settings;
- AWS region/workspace and authentication mode for Claude Platform on AWS.

Model IDs are sent on the wire and included in native-context scope, but **never select
capabilities, limits, thinking behavior, sampling behavior, media support, or
any other adapter branch**. Future IDs work without a crate release when the
caller provides the correct declaration.

```rust,no_run
use oven_sdk::{
    ApiEndpoint, Capability, CancellationCapability, CompactionCapability,
    HeaderConfig, Modalities, Modality, ModelCapabilities, ModelConfig,
    ModelDeclaration, ModelId, ModelLimits, ProviderConfig, ProviderId,
    ReplayCapability, ReplayDeclaration, ReplayPolicy, ResourceId, SecretString,
};
use oven_sdk_anthropic::{
    AnthropicAuth, AnthropicModel, AnthropicProtocolSettings, AnthropicSettings,
    AnthropicThinkingSupport, AnthropicTimeouts,
};

let capabilities = ModelCapabilities {
    features: Capability::TOOL_CALLING
        | Capability::PARALLEL_TOOLS
        | Capability::TOOL_INPUT_DELTAS
        | Capability::REASONING
        | Capability::TEMPERATURE
        | Capability::TOP_P
        | Capability::MAX_OUTPUT_TOKENS
        | Capability::USAGE,
    limits: ModelLimits::new(Some(200_000), None, Some(64_000)),
    modalities: Modalities::new([Modality::text()], [Modality::text()]),
    media: Default::default(),
    cancellation: CancellationCapability::LocalOnly,
    compaction: CompactionCapability::Unsupported,
    replay: ReplayDeclaration {
        policy: ReplayPolicy::IfValid,
        capability: ReplayCapability::Required,
        reasoning: true,
    },
};

let model = AnthropicModel::new(ModelConfig::new(
    ProviderConfig::new(
        ProviderId::new("anthropic"),
        ApiEndpoint::parse("https://api.anthropic.com/v1")?,
        AnthropicAuth::ApiKey(SecretString::new("resolved-api-key")),
        HeaderConfig::empty(),
    )?,
    ModelDeclaration::new(ModelId::new("caller-selected-model-id"), capabilities)?,
    AnthropicSettings {
        client: reqwest::Client::new(),
        timeouts: AnthropicTimeouts::default(),
        protocol: AnthropicProtocolSettings {
            thinking: AnthropicThinkingSupport::Extended,
            thinking_default_active: false,
            thinking_disable_allowed: true,
            thinking_disable_forbidden_efforts: Default::default(),
            effort: true,
            assistant_prefill: true,
            reject_non_default_sampling: false,
        },
        native_context_discriminator: Some(ResourceId::new("production-tenant")?),
    },
))?;
# let _ = model;
# Ok::<(), oven_sdk::ModelError>(())
```

Declare media through `ModelCapabilities::modalities` and
`ModelCapabilities::media`. Core validates the declared modality, exact MIME
pattern, and source form before the adapter applies protocol-specific size and
wire-shape checks.

## Direct Anthropic

Direct models require provider ID `anthropic` and use adapter ID
`oven.anthropic.messages`. The configured endpoint must include the desired API
prefix; the adapter appends `/messages`. `AnthropicAuth::ApiKey` injects
`x-api-key` unless caller headers already provide `x-api-key` or
`Authorization`. `AnthropicAuth::None` performs no auth injection.

The adapter preserves Messages SSE, client tools, structured output, prompt
cache markers, JPEG/PNG/GIF/WebP image forms, PDF/plain-text document forms,
thinking, open-string effort, checked token arithmetic, request-size limits,
structured errors, phase timeouts, and strict stream lifecycle handling.
These features are accepted only when the explicit declaration and protocol
settings permit them.

Construction enforces an immutable protocol ceiling: declarations may remove
features, modalities, MIME types, and source forms, but cannot claim behavior
outside the direct Anthropic wire implementation. Base endpoints containing a
query are rejected, and `/messages` is appended with URL path APIs. The caller's
`reqwest::Client` owns connection timeout policy; `AnthropicTimeouts` contains
only adapter-enforced header, credential-provider, and stream-idle timeouts.

`AnthropicProtocolSettings` explicitly declares manual/adaptive thinking
support, default-active thinking, disabled-mode support, open effort labels
forbidden while disabled, general effort support, assistant prefill, and the
sampling rule. Request `thinking.display` and `effort` remain open strings and
are forwarded unchanged after structural validation.

## Generic Anthropic Messages compatibility

`AnthropicCompatibleModel` accepts any validated caller-selected provider ID
and explicit base endpoint, then appends `/messages` with the same URL semantics
as `AnthropicModel`. Its caller-owned `adapter_id` must be valid and cannot use
the reserved `oven.anthropic.messages` or `oven.minimax.messages` identities.

The compatible model uses the direct Anthropic Messages request, SSE, tool,
error, beta-header, version-header, capability-ceiling, and protocol-settings
implementation without provider-name or model-name inference. Authentication is
explicitly one of `ApiKey`, `Bearer`, or `None`; the first two inject
`x-api-key` or `Authorization: Bearer ...` respectively unless caller headers
already authenticate. Capabilities remain entirely caller-declared.

## MiniMax Messages compatibility

MiniMax models require provider ID `minimax` and use adapter ID
`oven.minimax.messages`. The commonly documented endpoint is
`https://api.minimax.io/anthropic/v1`; callers provide it explicitly.
`MiniMaxAuth::Bearer` injects `Authorization: Bearer ...`, while
`MiniMaxAuth::None` leaves authentication to caller headers.

`MiniMaxProtocolSettings` explicitly controls whether thinking is supported
and whether it can be disabled. Image/video support is determined solely by
the caller's modality and media declaration, not by a MiniMax model name.
MiniMax's automatic/disabled tool-choice restrictions, temperature range,
media wire forms, open-string service tier/detail, provider file references,
10 MiB inline-image limit, 50 MiB inline-video limit, and 64 MiB serialized
request limit remain enforced.

MiniMax parses only the `minimax` request namespace. Anthropic and AWS request
options cannot alter its request behavior.

## Claude Platform on AWS

Claude Platform on AWS models require provider ID `anthropic-aws` and use
adapter ID `oven.anthropic.aws.messages`. This adapter is **not** Bedrock
Converse or InvokeModel.

The caller explicitly supplies the regional endpoint, region, workspace ID,
and one `AnthropicAwsAuth` mode:

- `BearerKey(SecretString)` for the platform `x-api-key` flow;
- `StaticCredentials(AnthropicAwsCredentials)` for AWS SigV4;
- `CredentialProvider(AnthropicAwsCredentialProvider)` for asynchronous,
  caller-managed credential resolution once per request.

Every request includes `anthropic-workspace-id`. SigV4 uses service
`aws-external-anthropic`, signs the exact serialized bytes that are sent,
supports session tokens, and applies static/dynamic caller headers before
signing. Caller `Authorization`, `x-api-key`, workspace, host, and
SigV4-controlled headers are rejected rather than overwritten.
AWS secret access keys and optional session tokens use core `SecretString`;
access key IDs remain non-secret strings.

## Native replay

Replay uses only current scope-aware private formats:

- direct Anthropic: `oven.anthropic.messages.assistant.v3`
- compatible Anthropic Messages: `oven.anthropic.messages.assistant.v3`, scoped to the caller adapter ID
- MiniMax: `oven.minimax.messages.assistant.v3`
- Claude Platform on AWS: `oven.anthropic.aws.messages.assistant.v3`

Each `NativeReplayArtifact` carries adapter identity plus a `NativeContextScope` made
from the configured provider ID, exact model/resource ID, and an internally
derived versioned SHA-256 `ResourceId`. Direct Anthropic and MiniMax hash the
canonical endpoint and concrete Messages surface. Claude Platform on AWS also
hashes region and workspace. An optional caller `native_context_discriminator` is
combined into the hash but is never trusted as the complete resource identity.
Endpoints, workspaces, discriminators, headers, and secrets are never copied
into replay metadata. Foreign adapters and foreign scopes are reported
separately and fall back to normalized reconstruction according to the declared
replay policy. There is no legacy replay decoder.

Signed/redacted provider reasoning is replayed only from a matching,
semantically valid current artifact. Normalized reconstruction never invents
provider-authoritative reasoning state. Replay payloads remain bounded and
omit model identity because identity now belongs to the core native-context scope.

## Provider-native compaction

Direct and compatible Anthropic Messages, MiniMax Messages compatibility, and
Claude Platform on AWS expose no provider-native compaction endpoint. All constructors
require `CompactionCapability::Unsupported` and reject `Native` declarations
before creating a model. The concrete models do not override core compaction
methods: `validate_compaction`, `supports_compaction`, and `compact` therefore
inherit core's default unsupported behavior and perform no provider I/O.

## Streaming, errors, and cancellation

All four concrete models share the incremental SSE parser and strict state
machine. Successful streams begin with `StreamStart` and end with exactly one
terminal `Finish`. Content block starts must be contiguous from zero; indices
cannot be duplicated or reused after finalization. Deltas and stops may
interleave, while normalized content and replay content retain validated start
order without null placeholders.

Malformed or unsigned finalized reasoning, invalid tool JSON, usage overflow,
terminal ordering violations, and semantic events after `message_stop` fail
before successful `Finish` or replay capture. HTTP diagnostics retain bounded,
selected safe fields. Cancellation stops local initiation or reading but does
not claim remote cancellation or billing termination.

## Live tests

Live tests are ignored and environment-gated:

```text
cargo test -p oven-sdk-anthropic --test live -- --ignored
```

- direct Anthropic: `ANTHROPIC_API_KEY`
- MiniMax: `MINIMAX_API_KEY`
- Claude Platform on AWS: `AWS_REGION`, `ANTHROPIC_AWS_WORKSPACE_ID`, and either
  `ANTHROPIC_AWS_API_KEY` or static AWS credential environment variables

## Differences from the Vercel AI SDK

Compared against Vercel AI SDK commit
`e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0`:

- Anthropic request/thinking/stream normalization:
  [`anthropic-language-model.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/anthropic/src/anthropic-language-model.ts)
- Anthropic prompt/media/reasoning conversion:
  [`convert-to-anthropic-prompt.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/anthropic/src/convert-to-anthropic-prompt.ts)
- Claude Platform on AWS transport:
  [`anthropic-aws-fetch.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/anthropic-aws/src/anthropic-aws-fetch.ts)
- Vercel's separate MiniMax provider:
  [`minimax-provider.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/minimax/src/minimax-provider.ts)

**Coverage gaps.** Vercel exposes additional hosted tools, Files/Skills,
citations, compaction/context management, count-token helpers, default AWS
credential-chain integration, and MiniMax video generation. They remain out of
scope. In particular, this crate has no compaction endpoint and rejects native
compaction declarations. Bedrock protocols are separate and unsupported.

**Intentional divergences.** oven-sdk requires explicit registry-free model
declarations instead of provider factories or model catalogs. It also requires
a terminal `Finish`, structured `ModelError`, explicit media/replay contracts,
    `NativeContextScope`-aware bounded native replay, no hidden retries, no URL downloads, and
separate credential/header/idle timeouts. MiniMax uses its documented bearer
header rather than Vercel's `x-api-key` choice. Provider-defined scalar labels
remain open strings. Generic Anthropic compatibility requires a caller-owned
provider ID, endpoint, adapter ID, authentication mode, capabilities, and
protocol settings rather than Vercel provider-package presets.

**Normalization differences.** Tool JSON finalizes only at
`content_block_stop`; malformed or non-object input is fatal. Thinking
signatures remain replay-only. Usage and terminal metadata become authoritative
at `Finish`; refusal maps to `FinishReason::Refused`; unknown stop labels remain
`FinishReason::Other(String)`. Lifecycle validation is stricter than Vercel's:
starts are contiguous, finalized indices cannot be reused, and post-terminal
semantic events are rejected.
