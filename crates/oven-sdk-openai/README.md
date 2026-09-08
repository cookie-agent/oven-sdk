# oven-sdk-openai

This source tree provides registry-free adapters for official OpenAI Chat
Completions, official OpenAI Responses, and explicitly configured
OpenAI-compatible Chat and Responses endpoints. Provider-native compaction is
available only through an explicitly declared official Responses surface.

```text
cargo add oven-sdk-openai@0.4.0
```

## Registry-free construction

Construct the protocol and authentication surface explicitly:

- `OpenAiChatModel::new(ModelConfig<OpenAiAuth, OpenAiChatSettings>)`
- `OpenAiChatModel::new_no_auth(ModelConfig<(), OpenAiChatSettings>)`
- `OpenAiResponsesModel::new(ModelConfig<OpenAiAuth, OpenAiResponsesSettings>)`
- `OpenAiResponsesModel::new_compatible(ModelConfig<OpenAiCompatibleAuth, OpenAiResponsesSettings>, AdapterId)`
- `OpenAiCompatibleChatModel::new(ModelConfig<OpenAiCompatibleAuth, OpenAiCompatibleChatSettings>)`

`new_compatible` and `new_no_auth` are source-tree additions, not claims about the published
0.4.0 version in the install command above. Depend on a committed revision
containing them until a release includes them. For `new_compatible`, the caller supplies the adapter ID,
provider/model identity, base endpoint, and capability declarations. Bearer,
header-provider, and no-auth endpoints use the same Responses encoder and SSE
decoder, including declared tool calls and block-aware native replay. The server must
actually implement that protocol; construction does not probe feature support.

`new_no_auth` constructs the official Chat codec directly with unit auth. Its
requests and captured artifacts retain the official Chat adapter identity;
there is no attribution-rewriting wrapper or fake credential.

There is no OpenAI factory, default `model()` surface, registry, model catalog,
compatible baseline, provider preset, endpoint inference, credential discovery,
or model-name rule table. Chat and Responses are separate Rust types. The model
ID is opaque and is used only for the request, descriptor identity, and native-context
scope; prefixes and names never select roles, token fields, sampling behavior,
reasoning, media, limits, or capabilities.

```rust,no_run
use oven_sdk::{
    ApiEndpoint, HeaderConfig, ModelCapabilities, ModelConfig, ModelDeclaration,
    ModelId, ProviderConfig, ProviderId, SecretString,
};
use oven_sdk_openai::{
    MaxTokensField, OpenAiAuth, OpenAiChatModel, OpenAiChatSettings,
    StructuredOutputSupport, SystemMessageRole,
};

let provider = ProviderConfig::new(
    ProviderId::new("openai"),
    ApiEndpoint::parse("https://api.openai.com/v1")?,
    OpenAiAuth::new(SecretString::new("caller-resolved-key")),
    HeaderConfig::empty(),
)?;
let declaration = ModelDeclaration::new(
    ModelId::new("caller-selected-model-id"),
    ModelCapabilities::conservative(),
)?;
let settings = OpenAiChatSettings::new(
    SystemMessageRole::System,
    MaxTokensField::MaxTokens,
    StructuredOutputSupport::Unsupported,
);
let model = OpenAiChatModel::new(ModelConfig::new(provider, declaration, settings))?;
# let _ = model;
# Ok::<(), oven_sdk::ModelError>(())
```

Callers declare every feature, limit, modality, media MIME/source rule,
cancellation mode, and replay policy through `ModelDeclaration`. Structural
wire choices remain typed settings: Chat system role, output-token field,
structured-output shape, visible reasoning field, streamed usage, compatible
adapter ID, query parameters, request-ID headers, and SSE strictness.
`ModelCapabilities.compaction` is also required. Chat and compatible Chat reject
`Native` at construction. Official Responses requires the capability to agree
exactly with `OpenAiResponsesSettings.compaction`: `Unsupported` or the explicit
`OpenAiResponsesCompaction::V1` standalone surface. Compatible Responses requires
both declarations to be unsupported and rejects native compaction at construction.

Authentication and endpoints are explicit. The crate never reads environment
variables. Official OpenAI authentication uses `OpenAiAuth`; compatible
endpoints use `OpenAiCompatibleAuth` for no auth, Bearer auth, or a caller-owned
header provider. Dynamic header/auth providers require a non-secret
`routing_discriminator` in the surface settings to record routing provenance.
It is not an identity gate for standard supported replay blocks. Protected transport/auth headers
cannot be overridden through ordinary provider headers.

## Mapping from models.dev

Applications may use models.dev or another catalog as configuration input, but
this crate has no dependency on, download from, or runtime integration with a
catalog. Map fields deliberately:

| models.dev concept | oven-sdk 0.4 configuration |
|---|---|
| Provider `id` | `ProviderConfig.id` |
| Model `id` | `ModelDeclaration.id` |
| Provider API URL | `ProviderConfig.api` |
| Context/input/output limits | `ModelCapabilities.limits` |
| Input/output modalities | `ModelCapabilities.modalities` |
| Attachment MIME/source support | `ModelCapabilities.media` |
| Tool/reasoning/structured/sampling flags | `ModelCapabilities.features` |
| Replay requirement | `ModelCapabilities.replay` |
| Native compaction support | `ModelCapabilities.compaction` |
| Chat versus Responses API | Concrete model type |

Do not infer adapter settings from the model ID. If a catalog says a model uses
the developer role, `max_completion_tokens`, or restricted sampling, map that
fact explicitly in application configuration and select only requests supported
by the declared capabilities.

## Requests and forward-open labels

Official endpoint options share `provider_options["openai"]`, with nested
`chat` and `responses` members. Compatible extra fields use
`provider_options["openai_compatible"]` on compatible Chat. Compatible Responses
uses the typed `openai.responses` options, not compatible Chat's `extra_body`.
Provider-defined scalar labels such as
reasoning effort/mode, service tier, verbosity, and truncation remain `String`
values and are forwarded unchanged. Compatible `extra_body` is extension-only:
it rejects every normalized or structural request key, including model,
messages, streaming, sampling, tools, token limits, reasoning, and response
format fields. Use the typed options for shared OpenAI fields.

Chat and Responses perform common `Request::validate_for` validation followed
by immutable protocol checks. Chat supports declared image inline/URL input and
inline PDF input. Responses supports declared image and PDF inline/URL input.
Unsupported MIME/source forms fail before dispatch. Assistant-history media is
rejected at request encoding rather than silently omitted.

## Streaming and replay

Both adapters preserve strict lifecycle behavior: one `StreamStart`, ID-scoped
text/reasoning/tool lifecycles, finalized calls exactly once, and one terminal
`Finish` on semantic success. Fatal transport, decode, event, or finalization
errors are typed stream errors. Responses requires a terminal completed or
incomplete event and treats terminal `response.output` as authoritative.

Capture uses the current private formats:

- `oven.openai.chat.assistant.v1`
- `oven.openai.responses.output.v1`

Standard supported Chat and Responses blocks are selected by target codec
support, not source adapter identity or provider/model/header/endpoint scope
equality. The supported OpenAI and Azure formats can exchange their standard
subsets; unknown custom formats are not guessed. Source payload integrity,
ordered content, and normalized semantics must still validate.

Responses `encrypted_content` requires equal known effective wire model IDs and
target reasoning support. Portable message siblings can survive exclusion of
ineligible encrypted reasoning when no required continuation is lost. Ordinary
tool-only history can normalize without an artifact; required native reasoning
or signature state cannot. Missing source identity or legacy missing evidence
does not provide magic recovery. The target can reject eligible crypto even for
the same wire ID, and the SDK never retries an HTTP 400 by stripping reasoning.
See the [core replay policy](../oven-sdk/README.md#native-replay-policy).

## Provider-native Responses compaction

When an official Responses declaration explicitly sets both
`CompactionCapability::Native` and `OpenAiResponsesCompaction::V1`,
`OpenAiResponsesModel` implements the core
`validate_compaction`, `supports_compaction`, and `compact` contract with one
`POST /responses/compact` call. Unsupported declarations fail before I/O.
Chat, compatible Chat, and compatible Responses never expose native compaction.

Compaction translates the complete normalized history using the same strict
Responses replay rules and accepts typed `OpenAiResponsesCompactionOptions` on
`CompactionRequest` for instructions, prompt-cache key, typed prompt-cache
controls, prompt-cache retention, and service tier. It is stateless: the body is
built only from the caller's local normalized/native context.
Provider-defined retention and service labels are bounded open strings and are
forwarded unchanged. The standalone endpoint's canonical user-message plus
terminal-compaction `output` array is retained without pruning, filtering,
reordering, or reinterpretation in the current private
`oven.openai.responses.compaction.v1` native-context payload. A continuation
prepends that exact array to newly encoded input. Canonical retained text,
image, and file content preserves provider-returned
`prompt_cache_breakpoint: {"mode":"explicit"}` exactly after strict shape and
type validation.

Native windows require the exact official Responses adapter and
`NativeContextScope`. Foreign scopes/adapters and malformed, scope-less, or
noncurrent payload shapes fail before dispatch as `NativeContext` errors.
Compaction requests are bounded to 32 MiB and successful response envelopes to
32 MiB plus 64 KiB. Successful responses require JSON, the exact current `response.compaction`
envelope, bounded canonical user messages, consistent usage totals, and a
terminal encrypted compaction item. The private payload binds output with a
SHA-256 fingerprint. Usage, request IDs, HTTP status, response metadata,
provider errors, phase timeouts, and cancellation remain structured.
There are no provider-undocumented item or content-count caps; traversal is
checked and the encoded body/native-context bounds are the safety limits.

Compaction coverage includes declaration honesty, unsupported-before-I/O,
canonical output and breakpoint preservation, large retained windows,
continuation, current-format/fingerprint rejection, usage and metadata, HTTP
errors, body limits, timeout, and cancellation fixtures. To run the ignored
official live round trip:

```text
OPENAI_API_KEY=... OPENAI_COMPACTION_MODEL=gpt-5-mini \
  cargo test -p oven-sdk-openai --test live \
  live_responses_native_compaction_round_trip -- --ignored --exact
```

The API key and model access are caller-provided; the live test performs a real
`/responses/compact` call followed by a `/responses` continuation.

## Timeouts, cancellation, and security

Defaults are 60 seconds for connect, 300 seconds for response headers, and 300
seconds for stream inactivity, with no total stream timeout. Cancellation is
local. Error bodies are bounded and sanitized, while byte counts, request IDs,
retry hints, stages, and provider codes remain structured diagnostics. Secrets,
request bodies, stream chunks, and replay payloads are never logged.

## Differences from the Vercel AI SDK

Compared against Vercel AI SDK commit
[`e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0`](https://github.com/vercel/ai/commit/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0),
especially:

- OpenAI Chat request and stream lifecycle:
  [`openai-chat-language-model.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/openai/src/chat/openai-chat-language-model.ts)
- OpenAI Responses request and stream lifecycle:
  [`openai-responses-language-model.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/openai/src/responses/openai-responses-language-model.ts)
- OpenAI-compatible Chat request and stream lifecycle:
  [`openai-compatible-chat-language-model.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/openai-compatible/src/chat/openai-compatible-chat-language-model.ts)
- Chat message conversion:
  [`convert-to-openai-chat-messages.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/openai/src/chat/convert-to-openai-chat-messages.ts)
- Responses input conversion:
  [`convert-to-openai-responses-input.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/openai/src/responses/convert-to-openai-responses-input.ts)
- OpenAI-compatible Chat message conversion:
  [`convert-to-openai-compatible-chat-messages.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/openai-compatible/src/chat/convert-to-openai-compatible-chat-messages.ts)
- Shared streaming tool-call tracking:
  [`streaming-tool-call-tracker.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/provider-utils/src/streaming-tool-call-tracker.ts)

**Coverage gaps.** Hosted tools may be captured from terminal Responses output,
but the supported Responses replay subset accepts only normalized message, reasoning,
and function-call allow-list. Logprobs, conversations, `store: true`, background
mode, realtime, embeddings, image/speech APIs, automatic capability probes,
catalog integration, and provider presets are out of scope. Vercel's compared
OpenAI provider supports server-managed `context_management` compaction items,
but does not expose oven-sdk's core `CompactionRequest`/`CompactionResult`
contract or a standalone `/responses/compact` model operation.

**Intentional divergences.** oven-sdk requires explicit registry-free
declarations, mandatory terminal `Finish`, structured `ModelError` diagnostics,
scope-aware bounded replay and native context, one provider call without hidden
retries, explicit phase timeouts, caller-owned authentication, and no URL
downloading. Standalone compact output is preserved as opaque canonical context
rather than normalized into ordinary generated content.

**Normalization differences.** Chat tool arguments finalize only at `[DONE]`
or clean EOF with a finish reason. Responses item-done and terminal output are
validated against one another, with terminal output as final authority.
Semantic `call_id` and provider item IDs remain distinct, Chat refusals become
`openai.refusal`, and reasoning summaries/raw text use strict ordered
lifecycles.
