# oven-sdk-google-vertex

Registry-free Google Vertex AI `generateContent` adapter version 0.4.0 for
`oven-sdk` 0.4.0.

```bash
cargo add oven-sdk@0.4.0 oven-sdk-google-vertex@0.4.0
```

## Explicit construction

The only model constructor is:

```rust,ignore
GoogleVertexModel::new(ModelConfig<VertexAuth, GoogleVertexSettings>)
```

The caller supplies every behavior-affecting value:

- `ProviderConfig`: provider ID, versioned API endpoint, caller-supplied access
  token or asynchronous token provider, and headers.
- `ModelDeclaration`: exact model ID, capabilities, token limits, modalities,
  media MIME/source declarations, cancellation, required compaction capability,
  and replay declaration.
- `GoogleVertexSettings`: project, location, typed publisher-model or endpoint
  resource, thinking family, tool behavior, streamed `partialArgs`, media limits
  and URL schemes, exact native-context scope, HTTP client, and phase timeouts.

There is no factory, model registry, model-name catalog, hostname selection, region
routing, generation lookup, or fallback declaration. The model ID is identity only:
it never selects capabilities, wire behavior, resource paths, media support,
thinking controls, tools, partial arguments, or replay policy.

The API endpoint is an explicit versioned base such as
`https://aiplatform.googleapis.com/v1beta1`. Private gateways and tests may provide
another HTTP(S) endpoint. The adapter appends only the configured project,
location, typed resource, and RPC method.

Authentication is also entirely caller-resolved. `VertexAuth` accepts only a
short-lived `AccessToken` or a caller-managed async `TokenProvider`. The crate does
not inspect environment variables, credential files, local SDK configuration, or
cloud metadata services.

Build `GoogleVertexSettings.native_context_scope` with
`google_vertex_native_context_scope`. Its
resource ID is a versioned, non-reversible SHA-256 fingerprint of the canonical
explicit API endpoint, project, location, and typed resource. Equivalent endpoint
URLs produce the same scope; different private gateways do not. Raw endpoint URLs,
headers, and authentication secrets are never serialized into replay artifacts.

## Mapping models.dev data

Applications may load models.dev themselves and map one record into the explicit
0.4 configuration. The crate does not download, cache, embed, or consult
models.dev.

| models.dev/application data | Oven core 0.4 destination |
| --- | --- |
| provider and model IDs | `ProviderConfig.id`, `ModelDeclaration.id` |
| API base selected by the application | `ProviderConfig.api` and `google_vertex_native_context_scope` input |
| context/input/output limits | `ModelCapabilities.limits` |
| tool, reasoning, structured-output, caching, usage, and source flags | `ModelCapabilities.features` |
| input/output modalities and MIME/source support | `ModelCapabilities.modalities` and `.media` |
| provider resource metadata | `GoogleVertexResource` plus project/location |
| generation-specific thinking/tool facts | `GoogleVertexThinkingMode` and `GoogleVertexToolSettings` |
| streamed function-argument support | `stream_function_call_arguments` plus `TOOL_INPUT_DELTAS` |
| resource/deployment identity | `GoogleVertexResource` plus project/location, fingerprinted with the API base by `google_vertex_native_context_scope` |

The application must keep the declaration and settings consistent. Construction
fails when native-context scope identity, provider-tool capability, partial-argument
capability, or thinking/tool settings disagree.
Vertex declares `CompactionCapability::Unsupported`; construction rejects
`CompactionCapability::Native` because this adapter does not implement provider-native
context compaction or accept native context windows.

## Current protocol behavior

- Uses only private replay format
  `oven.google.vertex.generate-content.assistant.v4` under adapter ID
  `oven.google.vertex.generate-content`; no earlier format is decoded.
- Replay is bound by core `NativeContextScope` to provider, model, canonical explicit API
  endpoint, project, location, and typed resource. Provider-supplied function IDs
  participate in semantic replay validation and remain consistent with function
  responses.
- Client functions use `parametersJsonSchema` with an exact object-only root and
  validated local `#/$defs/` references.
- Recognized cached-content resources must match the explicitly configured project
  and location, and require declared `PROMPT_CACHING` support.
- Typed JSONPath `partialArgs` preserve nested objects, arrays, string fragments,
  interleaving, conflict rejection, and composable deltas when explicitly enabled.
- Web, Maps, image, and retrieved-context/RAG grounding normalize to `SourcePart`.
- Media MIME/source support comes from `ModelDeclaration`; adapter-specific count,
  inline-size, HTTPS-count, and URL-scheme limits come from settings. URLs are never
  downloaded.
- Every successful stream ends with exactly one `Finish`. In-band provider failures
  use `Error`, `Finish(Error)`, then EOF, including malformed-function,
  unexpected-tool, and missing-thought-signature finishes.

## Differences from the Vercel AI SDK

Compared against Vercel AI SDK commit
[`e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0`](https://github.com/vercel/ai/commit/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0), especially:

- [`packages/google/src/google-language-model.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/google/src/google-language-model.ts)
- [`packages/google/src/convert-to-google-messages.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/google/src/convert-to-google-messages.ts)
- [`packages/google/src/google-prepare-tools.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/google/src/google-prepare-tools.ts)
- [`packages/google-vertex/src/google-vertex-provider-base.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/google-vertex/src/google-vertex-provider-base.ts)

Vercel provides provider factories, environment-driven routing and credential
discovery, model-specific logic, and broader URL handling. Oven intentionally
requires caller-resolved configuration and OAuth tokens, does no registry,
environment, filesystem, or cloud-metadata lookup, accepts only declared media and
URL schemes, never downloads URLs, emits strict lifecycle parts and structured
errors, and captures bounded endpoint-scoped replay artifacts. Provider-executed
tools are normalized as safe custom parts; citations become sources rather than
executable client calls.
