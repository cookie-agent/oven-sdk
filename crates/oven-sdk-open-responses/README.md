# oven-sdk-open-responses

Version 0.2.0 is a registry-free client for the standardized Open Responses
HTTP/SSE protocol on `oven-sdk` 0.4.0, with generic bearer transport and an
explicit Hugging Face profile. Construct models only with
`OpenResponsesModel::new(ModelConfig<OpenResponsesAuth, OpenResponsesSettings>)`.
The caller supplies the complete `/v1/responses` endpoint, resolved bearer
token, headers, exact routed model ID, limits, capabilities, modalities, media,
replay/cancellation/compaction declarations, transport profile, and settings.
Both transports require `CompactionCapability::Unsupported`: standardized Open
Responses and the Hugging Face router do not inherit OpenAI's native compact
endpoint or native-context continuation format.

There is no model registry, catalog, preset, environment lookup, name
inference, automatic probe, runtime models.dev access, OpenAI-only hosted-tool
assumption, or implicit Hugging Face routing. For Hugging Face, the exact model
ID sent on the wire must already contain any desired `:<provider>` or routing
policy suffix; `HuggingFaceTransport.routing` explicitly binds that caller
choice into descriptor metadata and native-context replay scope without
rewriting the ID.

The production crate never reads environment variables. Ignored live tests use
`OPEN_RESPONSES_TOKEN`, `OPEN_RESPONSES_MODEL`, `OPEN_RESPONSES_ENDPOINT`, and
`OPEN_RESPONSES_PROFILE` in test code only.

## Protocol sources

- [Open Responses 2026-04-24 specification](https://www.openresponses.org/specification/2026-04-24)
- [Open Responses 2026-04-24 reference and OpenAPI](https://www.openresponses.org/reference/2026-04-24)
- [Open Responses 2026-01-15 reference](https://www.openresponses.org/reference/2026-01-15)
- [Open Responses acceptance tests](https://www.openresponses.org/compliance)
- Hugging Face Responses API guide at pinned `hub-docs` commit
  [`b05db9f4f79a767f4d1f74c315a1fb8e0b9e2b36`](https://github.com/huggingface/hub-docs/blob/b05db9f4f79a767f4d1f74c315a1fb8e0b9e2b36/docs/inference-providers/guides/responses-api.md),
  including its Remote MCP and server-side tool coverage
- models.dev terminology reference only, commit
  [`c3057690bbb8bd41cafdefadcd2a7b958e2a4642`](https://github.com/anomalyco/models.dev/tree/c3057690bbb8bd41cafdefadcd2a7b958e2a4642)

## Differences from the Vercel AI SDK

Compared against Vercel AI SDK commit
[`e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0`](https://github.com/vercel/ai/tree/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0),
including
[`openai-responses-language-model.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/openai/src/responses/openai-responses-language-model.ts),
[`convert-to-openai-responses-input.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/openai/src/responses/convert-to-openai-responses-input.ts),
[`huggingface-provider.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/huggingface/src/huggingface-provider.ts),
and
[`huggingface-responses-language-model.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/huggingface/src/responses/huggingface-responses-language-model.ts).

- **Coverage gaps:** HTTP/SSE only; no WebSocket or native compaction endpoint,
  background mode, previous-response persistence, OpenAI hosted tools,
  Hugging Face Remote MCP tools or other provider-hosted/server-side tools,
  provider file IDs, logprobs, or automatic provider routing. The unsupported
  Hugging Face tool surface is documented by the pinned `hub-docs` source
  above. Standard message, reasoning/summary, refusal, function-call/output,
  structured-output, image/PDF input, URL-citation, assistant-phase, usage,
  error, and replay flows are covered. Unsupported normalized content is
  rejected before dispatch.
- **Intentional divergences:** the Open Responses specification—not OpenAI
  model tables—is authoritative; endpoint/auth/profile are mandatory; every
  event and item lifecycle is checked; `event:` must equal payload `type`;
  sequence numbers are contiguous; `[DONE]` is mandatory after a terminal
  response event; failures use structured `ModelError`; cancellation is
  explicitly local-only; and `CompactionCapability::Native` declarations are
  rejected rather than borrowing OpenAI-specific `/responses/compact`
  behavior.
- **Normalization differences:** standard text/raw-reasoning blocks use the
  content-part lifecycle; reasoning summaries use
  `response.reasoning_summary_part.*` plus
  `response.reasoning_summary_text.*`; function calls require the authoritative
  arguments-done event before item completion; URL annotations become ordered
  `Source` parts; assistant `phase` is retained in part metadata; unknown
  unprefixed events are rejected; and native replay captures only the current
  private `open.responses.items.v2` format, tied to an exact cryptographic
  `NativeContextScope`. The explicitly selected
  `HuggingFace` transport additionally accepts the router's pinned legacy
  `response.reasoning_text.delta/done` events and `reasoning_summary` content;
  the generic transport never accepts those variants.

Replay resource IDs use the current
`open.responses.native_context_scope.v2` namespace and are versioned SHA-256
fingerprints over the full endpoint, provider/model identity, declared
capabilities, transport/profile/routing, wire settings, resolved
authentication, and caller headers. Raw secrets never appear in descriptors,
resource IDs, diagnostics, or debug output. Legacy v1 private replay payloads
are not decoded.
