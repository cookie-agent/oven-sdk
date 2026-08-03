# oven-sdk-cohere

Registry-free Cohere native v2 Chat adapter for `oven-sdk` 0.4.0. Construction
uses only `CohereModel::new(ModelConfig<CohereAuth, CohereSettings>)`. The
caller supplies the complete `/v2/chat` endpoint, bearer token, headers, exact
model ID, limits, capabilities, modalities, media rules, replay/cancellation
declarations, and wire settings. This crate has no registry, presets,
environment lookup, model-name inference, probes, or models.dev dependency.

The production adapter never reads process environment variables. The ignored
live test reads `COHERE_API_KEY`, `COHERE_MODEL`, and `COHERE_ENDPOINT` in test
code only.

## Protocol sources

- [Cohere v2 Chat reference](https://docs.cohere.com/reference/chat)
- [Cohere v2 streaming reference](https://docs.cohere.com/reference/chat-stream)
- [Cohere streaming guide](https://docs.cohere.com/v2/docs/streaming)
- [Cohere tool-use guide](https://docs.cohere.com/v2/docs/tool-use-overview)
- [Cohere structured outputs](https://docs.cohere.com/v2/docs/structured-outputs)
- [Cohere image-input rules](https://docs.cohere.com/docs/image-inputs)
- models.dev terminology reference only, commit
  [`c3057690bbb8bd41cafdefadcd2a7b958e2a4642`](https://github.com/anomalyco/models.dev/commit/c3057690bbb8bd41cafdefadcd2a7b958e2a4642)

## Differences from the Vercel AI SDK

Compared against Vercel AI SDK commit
[`e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0`](https://github.com/vercel/ai/tree/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0),
primarily
[`cohere-chat-language-model.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/cohere/src/cohere-chat-language-model.ts),
[`convert-to-cohere-chat-prompt.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/cohere/src/convert-to-cohere-chat-prompt.ts),
and
[`cohere-prepare-tools.ts`](https://github.com/vercel/ai/blob/e84b8bc8154030cdb7469b0e0b8cd8b9354f19a0/packages/cohere/src/cohere-prepare-tools.ts).

- **Coverage gaps:** no V1 API, model catalog, embedding/rerank APIs, automatic
  URL download, logprobs normalization, or SDK serialization hooks. Documents,
  citation mode, priority, stop sequences, penalties, seed, top-k, image detail,
  strict tools, structured output, thinking, tool plan, parallel calls, and
  citations are covered.
- **Intentional divergences:** endpoint and credentials are mandatory; model
  IDs never select behavior; stream EOF before `message-end` is an error;
  provider failures use structured `ModelError`; every successful stream ends
  in exactly one `Finish`; cancellation is explicitly local-only; and native
  assistant-message replay is bounded and bound to an exact
  `NativeContextScope`. Cohere declares `CompactionCapability::Unsupported`,
  rejects native-context requests before network I/O, and rejects model
  declarations that claim native compaction. Normalized
  `reasoning_effort` values are accepted only through caller-configured
  effort-to-thinking mappings; Cohere receives only `enabled` or `disabled`
  thinking types. Schema-constrained output and strict tools reject every JSON
  Schema keyword outside Cohere's documented subset before dispatch.
- **Normalization differences:** thinking and tool-plan text are distinct
  reasoning blocks identified by metadata; citation events become `Source`
  parts; tool calls are emitted only after the indexed argument stream closes;
  inline assistant tool results become subsequent Cohere tool messages; and
  Cohere `message-end` directly determines finish and usage rather than a
  transform flush synthesizing success after arbitrary EOF. When reasoning
  replay is not declared, thinking blocks remain normalized output but are
  omitted from native capture and exact native replay.

Replay resource IDs are versioned SHA-256 fingerprints over the exact endpoint,
provider/model identity, capabilities, all Cohere settings including transport
timeouts and reasoning-effort mappings, and resolved caller
headers/authentication. Raw tokens and header values are never exposed in
descriptors, resource IDs, diagnostics, or debug output. Optional replay
capture discards an oversized artifact with a safe provider decision while
preserving `Finish`; required replay capture fails closed.
