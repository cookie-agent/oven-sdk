# oven-sdk-conformance

Conformance helpers for `oven-sdk` adapter authors. Use this crate in adapter
dev-dependencies to validate lifecycle, replay, capability, and SSE behavior.

## Installation

```bash
cargo add --dev oven-sdk-conformance@0.4.0
```

Version 0.4.0 targets `oven-sdk` 0.4.0 and adds provider-native compaction
declaration, cancellation, round-trip, context-shape, model-ID independence,
and unsupported-before-I/O assertions.

The dependency is pinned exactly to `oven-sdk = "=0.4.0"`. Media fixtures cover
every bounded MIME-pattern/value and source-form combination, including negative
undeclared cases. Replay and native-context assertions require an exact
configured `NativeContextScope`.
Model-ID independence checks are asynchronous, compare normalized
stream/complete results, and accept additional provider-specific probes.
`MockLanguageModel` supplies queued compaction results and captured compaction
requests for deterministic adapter tests.
