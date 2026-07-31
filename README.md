# oven-sdk

oven-sdk is a runtime-neutral Rust SDK for language-model providers: it owns normalized model contracts, typed streaming, cancellation, capabilities, structured errors, and opaque native replay artifacts while leaving persistence, fallback policy, permissions, and the agent loop to the calling harness. An artifact a provider cannot decode is discarded and normalized content is reconstructed exactly as for a provider swap. It is early-stage and its public API is still being established.
