# AGENTS.md

## ARCHITECTURE.md is the source of truth

`ARCHITECTURE.md` is the authoritative design record for this project.

**Rule:** if you change anything that alters **any architecture decision
recorded in `ARCHITECTURE.md`**, you **must** update `ARCHITECTURE.md` in the
same commit so the doc never drifts from the implementation. This includes,
without limitation: the `LanguageModel` trait, stream lifecycle, content
model, replay semantics, error taxonomy, capability model, crate boundaries,
adapter tiers, licensing, CI/publishing policy, timeout/cancellation policy,
and migration plans.

Minor implementation details that do not drift from the documented plan
(helper functions, internal refactors within a crate, performance tweaks,
bug fixes that restore documented behavior) do **not** require a doc update.

When in doubt whether a change is architectural: it is. Update the doc.

## Zero warnings

All warnings are addressed, not ignored: `cargo build`, `cargo clippy`, and
`cargo test` across the workspace must produce **zero** warnings (rustc,
clippy, and macro-expansion warnings alike). Do not silence warnings with
`#[allow]` unless the warning is a proven false positive and the allowance
is narrowly scoped with a comment explaining why. If a dependency upgrade or
new code introduces a warning, fix it in the same change.

## Provider crates: document the Vercel AI SDK delta

Provider crates (`oven-sdk-anthropic`, `oven-sdk-openai`, and any future
adapter) are implemented against the provider's **official API
documentation**, with the **Vercel AI SDK** (`github.com/vercel/ai`,
provider packages under `packages/`) as the reference for battle-tested
normalization decisions. The cookie_agent MVP providers crate is **not** a
protocol reference — its only authoritative content is the legacy
`TurnOpaque` journal shapes that tolerant decoders must accept.

**Rule:** whenever a provider crate is **created or modified**, its README
must contain a section (e.g. `## Differences from the Vercel AI SDK`)
documenting how the implementation differs from the corresponding Vercel AI
SDK provider package. Cover at minimum:

- **Coverage gaps:** Vercel features/options not yet implemented (and why:
out of scope, deferred, or intentionally unsupported).
- **Intentional divergences:** places where oven-sdk deliberately behaves
  differently (e.g. mandatory `Finish` terminal, structured `ModelError`
  taxonomy, native replay artifact capture, capability bitset, no
  auto-download of URLs, timeout policy).
- **Normalization differences:** any part-type mapping, finish-reason
  mapping, or stream-part lifecycle choice that does not match Vercel's
  behavior, with a pointer to the Vercel source file that informed it.

Keep the section current in the same commit as the code change. When
consulting the Vercel repo, record the commit SHA you compared against.
