# Kuramoto Painter – Multi‑level, token‑centric architecture

Why multi‑level?
Our system treats high‑level, efficiently computable objects—Phase–Amplitude Tokens (PAT), the ReflectorGraph, and affect headers—as part of the canonical state, not mere logs. Motivated by “homomorphically encrypted consciousness” style arguments, we design for representation/virtualization invariance: under gauge shifts, torus translations, or reflections (and eventually arbitrary obfuscations that preserve information), mind‑relevant semantics should remain available efficiently at the right level. In practice, extracting semantics directly from lattice microstate can be computationally hard, while tokens and graph provide an efficient bridge from physics to meaning. This is why we promote tokens/graphs/affect to first‑class, deterministic, replayable state.

What this buys us
- Deterministic, representation‑invariant editing and evaluation: high‑level distributions (e.g., PAT code histograms) and affect are stable under substrate‑preserving transforms.
- Token‑only (opaque substrate) mode: downstream modulators operate on PAT/affect/graph without reading raw phases, enabling privacy‑preserving operation while maintaining functional steering with bounded divergence.
- Clean theory‑to‑practice hooks for attention/orientation/valence and “virtualized scene” export, so we can share the “experience stream” (tokens/affect/graph snapshots) without raw phases. In short, we treat the computationally relevant parts of the mind‑like field as first‑class citizens.

New capabilities added
- Substrate invariance tests: src/kuramoto/substrate_invariance.test.ts verifies that gauge shifts, torus translations, and reflections keep PAT (minus gauge bin) and affect invariant within tight thresholds.
- Token‑only (opaque substrate) mode: wire flag and proxy attention fields from tokens.
  - Config: KuramotoEngineConfig includes opaqueSubstrateMode?: boolean and patScale?: 32 | 64
  - When enabled, internal modulators receive {Aact, Uact, lapA, divU} derived from PAT via src/tokens/tokenOnly.ts
- Epistemic‑gap benchmark: bench/epistemic_gap.ts trains tiny logistic probes to compare token vs lattice feature efficiency.
- Virtualization helpers and export:
  - src/kuramoto/virtualization.ts: gauge/translate/reflect transforms + invariance summaries
  - src/kuramoto/sceneVirtualize.ts: exportVirtualizedScene to store only tokens/affect/(optional)ReflectorGraph nodes; replayHighLevel to inspect streams without raw phases

How to run tests
- npm run test:run
  - substrate invariance: src/kuramoto/substrate_invariance.test.ts
  - token‑only approximation: src/kuramoto/token_only_mode.test.ts

Notes
- PAT invariance compares code histograms with the gauge‑sensitive bin dropped and uses strict affect thresholds (≈1e‑4) for FP stability.
- Token‑only proxy targets functional similarity, not bit‑equality; tests use RMSE bounds.
- The benchmark script can be executed with a TS runner (e.g., ts-node) if available, or compiled and run with node.
