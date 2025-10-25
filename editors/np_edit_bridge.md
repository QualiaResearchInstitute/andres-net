# NP‑Edit bridge over PAT (pairless editors)

This document describes how to pack/export PAT sequences, train a pairless NP‑Edit editor to emit ΔPAT (+ optional ΔA head hints), and evaluate edits with EditReward. Everything is deterministic and token‑space only.

## 0) Prereqs

- PAT tokens already include global affect headers per frame (arousal, valence) and patch‑local stats, plus a tiny LFQ codepath:
  - File: `src/tokens/pat.ts`
  - Type: `PAToken`:
    - `hdr:{ t:number; scale:32|64; patchId:string }`
    - `geom:{ A:number; phi_gf:number; theta:number; gradPhi:number; kappa:number }`
    - `stats:{ rho:number; energy:number; entropyRate:number; arousal:number; valence:number; attnMean:number; attnVar:number }`
    - `heads?: number[]` (optional per-frame head phases you may attach)
    - `topo:{ vortex:-1|0|1; chirality:-1|0|1 }`
    - `code?: { rvq:number[] }` (single‑stage low‑bit code vector, per‑dim uniform bins)

- Token‑only attention & replay:
  - Toggle “Token‑only (PAT)” in the UI to reconstruct attention fields from tokens via `src/tokens/tokenOnly.ts`.
  - This path is deterministic and can be used to compare “full vs token‑only” outputs.

## 1) Export a dataset (CBOR shards)

We provide a headless exporter that runs the Kuramoto simulation and dumps PAT streams:

- Script: `scripts/export_pat_dataset.ts`
- Build and run:
  - `npm run export:pat -- --out out/pat_shards --seeds 1234,5678 --W 96 --H 96 --frames 32 --scale 32`
- Artifacts:
  - `manifest.json` with grid size, seeds, frames, scale.
  - Shards: `pat_<seed>-<frame>.cbor` (and a small JSON sidecar).
  - Each shard payload:
    ```
    {
      kind: "PAT_SHARD",
      version: 1,
      shard: "1234-0008",
      count: N, // frames in shard
      items: [
        { seed: 1234, frame: 8, tokens: PAToken[] },
        ...
      ]
    }
    ```

Determinism:
- Export runs on CPU stepping only and is deterministic for fixed `W,H,seed,dt,stepsPerFrame`.

## 2) Packing PAT + instructions (pairless)

Goal: train NP‑Edit to map (Prompt text, PAT sequence) → ΔPAT (+ optional ΔA head hints)

Recommended sequence packing:
- For each scene/seed:
  - Text prompt describing the intended edit (e.g., “increase coherence on rim, reduce entropy near Plane #1, flip chirality”).
  - Input sequence: `[{ frame: k, tokens: PAToken[] }]` for K frames before the edit.
  - Output target: ΔPAT on selected future frames (or the entire window) with a compact format.

ΔPAT representation (token‑space deltas):
- For each PAToken, define a sparse delta message:
  ```
  {
    patchId: string, // same as hdr.patchId
    code?: { rvqDelta?: number[] }, // small integer steps in code space (e.g., {-1,0,+1})
    stats?: { rho?: number, entropyRate?: number, ... }, // small floating deltas
    geom?: { theta?: number, gradPhi?: number, ... }     // small floating deltas
  }
  ```
- Keep deltas small and bounded to preserve determinism and avoid drift.

Optional ΔA head hints:
- A tiny vector with suggested per‑head weights/phases:
  ```
  { heads: [{ weight?: number, phase?: number }] }
  ```
- These are optional; the core evaluation should not require them.

Suggested serialization:
- Train/val sets as CBOR or JSONL with:
  ```
  {
    prompt: string,
    inputs: [{ frame: number, tokens: PAToken[] }],
    targets: [{ frame: number, dpat: DPat[] , dA?: { heads?: {weight?:number,phase?:number}[] } }]
  }
  ```

## 3) Deterministic ΔPAT application (replay)

To evaluate an edit without pairs:
1) Load original sequence (tokens).
2) Apply ΔPAT to the target frames in a deterministic order:
   - For each token, locate by `patchId` and apply bounded deltas.
   - Clamp/quantize consistently: code deltas saturate in `[0..bins-1]`, numeric fields clamp to [0,1] or angle wrap as applicable.
3) Reconstruct attention fields (token‑only path), step the sim with fixed `dt`, `seed`, and capture frames.
4) All runs with same ΔPAT and seeds should be identical (CPU↔GPU within tolerance).

Pseudocode:
```
const tokens = loadTokens();
const edit = loadDelta();
for (const dp of edit.dpat) {
  const t = indexByPatchId[tokens][dp.patchId];
  if (!t) continue;
  if (dp.code?.rvqDelta) {
    for (let d=0; d<t.code.rvq.length; d++) {
      t.code.rvq[d] = clamp(t.code.rvq[d] + dp.code.rvqDelta[d], 0, bins-1);
    }
  }
  if (dp.stats?.rho !== undefined) t.stats.rho = clamp01(t.stats.rho + dp.stats.rho);
  // same for entropyRate, etc.
  // angles: wrap to (-π, π] if present in geom
}
rebuildAttentionFromTokens(tokens);
stepSimDeterministic(sim, dt, steps);
```

## 4) EditReward evaluation

We recommend the following protocol:
- Faithfulness: Compare edited rendering vs. a simple reward model over the prompt. At minimum, use deterministic feature‑space proxies (ρ, defect density, entropy rate) in attended regions vs. masked backgrounds.
- Preservation: Compute a mask of non‑edited regions (e.g., outside attended heads or away from specified planes) and penalize drift in those regions.
- Replay determinism: Run two identical decodes from the edited tokens and ensure bit‑identical (or within very small RMSE) outputs.
- Token metrics:
  - PAT recon MSE proxy: Use `rvq` code usage change magnitude as a proxy for edit size.
  - Perplexity shift: `TokenAtlas` perplexity before/after should stay within healthy bands unless the edit demands otherwise.

Minimum reported metrics:
- EditReward score (↑ is better): e.g., weighted sum of faithfulness − λ·preservation drift.
- Determinism RMSE (should be ~0 with noise=0).
- PAT usage and perplexity deltas (from `TokenAtlas`).

## 5) Baseline heuristics (no NN)

Provide a scripts/baselines/ example that:
- Parses prompts for simple intents (“increase coherence on rim”, “lower entropy near X”).
- Produces ΔPAT by nudging:
  - `stats.rho += α`
  - `stats.entropyRate -= β`
  - small `code.rvqDelta` on gradient/curvature dims.
- Evaluates with the same EditReward path.
This creates a non‑learning baseline to beat.

## 6) Repro details

- Deterministic seeds:
  - Use the “Step seed” from the UI, or set `noiseSeed` and schedule seeds explicitly in headless scripts.
- Time stepping:
  - Fix `dt` and `stepsPerFrame`; record them with the ΔPAT package.
- Token scale:
  - Record `scale = 32|64` per frame and keep consistent across training and evaluation.

## 7) Quick start

- Export a dataset:
  - `npm run export:pat -- --out out/pat_shards --seeds 1337 --W 96 --H 96 --frames 64 --scale 32`
- Train NP‑Edit (external):
  - Input sequences: CBOR → tensors
  - Objective: map (prompt, tokens[0..T]) → ΔPAT on frames T..T+K
- Evaluate:
  - Apply ΔPAT with bounded deltas
  - Token‑only replay
  - Compute EditReward + preservation + determinism

## 8) Notes

- Do not modify the underlying lattice parameters for eval; the point is token‑space editing only.
- Angles: use `wrapAngle` when present (e.g., `geom.theta`, `geom.phi_gf`).
- If adding multi‑stage RVQ later, extend `code` with `{ rvq: number[], stage?: number }` while keeping old consumers backward compatible.
