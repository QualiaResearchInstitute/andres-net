import { describe, it, expect } from "vitest";
import { applyAnnealModifiers } from "./annealing";

/**
 * Deterministic surrogate for rho(T) given anneal-modified K(T), D(T).
 * Idea: rho increases when coupling weakens slightly (less frustration) and noise rises a bit (helps shake out knots),
 * up to a point. We keep it simple and monotone here to test T90 timing differences between forward and reverse ramps.
 */
function rhoSurrogate(K: number, D: number, a = 0.25, b = 0.15) {
  const r = 0.5 + (-a * K) + (b * D);
  return Math.max(0, Math.min(1, r));
}

/** Compute T90 index for a ramp sequence based on first time rho crosses 90% of its span.
 * Handles both increasing and decreasing sequences deterministically.
 */
function t90Index(rhos: number[]): number {
  if (rhos.length === 0) return -1;
  const r0 = rhos[0];
  const r1 = rhos[rhos.length - 1];
  const span = r1 - r0;
  if (Math.abs(span) < 1e-6) return -1;
  const thr = r0 + 0.9 * span;
  // Use a tiny epsilon to make forward/reverse crossings symmetric under float rounding
  const eps = Math.max(1e-12, Math.abs(span) * 1e-12);
  if (span >= 0) {
    for (let i = 0; i < rhos.length; i++) {
      if (rhos[i] >= thr - eps) return i;
    }
  } else {
    for (let i = 0; i < rhos.length; i++) {
      if (rhos[i] <= thr + eps) return i;
    }
  }
  return -1;
}

describe("Forward vs Reverse anneal: T90-ρ improvement", () => {
  it("Forward and reverse T ramps reach 90% of rho-surrogate at the same index for linear schedules", () => {
    // Base parameters and anneal gains aligned with annealing.ts semantics
    const baseK = 1.0;
    const baseD = 0.05;
    const gains = { k: 0.5, d: 0.6 };

    // Deterministic schedule of T in [0,1]
    const steps = 21;
    const Tfw: number[] = [];
    for (let i = 0; i < steps; i++) Tfw.push(i / (steps - 1));
    const Trev = [...Tfw].reverse();

    // Build forward and reverse rho tracks using anneal modifiers
    const rhoFwd: number[] = [];
    const rhoRev: number[] = [];
    for (let i = 0; i < steps; i++) {
      const { K: Kf, D: Df } = applyAnnealModifiers(baseK, baseD, Tfw[i], gains);
      rhoFwd.push(rhoSurrogate(Kf, Df));
      const { K: Kr, D: Dr } = applyAnnealModifiers(baseK, baseD, Trev[i], gains);
      rhoRev.push(rhoSurrogate(Kr, Dr));
    }

    // Sanity: forward ramp increases rho; reverse ramp decreases rho under this surrogate
    for (let i = 1; i < steps; i++) {
      expect(rhoFwd[i]).toBeGreaterThanOrEqual(rhoFwd[i - 1]);
      expect(rhoRev[i]).toBeLessThanOrEqual(rhoRev[i - 1]);
    }

    // Compute T90 indices
    const i90F = t90Index(rhoFwd);
    const i90R = t90Index(rhoRev);

    // Both should exist
    expect(i90F).toBeGreaterThanOrEqual(0);
    expect(i90R).toBeGreaterThanOrEqual(0);

    // With linear K(T), D(T) and a linear rho surrogate, both forward-increasing and reverse-decreasing tracks
    // cross their own 90% span threshold at the same relative position in index space.
    expect(i90F).toBe(i90R);
  });
});
