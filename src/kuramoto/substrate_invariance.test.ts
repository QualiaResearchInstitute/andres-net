import { describe, it, expect } from "vitest";
import { createSimulation, type SimulationInitConfig } from "./simulation";
import { buildPATokens, type PAToken } from "../tokens/pat";
import { computeAffect } from "./affect";
import {
  applyGaugeShift,
  translateSnapshot,
  reflectSnapshot,
  patRVQHistogram,
  summarizeAffect,
} from "./virtualization";

/**
 * Helpers
 */

function makeInitConfig(W = 24, H = 24, reseedGraphKey = 777): SimulationInitConfig {
  return {
    W,
    H,
    wrap: true,
    omegaSpread: 0.25,
    swProb: 0.04,
    swEdgesPerNode: 2,
    swMinDist: 6,
    swMaxDist: 20,
    swNegFrac: 0.2,
    energyBaseline: 0.35,
    reseedGraphKey,
  };
}

function rvqHistogramDropPhi(tokens: PAToken[]) {
  // PAT.rvq index 1 corresponds to phi_gf angle bin in current definition; drop it
  return patRVQHistogram(tokens, [1]);
}

function affectClose(a: { arousal: number; valence: number }, b: { arousal: number; valence: number }, eps = 1e-4) {
  const da = Math.abs(a.arousal - b.arousal);
  const dv = Math.abs(a.valence - b.valence);
  return da <= eps && dv <= eps;
}

describe("Substrate invariance (representation-preserving transforms)", () => {
  it("Gauge shift leaves PAT code distribution (w/o phi bin) and affect invariant", () => {
    const cfg = makeInitConfig(32, 32, 1337);
    const sim = createSimulation(null, cfg).simulation;

    // Baseline high-level summaries
    const tokA = buildPATokens(sim, 32);
    const affA = summarizeAffect(computeAffect(sim));
    const histA = rvqHistogramDropPhi(tokA);

    // Apply a global gauge shift φ0
    applyGaugeShift(sim, 1.2345);

    const tokB = buildPATokens(sim, 32);
    const affB = summarizeAffect(computeAffect(sim));
    const histB = rvqHistogramDropPhi(tokB);

    // Affect within tight tolerance
    expect(affectClose(affA, affB, 1e-4)).toBe(true);

    // PAT rvq histogram equality (exact)
    expect(histB).toEqual(histA);
  });

  it("Torus translation leaves PAT code distribution (w/o phi bin) and affect invariant", () => {
    const cfg = makeInitConfig(24, 24, 2468);
    const base = createSimulation(null, cfg).simulation;

    // Build simB as a fresh sim and copy translated snapshot into phases
    const simB = createSimulation(null, cfg).simulation;
    const dx = 5, dy = -7;
    simB.phases = translateSnapshot(base.phases, cfg.W, cfg.H, dx, dy);

    // Summaries
    const tokA = buildPATokens(base, 32);
    const affA = summarizeAffect(computeAffect(base));
    const histA = rvqHistogramDropPhi(tokA);

    const tokB = buildPATokens(simB, 32);
    const affB = summarizeAffect(computeAffect(simB));
    const histB = rvqHistogramDropPhi(tokB);

    expect(affectClose(affA, affB, 1e-4)).toBe(true);
    expect(histB).toEqual(histA);
  });

  it("Reflection (x-axis) leaves PAT code distribution (w/o phi bin) and affect invariant", () => {
    const cfg = makeInitConfig(24, 24, 97531);
    const base = createSimulation(null, cfg).simulation;

    const simB = createSimulation(null, cfg).simulation;
    simB.phases = reflectSnapshot(base.phases, cfg.W, cfg.H, "x");

    const tokA = buildPATokens(base, 32);
    const affA = summarizeAffect(computeAffect(base));
    const histA = rvqHistogramDropPhi(tokA);

    const tokB = buildPATokens(simB, 32);
    const affB = summarizeAffect(computeAffect(simB));
    const histB = rvqHistogramDropPhi(tokB);

    expect(affectClose(affA, affB, 1e-4)).toBe(true);
    expect(histB).toEqual(histA);
  });

  it("Reflection (y-axis) leaves PAT code distribution (w/o phi bin) and affect invariant", () => {
    const cfg = makeInitConfig(24, 24, 86420);
    const base = createSimulation(null, cfg).simulation;

    const simB = createSimulation(null, cfg).simulation;
    simB.phases = reflectSnapshot(base.phases, cfg.W, cfg.H, "y");

    const tokA = buildPATokens(base, 32);
    const affA = summarizeAffect(computeAffect(base));
    const histA = rvqHistogramDropPhi(tokA);

    const tokB = buildPATokens(simB, 32);
    const affB = summarizeAffect(computeAffect(simB));
    const histB = rvqHistogramDropPhi(tokB);

    expect(affectClose(affA, affB, 1e-4)).toBe(true);
    expect(histB).toEqual(histA);
  });
});
