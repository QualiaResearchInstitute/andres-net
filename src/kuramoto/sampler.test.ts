import { describe, it, expect } from "vitest";
import { createSimulation } from "./simulation";
import {
  kappaSchedule,
  parzenScore,
  reverseFlowStep,
  buildPatchesFromNeighbors,
  localOrderParameter,
  defectDensityEstimate,
  type Patch,
} from "./reverseSampler";

describe("reverseSampler utilities", () => {
  it("kappaSchedule decreases with t and stays positive", () => {
    const k0 = kappaSchedule(0.0);
    const k1 = kappaSchedule(0.5);
    const k2 = kappaSchedule(1.0);
    expect(k0).toBeGreaterThan(k1);
    expect(k1).toBeGreaterThan(k2);
    expect(k2).toBeGreaterThan(0);
  });

  it("parzenScore is zero when all neighbors equal self angle", () => {
    const N = 16;
    const theta = new Float32Array(N).fill(0.5);
    // Make trivial patches where each node sees itself only
    const patches: Patch[] = Array.from({ length: N }, (_, i) => ({
      idx: Int32Array.from([i]),
      weights: Float32Array.from([1]),
    }));
    const s = parzenScore(theta, patches, 0.5);
    for (let i = 0; i < N; i++) {
      expect(Math.abs(s[i])).toBeLessThan(1e-7);
    }
  });

  it("reverseFlowStep wraps angles to (-π, π]", () => {
    const N = 8;
    const theta = new Float32Array(N).fill(Math.PI - 0.01);
    const zeroVec = (len: number) => new Float32Array(len).fill(0);
    const drift = (th: Float32Array, t: number) => zeroVec(th.length);
    const D = (t: number) => 0.0;
    const score = (th: Float32Array, t: number) => zeroVec(th.length);
    const out = reverseFlowStep(theta, drift, D, score, 0.5, 1.0);
    for (let i = 0; i < N; i++) {
      expect(out[i]).toBeGreaterThan(-Math.PI);
      expect(out[i]).toBeLessThanOrEqual(Math.PI);
    }
  });

  it("buildPatchesFromNeighbors returns normalized weights (~1)", () => {
    const cfg = {
      W: 8, H: 8, wrap: true,
      omegaSpread: 0.25,
      swProb: 0.0, swEdgesPerNode: 0, swMinDist: 6, swMaxDist: 20, swNegFrac: 0.0,
      energyBaseline: 0.35,
      reseedGraphKey: 999,
    };
    const sim = createSimulation(null, cfg).simulation;
    const patches = buildPatchesFromNeighbors(sim, true);
    expect(patches.length).toBe(sim.N);
    for (const p of patches) {
      let sum = 0;
      for (let k = 0; k < p.weights.length; k++) sum += p.weights[k];
      // Allow small numeric tolerance
      expect(Math.abs(sum - 1)).toBeLessThan(1e-5);
    }
  });

  it("localOrderParameter stays within [0,1]", () => {
    const cfg = {
      W: 6, H: 6, wrap: true,
      omegaSpread: 0.25,
      swProb: 0.0, swEdgesPerNode: 0, swMinDist: 6, swMaxDist: 20, swNegFrac: 0.0,
      energyBaseline: 0.35,
      reseedGraphKey: 2024,
    };
    const sim = createSimulation(null, cfg).simulation;
    const patches = buildPatchesFromNeighbors(sim, true);
    const rho = localOrderParameter(sim.phases, patches);
    expect(rho.length).toBe(sim.N);
    for (let i = 0; i < rho.length; i++) {
      expect(rho[i]).toBeGreaterThanOrEqual(0);
      expect(rho[i]).toBeLessThanOrEqual(1 + 1e-6);
    }
  });

  it("defectDensityEstimate returns non-negative density within [0, 1]", () => {
    const cfg = {
      W: 10, H: 10, wrap: true,
      omegaSpread: 0.25,
      swProb: 0.0, swEdgesPerNode: 0, swMinDist: 6, swMaxDist: 20, swNegFrac: 0.0,
      energyBaseline: 0.35,
      reseedGraphKey: 13579,
    };
    const sim = createSimulation(null, cfg).simulation;
    const out = defectDensityEstimate(sim.phases, sim.W, sim.H, true);
    expect(out.density).toBeGreaterThanOrEqual(0);
    expect(out.density).toBeLessThanOrEqual(1);
    expect(out.countPlus + out.countMinus).toBeGreaterThanOrEqual(0);
  });
});
