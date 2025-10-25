import { describe, it, expect } from "vitest";
import { createSimulation, type SimulationInitConfig } from "./simulation";
import { reverseStepInPlace } from "./reverseRunner";
import type { StepConfig } from "./stepSimulation";

function makeInitConfig(W = 16, H = 16, reseedGraphKey = 20251024): SimulationInitConfig {
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

function buildStepConfig(): StepConfig {
  return {
    dt: 0.03,
    wrap: true,
    // Couplings
    Kbase: 0.8,
    K1: 1.0,
    K2: -0.3,
    K3: 0.15,
    KS1: 0.9,
    KS2: 0.0,
    KS3: 0.0,
    KH1: 0.7,
    KH2: 0.0,
    KH3: 0.0,
    // Cross-layer gains
    alphaSurfToField: 0.6,
    alphaFieldToSurf: 0.5,
    alphaHypToField: 0.8,
    alphaFieldToHyp: 0.4,
    // SW
    swWeight: 0.25,
    // Environment/energy
    wallBarrier: 0.5,
    emGain: 0.6,
    energyBaseline: 0.35,
    energyLeak: 0.02,
    energyDiff: 0.12,
    sinkLine: 0.03,
    sinkSurf: 0.02,
    sinkHyp: 0.02,
    trapSurf: 0.2,
    trapHyp: 0.35,
    minEnergySurf: 0.25,
    minEnergyHyp: 0.4,
    // noiseAmp is unused when we pass explicit D(t) to reverseStepInPlace,
    // but keep small to avoid side effects elsewhere
    noiseAmp: 0.0,
    // DAG off (match non-inplace drift path)
    dagSweeps: 0,
    dagDepthOrdering: true,
    dagDepthFiltering: true,
    dagLogStats: false,
    // Attention off for determinism baseline
    attentionMods: undefined,
    // Horizon: transparent (no events)
    horizonFactor: (iIn: boolean, jIn: boolean, recvIn: boolean) => {
      if (iIn === jIn) return 1.0;
      if (recvIn && !jIn) return 1.0;
      if (!recvIn && jIn) return 1.0;
      return 1.0;
    },
  };
}

describe("Reverse probability-flow ODE determinism (CPU↔CPU)", () => {
  it("produces identical phases across two identical runs (noise-free D(t)=0)", () => {
    const cfgInit = makeInitConfig(24, 24, 7777);
    const A = createSimulation(null, cfgInit).simulation;
    const B = createSimulation(null, cfgInit).simulation;

    // initial phases identical
    expect(A.N).toBe(B.N);
    expect(Array.from(A.phases)).toEqual(Array.from(B.phases));

    const stepCfg: StepConfig = buildStepConfig();
    const frames = 6;
    const D = (_t: number) => 0.0; // no diffusion/drift from score multiplier

    for (let s = 0; s < frames; s++) {
      reverseStepInPlace(A, stepCfg, 0.5, { D });
      reverseStepInPlace(B, stepCfg, 0.5, { D });
    }

    // exact equality (Float32)
    expect(Array.from(A.phases)).toEqual(Array.from(B.phases));
  });
});

// Optional placeholder (skipped) for future CPU↔GPU tolerance test if a GPU path is added.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
describe.skip("Reverse ODE CPU↔GPU tolerance (RMSE < 1e-5)", () => {
  it("matches CPU and GPU implementations within tolerance", () => {
    // Placeholder: once a GPU reverse path exists, compute RMSE(θ_cpu, θ_gpu) < 1e-5
  });
});
