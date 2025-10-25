import { describe, it, expect } from "vitest";
import { createSimulation, type SimulationInitConfig } from "./simulation";
import { stepSimulation, type StepConfig } from "./stepSimulation";

function buildStepConfig(dt: number): StepConfig {
  return {
    dt,
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
    // Small-world weight
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
    // Noise OFF for determinism
    noiseAmp: 0.0,
    // DAG off (no in-place ordering)
    dagSweeps: 0,
    dagDepthOrdering: true,
    dagDepthFiltering: true,
    dagLogStats: false,
    // No attention mods
    attentionMods: undefined,
    // Simple horizon
    horizonFactor: (iIn: boolean, jIn: boolean, recvIn: boolean) => {
      if (iIn === jIn) return 1.0;
      if (recvIn && !jIn) return 1.0;
      if (!recvIn && jIn) return 1.0;
      return 1.0;
    },
  };
}

function makeInitConfig(W = 16, H = 16, reseedGraphKey = 123): SimulationInitConfig {
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

describe("Simulation determinism (noise-free)", () => {
  it("produces identical phases across two identical runs", () => {
    const cfg = makeInitConfig(24, 24, 4242);
    const A = createSimulation(null, cfg).simulation;
    const B = createSimulation(null, cfg).simulation;

    // Sanity: initial phases identical
    expect(A.N).toBe(B.N);
    expect(Array.from(A.phases)).toEqual(Array.from(B.phases));

    const stepCfg = buildStepConfig(0.03);
    const frames = 6;

    for (let s = 0; s < frames; s++) {
      stepSimulation(A, stepCfg);
      stepSimulation(B, stepCfg);
    }

    // Bitwise equality on Float32 values
    expect(Array.from(A.phases)).toEqual(Array.from(B.phases));
  });
});
