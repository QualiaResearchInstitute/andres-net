import { describe, it, expect } from "vitest";
import { createSimulation, type SimulationInitConfig } from "./simulation";
import { updateAttentionFields, type AttentionParams } from "./attention";

function makeInitConfig(W = 32, H = 32, reseedGraphKey = 999): SimulationInitConfig {
  return {
    W,
    H,
    wrap: true,
    omegaSpread: 0.25,
    swProb: 0.0,
    swEdgesPerNode: 0,
    swMinDist: 6,
    swMaxDist: 20,
    swNegFrac: 0.0,
    energyBaseline: 0.35,
    reseedGraphKey,
  };
}

function makeAttentionParams(): AttentionParams {
  return {
    // Diffusion/decay
    DA: 0.2,
    DU: 0.2,
    muA: 0.1,
    muU: 0.1,
    // Source gains
    lambdaS: 1.0,
    lambdaC: 0.6,
    topoGain: 0.0,
    // Modulator gains (not used by stepSimulation here)
    gammaK: 0.0,
    betaK: 0.0,
    gammaAlpha: 0.0,
    betaAlpha: 0.0,
    gammaD: 0.0,
    deltaD: 0.0,
    // Salience weights
    wGrad: 1.0,
    wLap: 0.4,
    wE: 0.3,
    wDef: 1.0,
    // Context blur
    contextRadius: 2,
    // Reflector disabled
    reflectorGain: 0.0,
    // clamps
    aClamp: 4.0,
    uClamp: 4.0,
  };
}

describe("orientation-aware α-bias numeric safety (near-isotropy)", () => {
  it("produces zero lapA payload when local orientation is flat (constant phases)", () => {
    const cfg = makeInitConfig(32, 32, 2024);
    const sim = createSimulation(null, cfg).simulation;

    // Force phases to a constant to induce near-isotropy (∇φ ≈ 0 => χ -> 0)
    for (let i = 0; i < sim.N; i++) sim.phases[i] = 0.5;

    const params = makeAttentionParams();
    const dt = 0.03;
    const t = 0.0;
    const wrapEdges = true;

    const out = updateAttentionFields(sim, [], params, dt, t, wrapEdges);

    // With χ_eff=0 the orientation-aware payload should be identically zero
    let maxAbs = 0;
    for (let i = 0; i < out.lapA.length; i++) {
      const v = Math.abs(out.lapA[i]);
      if (v > maxAbs) maxAbs = v;
    }
    expect(maxAbs).toBeLessThanOrEqual(1e-12);
  });
});
