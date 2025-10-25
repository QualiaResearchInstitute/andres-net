import { describe, it, expect } from "vitest";
import { createSimulation, type SimulationInitConfig } from "./simulation";
import { updateAttentionFields, type AttentionParams } from "./attention";
import { rmseAttentionOutputs } from "./eval";
import { buildPATokens } from "../tokens/pat";
import { tokensToAttentionFields } from "../tokens/tokenOnly";

function makeInitConfig(W = 32, H = 32, reseedGraphKey = 424242): SimulationInitConfig {
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

function makeAttentionParams(): AttentionParams {
  return {
    // Diffusion/decay
    DA: 0.8,
    DU: 0.6,
    muA: 0.2,
    muU: 0.15,
    // Sources
    lambdaS: 0.9,
    lambdaC: 0.6,
    topoGain: 0.0, // disable topo for simpler comparison
    // Modulator gains into physics (kept for signature compatibility)
    gammaK: 0.3,
    betaK: 0.15,
    gammaAlpha: 0.25,
    betaAlpha: 0.1,
    gammaD: 0.2,
    deltaD: 0.05,
    // Salience weights
    wGrad: 0.6,
    wLap: 0.2,
    wE: 0.2,
    wDef: 0.0,
    // Context blur
    contextRadius: 3,
    // Reflector disabled for this test to avoid dependency
    reflectorGain: 0.0,
    // clamps
    aClamp: 1.0,
    uClamp: 1.0,
  };
}

describe("Token-only mode: proxy attention fields approximate full attention outputs", () => {
  it("Aact proxy RMSE within tolerance at a snapshot (heads disabled)", () => {
    const cfg = makeInitConfig(32, 32, 1312);
    const sim = createSimulation(null, cfg).simulation;

    const params = makeAttentionParams();
    const dt = 0.03;
    const t = 0.0;
    const wrapEdges = true;

    // Full-visibility attention outputs (no heads for a fairer comparison)
    const outFull = updateAttentionFields(sim, [], params, dt, t, wrapEdges);

    // Token-only proxy derived from PAT (no Aact provided to PAT so it falls back to entropyRate)
    const tokens = buildPATokens(sim, 32);
    const proxy = tokensToAttentionFields(tokens, cfg.W, cfg.H, 32);

    // Compare Aact fields
    const rmseA = rmseAttentionOutputs(outFull.Aact, proxy.Aact);

    // Phase-1 proxy tolerance (coarse approximation is acceptable)
    expect(rmseA).toBeLessThanOrEqual(0.15);
  });
});
