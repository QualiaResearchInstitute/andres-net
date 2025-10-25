import { describe, it, expect } from "vitest";
import { createSimulation } from "./simulation";
import { updateAttentionFields, type AttentionParams } from "./attention";
import { ReflectorGraph } from "./graph/ReflectorGraph";

function makeParams(overrides: Partial<AttentionParams> = {}): AttentionParams {
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
    // Optional ReflectorGraph gain
    reflectorGain: 0.0,
    // clamps
    aClamp: 4.0,
    uClamp: 4.0,
    ...overrides,
  };
}

describe("attention/updateAttentionFields", () => {
  it("is deterministic given the same initial sim state and params", () => {
    const cfg = {
      W: 16, H: 16, wrap: true,
      omegaSpread: 0.25,
      swProb: 0.0, swEdgesPerNode: 0, swMinDist: 6, swMaxDist: 20, swNegFrac: 0.0,
      energyBaseline: 0.35,
      reseedGraphKey: 777,
    };
    const A = createSimulation(null, cfg).simulation;
    const B = createSimulation(null, cfg).simulation;

    // Attach ReflectorGraph for both to keep code paths identical (gain 0 so it's inactive)
    (A as any)._reflectorGraph = new ReflectorGraph(A.W, A.H);
    (B as any)._reflectorGraph = new ReflectorGraph(B.W, B.H);

    const heads: any[] = []; // no synthetic heads
    const p = makeParams({ reflectorGain: 0.0 });

    const outA = updateAttentionFields(A, heads, p, 0.03, 0.0, true);
    const outB = updateAttentionFields(B, heads, p, 0.03, 0.0, true);

    expect(Array.from(outA.A)).toEqual(Array.from(outB.A));
    expect(Array.from(outA.U)).toEqual(Array.from(outB.U));
    expect(Array.from(outA.Aact)).toEqual(Array.from(outB.Aact));
    expect(Array.from(outA.Uact)).toEqual(Array.from(outB.Uact));
    expect(Array.from(outA.lapA)).toEqual(Array.from(outB.lapA));
    expect(Array.from(outA.divU)).toEqual(Array.from(outB.divU));
  });

  it("reflectorGain>0 changes A compared to reflectorGain=0 (with ReflectorGraph attached)", () => {
    const cfg = {
      W: 16, H: 16, wrap: true,
      omegaSpread: 0.25,
      swProb: 0.0, swEdgesPerNode: 0, swMinDist: 6, swMaxDist: 20, swNegFrac: 0.0,
      energyBaseline: 0.35,
      reseedGraphKey: 1313,
    };
    const sim0 = createSimulation(null, cfg).simulation;
    const sim1 = createSimulation(null, cfg).simulation;

    // Attach helper to both
    (sim0 as any)._reflectorGraph = new ReflectorGraph(sim0.W, sim0.H);
    (sim1 as any)._reflectorGraph = new ReflectorGraph(sim1.W, sim1.H);

    const heads: any[] = [];
    const p0 = makeParams({ reflectorGain: 0.0 });
    const p1 = makeParams({ reflectorGain: 1.0 });

    const out0 = updateAttentionFields(sim0, heads, p0, 0.03, 0.0, true);
    const out1 = updateAttentionFields(sim1, heads, p1, 0.03, 0.0, true);

    // Expect some difference introduced by splatted reflector field
    expect(Array.from(out0.A)).not.toEqual(Array.from(out1.A));
  });
});
