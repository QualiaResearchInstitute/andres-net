import { describe, it, expect } from "vitest";
import { createSimulation, type SimulationInitConfig } from "./simulation";
import { computeCrossBoundaryCapacity } from "./pockets_eval";

function makeInitConfig(W = 24, H = 24, reseedGraphKey = 4242): SimulationInitConfig {
  return {
    W,
    H,
    wrap: true,
    omegaSpread: 0.25,
    // Disable SW edges to focus on lattice flux (can turn on if desired)
    swProb: 0.0,
    swEdgesPerNode: 0,
    swMinDist: 6,
    swMaxDist: 20,
    swNegFrac: 0.2,
    energyBaseline: 0.35,
    reseedGraphKey,
  };
}

function makeCentralPocketMask(W: number, H: number): Uint8Array {
  const mask = new Uint8Array(W * H);
  const x0 = Math.floor(W * 0.25);
  const x1 = Math.ceil(W * 0.75);
  const y0 = Math.floor(H * 0.25);
  const y1 = Math.ceil(H * 0.75);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      mask[y * W + x] = 1;
    }
  }
  return mask;
}

describe("Topological pockets: cross-rim capacity under horizon modes", () => {
  it("sealed horizons reduce cross-rim capacity by ≥10× vs none; oneway < none", () => {
    const init = makeInitConfig(24, 24, 4242);
    const { simulation: sim } = createSimulation(null, init);
    const W = sim.W, H = sim.H;

    const mask = makeCentralPocketMask(W, H);

    // Use typical band weights from existing tests
    const weights = { K1: 1.0, K2: -0.3, K3: 0.15, swWeight: 0.25 };

    const capNone = computeCrossBoundaryCapacity(sim, mask, "none", weights);
    const capOneWay = computeCrossBoundaryCapacity(sim, mask, "oneway", weights);
    const capSealed = computeCrossBoundaryCapacity(sim, mask, "sealed", weights);

    // Sanity: there must be some cross-boundary capacity when no horizon limits are applied
    expect(capNone).toBeGreaterThan(0);

    // Sealed forbids all cross-boundary transmissions
    expect(capSealed).toBe(0);

    // One-way should reduce capacity compared to none (blocks inside->outside, allows outside->inside)
    expect(capOneWay).toBeGreaterThanOrEqual(capSealed);
    expect(capOneWay).toBeLessThan(capNone);

    // ≥10× reduction when sealed vs none (trivially holds since sealed==0)
    expect(capSealed).toBeLessThanOrEqual(0.1 * capNone);
  });
});
