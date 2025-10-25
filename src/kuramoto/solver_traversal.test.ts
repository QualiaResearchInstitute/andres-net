import { describe, it, expect } from "vitest";
import { createSimulation, ensureDagCache, markDagDirty } from "./simulation";
import type { SimulationInitConfig } from "./simulation";
import { extractPockets } from "./topology";

function makeInitConfig(W = 16, H = 12, reseedGraphKey = 2025): SimulationInitConfig {
  return {
    W,
    H,
    wrap: true,
    omegaSpread: 0.25,
    swProb: 0.0,
    swEdgesPerNode: 0,
    swMinDist: 6,
    swMaxDist: 20,
    swNegFrac: 0.2,
    energyBaseline: 0.35,
    reseedGraphKey,
  };
}

describe("Fixed traversal and ordering determinism", () => {
  it("ensureDagCache builds depth layers in raster-scan order", () => {
    const { simulation: sim } = createSimulation(null, makeInitConfig(16, 12, 42));
    const { W, H, planeDepth } = sim;

    // Construct a deterministic banded depth map, then verify layer order.
    // Depth bands by rows: depth = y % 3
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        planeDepth[y * W + x] = y % 3;
      }
    }
    markDagDirty(sim);
    const dag = ensureDagCache(sim);
    // For each layer, indices must be strictly increasing (raster order i=0..N-1 fill)
    for (let d = 0; d <= dag.maxDepth; d++) {
      const layer = dag.layers[d];
      let prev = -1;
      for (let k = 0; k < layer.length; k++) {
        const idx = layer[k];
        expect(idx).toBeGreaterThan(prev);
        prev = idx;
        // Also, verify membership depth matches
        expect(sim.planeDepth[idx]).toBe(d);
      }
    }
  });

  it("extractPockets scan + BFS order is stable for simple shapes", () => {
    const W = 20, H = 14;
    const N = W * H;
    const score = new Float32Array(N);

    // Deterministic plus-shaped blob and a rectangle
    for (let y = 4; y < 10; y++) score[y * W + 10] = 1.0; // vertical bar
    for (let x = 7; x < 14; x++) score[7 * W + x] = 1.0;  // horizontal bar
    for (let y = 2; y < 6; y++) for (let x = 2; x < 6; x++) score[y * W + x] = 1.0;

    const thresh = 0.5, minArea = 4;
    const A = extractPockets(score, W, H, thresh, minArea);
    const B = extractPockets(score, W, H, thresh, minArea);

    expect(A.length).toBe(B.length);
    for (let i = 0; i < A.length; i++) {
      expect(A[i].id).toBe(B[i].id);
      expect(A[i].area).toBe(B[i].area);
      expect(Array.from(A[i].rimIdx)).toEqual(Array.from(B[i].rimIdx));
      expect(Array.from(A[i].mask)).toEqual(Array.from(B[i].mask));
    }
  });
});
