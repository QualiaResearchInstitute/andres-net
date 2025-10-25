import { describe, it, expect } from "vitest";
import { top1PCAPowerXY } from "./pockets_eval";

/**
 * Build a synthetic point cloud with an elongated cluster in the center.
 * Also return a pocketMask selecting that elongated cluster deterministically.
 */
function makeElongatedPoints(W = 32, H = 24) {
  const N = W * H;
  const xs = new Float32Array(N);
  const ys = new Float32Array(N);
  const pocketMask = new Uint8Array(N);

  // Deterministic construction:
  // - Inner rectangle: elongated horizontally, higher variation in x than y
  // - Outside: near-isotropic scatter
  const x0 = Math.floor(W * 0.2);
  const x1 = Math.ceil(W * 0.8);
  const y0 = Math.floor(H * 0.4);
  const y1 = Math.ceil(H * 0.6);

  let idx = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++, idx++) {
      // Normalize to [-1,1] box
      const xn = (x / Math.max(1, W - 1)) * 2 - 1;
      const yn = (y / Math.max(1, H - 1)) * 2 - 1;

      const inPocket = x >= x0 && x < x1 && y >= y0 && y < y1;
      pocketMask[idx] = inPocket ? 1 : 0;

      if (inPocket) {
        // Elongated: strong x spread, weaker y spread (deterministic, not random)
        xs[idx] = 1.5 * xn + 0.05 * (y % 3) - 0.05 * ((x % 5) - 2);
        ys[idx] = 0.25 * yn + 0.02 * ((x + y) % 4) - 0.01 * ((x % 3) - 1);
      } else {
        // More isotropic, smaller overall magnitude
        xs[idx] = 0.5 * xn + 0.05 * ((x + 2 * y) % 5);
        ys[idx] = 0.5 * yn + 0.05 * ((2 * x + y) % 5);
      }
    }
  }

  return { xs, ys, pocketMask, W, H, N };
}

/**
 * Deterministic synthetic rho field:
 * - In-pocket cells get higher rho, outside slightly lower.
 * - Shape chosen to be independent of phases; this test is about relationships, not sim dynamics.
 */
function makeSyntheticRho(W: number, H: number, pocketMask: Uint8Array) {
  const N = W * H;
  const rho = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const base = 0.4 + 0.1 * ((i % 7) / 7); // deterministic gentle variation
    rho[i] = base + (pocketMask[i] ? 0.3 : -0.05);
  }
  return rho;
}

describe("Pocket resonance: PCA power and rho relationships", () => {
  it("In-pocket top-1 PCA power exceeds outside and whole-field; in-pocket rho mean exceeds outside", () => {
    const { xs, ys, pocketMask, W, H, N } = makeElongatedPoints();

    // PCA power comparisons
    const pcaAll = top1PCAPowerXY(xs, ys);
    const pcaIn = top1PCAPowerXY(xs, ys, pocketMask);

    // Build outside mask
    const outsideMask = new Uint8Array(N);
    for (let i = 0; i < N; i++) outsideMask[i] = pocketMask[i] ? 0 : 1;
    const pcaOut = top1PCAPowerXY(xs, ys, outsideMask);

    // The elongated pocket should exhibit a larger principal variance than the outside
    expect(pcaIn).toBeGreaterThan(pcaOut);
    // And should dominate the aggregate (all) due to being the strongest coherent structure
    expect(pcaIn).toBeGreaterThan(pcaAll);

    // Synthetic rho relationship: in-pocket mean rho > outside mean rho
    const rho = makeSyntheticRho(W, H, pocketMask);
    let sumIn = 0, cntIn = 0, sumOut = 0, cntOut = 0;
    for (let i = 0; i < N; i++) {
      if (pocketMask[i]) {
        sumIn += rho[i];
        cntIn++;
      } else {
        sumOut += rho[i];
        cntOut++;
      }
    }
    const meanIn = cntIn > 0 ? sumIn / cntIn : 0;
    const meanOut = cntOut > 0 ? sumOut / cntOut : 0;
    expect(meanIn).toBeGreaterThan(meanOut);
  });
});
