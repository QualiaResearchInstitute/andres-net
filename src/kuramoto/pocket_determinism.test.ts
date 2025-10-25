import { describe, it, expect } from "vitest";
import { extractPockets } from "./topology";

/**
 * Build a deterministic score field with two rectangular blobs well above threshold.
 * This avoids dependence on dynamic sim fields and keeps CC/rim purely morphological.
 */
function makeScoreField(W = 32, H = 24) {
  const N = W * H;
  const score = new Float32Array(N);
  // Pocket A: centered rectangle
  for (let y = 6; y < 14; y++) {
    for (let x = 8; x < 16; x++) {
      score[y * W + x] = 1.0;
    }
  }
  // Pocket B: right rectangle
  for (let y = 10; y < 18; y++) {
    for (let x = 20; x < 27; x++) {
      score[y * W + x] = 1.0;
    }
  }
  return { score, W, H };
}

function pocketsSnapshot(pxs: ReturnType<typeof extractPockets>) {
  // Normalize to a JSON-serializable deterministic snapshot.
  // Include: ids in order, area, rimIdx contents, and a compact hash of mask.
  return pxs.map((p) => {
    // Simple mask hash: sum(i * mask[i]) to capture both membership and spread
    let h = 0;
    for (let i = 0; i < p.mask.length; i++) {
      if (p.mask[i]) h = (h + i) >>> 0;
    }
    return {
      id: p.id,
      area: p.area,
      rim: Array.from(p.rimIdx), // deterministic due to raster-scan push order
      maskHash: h,
    };
  });
}

describe("Pockets: deterministic labeling, area, and rims", () => {
  it("extractPockets produces byte-stable pockets across replays", () => {
    const { score, W, H } = makeScoreField(32, 24);
    const thresh = 0.5;
    const minArea = 8;

    const A = extractPockets(score, W, H, thresh, minArea);
    const B = extractPockets(score, W, H, thresh, minArea);

    // Count and ids
    expect(A.length).toBe(B.length);
    for (let k = 0; k < A.length; k++) {
      expect(A[k].id).toBe(B[k].id);
      // Masks equal
      expect(Array.from(A[k].mask)).toEqual(Array.from(B[k].mask));
      // Areas equal
      expect(A[k].area).toBe(B[k].area);
      // Rims equal (order preserved by raster scan)
      expect(Array.from(A[k].rimIdx)).toEqual(Array.from(B[k].rimIdx));
    }

    // Full snapshot equality for extra guard
    const snapA = JSON.stringify(pocketsSnapshot(A));
    const snapB = JSON.stringify(pocketsSnapshot(B));
    expect(snapA).toBe(snapB);
  });
});
