import { describe, it, expect } from "vitest";
import { clamp } from "./math";

/**
 * Reproduces the lift/disparity logic from renderFrame.ts in a pure function:
 *   lift = depth > 0 ? ((clamp(depth * 0.5, 0, 4) / 4) ** hyperCurve) * liftGain : 0
 *   disp  = round(ipd * lift)
 */
function liftValue(depth: number, hyperCurve: number, liftGain: number): number {
  if (depth <= 0) return 0;
  const v = clamp(depth * 0.5, 0, 4) / 4;
  return Math.pow(v, hyperCurve) * liftGain;
}
function disparity(depth: number, ipd: number, hyperCurve: number, liftGain: number): number {
  const L = liftValue(depth, hyperCurve, liftGain);
  return Math.round(ipd * L);
}

describe("Hyperbolic depth → stereo parallax monotonicity and determinism", () => {
  it("disparity is non-decreasing with depth and stable across replays", () => {
    const hyperCurve = 0.6;
    const liftGain = 0.8;
    const ipd = 2.0;

    // Synthetic depths (including 0 and clamped high values)
    const depths = [0, 1, 2, 3, 4, 6, 8];

    // First computation
    const dispA = depths.map((d) => disparity(d, ipd, hyperCurve, liftGain));
    // Monotone non-decreasing
    for (let i = 1; i < dispA.length; i++) {
      expect(dispA[i]).toBeGreaterThanOrEqual(dispA[i - 1]);
    }

    // Replay identical computation
    const dispB = depths.map((d) => disparity(d, ipd, hyperCurve, liftGain));
    expect(dispB).toEqual(dispA);

    // Sanity: non-zero disparity for some positive depth (given gains above)
    expect(dispA.some((v) => v > 0)).toBe(true);
  });
});
