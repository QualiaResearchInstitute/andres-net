import { describe, it, expect } from "vitest";
import { updateFrustration } from "./annealing";
import type { SimulationState } from "./types";

/**
 * Build a minimal sim stub with only W,H (updateFrustration only uses W*H).
 */
function makeSimStub(W = 32, H = 24) {
  return { W, H } as unknown as SimulationState;
}

/** Count ones deterministically (avoid typed-array reduce typing issues) */
function countOnes(u8: ArrayLike<number>): number {
  let c = 0;
  for (let i = 0; i < u8.length; i++) c += u8[i] ? 1 : 0;
  return c;
}

/** Clone Uint8Array */
function cloneU8(u8: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(u8.length);
  out.set(u8);
  return out;
}

/** Equality on contents (TypedArray or any ArrayLike<number>) */
function equalsU8(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * Construct a Dfield where every Q-th cell is above threshold deterministically.
 */
function makeDfieldAboveThresh(N: number, stride = 3, base = 0.0, high = 1.0) {
  const D = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    D[i] = (i % stride) === 0 ? high : base;
  }
  return D;
}

describe("Annealing: deterministic knot release with T ramp", () => {
  it("OFF (T=0) => stuck; ON with ramp => frustration flags drop after T≥T*; release frame identical on replay", () => {
    const W = 32, H = 24;
    const N = W * H;
    const sim = makeSimStub(W, H);

    // Initial frustration: all clenched (1)
    const prev = new Uint8Array(N);
    prev.fill(1);

    // Dfield: periodic spikes > thresh
    const releaseThresh = 0.5;
    const Dfield = makeDfieldAboveThresh(N, /*stride*/ 3, /*base*/ 0.0, /*high*/ 1.0);

    // OFF: T=0 => unchanged
    const off = updateFrustration(sim, 0, Dfield, releaseThresh, prev);
    expect(equalsU8(off, prev)).toBe(true);

    // ON: ramp T from 0 -> 1 in 9 steps; qT in [1..8] due to quantization
    const steps = 9;
    const Ts: number[] = [];
    for (let i = 0; i < steps; i++) Ts.push(i / (steps - 1));

    // First run: record per-step buffers and first drop frame
    const firstRunCounts: number[] = [];
    const firstRunStates: Uint8Array[] = [];
    let firstReleaseIdx: number | null = null;
    let cur = prev;
    for (let s = 0; s < Ts.length; s++) {
      cur = updateFrustration(sim, Ts[s], Dfield, releaseThresh, cur);
      firstRunStates.push(cloneU8(cur));
      const cnt = countOnes(cur);
      firstRunCounts.push(cnt);
      if (firstReleaseIdx === null && cnt < N) {
        firstReleaseIdx = s;
      }
    }
    expect(firstReleaseIdx).not.toBeNull();

    // Replay: identical sequence and result (byte-equal per step)
    let curReplay = prev;
    for (let s = 0; s < Ts.length; s++) {
      curReplay = updateFrustration(sim, Ts[s], Dfield, releaseThresh, curReplay);
      expect(equalsU8(curReplay, firstRunStates[s])).toBe(true);
    }

    // Monotone non-increasing after first release
    if (firstReleaseIdx !== null) {
      for (let s = firstReleaseIdx + 1; s < firstRunCounts.length; s++) {
        expect(firstRunCounts[s]).toBeLessThanOrEqual(firstRunCounts[s - 1]);
      }
    }
  });
});
