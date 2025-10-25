import { wrapAngle } from "./diffusion";
import type { SimulationState } from "./types";

/** Neighborhood patch for local circular KDE score */
export type Patch = { idx: Int32Array; weights: Float32Array };

/** Modest concentration ramp as τ ↓ 0 */
export function kappaSchedule(t: number) {
  // Higher concentration early (t≈0), smoother late (t≈1)
  // Range ~[8, 24] → tuneable if needed
  return 8.0 + 16.0 * (1 - t);
}

/** Non-ML local Parzen score on S¹ using von-Mises KDE gradient */
export function parzenScore(theta: Float32Array, patches: Patch[], t: number): Float32Array {
  const N = theta.length;
  const s = new Float32Array(N);
  const kap = kappaSchedule(t);
  for (let i = 0; i < N; i++) {
    const P = patches[i];
    const idx = P.idx;
    const w = P.weights;
    let num = 0;
    let den = 0;
    for (let k = 0; k < idx.length; k++) {
      const j = idx[k];
      const d = theta[j] - theta[i];
      const ew = Math.exp(kap * Math.cos(d));
      num += w[k] * ew * (kap * Math.sin(d));
      den += w[k] * ew;
    }
    s[i] = den > 1e-8 ? num / den : 0;
  }
  return s;
}

/** One explicit Euler step of the probability-flow ODE on S¹ */
export function reverseFlowStep(
  theta: Float32Array,
  drift: (th: Float32Array, t: number) => Float32Array,
  D: (t: number) => number,
  score: (th: Float32Array, t: number) => Float32Array,
  t: number,
  dt: number,
): Float32Array {
  const f = drift(theta, t);
  const s = score(theta, t);
  const out = new Float32Array(theta.length);
  const Dt = D(t);
  for (let i = 0; i < theta.length; i++) {
    out[i] = wrapAngle(theta[i] + dt * (f[i] - Dt * s[i]));
  }
  return out;
}

/** Utility: build local patches from the simulation's neighbor lists */
export function buildPatchesFromNeighbors(sim: SimulationState, normalize = true): Patch[] {
  const { N, neighbors, neighborBands } = sim;
  const patches: Patch[] = new Array(N);
  for (let i = 0; i < N; i++) {
    const idx = neighbors[i];
    const bands = neighborBands[i];
    const w = new Float32Array(idx.length);
    // ring-based weights: closer rings weigh more (1/r), fallback=1
    for (let k = 0; k < idx.length; k++) {
      const r = bands[k] || 1;
      w[k] = 1 / r;
    }
    if (normalize) {
      let sum = 0;
      for (let k = 0; k < w.length; k++) sum += w[k];
      const inv = sum > 1e-8 ? 1 / sum : 1;
      for (let k = 0; k < w.length; k++) w[k] *= inv;
    }
    patches[i] = { idx, weights: w };
  }
  return patches;
}

/** Utility: compute local order parameter ρ per-index given patches */
export function localOrderParameter(theta: Float32Array, patches: Patch[]): Float32Array {
  const N = theta.length;
  const rho = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const P = patches[i];
    let cx = 0,
      sx = 0;
    for (let k = 0; k < P.idx.length; k++) {
      const j = P.idx[k];
      const w = P.weights[k];
      const th = theta[j];
      cx += w * Math.cos(th);
      sx += w * Math.sin(th);
    }
    rho[i] = Math.hypot(cx, sx);
  }
  return rho;
}

/** Utility: simple plaquette winding-number defect detector (±1 defects) */
export function defectDensityEstimate(
  theta: Float32Array,
  W: number,
  H: number,
  wrap = true,
): { density: number; countPlus: number; countMinus: number } {
  const wrapDiff = (a: number, b: number) => {
    // smallest signed difference on circle
    let d = a - b;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d <= -Math.PI) d += 2 * Math.PI;
    return d;
  };
  let plus = 0;
  let minus = 0;
  // iterate 1x1 plaquettes
  for (let y = 0; y < H - (wrap ? 0 : 1); y++) {
    const yy = wrap ? y % H : y;
    const y1 = wrap ? (y + 1) % H : y + 1;
    if (!wrap && y1 >= H) continue;
    for (let x = 0; x < W - (wrap ? 0 : 1); x++) {
      const xx = wrap ? x % W : x;
      const x1 = wrap ? (x + 1) % W : x + 1;
      if (!wrap && x1 >= W) continue;
      const i00 = yy * W + xx;
      const i10 = yy * W + x1;
      const i11 = y1 * W + x1;
      const i01 = y1 * W + xx;
      const d1 = wrapDiff(theta[i10], theta[i00]);
      const d2 = wrapDiff(theta[i11], theta[i10]);
      const d3 = wrapDiff(theta[i01], theta[i11]);
      const d4 = wrapDiff(theta[i00], theta[i01]);
      const sum = d1 + d2 + d3 + d4; // ~ 2π * winding
      if (sum > Math.PI) plus++;
      else if (sum < -Math.PI) minus++;
    }
  }
  const area = W * H;
  const count = plus + minus;
  return { density: count / area, countPlus: plus, countMinus: minus };
}
