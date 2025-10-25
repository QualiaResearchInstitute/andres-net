import type { SimulationState } from "./types";

/**
 * Compute the total allowed cross-boundary coupling "capacity" across an in/out pocket rim.
 * - Iterates ordered neighbor pairs (i <- j) using sim.neighbors and sim.neighborBands, plus sim.swEdges.
 * - Applies horizon gating identical in spirit to stepSimulation:
 *    sealed:    forbid all cross-boundary transmissions (iIn != jIn) => 0
 *    oneway:    forbid leaks from inside->outside (jIn && !iIn) => 0, allow outside->inside
 *    none:      allow all
 * - Returns the sum of |w_eff| across all ordered cross-boundary links. Phase-independent and deterministic.
 */
export function computeCrossBoundaryCapacity(
  sim: SimulationState,
  pocketMask: Uint8Array,
  mode: "sealed" | "oneway" | "none",
  weights: { K1: number; K2: number; K3: number; swWeight: number },
): number {
  const { neighbors, neighborBands, swEdges, N } = sim;
  const { K1, K2, K3, swWeight } = weights;
  let acc = 0;

  const horizonPair = (i: number, j: number) => {
    const iIn = !!pocketMask[i];
    const jIn = !!pocketMask[j];
    if (mode === "none") return 1.0;
    if (mode === "sealed") {
      // forbid all cross-boundary transmissions
      return iIn === jIn ? 1.0 : 0.0;
    }
    // oneway: allow into pocket, block out of pocket
    // receiver = i, source = j
    if (mode === "oneway") {
      if (jIn && !iIn) return 0.0; // block leak from inside to outside
      return 1.0;
    }
    return 1.0;
  };

  for (let i = 0; i < N; i++) {
    const iIn = !!pocketMask[i];

    // Lattice neighbors (3 rings)
    const ns = neighbors[i] as Int32Array;
    const bands = neighborBands[i] as Uint8Array;
    for (let k = 0; k < ns.length; k++) {
      const j = ns[k];
      const jIn = !!pocketMask[j];
      if (iIn === jIn) continue; // only cross-boundary links
      const band = bands[k];
      const w = band === 1 ? K1 : band === 2 ? K2 : K3;
      if (w === 0) continue;
      const h = horizonPair(i, j);
      if (h <= 0) continue;
      acc += Math.abs(w) * h;
    }

    // Small-world edges
    const sw = (swEdges[i] as Array<[number, number]>);
    for (let k = 0; k < sw.length; k++) {
      const [j, sign] = sw[k];
      const jIn = !!pocketMask[j];
      if (iIn === jIn) continue;
      const w = swWeight * sign;
      if (w === 0) continue;
      const h = horizonPair(i, j);
      if (h <= 0) continue;
      acc += Math.abs(w) * h;
    }
  }

  return acc;
}

/**
 * Compute the top-1 PCA power (largest eigenvalue of 2x2 covariance) of points (x,y) within an optional mask.
 * - Deterministic, uses exact 2x2 eigensolution: for symmetric C = [[a,b],[b,c]],
 *   lambda_max = 0.5 * ((a+c) + sqrt((a-c)^2 + 4 b^2)).
 * - Returns lambda_max (σ1^2), which is monotonic in σ1 and suitable for growth comparisons.
 */
export function top1PCAPowerXY(
  xs: Float32Array,
  ys: Float32Array,
  mask?: Uint8Array,
): number {
  const n = Math.min(xs.length, ys.length);
  let cnt = 0;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const x = xs[i] ?? 0;
    const y = ys[i] ?? 0;
    mx += x;
    my += y;
    cnt++;
  }
  if (cnt === 0) return 0;
  mx /= cnt;
  my /= cnt;

  // Covariance accumulation
  let a = 0, b = 0, c = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const dx = (xs[i] ?? 0) - mx;
    const dy = (ys[i] ?? 0) - my;
    a += dx * dx; // var(x)
    b += dx * dy; // cov(x,y)
    c += dy * dy; // var(y)
  }
  a /= cnt;
  b /= cnt;
  c /= cnt;

  const trace = a + c;
  const delta = (a - c) * (a - c) + 4 * b * b;
  const sqrtDelta = Math.sqrt(Math.max(0, delta));
  const lambdaMax = 0.5 * (trace + sqrtDelta);
  return lambdaMax;
}
