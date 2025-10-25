import { TAU } from "./math";
import type { SimulationState } from "./types";
import type { PAToken } from "../tokens/pat";
import { ReflectorGraph } from "./graph/ReflectorGraph";

/**
 * Normalize angle to [0, 2π)
 */
function wrapToTau(x: number) {
  let y = x % TAU;
  if (y < 0) y += TAU;
  return y;
}

/**
 * Positive modulo for indices
 */
function posMod(n: number, m: number) {
  return ((n % m) + m) % m;
}

/**
 * Apply a global gauge shift φ0 to the simulation phases (and optionally layer phases).
 * Mutates the provided sim in-place.
 */
export function applyGaugeShift(
  sim: SimulationState,
  phi0: number,
  opts?: { includeLayers?: boolean },
) {
  const { phases, N } = sim;
  const includeLayers = opts?.includeLayers !== false;

  for (let i = 0; i < N; i++) {
    phases[i] = wrapToTau(phases[i] + phi0);
  }

  if (includeLayers) {
    const { surfPhi, hypPhi } = sim;
    for (let i = 0; i < N; i++) {
      surfPhi[i] = wrapToTau(surfPhi[i] + phi0);
      hypPhi[i] = wrapToTau(hypPhi[i] + phi0);
    }
  }
}

/**
 * Torus-translate a snapshot by integer offsets (dx, dy).
 * Returns a new Float32Array with translated values.
 */
export function translateSnapshot(
  field: Float32Array,
  W: number,
  H: number,
  dx: number,
  dy: number,
): Float32Array {
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const ys = posMod(y - dy, H);
    for (let x = 0; x < W; x++) {
      const xs = posMod(x - dx, W);
      out[y * W + x] = field[ys * W + xs];
    }
  }
  return out;
}

/**
 * Reflect a snapshot across x or y axis.
 * axis = "x" (horizontal mirror) or "y" (vertical mirror).
 * Returns a new Float32Array with reflected values.
 */
export function reflectSnapshot(
  field: Float32Array,
  W: number,
  H: number,
  axis: "x" | "y",
): Float32Array {
  const out = new Float32Array(W * H);
  if (axis === "x") {
    // mirror horizontally
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const xr = W - 1 - x;
        out[y * W + x] = field[y * W + xr];
      }
    }
  } else {
    // mirror vertically
    for (let y = 0; y < H; y++) {
      const yr = H - 1 - y;
      for (let x = 0; x < W; x++) {
        out[y * W + x] = field[yr * W + x];
      }
    }
  }
  return out;
}

/**
 * Build a histogram (multiset) over PAT rvq codes with optional dropped dimensions.
 * Returns a Map serialized as a plain object for ease of test comparison.
 */
export function patRVQHistogram(
  tokens: PAToken[],
  dropDims: number[] = [],
): Record<string, number> {
  const counts = new Map<string, number>();
  for (const t of tokens) {
    const rvq = t.code?.rvq ?? [];
    const proj: number[] = [];
    for (let i = 0; i < rvq.length; i++) {
      if (dropDims.includes(i)) continue;
      proj.push(rvq[i]);
    }
    const key = JSON.stringify(proj);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const obj: Record<string, number> = {};
  for (const [k, v] of counts.entries()) obj[k] = v;
  return obj;
}

/**
 * Produce a reduced, invariance-oriented summary per token that drops gauge-sensitive fields.
 * - Drops hdr, geom.phi_gf explicitly (encoded in rvq[1] in current PAT)
 * - Keeps rvq without the phi_gf bin, plus selected stats and topo signs.
 */
export function normalizePATForInvariance(tokens: PAToken[]) {
  const out = tokens.map((t) => {
    const rvq = t.code?.rvq ?? [];
    // Drop index 1 (phi_gf bin) from current PAT definition
    const proj = rvq.filter((_v, i) => i !== 1);
    const vortex = t.topo?.vortex ?? 0;
    return {
      rvq: proj,
      rho: t.stats.rho,
      entropyRate: t.stats.entropyRate,
      attnMean: t.stats.attnMean,
      attnVar: t.stats.attnVar,
      vortex,
    };
  });
  return out;
}

/**
 * Summarize affect headers for invariance testing.
 */
export function summarizeAffect(a: { arousal: number; valence: number }) {
  return { arousal: a.arousal, valence: a.valence };
}

/**
 * Summarize ReflectorGraph activities for invariance testing.
 */
export function summarizeReflector(refl: ReflectorGraph) {
  const nodes = refl.getNodes();
  const n = nodes.length;
  let sumA = 0;
  for (const nd of nodes) sumA += nd.a;
  const meanA = n > 0 ? sumA / n : 0;
  let varA = 0;
  if (n > 0) {
    for (const nd of nodes) {
      const d = nd.a - meanA;
      varA += d * d;
    }
    varA /= n;
  }
  return { n, sumA, meanA, varA };
}
