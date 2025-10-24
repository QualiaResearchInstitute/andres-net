import { TAU } from "./math";
import type { SimulationState } from "./types";
import { buildPatchesFromNeighbors, localOrderParameter, defectDensityEstimate } from "./reverseSampler";

/** Metrics captured for physics-aware evaluation */
export type EvalMetrics = {
  rhoMean: number;
  rhoStd: number;
  defectDensity: number;
  defectsPlus: number;
  defectsMinus: number;
  entropyGrad: number;      // mean gradient magnitude (phase diffs)
  anisotropyHV: number;     // (|dx|-|dy|)/(|dx|+|dy|) averaged
};

/** Clamp to [0,1] */
function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

/** Smallest signed difference on the circle */
function angleDiff(a: number, b: number) {
  let d = a - b;
  while (d > Math.PI) d -= TAU;
  while (d <= -Math.PI) d += TAU;
  return d;
}

/** Compute physics-aware metrics from the current simulation snapshot */
export function computeMetrics(sim: SimulationState): EvalMetrics {
  const { W, H, phases } = sim;
  const N = phases.length;

  // Local order parameter ρ via patches
  const patches = buildPatchesFromNeighbors(sim);
  const rho = localOrderParameter(phases, patches);
  let sumR = 0, sumR2 = 0;
  for (let i = 0; i < N; i++) {
    const r = clamp01(rho[i]);
    sumR += r;
    sumR2 += r * r;
  }
  const rhoMean = sumR / Math.max(1, N);
  const varR = Math.max(0, sumR2 / Math.max(1, N) - rhoMean * rhoMean);
  const rhoStd = Math.sqrt(varR);

  // Defect density (±1 vortices via plaquette winding test)
  const { density, countPlus, countMinus } = defectDensityEstimate(phases, W, H, true);

  // Entropy-rate proxy and anisotropy: use simple finite differences
  let sumAbsDx = 0, sumAbsDy = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const iL = y * W + ((x - 1 + W) % W);
      const iU = ((y - 1 + H) % H) * W + x;
      const dx = angleDiff(phases[i], phases[iL]);
      const dy = angleDiff(phases[i], phases[iU]);
      sumAbsDx += Math.abs(dx);
      sumAbsDy += Math.abs(dy);
    }
  }
  const denom = Math.max(1e-6, W * H);
  const meanAbsDx = sumAbsDx / denom;
  const meanAbsDy = sumAbsDy / denom;
  const entropyGrad = 0.5 * (meanAbsDx + meanAbsDy);
  const anisotropyHV = (meanAbsDx - meanAbsDy) / Math.max(1e-6, meanAbsDx + meanAbsDy);

  return {
    rhoMean,
    rhoStd,
    defectDensity: density,
    defectsPlus: countPlus,
    defectsMinus: countMinus,
    entropyGrad,
    anisotropyHV,
  };
}

/**
 * Correlate orientation-aware α-bias (lapA payload) with rim-aligned swirl.
 * Samples points where |∇A| is above a threshold (attention rims) and reports stats over lapA_orient.
 */
export function orientationBiasScore(
  lapA_orient: Float32Array,
  A: Float32Array,
  phases: Float32Array,
  W: number,
  H: number,
  threshA = 0.3
) {
  const N = W * H;
  let sum = 0, sum2 = 0, cnt = 0;
  for (let y = 0; y < H; y++) {
    const ym = (y - 1 + H) % H, yp = (y + 1) % H;
    for (let x = 0; x < W; x++) {
      const xm = (x - 1 + W) % W, xp = (x + 1) % W;
      const i = y * W + x;
      const gx = 0.5 * (A[y * W + xp] - A[y * W + xm]);
      const gy = 0.5 * (A[yp * W + x] - A[ym * W + x]);
      const g = Math.hypot(gx, gy);
      if (g > threshA) {
        const v = lapA_orient[i];
        sum += v;
        sum2 += v * v;
        cnt++;
      }
    }
  }
  const mean = cnt > 0 ? sum / cnt : 0;
  const varr = Math.max(0, cnt > 0 ? sum2 / cnt - mean * mean : 0);
  return { rimCount: cnt, mean, std: Math.sqrt(varr) };
}

/** RMSE helper for determinism checks over Float32Array fields */
export function rmseAttentionOutputs(a: Float32Array, b: Float32Array) {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s / Math.max(1, n));
}
