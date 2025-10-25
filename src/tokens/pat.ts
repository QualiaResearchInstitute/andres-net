import type { SimulationState } from "../kuramoto/types";
import { computeAffect } from "../kuramoto/affect";

/**
 * Phase–Amplitude Token (PAT)
 */
export interface PAToken {
  hdr: { t: number; scale: 32 | 64; patchId: string };
  geom: { A: number; phi_gf: number; theta: number; gradPhi: number; kappa: number };
  stats: {
    rho: number;
    energy: number;
    entropyRate: number;
    arousal: number;
    valence: number;
    attnMean: number;
    attnVar: number;
    reconMSE?: number; // optional reconstruction MSE if available from upstream
  };
  heads?: number[]; // phases of enabled heads (if provided)
  topo: { vortex: -1 | 0 | 1; chirality: -1 | 0 | 1 };
  code?: { rvq: number[] };
}

type BuildOpts = {
  Aact?: Float32Array | null; // optional attention amplitude field
  heads?: number[]; // optional enabled head phases
};

/**
 * Build PAT tokens on a uniform grid of square patches of size = scale (32 or 64).
 * Deterministic, no randomness. All outputs normalized to [0,1] where applicable.
 */
export function buildPATokens(sim: SimulationState, scale: 32 | 64, opts?: BuildOpts): PAToken[] {
  const { W, H, phases, energy } = sim;
  const Aact = opts?.Aact ?? null;
  const heads = opts?.heads ?? undefined;

  // Global affect headers (Phase-1)
  const affect = computeAffect(sim);
  const arousal = isFinite(affect.arousal) ? affect.arousal : 0;
  const valence = isFinite(affect.valence) ? affect.valence : 0;

  const tokens: PAToken[] = [];
  const now = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();

  // Precompute cos/sin for wrapped gradients
  const N = W * H;
  const cosTh = new Float32Array(N);
  const sinTh = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const th = phases[i];
    cosTh[i] = Math.cos(th);
    sinTh[i] = Math.sin(th);
  }

  // Iterate patches
  for (let y0 = 0; y0 < H; y0 += scale) {
    const y1 = Math.min(H, y0 + scale);
    for (let x0 = 0; x0 < W; x0 += scale) {
      const x1 = Math.min(W, x0 + scale);

      // Accumulators
      let sumE = 0;
      let cnt = 0;

      // Mean complex phase for rho and phi_gf
      let cx = 0, sx = 0;

      // Gradient accumulators for |∇φ| and curvature κ (finite differences on wrapped phase)
      let sumGrad = 0;
      let sumLap = 0;

      // Attention stats (if provided)
      let sumA = 0, sumA2 = 0;

      // Structure tensor for orientation from ∇φ (J = [[gx^2, gxgy], [gxgy, gy^2]])
      let Sxx = 0, Sxy = 0, Syy = 0;

      // Vortex/chirality counts via plaquette winding restricted to patch
      let plus = 0, minus = 0;

      for (let y = y0; y < y1; y++) {
        const ym = clampWrap(y - 1, 0, H - 1);
        const yp = clampWrap(y + 1, 0, H - 1);
        for (let x = x0; x < x1; x++) {
          const xm = clampWrap(x - 1, 0, W - 1);
          const xp = clampWrap(x + 1, 0, W - 1);
          const i = y * W + x;

          // Energy
          const e = energy[i];
          sumE += e;

          // Phase complex
          const c = cosTh[i];
          const s = sinTh[i];
          cx += c;
          sx += s;

          // Wrapped gradient of phase using cos/sin identity:
          // ∇θ = (-sinθ ∇cosθ + cosθ ∇sinθ)
          const iL = y * W + xm;
          const iR = y * W + xp;
          const iU = ym * W + x;
          const iD = yp * W + x;

          const dcosdx = 0.5 * (cosTh[iR] - cosTh[iL]);
          const dsindx = 0.5 * (sinTh[iR] - sinTh[iL]);
          const dcosdy = 0.5 * (cosTh[iD] - cosTh[iU]);
          const dsindy = 0.5 * (sinTh[iD] - sinTh[iU]);

          const gx = -s * dcosdx + c * dsindx;
          const gy = -s * dcosdy + c * dsindy;

          const gradMag = Math.hypot(gx, gy);
          sumGrad += gradMag;

          // Structure tensor
          Sxx += gx * gx;
          Sxy += gx * gy;
          Syy += gy * gy;

          // Laplacian (second differences on θ via cos/sin)
          const iLL = y * W + clampWrap(x - 2, 0, W - 1);
          const iRR = y * W + clampWrap(x + 2, 0, W - 1);
          const iUU = clampWrap(y - 2, 0, H - 1) * W + x;
          const iDD = clampWrap(y + 2, 0, H - 1) * W + x;

          // approximate θ via atan2 and compute 5-point laplacian on θ
          const thetaCenter = Math.atan2(sinTh[i], cosTh[i]);
          const thetaL = Math.atan2(sinTh[iL], cosTh[iL]);
          const thetaR = Math.atan2(sinTh[iR], cosTh[iR]);
          const thetaU = Math.atan2(sinTh[iU], cosTh[iU]);
          const thetaD = Math.atan2(sinTh[iD], cosTh[iD]);
          const lap = wrapAngle(thetaL - thetaCenter) +
            wrapAngle(thetaR - thetaCenter) +
            wrapAngle(thetaU - thetaCenter) +
            wrapAngle(thetaD - thetaCenter);
          sumLap += Math.abs(lap);

          // Attention stats
          if (Aact) {
            const a = Aact[i] ?? 0;
            sumA += a;
            sumA2 += a * a;
          }

          cnt++;
        }
      }

      // Plaquette winding for ±1 vortices within patch
      for (let y = y0; y < y1 - 1; y++) {
        const y_ = clampWrap(y, 0, H - 1);
        const y1_ = clampWrap(y + 1, 0, H - 1);
        for (let x = x0; x < x1 - 1; x++) {
          const x_ = clampWrap(x, 0, W - 1);
          const x1_ = clampWrap(x + 1, 0, W - 1);
          const i00 = y_ * W + x_;
          const i10 = y_ * W + x1_;
          const i11 = y1_ * W + x1_;
          const i01 = y1_ * W + x_;

          const d1 = wrapAngle(Math.atan2(sinTh[i10], cosTh[i10]) - Math.atan2(sinTh[i00], cosTh[i00]));
          const d2 = wrapAngle(Math.atan2(sinTh[i11], cosTh[i11]) - Math.atan2(sinTh[i10], cosTh[i10]));
          const d3 = wrapAngle(Math.atan2(sinTh[i01], cosTh[i01]) - Math.atan2(sinTh[i11], cosTh[i11]));
          const d4 = wrapAngle(Math.atan2(sinTh[i00], cosTh[i00]) - Math.atan2(sinTh[i01], cosTh[i01]));
          const sum = d1 + d2 + d3 + d4;
          if (sum > Math.PI) plus++;
          else if (sum < -Math.PI) minus++;
        }
      }

      // Patch-reduced stats
      const meanE = cnt > 0 ? sumE / cnt : 0;
      const meanGrad = cnt > 0 ? sumGrad / cnt : 0;
      const meanLap = cnt > 0 ? sumLap / cnt : 0;

      const R = cnt > 0 ? Math.hypot(cx, sx) / cnt : 0;
      const phi_gf = Math.atan2(sx, cx); // global frame mean phase ([-π, π])

      // Orientation from structure tensor principal direction
      let thetaOrient = 0;
      if (Sxx + Syy > 1e-9) {
        // eigenvector of largest eigenvalue
        // tan(2θ) = 2Sxy / (Sxx - Syy)
        thetaOrient = 0.5 * Math.atan2(2 * Sxy, Sxx - Syy); // [-π/2, π/2]
      }

      // Attention mean/var
      let attnMean = 0, attnVar = 0;
      if (Aact && cnt > 0) {
        attnMean = sumA / cnt;
        const m2 = Math.max(0, sumA2 / cnt - attnMean * attnMean);
        attnVar = Math.sqrt(m2);
      }

      // Topology summary
      let vortex: -1 | 0 | 1 = 0;
      if (plus > minus) vortex = 1;
      else if (minus > plus) vortex = -1;
      const chirality: -1 | 0 | 1 = vortex;

      // Geom feature vector (normalized for simple LFQ)
      const geom = {
        A: 1.0, // amplitude proxy (Phase-1 baseline)
        phi_gf, // mean phase (radians)
        theta: thetaOrient, // orientation angle (radians)
        gradPhi: meanGrad, // average |∇θ|
        kappa: meanLap, // curvature proxy
      };

      // Simple local headers
      const stats = {
        rho: clamp01(R),
        energy: clamp01(meanE), // energy roughly in [0,1]
        entropyRate: clamp01(meanGrad / Math.PI), // normalize by π
        arousal,
        valence,
        attnMean,
        attnVar,
      };

      // Tiny LFQ (per-dim 4 bins), reported as indices per-dimension (rvq single stage)
      const rvq = lfqEncode([
        clamp01(geom.A),
        angle01(geom.phi_gf),
        orientRef01(geom.theta),
        clamp01(stats.entropyRate),
        clamp01(Math.min(1, Math.abs(geom.kappa) / Math.PI)),
      ], 4);

      tokens.push({
        hdr: { t: now, scale, patchId: `${x0},${y0},${scale}` },
        geom,
        stats,
        heads,
        topo: { vortex, chirality },
        code: { rvq },
      });
    }
  }

  return tokens;
}

// Helpers

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function clampWrap(v: number, lo: number, hi: number) {
  // Wrap index v into [lo, hi] (torus topology)
  const n = hi - lo + 1;
  let x = (v - lo) % n;
  if (x < 0) x += n;
  return lo + x;
}

function wrapAngle(x: number) {
  let y = (x + Math.PI) % (2 * Math.PI);
  if (y < 0) y += 2 * Math.PI;
  return y - Math.PI;
}

// Map angle in [-π, π] to [0,1]
function angle01(a: number) {
  return (wrapAngle(a) + Math.PI) / (2 * Math.PI);
}

/* Map angle in [-π/2, π/2] to [0,1] */
function angleHalf01(a: number) {
  const h = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, a));
  return (h + Math.PI / 2) / Math.PI;
}

/* Reflection-invariant orientation encoding in [0,1]
   Folds θ and -θ to the same value; 0 -> 0, ±π/2 -> 1. */
function orientRef01(a: number) {
  const u = angleHalf01(a); // [0,1], 0.5 at 0
  return Math.abs(u - 0.5) * 2;
}

// Simple LFQ: per-dimension uniform bins -> return indices per-dimension
function lfqEncode(vec: number[], bins: number): number[] {
  const out: number[] = new Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    const v = clamp01(vec[i] ?? 0);
    let idx = Math.floor(v * bins);
    if (idx >= bins) idx = bins - 1;
    out[i] = idx;
  }
  return out;
}
