import { TAU } from "./math";
import { localOrderParameter, buildPatchesFromNeighbors } from "./reverseSampler";
import type { SimulationState } from "./types";

/** Ridge regression head: y = W^T x + b */
export type RidgeHead = {
  inDim: number;
  outDim: number;
  lambda: number;
  W: Float32Array; // [inDim*outDim], column-major by output: W[o + i*outDim]
  b: Float32Array; // [outDim]
};

/** Simple codebook for PAT tokens */
export type PatCodebook = {
  dim: number;
  size: number;
  table: Float32Array; // [size*dim], row-major: table[c*dim + d]
};

export type PatSnapshot = {
  tokens: Int32Array; // [N]
  distances: Float32Array; // [N]
  arousal: number;
  valence: number;
  rhoMean: number;
  rhoStd: number;
  W: number;
  H: number;
};

/** Build features for RGB/PAT heads from sim fields */
export function buildFeatureMatrix(
  sim: SimulationState,
  opts?: { includeAE?: boolean; includeRho?: boolean; patches?: ReturnType<typeof buildPatchesFromNeighbors> },
): { X: Float32Array; inDim: number } {
  const { phases, energy, W, H } = sim;
  const N = phases.length;
  const includeAE = opts?.includeAE ?? true;
  const includeRho = opts?.includeRho ?? true;
  const dims: number[] = [];
  // base circular embedding
  const hasCosSin = true;
  let inDim = 0;
  if (hasCosSin) inDim += 2;
  if (includeAE) inDim += 2; // A,E
  if (includeRho) inDim += 1; // rho
  const X = new Float32Array(N * inDim);

  const patches = includeRho ? opts?.patches ?? buildPatchesFromNeighbors(sim) : null;
  const rhoField = includeRho && patches ? localOrderParameter(phases, patches) : null;

  // Amplitude proxy A: use 1.0 baseline (can be replaced by lift or planeQuickR proxy)
  for (let i = 0; i < N; i++) {
    const th = phases[i];
    let col = 0;
    if (hasCosSin) {
      X[i * inDim + col++] = Math.cos(th);
      X[i * inDim + col++] = Math.sin(th);
    }
    if (includeAE) {
      const A = 1.0;
      const E = energy[i];
      X[i * inDim + col++] = A;
      X[i * inDim + col++] = E;
    }
    if (includeRho) {
      const r = rhoField ? rhoField[i] : 0.0;
      X[i * inDim + col++] = r;
    }
  }
  return { X, inDim };
}

/** Solve (XtX + λI)W = XtY, with mean-centering for bias. */
export function fitRidge(
  X: Float32Array,
  Y: Float32Array,
  N: number,
  inDim: number,
  outDim: number,
  lambda = 1e-3,
): RidgeHead {
  // compute means
  const meanX = new Float32Array(inDim);
  const meanY = new Float32Array(outDim);
  for (let n = 0; n < N; n++) {
    for (let i = 0; i < inDim; i++) meanX[i] += X[n * inDim + i];
    for (let o = 0; o < outDim; o++) meanY[o] += Y[n * outDim + o];
  }
  for (let i = 0; i < inDim; i++) meanX[i] /= N;
  for (let o = 0; o < outDim; o++) meanY[o] /= N;

  // compute XtX and XtY on centered data
  const XtX = new Float64Array(inDim * inDim);
  const XtY = new Float64Array(inDim * outDim);
  for (let n = 0; n < N; n++) {
    for (let i = 0; i < inDim; i++) {
      const xi = X[n * inDim + i] - meanX[i];
      for (let j = 0; j < inDim; j++) {
        const xj = X[n * inDim + j] - meanX[j];
        XtX[i * inDim + j] += xi * xj;
      }
      for (let o = 0; o < outDim; o++) {
        const yo = Y[n * outDim + o] - meanY[o];
        XtY[i * outDim + o] += xi * yo;
      }
    }
  }
  // add λI
  for (let i = 0; i < inDim; i++) {
    XtX[i * inDim + i] += lambda;
  }

  // solve for each output: XtX * w_o = XtY[:,o]
  const W = new Float32Array(inDim * outDim);
  const A = XtX; // alias
  // precompute LU (Gaussian elimination) - naive but fine for small inDim
  // We'll solve per o using a copy of RHS
  const solve = (rhs: Float64Array): Float64Array => {
    const n = inDim;
    // make copies
    const M = new Float64Array(A);
    const b = new Float64Array(rhs);

    // forward elimination with partial pivoting
    for (let k = 0; k < n; k++) {
      // pivot
      let piv = k;
      let pivVal = Math.abs(M[k * n + k]);
      for (let r = k + 1; r < n; r++) {
        const v = Math.abs(M[r * n + k]);
        if (v > pivVal) {
          piv = r;
          pivVal = v;
        }
      }
      if (piv !== k) {
        for (let c = k; c < n; c++) {
          const tmp = M[k * n + c];
          M[k * n + c] = M[piv * n + c];
          M[piv * n + c] = tmp;
        }
        const tb = b[k];
        b[k] = b[piv];
        b[piv] = tb;
      }
      const diag = M[k * n + k] || 1e-12;
      // eliminate
      for (let r = k + 1; r < n; r++) {
        const factor = M[r * n + k] / diag;
        if (factor === 0) continue;
        for (let c = k; c < n; c++) {
          M[r * n + c] -= factor * M[k * n + c];
        }
        b[r] -= factor * b[k];
      }
    }

    // back substitution
    const x = new Float64Array(n);
    for (let r = n - 1; r >= 0; r--) {
      let acc = b[r];
      for (let c = r + 1; c < n; c++) acc -= M[r * n + c] * x[c];
      const diag = M[r * n + r] || 1e-12;
      x[r] = acc / diag;
    }
    return x;
  };

  for (let o = 0; o < outDim; o++) {
    const rhs = new Float64Array(inDim);
    for (let i = 0; i < inDim; i++) rhs[i] = XtY[i * outDim + o];
    const wo = solve(rhs);
    for (let i = 0; i < inDim; i++) W[o + i * outDim] = wo[i];
  }

  // compute intercept: b = meanY - W^T * meanX
  const b = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let v = meanY[o];
    let corr = 0;
    for (let i = 0; i < inDim; i++) corr += W[o + i * outDim] * meanX[i];
    b[o] = v - corr;
  }

  return { inDim, outDim, lambda, W, b };
}

export function applyRidge(head: RidgeHead, x: Float32Array): Float32Array {
  const { inDim, outDim, W, b } = head;
  if (x.length !== inDim) {
    throw new Error(`applyRidge: expected inDim=${inDim}, got ${x.length}`);
  }
  const out = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let v = b[o];
    for (let i = 0; i < inDim; i++) v += W[o + i * outDim] * x[i];
    out[o] = v;
  }
  return out;
}

/** Quantize feature vectors to nearest codebook rows (L2) */
export function quantizeToCodebook(
  V: Float32Array, // [N * dim]
  dim: number,
  codebook: PatCodebook,
): { tokens: Int32Array; dists: Float32Array } {
  if (dim !== codebook.dim) throw new Error("quantize: dim mismatch");
  const N = Math.floor(V.length / dim);
  const tokens = new Int32Array(N);
  const dists = new Float32Array(N);
  const { table, size } = codebook;
  for (let n = 0; n < N; n++) {
    let best = -1;
    let bestD = Number.POSITIVE_INFINITY;
    const base = n * dim;
    for (let c = 0; c < size; c++) {
      let d = 0;
      const cb = c * dim;
      for (let k = 0; k < dim; k++) {
        const diff = V[base + k] - table[cb + k];
        d += diff * diff;
      }
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
    tokens[n] = best;
    dists[n] = Math.sqrt(bestD);
  }
  return { tokens, dists };
}

/** Derive a simple PAT snapshot (tokens + affect headers) */
export function patSnapshot(
  sim: SimulationState,
  codebook: PatCodebook,
  opts?: { patches?: ReturnType<typeof buildPatchesFromNeighbors> },
): PatSnapshot {
  const { W, H, phases, energy } = sim;
  const patches = opts?.patches ?? buildPatchesFromNeighbors(sim);
  const rhoField = localOrderParameter(phases, patches);
  const { X, inDim } = buildFeatureMatrix(sim, { includeAE: true, includeRho: true, patches });

  // We will quantize the feature vector itself as a Phase-Amplitude Token (PAT) proxy
  const { tokens, dists } = quantizeToCodebook(X, inDim, codebook);

  // Affect headers
  const stats = reduceStats(rhoField);
  const { arousal, valence } = computeAffectHeaders(sim, rhoField);

  return {
    tokens,
    distances: dists,
    arousal,
    valence,
    rhoMean: stats.mean,
    rhoStd: stats.std,
    W,
    H,
  };
}

/** Compute arousal/valence scalars (Phase-1 lightweight) */
export function computeAffectHeaders(sim: SimulationState, rhoField?: Float32Array): { arousal: number; valence: number } {
  const { phases, energy, W, H } = sim;
  const N = phases.length;

  // Order parameter (global) from rhoField or crude global from cos/sin
  let rhoMean = 0;
  if (rhoField) {
    for (let i = 0; i < rhoField.length; i++) rhoMean += rhoField[i];
    rhoMean /= rhoField.length;
  } else {
    let cr = 0, ci = 0;
    for (let i = 0; i < N; i++) {
      cr += Math.cos(phases[i]);
      ci += Math.sin(phases[i]);
    }
    rhoMean = Math.hypot(cr, ci) / N;
  }

  // Defect density proxy (approximate)
  const { density } = defectDensityApprox(phases, W, H);

  // Entropy-rate proxy: mean gradient magnitude
  let grad = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const il = y * W + Math.max(0, x - 1);
      const iu = Math.max(0, y - 1) * W + x;
      const dl = angleDiff(phases[i], phases[il]);
      const du = angleDiff(phases[i], phases[iu]);
      grad += Math.abs(dl) + Math.abs(du);
    }
  }
  grad /= (W * H * 2);

  // Energy statistic
  let eMean = 0;
  for (let i = 0; i < N; i++) eMean += energy[i];
  eMean /= N;

  // Arousal: combine energy, gradient, and 1 - rho
  const arousal = clamp01(0.5 * norm01(eMean, 0.1, 1.0) + 0.3 * norm01(grad, 0.0, Math.PI / 2) + 0.2 * (1 - rhoMean));

  // Valence: high with coherence and low defect density
  const valence = clamp01(0.7 * rhoMean + 0.3 * (1 - norm01(density, 0, 0.01)));

  return { arousal, valence };
}

function angleDiff(a: number, b: number) {
  let d = a - b;
  while (d > Math.PI) d -= TAU;
  while (d <= -Math.PI) d += TAU;
  return d;
}

function reduceStats(arr: Float32Array) {
  let mean = 0, m2 = 0;
  for (let i = 0; i < arr.length; i++) {
    const x = arr[i];
    mean += x;
    m2 += x * x;
  }
  mean /= arr.length;
  const varr = Math.max(0, m2 / arr.length - mean * mean);
  return { mean, std: Math.sqrt(varr) };
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function norm01(x: number, lo: number, hi: number) {
  if (hi <= lo) return 0;
  const t = (x - lo) / (hi - lo);
  return clamp01(t);
}

/** Fast approximate defect density from plaquette winding using coarse stride */
export function defectDensityApprox(theta: Float32Array, W: number, H: number, stride = 1) {
  let plus = 0, minus = 0, count = 0;
  for (let y = 0; y < H - stride; y += stride) {
    for (let x = 0; x < W - stride; x += stride) {
      const i00 = y * W + x;
      const i10 = y * W + (x + stride);
      const i11 = (y + stride) * W + (x + stride);
      const i01 = (y + stride) * W + x;
      const d1 = angleDiff(theta[i10], theta[i00]);
      const d2 = angleDiff(theta[i11], theta[i10]);
      const d3 = angleDiff(theta[i01], theta[i11]);
      const d4 = angleDiff(theta[i00], theta[i01]);
      const sum = d1 + d2 + d3 + d4;
      if (sum > Math.PI) plus++;
      else if (sum < -Math.PI) minus++;
      count++;
    }
  }
  const density = count > 0 ? (plus + minus) / (count * stride * stride) : 0;
  return { density, countPlus: plus, countMinus: minus };
}

/** Example tiny codebook constructor (grid on unit hypercube) */
export function makeToyPatCodebook(dim: number, perAxis = 4): PatCodebook {
  const size = Math.max(1, Math.pow(perAxis, dim) | 0);
  const table = new Float32Array(size * dim);
  const coords: number[][] = [[]];
  for (let d = 0; d < dim; d++) {
    const newCoords: number[][] = [];
    for (const base of coords) {
      for (let k = 0; k < perAxis; k++) {
        newCoords.push([...base, k / (perAxis - 1)]);
      }
    }
    coords.splice(0, coords.length, ...newCoords);
  }
  for (let i = 0; i < size; i++) {
    const row = coords[i] ?? new Array(dim).fill(0);
    for (let d = 0; d < dim; d++) table[i * dim + d] = row[d];
  }
  return { dim, size, table };
}

/** Batch apply ridge head to a [N,inDim] matrix X, returns [N,outDim] */
export function applyRidgeBatch(head: RidgeHead, X: Float32Array, N: number): Float32Array {
  const { inDim, outDim, W, b } = head;
  const Y = new Float32Array(N * outDim);
  for (let n = 0; n < N; n++) {
    for (let o = 0; o < outDim; o++) {
      let v = b[o];
      for (let i = 0; i < inDim; i++) {
        v += W[o + i * outDim] * X[n * inDim + i];
      }
      Y[n * outDim + o] = v;
    }
  }
  return Y;
}

/** RGB targets consistent with hsv(θ, 0.95, v(E)) used by the renderer */
export function computeRGBTargets(
  phases: Float32Array,
  energy: Float32Array,
  brightnessBase: number,
  energyGamma: number,
): Float32Array {
  const N = phases.length;
  const outDim = 3;
  const Y = new Float32Array(N * outDim);
  for (let i = 0; i < N; i++) {
    const hue = phases[i] / (2 * Math.PI);
    const sat = 0.95;
    const v = Math.max(0, Math.min(1, brightnessBase * (0.25 + 0.75 * Math.pow(energy[i], energyGamma))));
    const i6 = Math.floor(hue * 6);
    const f = hue * 6 - i6;
    const p = v * (1 - sat);
    const q = v * (1 - f * sat);
    const t = v * (1 - (1 - f) * sat);
    let R = 0, G = 0, B = 0;
    switch (i6 % 6) {
      case 0: R = v; G = t; B = p; break;
      case 1: R = q; G = v; B = p; break;
      case 2: R = p; G = v; B = t; break;
      case 3: R = p; G = q; B = v; break;
      case 4: R = t; G = p; B = v; break;
      case 5: R = v; G = p; B = q; break;
    }
    Y[i * outDim + 0] = R;
    Y[i * outDim + 1] = G;
    Y[i * outDim + 2] = B;
  }
  return Y;
}

/** Compute MSE between two same-shaped arrays */
export function mse(A: Float32Array, B: Float32Array): number {
  const n = Math.min(A.length, B.length);
  let se = 0;
  for (let i = 0; i < n; i++) {
    const d = A[i] - B[i];
    se += d * d;
  }
  return se / Math.max(1, n);
}

/** PSNR from MSE, assumes values in [0,1] by default */
export function psnrFromMSE(mseValue: number, maxVal = 1.0): number {
  if (mseValue <= 1e-12) return 99.0;
  return 10 * Math.log10((maxVal * maxVal) / mseValue);
}

/** Compute PAT reconstruction MSE from quantization distances (root squared distances) */
export function patReconMSEFromDists(dists: Float32Array, dim: number): number {
  if (dim <= 0) return 0;
  let sum = 0;
  for (let i = 0; i < dists.length; i++) {
    const sq = dists[i] * dists[i]; // bestD
    sum += sq / dim;
  }
  return sum / Math.max(1, dists.length);
}
