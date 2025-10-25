import type { SimulationState } from "./types";
import { computeMetrics } from "./eval";

/**
 * Affect header exposed to UI/logging
 */
export interface Affect {
  arousal: number; // [0,1]
  valence: number; // [0,1]
  // Flat headers (mirrored in metrics for back-compat)
  rhoMean: number;
  defectDensity: number;
  entropyRate: number;
  symmetry: number;
  chiralityMatch: number; // [-1,1]
  metrics: {
    rhoMean: number;
    defectDensity: number;
    entropyRate: number;
    symmetry: number;
    chiralityMatch: number; // [-1,1]
  };
}

type AffectWeights = {
  arousal: { r: number; e: number; h: number };
  valence: { r: number; def: number; h: number; sym: number; chi: number };
};

type AffectParams = {
  weights?: Partial<AffectWeights>;
  // Optional expected ranges for normalization
  ranges?: {
    energyMean?: [number, number];     // default [0.1, 1.0]
    entropyRate?: [number, number];    // default [0, π/2]
    defectDensity?: [number, number];  // default [0, 0.02]
  };
};

/**
 * Main entry: compute normalized arousal/valence and supporting metrics from the sim snapshot.
 * Deterministic, branch-free accumulation and fixed iteration order.
 */
export function computeAffect(sim: SimulationState, _attOut?: unknown, params?: AffectParams): Affect {
  const { energy, W, H } = sim;

  // Physics-aware metrics (ρ, defects, entropy-rate proxy)
  const m = computeMetrics(sim);
  const rhoMean = clamp01(m.rhoMean);
  const defectDensity = Math.max(0, m.defectDensity);
  const entropyRate = Math.max(0, m.entropyGrad);

  // Symmetry and chirality proxies (Phase-1: lightweight)
  const symmetry = symmetryScoreEnergy(energy, W, H);           // [0,1] high = more symmetric
  const chiralityMatch = chiralityFromDefects(m.defectsPlus, m.defectsMinus); // [-1,1]

  // Energy mean
  const eMean = meanArray(energy);

  // Normalization ranges (heuristics aligned to existing code)
  const rngEnergy = params?.ranges?.energyMean ?? [0.1, 1.0] as [number, number];
  const rngEntropy = params?.ranges?.entropyRate ?? [0, Math.PI / 2] as [number, number];
  const rngDefect = params?.ranges?.defectDensity ?? [0, 0.02] as [number, number];

  const nEnergy = norm01(eMean, rngEnergy[0], rngEnergy[1]);
  const nEntropy = norm01(entropyRate, rngEntropy[0], rngEntropy[1]);
  const nDefect = norm01(defectDensity, rngDefect[0], rngDefect[1]);

  // Weights (sensible defaults)
  const weights: AffectWeights = {
    arousal: { r: 0.4, e: 0.4, h: 0.2 },
    valence: { r: 0.5, def: 0.25, h: 0.15, sym: 0.05, chi: 0.05 },
  };
  if (params?.weights) {
    if (params.weights.arousal) Object.assign(weights.arousal, params.weights.arousal);
    if (params.weights.valence) Object.assign(weights.valence, params.weights.valence);
  }

  // Arousal: ↑ with energy, entropy-rate; ↓ with coherence (ρ)
  const arousalRaw =
    weights.arousal.e * nEnergy +
    weights.arousal.h * nEntropy +
    weights.arousal.r * (1 - rhoMean);

  // Valence: ↑ with coherence and symmetry; ↓ with defects/entropy-rate; ± with chirality match
  const valenceRaw =
    weights.valence.r * rhoMean +
    weights.valence.sym * symmetry +
    weights.valence.chi * (0.5 + 0.5 * chiralityMatch) + // map [-1,1]→[0,1]
    (-weights.valence.def) * nDefect +
    (-weights.valence.h) * nEntropy;

  const arousal = clamp01(arousalRaw);
  const valence = clamp01(valenceRaw);

  return {
    arousal,
    valence,
    rhoMean,
    defectDensity,
    entropyRate,
    symmetry,
    chiralityMatch,
    metrics: {
      rhoMean,
      defectDensity,
      entropyRate,
      symmetry,
      chiralityMatch,
    },
  };
}

/**
 * Simple horizontal-reflection and 90°-rotation symmetry proxy on the energy field.
 * Returns [0,1], higher = more symmetric.
 */
function symmetryScoreEnergy(E: Float32Array, W: number, H: number): number {
  // Use mean absolute difference normalized by dynamic scale.
  const N = W * H;
  if (N === 0) return 0;
  // Precompute a robust scale (mean absolute energy)
  let meanAbs = 0;
  for (let i = 0; i < N; i++) meanAbs += Math.abs(E[i]);
  meanAbs = Math.max(1e-6, meanAbs / N);

  // Horizontal reflection difference
  let diffRef = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const j = y * W + (W - 1 - x);
      diffRef += Math.abs(E[i] - E[j]);
    }
  }
  const normRef = diffRef / (N * meanAbs);

  // 90° rotation difference (only compare overlapping indices)
  let diffRot = 0;
  const M = Math.min(W, H);
  const areaRot = M * M;
  for (let y = 0; y < M; y++) {
    for (let x = 0; x < M; x++) {
      const i = y * W + x;
      const jx = y; // transpose index
      const jy = x;
      const j = jy * W + jx;
      diffRot += Math.abs(E[i] - E[j]);
    }
  }
  const normRot = areaRot > 0 ? diffRot / (areaRot * meanAbs) : 1;

  // Convert differences to similarity in [0,1], then average
  const sRef = 1 - clamp01(normRef / 2); // divide by 2 to soften
  const sRot = 1 - clamp01(normRot / 2);
  return clamp01(0.5 * (sRef + sRot));
}

/**
 * Net chirality estimate from defect counts. [-1,1]
 */
function chiralityFromDefects(plus: number, minus: number): number {
  const num = (plus | 0) - (minus | 0);
  const den = (plus | 0) + (minus | 0);
  if (den <= 0) return 0;
  const v = num / den;
  // clamp just in case
  return Math.max(-1, Math.min(1, v));
}

function meanArray(A: Float32Array): number {
  if (A.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < A.length; i++) s += A[i];
  return s / A.length;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function norm01(x: number, lo: number, hi: number): number {
  if (hi <= lo) return 0;
  return clamp01((x - lo) / (hi - lo));
}
