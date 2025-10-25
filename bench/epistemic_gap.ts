/* eslint-disable no-console */
import { createSimulation, type SimulationInitConfig } from "../src/kuramoto/simulation";
import { stepSimulation, type StepConfig } from "../src/kuramoto/stepSimulation";
import { buildPATokens } from "../src/tokens/pat";
import { computeMetrics } from "../src/kuramoto/eval";

/**
 * Epistemic-gap benchmark (Phase-1 skeleton)
 * - Two regimes (labels): different coupling strengths induce distinct dynamics
 * - Feature sets:
 *   (A) Lattice: coarse physics metrics (rhoMean, defectDensity, entropyGrad, anisotropyHV)
 *   (B) Tokens: PAT rvq histogram + per-token stats means
 * - Train tiny logistic probes and report accuracies and wall-clock.
 *
 * This script is a standalone Node/TS script. You can run with:
 *   npx ts-node bench/epistemic_gap.ts
 * or compile with tsc and run node on the output.
 */

type Sample = { x: Float32Array; y: number };

function makeInitConfig(W = 24, H = 24, reseedGraphKey = 1000): SimulationInitConfig {
  return {
    W,
    H,
    wrap: true,
    omegaSpread: 0.25,
    swProb: 0.04,
    swEdgesPerNode: 2,
    swMinDist: 6,
    swMaxDist: 20,
    swNegFrac: 0.2,
    energyBaseline: 0.35,
    reseedGraphKey,
  };
}

function makeStepConfig(Kbase: number): StepConfig {
  return {
    dt: 0.03,
    wrap: true,
    // Couplings
    Kbase,
    K1: 1.0,
    K2: -0.3,
    K3: 0.15,
    KS1: 0.9,
    KS2: 0.0,
    KS3: 0.0,
    KH1: 0.7,
    KH2: 0.0,
    KH3: 0.0,
    // Cross-layer gains
    alphaSurfToField: 0.6,
    alphaFieldToSurf: 0.5,
    alphaHypToField: 0.8,
    alphaFieldToHyp: 0.4,
    // Small-world weight
    swWeight: 0.25,
    // Environment/energy
    wallBarrier: 0.5,
    emGain: 0.6,
    energyBaseline: 0.35,
    energyLeak: 0.02,
    energyDiff: 0.12,
    sinkLine: 0.03,
    sinkSurf: 0.02,
    sinkHyp: 0.02,
    trapSurf: 0.2,
    trapHyp: 0.35,
    minEnergySurf: 0.25,
    minEnergyHyp: 0.4,
    // Noise moderate for diversity
    noiseAmp: 0.15,
    // DAG off (non-inplace)
    dagSweeps: 0,
    dagDepthOrdering: true,
    dagDepthFiltering: true,
    dagLogStats: false,
    // No attention mods in this benchmark
    attentionMods: undefined,
    horizonFactor: () => 1.0,
  };
}

/** Lattice features: coarse physics metrics vector (length 4) */
function latticeFeatures(sim: ReturnType<typeof createSimulation>["simulation"]): Float32Array {
  const m = computeMetrics(sim);
  return new Float32Array([m.rhoMean, m.defectDensity, m.entropyGrad, m.anisotropyHV]);
}

/** Token features: rvq histogram (bins^dims) flattened + means of token stats */
function tokenFeatures(sim: ReturnType<typeof createSimulation>["simulation"]): Float32Array {
  const tokens = buildPATokens(sim, 32);
  // rvq is per-token array (length 5 with bins=4). Build histogram over tuples.
  const bins = 4;
  const dims = 5;
  const histSize = Math.pow(bins, dims);
  const hist = new Float32Array(histSize);
  const idxOf = (rvq: number[]) => {
    let idx = 0;
    for (let i = 0; i < dims; i++) idx = idx * bins + (rvq[i] ?? 0);
    return idx | 0;
  };
  let sumRho = 0, sumEnt = 0, sumAttnM = 0, sumAttnV = 0, n = 0;
  for (const t of tokens) {
    const rvq = t.code?.rvq ?? [0, 0, 0, 0, 0];
    const id = idxOf(rvq);
    if (id >= 0 && id < hist.length) hist[id] += 1;
    sumRho += t.stats.rho;
    sumEnt += t.stats.entropyRate;
    sumAttnM += t.stats.attnMean;
    sumAttnV += t.stats.attnVar;
    n++;
  }
  // Normalize histogram
  if (n > 0) {
    for (let i = 0; i < hist.length; i++) hist[i] /= n;
  }
  const means = new Float32Array([
    n > 0 ? sumRho / n : 0,
    n > 0 ? sumEnt / n : 0,
    n > 0 ? sumAttnM / n : 0,
    n > 0 ? sumAttnV / n : 0,
  ]);
  const out = new Float32Array(hist.length + means.length);
  out.set(hist, 0);
  out.set(means, hist.length);
  return out;
}

/** Tiny logistic regression (binary) */
function fitLogisticProbe(samples: Sample[], epochs = 200, lr = 0.5, l2 = 1e-4) {
  if (samples.length === 0) throw new Error("No samples");
  const d = samples[0].x.length;
  let W = new Float32Array(d);
  let b = 0;

  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

  for (let ep = 0; ep < epochs; ep++) {
    let dW = new Float32Array(d);
    let db = 0;
    for (const s of samples) {
      let z = b;
      const x = s.x;
      for (let i = 0; i < d; i++) z += W[i] * x[i];
      const p = sigmoid(z);
      const err = p - s.y; // derivative of log-loss
      for (let i = 0; i < d; i++) dW[i] += err * x[i];
      db += err;
    }
    // L2
    for (let i = 0; i < d; i++) dW[i] = dW[i] / samples.length + l2 * W[i];
    db = db / samples.length;
    for (let i = 0; i < d; i++) W[i] -= lr * dW[i];
    b -= lr * db;
  }

  const predict = (x: Float32Array) => {
    let z = b;
    for (let i = 0; i < x.length; i++) z += W[i] * x[i];
    return z >= 0 ? 1 : 0;
  };
  return { W, b, predict };
}

function accuracy(probe: { predict: (x: Float32Array) => number }, data: Sample[]) {
  let correct = 0;
  for (const s of data) {
    if (probe.predict(s.x) === s.y) correct++;
  }
  return correct / Math.max(1, data.length);
}

function buildDataset(nPerClass = 24) {
  const X_lat: Sample[] = [];
  const X_tok: Sample[] = [];

  // Class 0: lower coupling
  for (let k = 0; k < nPerClass; k++) {
    const sim = createSimulation(null, makeInitConfig(24, 24, 1000 + k)).simulation;
    const cfg = makeStepConfig(0.7);
    for (let s = 0; s < 8; s++) stepSimulation(sim, cfg);
    X_lat.push({ x: latticeFeatures(sim), y: 0 });
    X_tok.push({ x: tokenFeatures(sim), y: 0 });
  }

  // Class 1: higher coupling
  for (let k = 0; k < nPerClass; k++) {
    const sim = createSimulation(null, makeInitConfig(24, 24, 2000 + k)).simulation;
    const cfg = makeStepConfig(1.3);
    for (let s = 0; s < 8; s++) stepSimulation(sim, cfg);
    X_lat.push({ x: latticeFeatures(sim), y: 1 });
    X_tok.push({ x: tokenFeatures(sim), y: 1 });
  }

  return { X_lat, X_tok };
}

function main() {
  const t0 = Date.now();
  const { X_lat, X_tok } = buildDataset(24);

  const split = (arr: Sample[], frac = 0.7) => {
    const n = arr.length;
    const k = Math.max(1, Math.floor(frac * n));
    return { train: arr.slice(0, k), test: arr.slice(k) };
    // Deterministic ordering due to deterministic sims
  };

  const { train: Ltr, test: Lte } = split(X_lat);
  const { train: Ttr, test: Tte } = split(X_tok);

  const pLat = fitLogisticProbe(Ltr, 400, 0.5, 1e-3);
  const pTok = fitLogisticProbe(Ttr, 400, 0.5, 1e-3);

  const accLat = accuracy(pLat, Lte);
  const accTok = accuracy(pTok, Tte);
  const t1 = Date.now();

  console.log("Epistemic-gap probe results");
  console.log("  Lattice feature probe accuracy:", accLat.toFixed(3));
  console.log("  Token feature probe accuracy  :", accTok.toFixed(3));
  console.log("  Δ (token - lattice)           :", (accTok - accLat).toFixed(3));
  console.log("  Wall-clock (ms)               :", (t1 - t0));

  // Soft expectation: token features >= lattice features
  if (accTok + 1e-6 < accLat) {
    console.warn("Note: token probe underperformed lattice probe in this run. Consider increasing dataset size or steps.");
  }
}

if (require.main === module) {
  main();
}
