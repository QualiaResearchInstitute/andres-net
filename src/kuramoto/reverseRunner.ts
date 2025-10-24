import { TAU } from "./math";
import type { SimulationState } from "./types";
import type { StepConfig } from "./stepSimulation";
import { buildPatchesFromNeighbors, parzenScore, reverseFlowStep } from "./reverseSampler";

/**
 * Compute Kuramoto drift f(theta,t) for the current sim state but using a provided theta field.
 * This mirrors the non-inplace branch in stepSimulation, without applying dt.
 */
export function computeDriftTheta(
  theta: Float32Array,
  sim: SimulationState,
  cfg: StepConfig,
): Float32Array {
  const {
    wrap,
    Kbase,
    K1,
    K2,
    K3,
    alphaSurfToField,
    alphaFieldToSurf,
    alphaHypToField,
    alphaFieldToHyp,
    swWeight,
    wallBarrier,
    horizonFactor,
  } = cfg;

  const {
    W,
    H,
    N,
    omegas,
    neighbors,
    neighborBands,
    swEdges,
    wall,
    ringOffsets,
    surfPhi,
    hypPhi,
    surfMask,
    hypMask,
  } = sim;

  const out = new Float32Array(N);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const th = theta[i];
      let sum = 0;

      const ns = neighbors[i] as Int32Array;
      const bands = neighborBands[i] as Uint8Array;
      const iIn = !!surfMask[i];

      for (let k = 0; k < ns.length; k++) {
        const j = ns[k];
        const band = bands[k];
        const w = band === 1 ? K1 : band === 2 ? K2 : K3;
        if (w === 0) continue;
        const jIn = !!surfMask[j];
        const barrierH = 1 - wallBarrier * (Math.max(wall[i], wall[j]) > 0 ? 1 : 0);
        const horizon = horizonFactor(iIn, jIn, iIn);
        sum += barrierH * horizon * w * Math.sin(theta[j] - th);
      }

      const sw = swEdges[i] as Array<[number, number]>;
      for (let k = 0; k < sw.length; k++) {
        const [j, sign] = sw[k];
        const w = swWeight * sign;
        const jIn = !!surfMask[j];
        const barrierH = 1 - wallBarrier * (Math.max(wall[i], wall[j]) > 0 ? 1 : 0);
        const horizon = horizonFactor(iIn, jIn, iIn);
        sum += barrierH * horizon * w * Math.sin(theta[j] - th);
      }

      let sumSurf = 0;
      if (surfMask[i]) sumSurf = alphaSurfToField * Math.sin(surfPhi[i] - th);
      let sumHyp = 0;
      if (hypMask[i]) sumHyp = alphaHypToField * Math.sin(hypPhi[i] - th);

      // dtheta/dt = omegas + couplings
      out[i] = omegas[i] + Kbase * sum + sumSurf + sumHyp;
    }
  }

  return out;
}

/**
 * Perform one explicit probability-flow ODE reverse step in-place on sim.phases
 * Uses a non-ML Parzen score (von-Mises KDE) over local patches.
 * D is derived from cfg.noiseAmp as a reasonable Phase-1 proxy.
 */
export function reverseStepInPlace(
  sim: SimulationState,
  cfg: StepConfig,
  t: number = 0.5,
) {
  const N = sim.N;
  const patches = buildPatchesFromNeighbors(sim);
  const Dconst = Math.max(1e-5, cfg.noiseAmp || 0.1);

  const drift = (th: Float32Array, _t: number) => computeDriftTheta(th, sim, cfg);
  const D = (_t: number) => Dconst;
  const score = (th: Float32Array, tt: number) => parzenScore(th, patches, tt);

  const next = reverseFlowStep(sim.phases, drift, D, score, t, cfg.dt);

  // normalize to [0, 2π)
  for (let i = 0; i < N; i++) {
    let v = next[i] % TAU;
    if (v < 0) v += TAU;
    sim.phases[i] = v;
  }
}
