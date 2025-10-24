import { clamp, idx, TAU } from "./math";
import { ensureDagCache, nextNoise } from "./simulation";
import { SimulationState } from "./types";

export type AttentionMods = {
  Aact: Float32Array;
  Uact: Float32Array;
  lapA: Float32Array;
  divU: Float32Array;
  gammaK: number;
  betaK: number;
  gammaAlpha: number;
  betaAlpha: number;
  gammaD: number;
  deltaD: number;
};

export type StepConfig = {
  dt: number;
  wrap: boolean;
  Kbase: number;
  K1: number;
  K2: number;
  K3: number;
  KS1: number;
  KS2: number;
  KS3: number;
  KH1: number;
  KH2: number;
  KH3: number;
  alphaSurfToField: number;
  alphaFieldToSurf: number;
  alphaHypToField: number;
  alphaFieldToHyp: number;
  swWeight: number;
  wallBarrier: number;
  emGain: number;
  energyBaseline: number;
  energyLeak: number;
  energyDiff: number;
  sinkLine: number;
  sinkSurf: number;
  sinkHyp: number;
  trapSurf: number;
  trapHyp: number;
  minEnergySurf: number;
  minEnergyHyp: number;
  noiseAmp: number;
  dagSweeps: number;
  dagDepthOrdering: boolean;
  dagDepthFiltering: boolean;
  dagLogStats: boolean;
  attentionMods?: AttentionMods;
  horizonFactor: (iInside: boolean, jInside: boolean, receiverInside: boolean) => number;
};

export function stepSimulation(sim: SimulationState, config: StepConfig) {
  const {
    dt,
    wrap,
    Kbase,
    K1,
    K2,
    K3,
    KS1,
    KS2,
    KS3,
    KH1,
    KH2,
    KH3,
    alphaSurfToField,
    alphaFieldToSurf,
    alphaHypToField,
    alphaFieldToHyp,
    swWeight,
    wallBarrier,
    emGain,
    energyBaseline,
    energyLeak,
    energyDiff,
    sinkLine,
    sinkSurf,
    sinkHyp,
    trapSurf,
    trapHyp,
    minEnergySurf,
    minEnergyHyp,
    noiseAmp,
    dagSweeps,
    dagDepthOrdering,
    dagDepthFiltering,
    dagLogStats,
    horizonFactor,
    attentionMods,
  } = config;

  const {
    W,
    H,
    N,
    phases,
    omegas,
    neighbors,
    neighborBands,
    swEdges,
    wall,
    pot,
    planeDepth,
    planeMeta,
    nextPhases,
    ringOffsets,
    surfPhi,
    surfOmega,
    hypPhi,
    hypOmega,
    surfMask,
    hypMask,
    energy,
    nextEnergy,
    dag,
  } = sim;

  const activePlaneSet = sim.activePlaneSet ?? new Set<number>();
  planeMeta.forEach((meta, planeId) => {
    if (activePlaneSet.size > 0 && !activePlaneSet.has(planeId)) {
      meta.R = 0;
      meta.psi = 0;
      return;
    }
    let cr = 0,
      ci = 0;
    const cells = meta.cells;
    const L = cells.length;
    for (let t = 0; t < L; t++) {
      const th = phases[cells[t]];
      cr += Math.cos(th);
      ci += Math.sin(th);
    }
    const R = L > 0 ? Math.hypot(cr, ci) / L : 0;
    const psi = Math.atan2(ci, cr);
    meta.R = R;
    meta.psi = psi;
  });

  const nextSurf = new Float32Array(N);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const phi = surfPhi[i];
      let dphi = surfOmega[i];
      if (surfMask[i]) {
        for (let r = 1; r <= 3; r++) {
          const weightR = r === 1 ? KS1 : r === 2 ? KS2 : KS3;
          if (weightR === 0) continue;
          const off = ringOffsets[r];
          for (let t = 0; t < off.length; t++) {
            const [dx, dy] = off[t];
            const j = idx(x + dx, y + dy, W, H, wrap);
            if (j >= 0 && surfMask[j]) dphi += weightR * Math.sin(surfPhi[j] - phi);
          }
        }
        dphi += alphaFieldToSurf * Math.sin(phases[i] - phi);
      } else {
        dphi += 0.4 * Math.sin(phases[i] - phi);
      }
      nextSurf[i] = (phi + dphi * dt) % TAU;
      if (nextSurf[i] < 0) nextSurf[i] += TAU;
    }
  }
  surfPhi.set(nextSurf);

  const nextHyp = new Float32Array(N);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const phi = hypPhi[i];
      let dphi = hypOmega[i];
      if (hypMask[i]) {
        for (let r = 1; r <= 3; r++) {
          const weightR = r === 1 ? KH1 : r === 2 ? KH2 : KH3;
          if (weightR === 0) continue;
          const off = ringOffsets[r];
          for (let t = 0; t < off.length; t++) {
            const [dx, dy] = off[t];
            const j = idx(x + dx, y + dy, W, H, wrap);
            if (j >= 0 && hypMask[j]) dphi += weightR * Math.sin(hypPhi[j] - phi);
          }
        }
        dphi += alphaFieldToHyp * Math.sin(phases[i] - phi);
      } else {
        dphi += 0.3 * Math.sin(phases[i] - phi);
      }
      nextHyp[i] = (phi + dphi * dt) % TAU;
      if (nextHyp[i] < 0) nextHyp[i] += TAU;
    }
  }
  hypPhi.set(nextHyp);

  const doInPlace = dagSweeps > 0;
  const depthOrdered = doInPlace && dagDepthOrdering;
  const depthFilterEnabled = depthOrdered && dagDepthFiltering;

  if (!doInPlace) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const th = phases[i];
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
          {
            let wEff = w;
            let alphaBias = 0;
            if (attentionMods) {
              const { Aact, Uact, lapA, divU, gammaK, betaK, gammaAlpha, betaAlpha } = attentionMods;
              wEff = w * (1 + gammaK * Aact[i] * Aact[j] - betaK * Uact[i] * Uact[j]);
              alphaBias = gammaAlpha * lapA[i] - betaAlpha * divU[i];
            }
            sum += barrierH * horizon * wEff * Math.sin(phases[j] - th - alphaBias);
          }
        }
        const sw = swEdges[i] as Array<[number, number]>;
        for (let k = 0; k < sw.length; k++) {
          const [j, sign] = sw[k];
          const w = swWeight * sign;
          const jIn = !!surfMask[j];
          const barrierH = 1 - wallBarrier * (Math.max(wall[i], wall[j]) > 0 ? 1 : 0);
          const horizon = horizonFactor(iIn, jIn, iIn);
          {
            let wEff = w;
            let alphaBias = 0;
            if (attentionMods) {
              const { Aact, Uact, lapA, divU, gammaK, betaK, gammaAlpha, betaAlpha } = attentionMods;
              wEff = w * (1 + gammaK * Aact[i] * Aact[j] - betaK * Uact[i] * Uact[j]);
              alphaBias = gammaAlpha * lapA[i] - betaAlpha * divU[i];
            }
            sum += barrierH * horizon * wEff * Math.sin(phases[j] - th - alphaBias);
          }
        }
        let sumSurf = 0;
        if (surfMask[i]) sumSurf = alphaSurfToField * Math.sin(surfPhi[i] - th);
        let sumHyp = 0;
        if (hypMask[i]) sumHyp = alphaHypToField * Math.sin(hypPhi[i] - th);
        const dtheta = (omegas[i] + Kbase * sum + sumSurf + sumHyp) * dt;
        nextPhases[i] = (th + dtheta) % TAU;
        if (nextPhases[i] < 0) nextPhases[i] += TAU;
      }
    }
    phases.set(nextPhases);
  } else {
    const dagCache = ensureDagCache(sim);
    if (dagCache.dirty) {
      for (let i = 0; i < N; i++) dagCache.snapshot[i] = phases[i];
      dagCache.stats = { step: 0, filtered: 0, used: 0, visited: 0, sweeps: 0 };
      dagCache.dirty = false;
    }
    const snapshot = dagCache.snapshot;
    const layers = dagCache.layers;
    for (let sweep = 0; sweep < dagSweeps; sweep++) {
      for (let depth = 0; depth < layers.length; depth++) {
        const layer = layers[depth];
        const depthMask = depthFilterEnabled ? depth : -1;
        for (let idxLayer = 0; idxLayer < layer.length; idxLayer++) {
          const i = layer[idxLayer];
          const th = phases[i];
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
            const source = depthOrdered ? (depthMask >= 0 ? snapshot[j] : phases[j]) : phases[j];
            {
              let wEff = w;
              let alphaBias = 0;
              if (attentionMods) {
                const { Aact, Uact, lapA, divU, gammaK, betaK, gammaAlpha, betaAlpha } = attentionMods;
                wEff = w * (1 + gammaK * Aact[i] * Aact[j] - betaK * Uact[i] * Uact[j]);
                alphaBias = gammaAlpha * lapA[i] - betaAlpha * divU[i];
              }
              sum += barrierH * horizon * wEff * Math.sin(source - th - alphaBias);
            }
          }
          const sw = swEdges[i] as Array<[number, number]>;
          for (let k = 0; k < sw.length; k++) {
            const [j, sign] = sw[k];
            const w = swWeight * sign;
            const jIn = !!surfMask[j];
            const barrierH = 1 - wallBarrier * (Math.max(wall[i], wall[j]) > 0 ? 1 : 0);
            const horizon = horizonFactor(iIn, jIn, iIn);
            const source = depthOrdered ? (depthMask >= 0 ? snapshot[j] : phases[j]) : phases[j];
            {
              let wEff = w;
              let alphaBias = 0;
              if (attentionMods) {
                const { Aact, Uact, lapA, divU, gammaK, betaK, gammaAlpha, betaAlpha } = attentionMods;
                wEff = w * (1 + gammaK * Aact[i] * Aact[j] - betaK * Uact[i] * Uact[j]);
                alphaBias = gammaAlpha * lapA[i] - betaAlpha * divU[i];
              }
              sum += barrierH * horizon * wEff * Math.sin(source - th - alphaBias);
            }
          }
          let sumSurf = 0;
          if (surfMask[i]) sumSurf = alphaSurfToField * Math.sin(surfPhi[i] - th);
          let sumHyp = 0;
          if (hypMask[i]) sumHyp = alphaHypToField * Math.sin(hypPhi[i] - th);
          const dtheta = (omegas[i] + Kbase * sum + sumSurf + sumHyp) * dt;
          phases[i] = (th + dtheta) % TAU;
          if (phases[i] < 0) phases[i] += TAU;
        }
      }
    }
  }

  if (noiseAmp > 0) {
    const mods = attentionMods;
    for (let i = 0; i < N; i++) {
      const rnd = nextNoise(sim) * 2 - 1;
      let amp = noiseAmp;
      if (mods) {
        amp = clamp(
          noiseAmp * (1 - mods.gammaD * (mods.Aact[i] ?? 0)) + mods.deltaD * (mods.Uact[i] ?? 0),
          0,
          1.5 * Math.max(1e-6, noiseAmp)
        );
      }
      phases[i] = (phases[i] + rnd * amp * dt) % TAU;
      if (phases[i] < 0) phases[i] += TAU;
    }
  }

  for (let i = 0; i < N; i++) {
    const surf = surfPhi[i];
    const hyp = hypPhi[i];
    const surfDiff = Math.sin(surf - phases[i]);
    const hypDiff = Math.sin(hyp - phases[i]);
    const sink = sinkLine + (surfMask[i] ? sinkSurf : 0) + (hypMask[i] ? sinkHyp : 0);
    const trap = (surfMask[i] ? trapSurf : 0) + (hypMask[i] ? trapHyp : 0);
    const minEnergy = surfMask[i] ? minEnergySurf : hypMask[i] ? minEnergyHyp : 0;
    let nextE =
      energy[i] +
      dt *
        (energyBaseline +
          emGain * pot[i] +
          energyDiff * (surfDiff + 0.5 * hypDiff) -
          sink * energy[i] +
          trap * (Math.max(minEnergy, energy[i]) - energy[i]));
    nextE = clamp(nextE, minEnergy, 4);
    nextEnergy[i] = nextE;
  }
  const tmp = sim.energy;
  sim.energy = sim.nextEnergy;
  sim.nextEnergy = tmp;

  if (dagLogStats && dag) {
    dag.stats.step++;
  }
}
