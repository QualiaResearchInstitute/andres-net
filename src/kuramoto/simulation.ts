import { clamp, idx, seedPRNG, TAU } from "./math";
import { PlaneMetadata, SimulationState } from "./types";

export type SimulationInitConfig = {
  W: number;
  H: number;
  wrap: boolean;
  omegaSpread: number;
  swProb: number;
  swEdgesPerNode: number;
  swMinDist: number;
  swMaxDist: number;
  swNegFrac: number;
  energyBaseline: number;
  reseedGraphKey: number;
};

export type SimulationInitResult = {
  simulation: SimulationState;
  shouldResetDrawing: boolean;
};

export function createSimulation(prev: SimulationState | null, config: SimulationInitConfig): SimulationInitResult {
  const {
    W,
    H,
    wrap,
    omegaSpread,
    swProb,
    swEdgesPerNode,
    swMinDist,
    swMaxDist,
    swNegFrac,
    energyBaseline,
    reseedGraphKey,
  } = config;

  const N = W * H;
  const prevKey = prev?.reseedKey ?? -1;
  const fullReset = !prev || prev.W !== W || prev.H !== H || prevKey !== reseedGraphKey;
  const rng = seedPRNG(12345 + W * 31 + H * 101 + reseedGraphKey * 997);

  const phases = new Float32Array(N);
  if (prev && prev.N === N && !fullReset) {
    phases.set(prev.phases);
  } else {
    for (let i = 0; i < N; i++) phases[i] = rng() * TAU;
  }

  const omegaSeeds = new Float32Array(N);
  if (prev && prev.N === N && !fullReset && prev.omegaSeeds) {
    omegaSeeds.set(prev.omegaSeeds);
  } else {
    for (let i = 0; i < N; i++) omegaSeeds[i] = rng() * 2 - 1;
  }

  const omegas = new Float32Array(N);
  for (let i = 0; i < N; i++) omegas[i] = omegaSeeds[i] * omegaSpread;

  const ringOffsets: Array<Array<[number, number]>> = [[], [], [], []];
  for (let r = 1; r <= 3; r++) {
    const off: Array<[number, number]> = [];
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx === 0 && dy === 0) continue;
        const d = Math.max(Math.abs(dx), Math.abs(dy));
        if (d === r) off.push([dx, dy]);
      }
    }
    ringOffsets[r] = off;
  }

  const neighbors: Array<Int32Array> = new Array(N);
  const neighborBands: Array<Uint8Array> = new Array(N);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const ns: number[] = [];
      const rs: number[] = [];
      for (let r = 1; r <= 3; r++) {
        const off = ringOffsets[r];
        for (let t = 0; t < off.length; t++) {
          const [dx, dy] = off[t];
          const j = idx(x + dx, y + dy, W, H, wrap);
          if (j >= 0) {
            ns.push(j);
            rs.push(r);
          }
        }
      }
      neighbors[i] = Int32Array.from(ns);
      neighborBands[i] = Uint8Array.from(rs);
    }
  }

  const swEdges = new Array(N);
  for (let i = 0; i < N; i++) swEdges[i] = [] as Array<[number, number]>;
  const inRange = (a: { x: number; y: number }, b: { x: number; y: number }) => {
    const dx = Math.abs(a.x - b.x);
    const dy = Math.abs(a.y - b.y);
    const d = Math.sqrt(dx * dx + dy * dy);
    return d >= swMinDist && d <= swMaxDist;
  };
  const pick = () => ({ x: Math.floor(rng() * W), y: Math.floor(rng() * H) });
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (rng() < swProb) {
        for (let e = 0; e < swEdgesPerNode; e++) {
          let tries = 0;
          while (tries++ < 50) {
            const q = pick();
            if (inRange({ x, y }, q)) {
              const j = idx(q.x, q.y, W, H, wrap);
              const sign = rng() < swNegFrac ? -1 : 1;
              swEdges[i].push([j, sign]);
              break;
            }
          }
        }
      }
    }
  }

  const wall = new Float32Array(N);
  const pot = new Float32Array(N);
  const planeDepth = new Uint8Array(N);
  const energy = new Float32Array(N);
  const nextEnergy = new Float32Array(N);
  if (prev && prev.N === N && !fullReset) {
    wall.set(prev.wall);
    pot.set(prev.pot);
    planeDepth.set(prev.planeDepth);
    energy.set(prev.energy);
    nextEnergy.set(prev.nextEnergy);
  } else {
    energy.fill(energyBaseline);
  }

  const surfPhi = new Float32Array(N);
  const surfOmega = new Float32Array(N);
  const hypPhi = new Float32Array(N);
  const hypOmega = new Float32Array(N);
  if (prev && prev.N === N && !fullReset) {
    surfPhi.set(prev.surfPhi);
    hypPhi.set(prev.hypPhi);
    surfOmega.set(prev.surfOmega);
    hypOmega.set(prev.hypOmega);
  } else {
    for (let i = 0; i < N; i++) {
      surfPhi[i] = phases[i];
      hypPhi[i] = phases[i];
      surfOmega[i] = (rng() * 2 - 1) * 0.1;
      hypOmega[i] = (rng() * 2 - 1) * 0.1;
    }
  }

  const surfMask = new Uint8Array(N);
  const hypMask = new Uint8Array(N);
  if (prev && prev.N === N && !fullReset) {
    surfMask.set(prev.surfMask);
    hypMask.set(prev.hypMask);
  }

  const planeMeta =
    prev && prev.N === N && !fullReset
      ? prev.planeMeta
      : new Map<number, PlaneMetadata>();

  const nextPhases = new Float32Array(N);

  const simulation: SimulationState = {
    W,
    H,
    N,
    phases,
    omegaSeeds,
    omegas,
    neighbors,
    neighborBands,
    swEdges,
    wall,
    pot,
    planeDepth,
    planeMeta,
    nextPhases,
    rng,
    ringOffsets,
    surfPhi,
    surfOmega,
    hypPhi,
    hypOmega,
    surfMask,
    hypMask,
    energy,
    nextEnergy,
    dag: {
      dirty: true,
      layers: [],
      maxDepth: 0,
      snapshot: new Float32Array(N),
      stats: { step: 0, filtered: 0, used: 0, visited: 0, sweeps: 0 },
    },
    reseedKey: reseedGraphKey,
    activePlaneIds:
      prev && prev.N === N && !fullReset && prev.activePlaneIds
        ? [...prev.activePlaneIds]
        : [],
    activePlaneSet:
      prev && prev.N === N && !fullReset && prev.activePlaneIds
        ? new Set(prev.activePlaneIds)
        : new Set<number>(),
  };

  const shouldResetDrawing = !prev || fullReset || prev.N !== N;

  return { simulation, shouldResetDrawing };
}

export function updateOmegaSpread(sim: SimulationState, omegaSpread: number) {
  const { omegas, omegaSeeds } = sim;
  for (let i = 0; i < omegas.length; i++) {
    omegas[i] = omegaSeeds[i] * omegaSpread;
  }
}

export function markDagDirty(sim: SimulationState | null) {
  if (sim && sim.dag) {
    sim.dag.dirty = true;
  }
}

export function ensureDagCache(sim: SimulationState) {
  const dag = sim.dag;
  if (!dag.snapshot || dag.snapshot.length !== sim.N) {
    dag.snapshot = new Float32Array(sim.N);
    dag.dirty = true;
  }
  if (dag.dirty) {
    const { planeDepth, N } = sim;
    let maxDepth = 0;
    for (let i = 0; i < N; i++) maxDepth = Math.max(maxDepth, planeDepth[i]);
    const counts = new Array(maxDepth + 1).fill(0);
    for (let i = 0; i < N; i++) counts[planeDepth[i]]++;
    const layers: Array<Int32Array> = new Array(maxDepth + 1);
    for (let d = 0; d <= maxDepth; d++) layers[d] = new Int32Array(counts[d]);
    counts.fill(0);
    for (let i = 0; i < N; i++) {
      const depth = planeDepth[i];
      layers[depth][counts[depth]++] = i;
    }
    dag.layers = layers;
    dag.maxDepth = maxDepth;
    dag.dirty = false;
  }
  return dag;
}

export function updateMasks(sim: SimulationState) {
  const { N, planeDepth, surfMask, hypMask } = sim;
  for (let i = 0; i < N; i++) {
    const depth = planeDepth[i];
    surfMask[i] = depth >= 1 ? 1 : 0;
    hypMask[i] = depth >= 2 ? 1 : 0;
  }
}

export function recomputePotential(sim: SimulationState, radius: number) {
  const { W, H, N, wall, pot } = sim;
  pot.fill(0);
  for (let y = 0; y < H; y++) {
    let acc = 0;
    const row = y * W;
    for (let x = 0; x < W; x++) {
      if (x === 0) {
        acc = 0;
        for (let dx = -radius; dx <= radius; dx++) {
          acc += wall[row + clamp(x + dx, 0, W - 1)];
        }
      } else {
        const prev = clamp(x - radius - 1, 0, W - 1);
        const next = clamp(x + radius, 0, W - 1);
        acc += wall[row + next] - wall[row + prev];
      }
      pot[row + x] = acc / (2 * radius + 1);
    }
  }

  const temp = new Float32Array(N);
  for (let x = 0; x < W; x++) {
    let acc = 0;
    for (let y = 0; y < H; y++) {
      if (y === 0) {
        acc = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          acc += pot[clamp(y + dy, 0, H - 1) * W + x];
        }
      } else {
        const prev = clamp(y - radius - 1, 0, H - 1);
        const next = clamp(y + radius, 0, H - 1);
        acc += pot[next * W + x] - pot[prev * W + x];
      }
      temp[y * W + x] = acc / (2 * radius + 1);
    }
  }
  let maxv = 1e-6;
  for (let i = 0; i < N; i++) maxv = Math.max(maxv, temp[i]);
  for (let i = 0; i < N; i++) pot[i] = temp[i] / maxv;
}

export function resetPhases(sim: SimulationState) {
  for (let i = 0; i < sim.N; i++) {
    sim.phases[i] = sim.rng() * TAU;
  }
}

export function clearWalls(sim: SimulationState) {
  sim.wall.fill(0);
  sim.pot.fill(0);
}
