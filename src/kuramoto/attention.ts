import { clamp, TAU } from "./math";
import type { SimulationState } from "./types";

export type AttentionHead = {
  enabled: boolean;
  weight: number; // [0..1]
  freq: number; // Hz-equivalent for UI
  phase: number; // radians
  radius: number; // pixels (gaussian sigma approx)
  bindPrimaryPlane: boolean; // follow primary active plane centroid if available
};

export type AttentionParams = {
  // Diffusion/decay
  DA: number;
  DU: number;
  muA: number;
  muU: number;
  // Source gains
  lambdaS: number; // bottom-up salience -> A
  lambdaC: number; // context echo -> U
  topoGain: number; // vortices/disclinations -> (+A, -U)
  // Modulator gains into physics
  gammaK: number;
  betaK: number;
  gammaAlpha: number;
  betaAlpha: number;
  gammaD: number;
  deltaD: number;
  // Optional wind advection gain (guarded, default 0 via engine)
  etaV?: number;
  // Salience weights
  wGrad: number;
  wLap: number;
  wE: number;
  wDef: number;
  // Context blur radius (pixels)
  contextRadius: number;
  // Optional ReflectorGraph gain (0 = disabled)
  reflectorGain?: number;
  // Clamp for stability
  aClamp: number;
  uClamp: number;
};

export type AttentionOutputs = {
  A: Float32Array;
  U: Float32Array;
  Aact: Float32Array; // σ(A)
  Uact: Float32Array; // σ(U)
  lapA: Float32Array; // orientation-aware payload used by gammaAlpha
  divU: Float32Array; // ΔU (divergence proxy)
  lapAraw: Float32Array; // raw ΔA
  advect: Float32Array; // ∇A ⋅ ∇θ
};

const sigma = (x: number) => Math.tanh(x);

function laplacian5(field: Float32Array, W: number, H: number, wrap = true) {
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const ym = wrap ? (y - 1 + H) % H : Math.max(0, y - 1);
    const yp = wrap ? (y + 1) % H : Math.min(H - 1, y + 1);
    for (let x = 0; x < W; x++) {
      const xm = wrap ? (x - 1 + W) % W : Math.max(0, x - 1);
      const xp = wrap ? (x + 1) % W : Math.min(W - 1, x + 1);
      const i = y * W + x;
      const l = y * W + xm;
      const r = y * W + xp;
      const u = ym * W + x;
      const d = yp * W + x;
      out[i] = field[l] + field[r] + field[u] + field[d] - 4 * field[i];
    }
  }
  return out;
}

function boxBlurSeparable(src: Float32Array, W: number, H: number, radius: number, wrap = true) {
  const N = W * H;
  if (radius <= 0) return new Float32Array(src);
  const tmp = new Float32Array(N);
  const out = new Float32Array(N);
  const r = Math.floor(radius);
  const win = 2 * r + 1;

  // horizontal
  for (let y = 0; y < H; y++) {
    let acc = 0;
    for (let dx = -r; dx <= r; dx++) {
      const x0 = clamp(dx, 0, W - 1);
      const i0 = y * W + (wrap ? (dx + W) % W : x0);
      acc += src[i0];
    }
    for (let x = 0; x < W; x++) {
      const xl = x - r - 1;
      const xr = x + r;
      const il = y * W + (wrap ? (xl + W) % W : clamp(xl, 0, W - 1));
      const ir = y * W + (wrap ? (xr + W) % W : clamp(xr, 0, W - 1));
      if (x === 0) {
        tmp[y * W + x] = acc / win;
      } else {
        acc += src[ir] - src[il];
        tmp[y * W + x] = acc / win;
      }
    }
  }

  // vertical
  for (let x = 0; x < W; x++) {
    let acc = 0;
    for (let dy = -r; dy <= r; dy++) {
      const y0 = clamp(dy, 0, H - 1);
      const i0 = (wrap ? (dy + H) % H : y0) * W + x;
      acc += tmp[i0];
    }
    for (let y = 0; y < H; y++) {
      const yu = y - r - 1;
      const yd = y + r;
      const iu = (wrap ? (yu + H) % H : clamp(yu, 0, H - 1)) * W + x;
      const id = (wrap ? (yd + H) % H : clamp(yd, 0, H - 1)) * W + x;
      if (y === 0) {
        out[y * W + x] = acc / win;
      } else {
        acc += tmp[id] - tmp[iu];
        out[y * W + x] = acc / win;
      }
    }
  }

  return out;
}

function detectDefects(phases: Float32Array, W: number, H: number) {
  const sign = new Int8Array(W * H); // store at top-left of each plaquette
  const wrapd = (d: number) => {
    while (d > Math.PI) d -= TAU;
    while (d <= -Math.PI) d += TAU;
    return d;
  };
  for (let y = 0; y < H - 1; y++) {
    for (let x = 0; x < W - 1; x++) {
      const i00 = y * W + x;
      const i10 = y * W + (x + 1);
      const i11 = (y + 1) * W + (x + 1);
      const i01 = (y + 1) * W + x;
      const d1 = wrapd(phases[i10] - phases[i00]);
      const d2 = wrapd(phases[i11] - phases[i10]);
      const d3 = wrapd(phases[i01] - phases[i11]);
      const d4 = wrapd(phases[i00] - phases[i01]);
      const sum = d1 + d2 + d3 + d4;
      if (sum > Math.PI) sign[i00] = +1;
      else if (sum < -Math.PI) sign[i00] = -1;
      else sign[i00] = 0;
    }
  }
  return sign;
}

function gradPhaseMagnitude(phases: Float32Array, W: number, H: number, wrapEdges = true) {
  // Use wrapped differences for robust gradient magnitude of θ
  const out = new Float32Array(W * H);
  const wrapd = (d: number) => {
    while (d > Math.PI) d -= TAU;
    while (d <= -Math.PI) d += TAU;
    return d;
  };
  for (let y = 0; y < H; y++) {
    const ym = wrapEdges ? (y - 1 + H) % H : Math.max(0, y - 1);
    const yp = wrapEdges ? (y + 1) % H : Math.min(H - 1, y + 1);
    for (let x = 0; x < W; x++) {
      const xm = wrapEdges ? (x - 1 + W) % W : Math.max(0, x - 1);
      const xp = wrapEdges ? (x + 1) % W : Math.min(W - 1, x + 1);
      const i = y * W + x;
      const im = y * W + xm;
      const ip = y * W + xp;
      const jm = ym * W + x;
      const jp = yp * W + x;
      const gx = 0.5 * wrapd(phases[ip] - phases[im]);
      const gy = 0.5 * wrapd(phases[jp] - phases[jm]);
      out[i] = Math.hypot(gx, gy);
    }
  }
  return out;
}

function synthHeads(sim: SimulationState, heads: AttentionHead[], t: number, defaultRadius: number) {
  const { W, H } = sim;
  const out = new Float32Array(W * H);
  const primary = sim.activePlaneIds && sim.activePlaneIds.length > 0 ? sim.activePlaneIds[0] : null;

  for (const h of heads) {
    if (!h.enabled || h.weight <= 0) continue;
    let cx = Math.floor(W / 2);
    let cy = Math.floor(H / 2);
    if (h.bindPrimaryPlane && primary !== null) {
      const meta = sim.planeMeta.get(primary);
      if (meta) {
        cx = clamp(Math.round(meta.centroid.x), 0, W - 1);
        cy = clamp(Math.round(meta.centroid.y), 0, H - 1);
      }
    }
    const sigmaPx = Math.max(1, h.radius || defaultRadius);
    const twoSig2 = 2 * sigmaPx * sigmaPx;
    const osc = Math.cos(2 * Math.PI * h.freq * t + h.phase);
    const amp = h.weight * osc;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const i = y * W + x;
        const g = Math.exp(-(dx * dx + dy * dy) / twoSig2);
        out[i] += amp * g;
      }
    }
  }
  return out;
}

export function updateAttentionFields(
  sim: SimulationState,
  heads: AttentionHead[],
  p: AttentionParams,
  dt: number,
  t: number,
  wrapEdges: boolean
): AttentionOutputs {
  const { W, H, N, phases, energy } = sim;
  // Lazily attach A/U to sim instance
  if (!(sim as any)._A) (sim as any)._A = new Float32Array(N);
  if (!(sim as any)._U) (sim as any)._U = new Float32Array(N);
  const A = (sim as any)._A as Float32Array;
  const U = (sim as any)._U as Float32Array;

  // 1) Bottom-up salience
  const gmag = gradPhaseMagnitude(phases, W, H, wrapEdges);
  const lapTheta = laplacian5(phases, W, H, wrapEdges); // rough curvature proxy of phase
  const defects = detectDefects(phases, W, H);
  const S = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const sGrad = p.wGrad * gmag[i];
    const sLap = p.wLap * Math.abs(lapTheta[i]);
    const sE = p.wE * energy[i];
    const sDef = p.wDef * Math.abs(defects[i]);
    S[i] = sGrad + sLap + sE + sDef;
  }

  // 2) Head synthesis (adds to A)
  const headSrc = synthHeads(sim, heads, t, Math.max(6, Math.round(0.05 * Math.max(W, H))));

  // 2b) Optional: ReflectorGraph source (opt-in via p.reflectorGain)
  // If present, update candidates from salience S, run one power iteration,
  // and splat node activities back as a gentle A source.
  {
    const refl = (sim as any)._reflectorGraph;
    const rgain = p.reflectorGain ?? 0.0;
    if (refl && rgain !== 0) {
      try {
        refl.updateCandidates(S, 16, 6);
        refl.step(0.9, 8, 4);
        const aField: Float32Array = refl.splatToField(rgain, 6);
        for (let i = 0; i < N; i++) headSrc[i] += aField[i];
      } catch {
        // no-op: keep attention deterministic even if helper absent/mismatched
      }
    }
  }

  // 3) Context echo (broader blur of A)
  const Ablur = boxBlurSeparable(A, W, H, p.contextRadius, wrapEdges);

  // 4) Topological source
  const topo = new Float32Array(N);
  if (p.topoGain !== 0) {
    // smear defect signs lightly to reduce checkerboarding
    for (let y = 0; y < H - 1; y++) {
      for (let x = 0; x < W - 1; x++) {
        const i = y * W + x;
        const s = defects[i];
        if (s === 0) continue;
        const w = p.topoGain * s;
        topo[i] += w;
        topo[i + 1] += 0.5 * w;
        topo[i + W] += 0.5 * w;
        topo[i + W + 1] += 0.25 * w;
      }
    }
  }

  // 5) Diffuse/decay + sources
  const lapA = laplacian5(A, W, H, wrapEdges);
  const lapU = laplacian5(U, W, H, wrapEdges);
  for (let i = 0; i < N; i++) {
    const dA = p.DA * lapA[i] + p.lambdaS * S[i] + headSrc[i] + topo[i] - p.muA * A[i];
    const dU = p.DU * lapU[i] + p.lambdaC * Ablur[i] - topo[i] - p.muU * U[i];
    A[i] = clamp(A[i] + dt * dA, -p.aClamp, p.aClamp);
    U[i] = clamp(U[i] + dt * dU, -p.uClamp, p.uClamp);
  }

  // 6) Modulators (bounded activities and proxies)
  const Aact = new Float32Array(N);
  const Uact = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    Aact[i] = sigma(A[i]);
    Uact[i] = sigma(U[i]);
  }

  // Orientation-aware α-bias:
  // Build structure tensor J over the phase field using wrapped central differences,
  // smooth J with a small separable box blur, get v_max and coherence χ,
  // then compute ∇⊥A ⋅ v_max and use that as the "lapA" payload (drop-in replacement).
  const Ax = new Float32Array(N);
  const Ay = new Float32Array(N);
  const phix = new Float32Array(N);
  const phiy = new Float32Array(N);

  const wrapd = (d: number) => {
    while (d > Math.PI) d -= TAU;
    while (d <= -Math.PI) d += TAU;
    return d;
  };

  // central differences with periodic wrap
  for (let y = 0; y < H; y++) {
    const ym = wrapEdges ? (y - 1 + H) % H : Math.max(0, y - 1);
    const yp = wrapEdges ? (y + 1) % H : Math.min(H - 1, y + 1);
    for (let x = 0; x < W; x++) {
      const xm = wrapEdges ? (x - 1 + W) % W : Math.max(0, x - 1);
      const xp = wrapEdges ? (x + 1) % W : Math.min(W - 1, x + 1);
      const i = y * W + x;

      const iL = y * W + xm;
      const iR = y * W + xp;
      const iU = ym * W + x;
      const iD = yp * W + x;

      // ∇A
      Ax[i] = 0.5 * (A[iR] - A[iL]);
      Ay[i] = 0.5 * (A[iD] - A[iU]);

      // ∇θ with wrap
      const gx = 0.5 * wrapd(sim.phases[iR] - sim.phases[iL]);
      const gy = 0.5 * wrapd(sim.phases[iD] - sim.phases[iU]);
      phix[i] = gx;
      phiy[i] = gy;
    }
  }

  // Structure tensor elements (unsmoothed)
  const j11 = new Float32Array(N);
  const j22 = new Float32Array(N);
  const j12 = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const gx = phix[i];
    const gy = phiy[i];
    j11[i] = gx * gx;
    j22[i] = gy * gy;
    j12[i] = gx * gy;
  }

  // Smooth the tensor fields with a tiny blur (fast, deterministic)
  const j11b = boxBlurSeparable(j11, W, H, 1, wrapEdges);
  const j22b = boxBlurSeparable(j22, W, H, 1, wrapEdges);
  const j12b = boxBlurSeparable(j12, W, H, 1, wrapEdges);

  // Compute dominant eigenvector and coherence; then ∇⊥A ⋅ v_max
  const lapU2 = laplacian5(U, W, H, wrapEdges); // keep divergence proxy for U (ΔU)
  const lapA2 = new Float32Array(N);            // will carry χ · (∇⊥A ⋅ v_max)

  for (let i = 0; i < N; i++) {
    const a = j11b[i];
    const b = j12b[i];
    const c = j22b[i];

    const t = a + c;
    const d = Math.hypot(a - c, 2 * b);
    const lamMax = 0.5 * (t + d);
    const lamMin = 0.5 * (t - d);

    // Numeric safety near isotropy: robust eigenvector and coherence
    const vx0 = Math.abs(b) > 1e-8 ? (lamMax - c) : 1.0;
    const vy0 = Math.abs(b) > 1e-8 ? b : 0.0;
    const norm = Math.max(1e-8, Math.hypot(vx0, vy0));
    const vxn = vx0 / norm;
    const vyn = vy0 / norm;

    // coherence χ in [0,1] with denom floor; fade out in weak-orientation regions
    const denom = Math.max(lamMax + lamMin, 1e-8);
    const chi = Math.max(0, Math.min(1, (lamMax - lamMin) / denom));
    const chiEff = chi < 0.05 ? 0 : chi;

    // ∇⊥A = (-Ay, Ax); swirl = ∇⊥A ⋅ v_max
    const swirl = (-Ay[i]) * vxn + (Ax[i]) * vyn;

    // final orientation-aware payload (dimensionless): stepSimulation multiplies by gammaAlpha
    lapA2[i] = chiEff * swirl;
  }

  // Wind advection scalar: ∇A ⋅ ∇θ
  const advect = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    advect[i] = Ax[i] * phix[i] + Ay[i] * phiy[i];
  }

  return { A, U, Aact, Uact, lapA: lapA2, divU: lapU2, lapAraw: lapA, advect };
}
