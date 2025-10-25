import type { SimulationState } from "./types";
import { stepSimulation, type StepConfig } from "./stepSimulation";
import { reverseStepInPlace } from "./reverseRunner";

export type Schedules = {
  K: (t: number) => number;
  Kref: (t: number) => number;
  D: (t: number) => number;
  psi: (t: number) => number; // global reference phase
};

/** Wrap angle to (-π, π] */
export function wrapAngle(x: number): number {
  let y = (x + Math.PI) % (2 * Math.PI);
  if (y < 0) y += 2 * Math.PI;
  return y - Math.PI;
}

/** Forward Kuramoto step (SDE/ODE variant) */
export function forwardStep(
  theta: Float32Array,
  omega: Float32Array,
  alpha: Float32Array,
  nbrIdx: Int32Array[],
  w: Float32Array[],
  t: number,
  dt: number,
  sched: Schedules,
  u: Float32Array | null,
  noise?: Float32Array | null
): Float32Array {
  const N = theta.length;
  const out = new Float32Array(N);
  const Kt = sched.K(t);
  const Kref = sched.Kref(t);
  const Dt = sched.D(t);
  const psi = sched.psi(t);

  for (let i = 0; i < N; i++) {
    let drift = omega[i];
    const ni = nbrIdx[i];
    const wi = w[i];
    for (let k = 0; k < ni.length; k++) {
      const j = ni[k];
      drift += Kt * wi[k] * Math.sin(theta[j] - theta[i] - alpha[i]);
    }
    drift += Kref * Math.sin(psi - theta[i]);
    if (u) drift += u[i];
    let dth = drift * dt;
    if (noise) dth += Math.sqrt(2 * Dt * dt) * noise[i];
    out[i] = wrapAngle(theta[i] + dth);
  }
  return out;
}

/** Example schedule factory */
export function runForward(
  sim: SimulationState,
  buildCfg: () => StepConfig,
  steps: number,
  sched?: { K: (t: number) => number; D: (t: number) => number },
  t0 = 0,
  t1 = 1,
) {
  const n = Math.max(1, steps | 0);
  for (let s = 0; s < n; s++) {
    const tau = n === 1 ? t1 : t0 + (t1 - t0) * (s / Math.max(1, n - 1));
    const cfg = buildCfg();
    if (sched) {
      cfg.Kbase = sched.K(tau);
      cfg.noiseAmp = Math.max(0, sched.D(tau));
    }
    stepSimulation(sim, cfg);
  }
}

export function runReverse(
  sim: SimulationState,
  buildCfg: () => StepConfig,
  steps: number,
  t: number = 0.5,
  opts?: {
    sched?: { K: (t: number) => number; D: (t: number) => number; Kref?: (t: number) => number; psi?: (t: number) => number };
    t0?: number;
    t1?: number;
  },
) {
  const n = Math.max(1, steps | 0);
  for (let s = 0; s < n; s++) {
    const tau = opts?.sched
      ? (n === 1 ? (opts.t1 ?? 1) : (opts.t0 ?? 0) + ((opts.t1 ?? 1) - (opts.t0 ?? 0)) * (s / Math.max(1, n - 1)))
      : t;
    const cfg = buildCfg();
    if (opts?.sched) {
      cfg.Kbase = opts.sched.K(tau);
    }
    reverseStepInPlace(sim, cfg, tau, opts?.sched ? { D: opts.sched.D } : undefined);
  }
}

export function runFwdRev(
  sim: SimulationState,
  buildCfg: () => StepConfig,
  fwd: { steps: number; sched?: { K: (t: number) => number; D: (t: number) => number } },
  rev: { steps: number; t?: number; sched?: { K: (t: number) => number; D: (t: number) => number; Kref?: (t: number) => number; psi?: (t: number) => number } },
) {
  runForward(sim, buildCfg, fwd.steps, fwd.sched);
  const rs = rev.sched ?? fwd.sched;
  if (rs) {
    runReverse(sim, buildCfg, rev.steps, rev.t ?? 0.5, { sched: rs, t0: 1, t1: 0 });
  } else {
    runReverse(sim, buildCfg, rev.steps, rev.t ?? 0.5);
  }
}

export function makeSchedules(K0 = 1, Kref0 = 0.5, D0 = 0.1, D1 = 0.5) {
  return {
    K: (t: number) => K0 * Math.pow(1 - t, 2),
    Kref: (t: number) => Kref0 * Math.pow(1 - t, 2),
    D: (t: number) => D0 + (D1 - D0) * Math.pow(t, 2),
    psi: (t: number) => 0.0,
  } as Schedules;
}
