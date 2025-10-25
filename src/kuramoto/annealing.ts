import type { SimulationState } from "./types";

/**
 * Dissonance field D = w.grad*|∇phi| + w.vort*[vortex] + w.fr*[frustration]
 * All operations are deterministic, single pass, fixed order.
 */
export function computeDissonance(
  gradPhi: Float32Array,
  vort: Uint8Array,
  frustration: Uint8Array,
  w: { grad: number; vort: number; fr: number },
): Float32Array {
  const N = gradPhi.length | 0;
  const out = new Float32Array(N);
  const wg = w.grad ?? 0;
  const wv = w.vort ?? 0;
  const wf = w.fr ?? 0;
  for (let i = 0; i < N; i++) {
    const g = gradPhi[i] ?? 0;
    const v = vort[i] ? 1 : 0;
    const f = frustration[i] ? 1 : 0;
    out[i] = wg * g + wv * v + wf * f;
  }
  return out;
}

/**
 * Apply anneal schedule modifiers to base K (coupling) and D (noise amplitude).
 * K' = K * (1 - g.k * T)
 * D' = D + g.d * T
 * Clamped to non-negative ranges deterministically.
 */
export function applyAnnealModifiers(
  baseK: number,
  baseD: number,
  T: number,
  g: { k: number; d: number },
): { K: number; D: number } {
  const K = Math.max(0, baseK * (1 - (g.k ?? 0) * T));
  const D = Math.max(0, baseD + (g.d ?? 0) * T);
  return { K, D };
}

/**
 * Deterministic frustration update.
 * When T is high and local dissonance remains above threshold, flip a local frustration flag.
 * Implementation is intentionally conservative and deterministic:
 * - Fixed scan order 0..N-1
 * - Flip at every Qth index to avoid cascading, where Q is derived from T quantization
 * - No randomness; no new RNG streams are created
 *
 * Returns a new Uint8Array of frustration flags (0/1).
 */
export function updateFrustration(
  sim: SimulationState,
  T: number,
  Dfield: Float32Array,
  releaseThresh: number,
  prev?: Uint8Array,
): Uint8Array {
  const N = sim.W * sim.H;
  const out = new Uint8Array(N);
  if (prev) out.set(prev);
  if (!Number.isFinite(T) || T <= 0) return out;
  // Quantize T deterministically to an integer in [1,8]
  const qT = Math.min(8, Math.max(1, Math.floor(T * 8 + 1e-6)));
  // Flip every qT-th index where dissonance is above threshold
  for (let i = 0; i < N; i++) {
    if (Dfield[i] > releaseThresh && (i % qT) === 0) {
      out[i] = 0; // release (unclench)
    }
  }
  return out;
}
