export interface ControlSchedule {
  seed: number;
  steps: number; // deterministic replay length
  // low-dim gains for nudging sim params (placeholders for Phase-2 wiring)
  K_gain: number;
  alpha_gain: number;
  noise_gain: number;
  smallWorldP: number;
  wallGain: number;
  horizon: "in" | "out" | "none";
  // affect targets (arousal/valence)
  targetArousal?: number;
  targetValence?: number;
}

export type ControllerOutput = {
  theta0: Float32Array;
  u: Float32Array; // per-node external control (v1 zeros)
  sched: ControlSchedule;
};

/**
 * h_phi: deterministic controller stub
 * - encodes prompt + knobs to (theta0, u(t), sched)
 * - Phase-1: rule-based presets, zeros for u, identity for theta0
 */
export function h_phi(
  prompt: string,
  knobs: Record<string, number>,
  opts?: { N?: number; seed?: number; defaultSteps?: number },
): ControllerOutput {
  const N = Math.max(0, Math.floor(opts?.N ?? 0));
  const theta0 = new Float32Array(N);
  const u = new Float32Array(N); // zeros by default
  const seed = (opts?.seed ?? 1234) | 0;
  const steps = Math.max(1, Math.floor(opts?.defaultSteps ?? 256));

  // Simple prompt-based preset (expand later)
  const sentiment = classifyPrompt(prompt);
  const sched: ControlSchedule = {
    seed,
    steps,
    K_gain: sentiment === "calm" ? 0.8 : sentiment === "tense" ? 1.2 : 1.0,
    alpha_gain: sentiment === "calm" ? 0.6 : 0.9,
    noise_gain: sentiment === "tense" ? 0.35 : 0.15,
    smallWorldP: knobs.smallWorldP ?? 0.05,
    wallGain: knobs.wallGain ?? 0.5,
    horizon: "none",
    targetArousal: sentiment === "tense" ? 0.75 : 0.35,
    targetValence: sentiment === "calm" ? 0.75 : 0.5,
  };

  return { theta0, u, sched };
}

function classifyPrompt(prompt: string): "calm" | "tense" | "neutral" {
  const p = (prompt || "").toLowerCase();
  if (/(calm|serene|smooth|soft|gentle|peace)/.test(p)) return "calm";
  if (/(tense|chaos|sharp|aggressive|noisy|storm)/.test(p)) return "tense";
  return "neutral";
}
