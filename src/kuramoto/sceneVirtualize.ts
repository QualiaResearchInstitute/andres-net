import type { SimulationState } from "./types";
import type { StepConfig } from "./stepSimulation";
import { stepSimulation } from "./stepSimulation";
import { computeAffect, type Affect } from "./affect";
import { buildPATokens, type PAToken } from "../tokens/pat";
import { ReflectorGraph } from "./graph/ReflectorGraph";

export type VirtualizedFrame = {
  t: number;
  affect: Affect;
  tokens: PAToken[];
  reflectorNodes?: { x: number; y: number; a: number; w: number }[];
};

export type VirtualizedScene = {
  // Metadata for deterministic replay compatibility
  schema: "vs1";
  repoCommit: string;
  seed: number;
  codebooks: { rvqSha: string[] };
  determinism: { fixedDt: boolean; cpuParity: boolean };

  // Scene dimensions and creation
  W: number;
  H: number;
  createdAt: number;
  gauge: number;

  // Stream frames
  frames: VirtualizedFrame[];

  // Optional notes
  note?: string;
};

/**
 * Export a privacy-preserving scene consisting only of high-level state:
 * - affect headers
 * - PAT tokens
 * - optional ReflectorGraph node snapshots
 * No raw phases are included.
 *
 * Mutates the provided sim by stepping it forward 'frames' times.
 */
export function exportVirtualizedScene(
  sim: SimulationState,
  frames: number,
  stepCfg: StepConfig,
  opts?: {
    patScale?: 32 | 64;
    includeReflector?: boolean;
    gauge?: number;
    startT?: number;
    // Optional metadata overrides/injection
    repoCommit?: string;
    codebooks?: { rvqSha: string[] };
    seed?: number;
    determinism?: { fixedDt?: boolean; cpuParity?: boolean };
    note?: string;
  }
): VirtualizedScene {
  const W = sim.W;
  const H = sim.H;
  const patScale = (opts?.patScale ?? 32) as 32 | 64;
  const includeReflector = !!opts?.includeReflector;
  const gauge = opts?.gauge ?? 0;
  const startT = opts?.startT ?? 0;

  // Optional: try to discover an attached reflector graph on the sim
  const attachedRefl = (sim as any)._reflectorGraph as ReflectorGraph | undefined;

  // Determinism metadata (best-effort defaults)
  const repoCommit = opts?.repoCommit ?? ((((globalThis as any)?.process?.env?.GIT_COMMIT) as string) ?? "unknown");
  const seed = typeof opts?.seed === "number" ? opts!.seed : (sim as any)?.noiseSeed ?? 0;
  const determinism = {
    fixedDt: opts?.determinism?.fixedDt ?? (typeof stepCfg?.dt === "number"),
    cpuParity: opts?.determinism?.cpuParity ?? false,
  };
  const codebooks = opts?.codebooks ?? { rvqSha: [] };

  const out: VirtualizedScene = {
    schema: "vs1",
    repoCommit,
    seed,
    codebooks,
    determinism,

    W,
    H,
    createdAt: Date.now(),
    gauge,
    frames: [],
    note: opts?.note,
  };

  let t = startT;

  for (let k = 0; k < Math.max(0, frames); k++) {
    const affect = computeAffect(sim);
    const tokens = buildPATokens(sim, patScale);
    let reflectorNodes: { x: number; y: number; a: number; w: number }[] | undefined = undefined;

    if (includeReflector && attachedRefl) {
      // Use the current nodes; do not mutate the graph here to keep determinism controlled by caller
      reflectorNodes = attachedRefl.getNodes().map((n) => ({ x: n.x, y: n.y, a: n.a, w: n.w }));
    }

    out.frames.push({ t, affect, tokens, reflectorNodes });

    // advance simulation exactly one frame as defined by stepCfg
    stepSimulation(sim, stepCfg);
    t += stepCfg.dt;
  }

  return out;
}

/**
 * Serialize VirtualizedScene to a JSON-friendly object (arrays instead of typed arrays in tokens).
 * PAT tokens are already plain JS structures; no special conversion required here beyond a shallow copy.
 */
export function serializeVirtualizedScene(scene: VirtualizedScene) {
  return JSON.parse(JSON.stringify(scene));
}

/**
 * Deserialize a VirtualizedScene from a parsed JSON object with shallow validation.
 * Backward compatible: fills reasonable defaults if fields are missing.
 */
export function deserializeVirtualizedScene(obj: any): VirtualizedScene {
  if (!obj || typeof obj !== "object" || !Array.isArray(obj.frames)) {
    throw new Error("Invalid virtualized scene");
  }
  const W = Number(obj.W) | 0;
  const H = Number(obj.H) | 0;
  const gauge = Number(obj.gauge) || 0;
  const createdAt = Number(obj.createdAt) || Date.now();

  // Metadata defaults for backward compatibility
  const schema: "vs1" = (obj.schema === "vs1" ? "vs1" : "vs1");
  const repoCommit: string = typeof obj.repoCommit === "string" ? obj.repoCommit : "unknown";
  const seed: number = typeof obj.seed === "number" ? obj.seed : 0;
  const codebooks = obj.codebooks && Array.isArray(obj.codebooks.rvqSha) ? { rvqSha: obj.codebooks.rvqSha as string[] } : { rvqSha: [] };
  const determinism = obj.determinism && typeof obj.determinism === "object"
    ? { fixedDt: !!obj.determinism.fixedDt, cpuParity: !!obj.determinism.cpuParity }
    : { fixedDt: true, cpuParity: false };

  const frames: VirtualizedFrame[] = obj.frames.map((f: any) => ({
    t: Number(f.t) || 0,
    affect: f.affect,
    tokens: f.tokens,
    reflectorNodes: Array.isArray(f.reflectorNodes) ? f.reflectorNodes : undefined,
  }));

  return {
    schema,
    repoCommit,
    seed,
    codebooks,
    determinism,
    W, H, gauge, createdAt, frames,
    note: obj.note,
  };
}

/**
 * Phase-1 replay helper:
 * - Returns the stored high-level stream (affect/tokens) from the virtualized scene as-is.
 * - Intended for privacy-preserving sharing/inspection without raw phases.
 * Future (Phase-2): map tokens back into a substrate via deterministic controllers to regenerate field frames.
 */
export function replayHighLevel(scene: VirtualizedScene): VirtualizedFrame[] {
  return scene.frames.map((f) => ({
    t: f.t,
    affect: f.affect,
    tokens: f.tokens,
    reflectorNodes: f.reflectorNodes ? [...f.reflectorNodes] : undefined,
  }));
}
