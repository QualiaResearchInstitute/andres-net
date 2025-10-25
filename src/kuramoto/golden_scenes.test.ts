import { describe, it, expect } from "vitest";
import path from "node:path";
import { promises as fs } from "node:fs";

import { createSimulation, type SimulationInitConfig } from "./simulation";
import { stepSimulation, type StepConfig } from "./stepSimulation";
import { startSceneRecorder, serializeScene, type RecorderOptions } from "./sceneRecorder";

// ---------------------------
// Config builders (shared style with determinism.test)
// ---------------------------
function buildStepConfig(dt: number): StepConfig {
  return {
    dt,
    wrap: true,
    // Couplings
    Kbase: 0.8,
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
    // Noise OFF for determinism
    noiseAmp: 0.0,
    // DAG off (no in-place ordering)
    dagSweeps: 0,
    dagDepthOrdering: true,
    dagDepthFiltering: true,
    dagLogStats: false,
    // No attention mods
    attentionMods: undefined,
    // Simple horizon
    horizonFactor: (iIn: boolean, jIn: boolean, recvIn: boolean) => {
      if (iIn === jIn) return 1.0;
      if (recvIn && !jIn) return 1.0;
      if (!recvIn && jIn) return 1.0;
      return 1.0;
    },
  };
}

function makeInitConfig(W = 16, H = 16, reseedGraphKey = 123): SimulationInitConfig {
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

// ---------------------------
// Golden scenes spec
// ---------------------------
type SceneSpec = {
  name: string;
  init: SimulationInitConfig;
  frames: number;
  stepCfg: StepConfig;
  recOpts?: RecorderOptions;
  perfBudgetMs: number;
};

const scenes: SceneSpec[] = [
  {
    name: "baseline_24x24_f6_dt003",
    init: makeInitConfig(24, 24, 4242),
    frames: 6,
    stepCfg: buildStepConfig(0.03),
    recOpts: { downsample: { strideX: 2, strideY: 2 } },
    perfBudgetMs: 1500,
  },
  {
    name: "baseline_16x16_f8_dt002",
    init: makeInitConfig(16, 16, 1111),
    frames: 8,
    stepCfg: buildStepConfig(0.02),
    recOpts: { downsample: false },
    perfBudgetMs: 1250,
  },
];

// ---------------------------
// Utilities
// ---------------------------
const GOLDEN_DIR = path.resolve(process.cwd(), "goldens");

async function ensureDir(dir: string) {
  await fs.mkdir(dir, { recursive: true });
}

function sanitizeForGolden(obj: any) {
  const copy = JSON.parse(JSON.stringify(obj));
  if (copy?.meta) {
    // Zero volatile timestamp to make goldens stable across runs
    copy.meta.createdAt = 0;
  }
  return copy;
}

async function generateSceneJSON(spec: SceneSpec) {
  const { simulation: sim } = createSimulation(null, spec.init);
  const rec = startSceneRecorder(() => sim, spec.recOpts);

  const t0 = globalThis.process?.hrtime?.bigint?.() ?? BigInt(Date.now());
  for (let s = 0; s < spec.frames; s++) {
    stepSimulation(sim, spec.stepCfg);
    rec.snapshot(s);
  }
  const t1 = globalThis.process?.hrtime?.bigint?.() ?? BigInt(Date.now());
  const elapsedNs = Number(t1 - t0);
  const elapsedMs = elapsedNs / 1e6;

  const record = rec.stop();
  // Stabilize metadata fields that are time-dependent
  record.meta.createdAt = 0;

  const serialized = serializeScene(record);
  const stable = sanitizeForGolden(serialized);
  const json = JSON.stringify(stable);
  return { json, ms: elapsedMs };
}

// ---------------------------
// Tests
// ---------------------------
describe("golden scenes", () => {
  it("match golden logs (set UPDATE_GOLDEN=1 to update)", async () => {
    await ensureDir(GOLDEN_DIR);

    for (const spec of scenes) {
      const { json } = await generateSceneJSON(spec);
      const file = path.join(GOLDEN_DIR, `${spec.name}.json`);

      if (process.env.UPDATE_GOLDEN) {
        await fs.writeFile(file, json, "utf8");
        // After update, verify the file reads back to the same bytes
        const back = await fs.readFile(file, "utf8");
        expect(back).toBe(json);
      } else {
        // If missing, fail with actionable hint to update the goldens
        let golden: string | null = null;
        try {
          golden = await fs.readFile(file, "utf8");
        } catch {
          golden = null;
        }
        expect(golden, `Missing golden file ${file}. Run: npm run update:golden`).toBeTruthy();
        if (golden) {
          // Byte-for-byte equality ensures strict determinism
          expect(json).toBe(golden);
        }
      }
    }
  });

  it("meets perf budgets for golden generation", async () => {
    for (const spec of scenes) {
      const { ms } = await generateSceneJSON(spec);
      expect(ms).toBeLessThan(spec.perfBudgetMs);
    }
  });
});
