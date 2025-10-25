import { describe, it, expect } from "vitest";
import { createSimulation, type SimulationInitConfig } from "./simulation";
import { exportVirtualizedScene, serializeVirtualizedScene, deserializeVirtualizedScene } from "./sceneVirtualize";
import type { StepConfig } from "./stepSimulation";

function buildStepConfig(dt: number): StepConfig {
  return {
    dt,
    wrap: true,
    // Couplings (minimal values; not relevant to metadata assertions)
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
    // Noise OFF for determinism of export
    noiseAmp: 0.0,
    // DAG off
    dagSweeps: 0,
    dagDepthOrdering: true,
    dagDepthFiltering: true,
    dagLogStats: false,
    // No attention mods
    attentionMods: undefined,
    // Simple horizon
    horizonFactor: (_iIn: boolean, _jIn: boolean, _recvIn: boolean) => 1.0,
  };
}

function makeInitConfig(W = 16, H = 16, reseedGraphKey = 7777): SimulationInitConfig {
  return {
    W,
    H,
    wrap: true,
    omegaSpread: 0.25,
    swProb: 0.0,
    swEdgesPerNode: 0,
    swMinDist: 6,
    swMaxDist: 20,
    swNegFrac: 0.0,
    energyBaseline: 0.35,
    reseedGraphKey,
  };
}

describe("VirtualizedScene metadata", () => {
  it("includes schema/versioning and determinism fields and roundtrips via serialize/deserialize", () => {
    const { simulation: sim } = createSimulation(null, makeInitConfig(16, 16, 2468));
    const stepCfg = buildStepConfig(0.03);

    const scene = exportVirtualizedScene(sim, 2, stepCfg, {
      patScale: 32,
      includeReflector: false,
      repoCommit: "deadbeefcafebabe",
      seed: 123456,
      determinism: { fixedDt: true, cpuParity: true },
      codebooks: { rvqSha: ["rvq:v1:a1b2", "rvq:v1:c3d4"] },
      note: "test-scene",
      gauge: 0,
      startT: 0,
    });

    expect(scene.schema).toBe("vs1");
    expect(scene.repoCommit).toBe("deadbeefcafebabe");
    expect(scene.seed).toBe(123456);
    expect(scene.codebooks).toBeTruthy();
    expect(Array.isArray(scene.codebooks.rvqSha)).toBe(true);
    expect(scene.codebooks.rvqSha.length).toBe(2);
    expect(scene.determinism.fixedDt).toBe(true);
    expect(scene.determinism.cpuParity).toBe(true);
    expect(scene.W).toBe(16);
    expect(scene.H).toBe(16);
    expect(scene.frames.length).toBe(2);

    // Roundtrip via JSON-friendly serializer
    const jsonish = serializeVirtualizedScene(scene);
    const rt = deserializeVirtualizedScene(jsonish);

    expect(rt.schema).toBe("vs1");
    expect(rt.repoCommit).toBe("deadbeefcafebabe");
    expect(rt.seed).toBe(123456);
    expect(rt.codebooks.rvqSha.length).toBe(2);
    expect(rt.determinism.fixedDt).toBe(true);
    expect(rt.determinism.cpuParity).toBe(true);
    expect(rt.W).toBe(16);
    expect(rt.H).toBe(16);
    expect(rt.frames.length).toBe(2);
  });

  it("fills reasonable defaults when metadata is missing (backward compatibility)", () => {
    // Simulate an older object without metadata keys
    const legacyLike: any = {
      W: 8,
      H: 8,
      createdAt: 0,
      gauge: 0,
      frames: [
        { t: 0, affect: { arousal: 0, valence: 0 }, tokens: [] },
        { t: 0.03, affect: { arousal: 0.1, valence: -0.1 }, tokens: [] },
      ],
    };

    const rt = deserializeVirtualizedScene(legacyLike);
    expect(rt.schema).toBe("vs1"); // defaulted
    expect(typeof rt.repoCommit).toBe("string");
    expect(typeof rt.seed).toBe("number");
    expect(Array.isArray(rt.codebooks.rvqSha)).toBe(true);
    expect(typeof rt.determinism.fixedDt).toBe("boolean");
    expect(typeof rt.determinism.cpuParity).toBe("boolean");
    expect(rt.W).toBe(8);
    expect(rt.H).toBe(8);
    expect(rt.frames.length).toBe(2);
  });
});
