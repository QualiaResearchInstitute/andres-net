/* eslint-disable no-console */
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { encode as cborEncode } from "cbor-x";
import { createHash } from "node:crypto";
import { execSync } from "node:child_process";

import { createSimulation, updateOmegaSpread } from "../src/kuramoto/simulation";
import { stepSimulation, type StepConfig } from "../src/kuramoto/stepSimulation";
import { buildPATokens } from "../src/tokens/pat";

/**
 * Export a deterministic PAT dataset as CBOR shards.
 *
 * Usage (after adding an npm script, e.g. `npm run export:pat`):
 *   npm run export:pat -- --out out/pat_shards --seeds 1234,5678 --W 96 --H 96 --frames 32 --scale 32 --shardSize 16
 *
 * Notes:
 * - Runs a headless Kuramoto simulation (no rendering) and emits PAT tokens each frame.
 * - Determinism: fixed traversal orders, sort by (seed, frame), fixed shard size per-seed, provenance recorded.
 */

type Cli = {
  out: string;
  seeds: number[];
  W: number;
  H: number;
  frames: number;
  scale: 32 | 64;
  dt: number;
  stepsPerFrame: number;
  shardSize: number;
  commit?: string;
};

function parseArgs(argv: string[]): Cli {
  const args = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const val = (i + 1 < argv.length && !argv[i + 1].startsWith("--")) ? argv[++i] : "true";
      args.set(key, val);
    }
  }
  const seedsArg = (args.get("seeds") ?? "1234,5678").split(",").map((s) => Number.parseInt(s.trim(), 10)).filter(Number.isFinite);
  const W = Number.parseInt(args.get("W") ?? "96", 10);
  const H = Number.parseInt(args.get("H") ?? "96", 10);
  const frames = Number.parseInt(args.get("frames") ?? "32", 10);
  const scaleRaw = Number.parseInt(args.get("scale") ?? "32", 10) as 32 | 64;
  const scale: 32 | 64 = scaleRaw === 64 ? 64 : 32;
  const dt = Number.parseFloat(args.get("dt") ?? "0.03");
  const stepsPerFrame = Number.parseInt(args.get("spf") ?? args.get("stepsPerFrame") ?? "1", 10);
  const shardSize = Number.parseInt(args.get("shardSize") ?? "16", 10);

  return {
    out: args.get("out") ?? "out/pat_shards",
    seeds: seedsArg.length > 0 ? seedsArg : [1234, 5678],
    W: Number.isFinite(W) ? W : 96,
    H: Number.isFinite(H) ? H : 96,
    frames: Number.isFinite(frames) ? frames : 32,
    scale,
    dt: Number.isFinite(dt) ? dt : 0.03,
    stepsPerFrame: Number.isFinite(stepsPerFrame) ? stepsPerFrame : 1,
    shardSize: Number.isFinite(shardSize) && shardSize > 0 ? shardSize : 16,
    commit: args.get("commit") ?? undefined,
  };
}

// Build a StepConfig similar to app defaults (no attention coupling here)
function makeStepConfig(dt: number): StepConfig {
  return {
    dt,
    wrap: true,
    // Base K-series (match painter defaults roughly)
    Kbase: 0.8,
    K1: 1.0,
    K2: -0.3,
    K3: 0.15,
    // Surface ↔ Field
    alphaSurfToField: 0.6,
    alphaFieldToSurf: 0.5,
    KS1: 0.9, KS2: 0.0, KS3: 0.0,
    // Hyper ↔ Field
    alphaHypToField: 0.8,
    alphaFieldToHyp: 0.4,
    KH1: 0.7, KH2: 0.0, KH3: 0.0,
    // Energy dynamics
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
    // Small-world coupling weight
    swWeight: 0.25,
    wallBarrier: 0.5,
    // Noise (transport)
    noiseAmp: 0.0,
    // DAG / debug
    dagSweeps: 0,
    dagDepthOrdering: true,
    dagDepthFiltering: true,
    dagLogStats: false,
    // Attention mods disabled for export (token-only datasets prefer raw physics)
    attentionMods: undefined,
    // Horizon factor unity (no barriers)
    horizonFactor: () => 1.0,
  };
}

// Stable description of the LFQ "codebook" used by PAT (bins/features ordering)
const CODEBOOK_DESC = "PAT_LFQ_v1|dims=5|bins=4|features=A,phi_gf,theta_ref,entropyRate,kappa_abs";

function computeCodebookSHA(): string {
  return createHash("sha256").update(CODEBOOK_DESC).digest("hex");
}

function getGitCommit(): string {
  try {
    return execSync("git rev-parse HEAD").toString().trim();
  } catch {
    return "unknown";
  }
}

async function main() {
  const cli = parseArgs(process.argv.slice(2));
  const outDir = resolve(process.cwd(), cli.out);
  mkdirSync(outDir, { recursive: true });

  const codebookSHA = computeCodebookSHA();
  const commit = (cli.commit && cli.commit.length > 0) ? cli.commit : getGitCommit();

  const manifest = {
    version: 1,
    W: cli.W,
    H: cli.H,
    frames: cli.frames,
    scale: cli.scale,
    shardSize: cli.shardSize,
    seeds: cli.seeds.slice(),
    codebookSHA,
    commit,
    createdAt: new Date().toISOString(),
    codebookDesc: CODEBOOK_DESC,
  };

  writeFileSync(resolve(outDir, "manifest.json"), JSON.stringify(manifest, null, 2));

  for (const seed of cli.seeds) {
    // Deterministic simulation per seed (reuse reseedGraphKey for network topology)
    const { simulation: sim } = createSimulation(null, {
      W: cli.W,
      H: cli.H,
      wrap: true,
      omegaSpread: 0.25,
      swProb: 0.04,
      swEdgesPerNode: 2,
      swMinDist: 6,
      swMaxDist: 20,
      swNegFrac: 0.2,
      energyBaseline: 0.35,
      reseedGraphKey: seed,
    });

    // Ensure omega spread applied
    updateOmegaSpread(sim, 0.25);

    const cfg = makeStepConfig(cli.dt);

    // Buffer items per-seed so shards are strictly fixed-size segments by frame index
    let segmentItems: Array<{ seed: number; frame: number; tokens: ReturnType<typeof buildPATokens> }> = [];

    for (let f = 0; f < cli.frames; f++) {
      for (let s = 0; s < cli.stepsPerFrame; s++) {
        stepSimulation(sim, cfg);
      }
      const tokens = buildPATokens(sim, cli.scale);
      segmentItems.push({ seed, frame: f, tokens });

      // When we complete a segment, write the shard
      if ((f + 1) % cli.shardSize === 0) {
        const segIdx = Math.floor(f / cli.shardSize);
        // Sort deterministically by (seed, frame)
        segmentItems.sort((a, b) => (a.seed - b.seed) || (a.frame - b.frame));
        writeShard(outDir, seed, segIdx, segmentItems, { commit, codebookSHA });
        segmentItems = [];
      }
    }

    // Flush the final partial segment (if any)
    if (segmentItems.length > 0) {
      const lastFrame = cli.frames - 1;
      const segIdx = Math.floor(lastFrame / cli.shardSize);
      segmentItems.sort((a, b) => (a.seed - b.seed) || (a.frame - b.frame));
      writeShard(outDir, seed, segIdx, segmentItems, { commit, codebookSHA });
      segmentItems = [];
    }
  }

  console.log(`[export_pat_dataset] Done. W=${cli.W} H=${cli.H} frames=${cli.frames} scale=${cli.scale} shardSize=${cli.shardSize} out=${outDir}`);
}

function writeShard(
  outDir: string,
  seed: number,
  segIdx: number,
  items: Array<{ seed: number; frame: number; tokens: ReturnType<typeof buildPATokens> }>,
  prov: { commit: string; codebookSHA: string }
) {
  const shardId = `${seed}-${String(segIdx).padStart(4, "0")}`;
  const payload = {
    kind: "PAT_SHARD",
    version: 1,
    shard: shardId,
    seed,
    segment: segIdx,
    count: items.length,
    provenance: {
      commit: prov.commit,
      codebookSHA: prov.codebookSHA,
    },
    items,
  };
  const data = cborEncode(payload);
  const file = resolve(outDir, `pat_${shardId}.cbor`);
  writeFileSync(file, data);
  // Sidecar JSON for quick inspection (does not impact CBOR determinism)
  const sidecar = {
    shard: shardId,
    seed,
    segment: segIdx,
    count: items.length,
    frames: items.map((it) => ({ seed: it.seed, frame: it.frame, tokenCount: it.tokens.length })),
    provenance: payload.provenance,
  };
  writeFileSync(resolve(outDir, `pat_${shardId}.json`), JSON.stringify(sidecar, null, 2));
  // Deterministic checksum of CBOR bytes
  const shaHex = createHash("sha256").update(data).digest("hex");
  writeFileSync(resolve(outDir, `pat_${shardId}.sha256`), shaHex + "\n");
}

main().catch((err) => {
  console.error("[export_pat_dataset] error:", err);
  process.exit(1);
});
