import type { SimulationState } from "./types";

/**
 * Scene Recorder scaffolding (Phase-1)
 * - Lightweight snapshotter for simulation state over time
 * - Focuses on reproducibility: captures phases and energy fields plus metadata
 */

export type SceneFrame = {
  t: number; // user-supplied timeline (seconds or steps)
  phases: Float32Array;
  energy: Float32Array;
};

export type SceneRecord = {
  meta: {
    W: number;
    H: number;
    createdAt: number;
    frames: number;
    // Optional schedules sampled for this scene (if recorded with schedules)
    schedules?: {
      t: number[];
      K: number[];
      D: number[];
      Kref?: number[];
      psi?: number[];
    };
  };
  frames: SceneFrame[];
};

export type RecorderHandle = {
  snapshot: (t: number) => void;
  stop: () => SceneRecord;
  isRecording: () => boolean;
};

export type RecorderOptions = {
  // Auto-capture every ms; if undefined, only manual snapshot() will capture
  intervalMs?: number;
  // If true, downsample energy to save memory (nearest neighbor by stride)
  downsample?: { strideX: number; strideY: number } | false;
};

/** Utility: downsample a field (row-major W×H) by integer strides (nearest) */
function downsampleField(src: Float32Array, W: number, H: number, sx = 2, sy = 2) {
  const W2 = Math.max(1, Math.floor(W / sx));
  const H2 = Math.max(1, Math.floor(H / sy));
  const out = new Float32Array(W2 * H2);
  for (let y = 0; y < H2; y++) {
    for (let x = 0; x < W2; x++) {
      const yy = Math.min(H - 1, y * sy);
      const xx = Math.min(W - 1, x * sx);
      out[y * W2 + x] = src[yy * W + xx];
    }
  }
  return { field: out, W: W2, H: H2 };
}

/**
 * Start a scene recorder bound to a running SimulationState provider
 * - Maintains an in-memory buffer of frames
 * - Caller controls timeline t (e.g. seconds elapsed or step count)
 */
export function startSceneRecorder(getSim: () => SimulationState | null, opts?: RecorderOptions): RecorderHandle {
  const frames: SceneFrame[] = [];
  let active = true;
  let timer: any = null;

  const snapshot = (t: number) => {
    if (!active) return;
    const sim = getSim();
    if (!sim) return;
    const { W, H, phases, energy } = sim;
    const copyTheta = new Float32Array(phases); // deep copy
    let copyEnergy = new Float32Array(energy);
    if (opts?.downsample) {
      const { strideX, strideY } = opts.downsample;
      const ds = downsampleField(copyEnergy, W, H, Math.max(1, strideX), Math.max(1, strideY));
      copyEnergy = ds.field;
    }
    frames.push({ t, phases: copyTheta, energy: copyEnergy });
  };

  if (typeof opts?.intervalMs === "number" && opts.intervalMs > 0) {
    timer = setInterval(() => {
      // Use frames.length as a default time axis if not provided externally
      snapshot(frames.length);
    }, opts.intervalMs);
  }

  const stop = (): SceneRecord => {
    if (timer) {
      try {
        clearInterval(timer);
      } catch {}
      timer = null;
    }
    active = false;
    const sim = getSim();
    const W = sim?.W ?? 0;
    const H = sim?.H ?? 0;
    return {
      meta: { W, H, createdAt: Date.now(), frames: frames.length },
      frames,
    };
  };

  return {
    snapshot,
    stop,
    isRecording: () => active,
  };
}

/**
 * Serialize a scene record to a JSON-friendly object (numbers arrays)
 * Note: Float32Array are converted to regular arrays for portability.
 */
export function serializeScene(rec: SceneRecord) {
  return {
    meta: rec.meta,
    frames: rec.frames.map((f) => ({
      t: f.t,
      phases: Array.from(f.phases),
      energy: Array.from(f.energy),
    })),
  };
}

/** Optional schedules payload for serialization */
export type SchedulesSeries = {
  t: number[];
  K: number[];
  D: number[];
  Kref?: number[];
  psi?: number[];
};

/** Serialize scene and attach schedules into meta.schedules */
export function serializeSceneWithSchedules(rec: SceneRecord, sched: SchedulesSeries) {
  const meta = { ...rec.meta, schedules: sched };
  return {
    meta,
    frames: rec.frames.map((f) => ({
      t: f.t,
      phases: Array.from(f.phases),
      energy: Array.from(f.energy),
    })),
  };
}

/**
 * Deserialize from serialized JSON-friendly object back to SceneRecord
 */
export function deserializeScene(obj: any): SceneRecord {
  if (!obj || typeof obj !== "object" || !Array.isArray(obj.frames)) {
    throw new Error("Invalid scene object");
  }
  const frames: SceneFrame[] = obj.frames.map((f: any) => ({
    t: Number(f.t) || 0,
    phases: Float32Array.from(f.phases || []),
    energy: Float32Array.from(f.energy || []),
  }));
  const meta = obj.meta || { W: 0, H: 0, createdAt: Date.now(), frames: frames.length };
  return { meta, frames };
}
