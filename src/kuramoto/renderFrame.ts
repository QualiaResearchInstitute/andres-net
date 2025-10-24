import { drawImageOutline, drawPolygon, drawPolyline } from "./drawing";
import { clamp, hsvToRgb, TAU } from "./math";
import { ensureDagCache } from "./simulation";
import { DrawingState, SimulationState } from "./types";

export type RenderConfig = {
  pixelSize: number;
  showWalls: boolean;
  showLines: boolean;
  showOutlines: boolean;
  showDagDepth: boolean;
  showImages: boolean;
  stereo: boolean;
  stereoAlpha: number;
  ipd: number;
  brightnessBase: number;
  energyGamma: number;
  liftGain: number;
  hyperCurve: number;
};

export type RenderTargets = {
  field: HTMLCanvasElement | null;
  overlay: HTMLCanvasElement | null;
};

export function renderFrame(
  sim: SimulationState | null,
  draw: DrawingState,
  targets: RenderTargets,
  config: RenderConfig,
) {
  if (!sim || !targets.field || !targets.overlay) return;
  const fc = targets.field;
  const oc = targets.overlay;
  const fctx = fc.getContext("2d");
  const octx = oc.getContext("2d");
  if (!fctx || !octx) return;

  const {
    pixelSize,
    showWalls,
    showLines,
    showOutlines,
    showDagDepth,
    showImages,
    stereo,
    stereoAlpha,
    ipd,
    brightnessBase,
    energyGamma,
    liftGain,
    hyperCurve,
  } = config;

  const { W, H, phases, wall, pot, planeDepth, planeMeta, energy } = sim;

  const selectedPlaneIds = draw.selectedPlaneIds;
  const selectedPlaneSet = new Set(selectedPlaneIds);
  const primarySelectedPlane = selectedPlaneIds.length > 0 ? selectedPlaneIds[0] : null;
  const activePlaneSet = sim.activePlaneSet ?? new Set<number>();

  const pw = pixelSize;
  const ph = pixelSize;
  fctx.clearRect(0, 0, fc.width, fc.height);

  const planeQuickR = new Float32Array(W * H);
  if (planeMeta.size > 0) {
    planeMeta.forEach((meta, planeId) => {
      if (activePlaneSet.size > 0 && !activePlaneSet.has(planeId)) return;
      const { R, centroid } = meta;
      const cx = centroid.x;
      const cy = centroid.y;
      const maxd = Math.max(W, H) * 0.33;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const k = y * W + x;
          const d = Math.hypot(x - cx, y - cy);
          const r = Math.max(0, 1 - d / maxd) * R;
          if (r > planeQuickR[k]) planeQuickR[k] = r;
        }
      }
    });
  }

  const liftField = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const k = y * W + x;
      const depth = planeDepth[k];
      const lift = depth > 0 ? Math.pow(clamp(depth * 0.5, 0, 4) / 4, hyperCurve) * liftGain : 0;
      liftField[k] = lift;
    }
  }

  const sampleColor = (X: number, Y: number) => {
    const xx = clamp(X, 0, W - 1);
    const yy = clamp(Y, 0, H - 1);
    const k = yy * W + xx;
    const th = phases[k];
    const hue = th / TAU;
    const e = energy[k];
    const vEnergy = Math.pow(e, energyGamma);
    const rLocal = planeQuickR[k];
    const v = clamp(
      brightnessBase * (0.25 + 0.75 * vEnergy) * (1 + 0.35 * liftField[k]) * (0.95 + 0.05 * rLocal) +
        0.05 * (pot[k] - 0.5),
      0,
      1,
    );
    const [R, G, B] = hsvToRgb(hue, 0.95, v);
    return [R, G, B] as [number, number, number];
  };

  if (!stereo) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const [R, G, B] = sampleColor(x, y);
        fctx.fillStyle = `rgb(${R},${G},${B})`;
        fctx.fillRect(x * pw, y * ph, pw, ph);
      }
    }
  } else {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const k = y * W + x;
        const disp = Math.round(-ipd * liftField[k]);
        const [R, G, B] = sampleColor(x + disp, y);
        fctx.fillStyle = `rgba(${R},${G},${B},1)`;
        fctx.fillRect(x * pw, y * ph, pw, ph);
      }
    }
    fctx.globalAlpha = clamp(stereoAlpha, 0, 1);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const k = y * W + x;
        const disp = Math.round(ipd * liftField[k]);
        const [R, G, B] = sampleColor(x + disp, y);
        fctx.fillStyle = `rgba(${R},${G},${B},1)`;
        fctx.fillRect(x * pw, y * ph, pw, ph);
      }
    }
    fctx.globalAlpha = 1;
  }

  octx.clearRect(0, 0, oc.width, oc.height);

  if (showDagDepth) {
    const dag = ensureDagCache(sim);
    if (dag.maxDepth > 0) {
      const denom = Math.max(1, dag.maxDepth);
      octx.save();
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const k = y * W + x;
          const depth = planeDepth[k];
          if (depth === 0) continue;
          const norm = depth / denom;
          const r = Math.floor(60 + 150 * norm);
          const g = Math.floor(160 + 80 * norm);
          const b = Math.floor(255 - 60 * norm);
          const alpha = clamp(0.15 + 0.35 * norm, 0, 0.6);
          octx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
          octx.fillRect(x * pw, y * ph, pw, ph);
        }
      }
      octx.restore();
    }
  }

  if (showWalls) {
    octx.save();
    octx.globalAlpha = 0.15;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const k = y * W + x;
        const v = sim.wall[k];
        if (v <= 0) continue;
        const a = clamp(v / 6, 0, 1);
        octx.fillStyle = `rgba(255,255,255,${a})`;
        octx.fillRect(x * pw, y * ph, pw, ph);
      }
    }
    octx.restore();
  }

  if (showImages) {
    draw.layers.forEach((layer) => {
      if (layer.kind !== "image" || !layer.loaded || !layer.image) return;
      const tx = layer.position.x * pixelSize;
      const ty = layer.position.y * pixelSize;
      const tw = Math.max(1, layer.width * pixelSize);
      const th = Math.max(1, layer.height * pixelSize);
      octx.save();
      octx.globalAlpha = 0.92;
      octx.drawImage(layer.image, tx, ty, tw, th);
      octx.restore();
    });
  }

  if (showLines) {
    const cs = draw.currentStroke;
    if (cs.length > 0) drawPolyline(octx, cs, pixelSize, "#fff", 2);
    draw.layers.forEach((layer) => {
      if (layer.kind === "stroke") {
        drawPolyline(octx, layer.points, pixelSize, "rgba(255,255,255,0.7)", 2);
      } else if (layer.kind === "image") {
        const dragging = draw.draggingImage?.layerId === layer.id;
        drawImageOutline(
          octx,
          layer,
          pixelSize,
          dragging ? "rgba(56,189,248,0.9)" : "rgba(255,255,255,0.65)",
        );
      }
    });
  }

  if (showOutlines) {
    draw.layers.forEach((layer) => {
      if (layer.kind !== "plane") return;
      const meta =
        sim.planeMeta.get(layer.planeId) ?? {
          centroid: layer.centroid,
          orientation: layer.orientation,
          orientationSign: layer.orientationSign,
          color: [255, 255, 255] as [number, number, number],
          solo: false,
          muted: false,
          locked: false,
        };
      const highlight = selectedPlaneSet.has(layer.planeId);
      const primary = primarySelectedPlane === layer.planeId;
      const muted = meta.muted ?? false;
      const active = activePlaneSet.size === 0 ? !muted : activePlaneSet.has(layer.planeId);
      drawPolygon(octx, layer.points, pixelSize, meta, showDagDepth, {
        highlight,
        primary,
        muted,
        active,
      });
    });
  }
}
