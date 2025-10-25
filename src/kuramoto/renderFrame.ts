import { drawImageOutline, drawPolygon, drawPolyline } from "./drawing";
import { clamp, hsvToRgb, TAU } from "./math";
import { ensureDagCache } from "./simulation";
import { DrawingState, SimulationState, TransformHandleType } from "./types";
import { computePlaneBounds } from "./layers";
import { buildPatchesFromNeighbors, localOrderParameter } from "./reverseSampler";

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
  showTransformHandles: boolean;
  showRhoOverlay: boolean;
  showDefectsOverlay: boolean;
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
    showTransformHandles,
    showRhoOverlay,
    showDefectsOverlay,
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

  // Physics-aware overlays
  if (showRhoOverlay) {
    const patches = buildPatchesFromNeighbors(sim);
    const rho = localOrderParameter(sim.phases, patches);
    octx.save();
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const k = y * W + x;
        // Cyan-tinted overlay scaled by local order
        const a = clamp(rho[k], 0, 1) * 0.35;
        if (a <= 1e-3) continue;
        octx.fillStyle = `rgba(56,189,248,${a})`;
        octx.fillRect(x * pw, y * ph, pw, ph);
      }
    }
    octx.restore();
  }

  if (showDefectsOverlay) {
    // Mark ±1 winding defects by plaquette test
    octx.save();
    const markSize = Math.max(1, Math.floor(Math.min(pw, ph) * 0.6));
    for (let y = 0; y < H - 1; y++) {
      for (let x = 0; x < W - 1; x++) {
        const i00 = y * W + x;
        const i10 = y * W + (x + 1);
        const i11 = (y + 1) * W + (x + 1);
        const i01 = (y + 1) * W + x;
        let d1 = sim.phases[i10] - sim.phases[i00];
        let d2 = sim.phases[i11] - sim.phases[i10];
        let d3 = sim.phases[i01] - sim.phases[i11];
        let d4 = sim.phases[i00] - sim.phases[i01];
        const wrap = (d: number) => {
          while (d > Math.PI) d -= TAU;
          while (d <= -Math.PI) d += TAU;
          return d;
        };
        d1 = wrap(d1); d2 = wrap(d2); d3 = wrap(d3); d4 = wrap(d4);
        const sum = d1 + d2 + d3 + d4;
        if (sum > Math.PI || sum < -Math.PI) {
          const cx = Math.round((x + 0.5) * pw);
          const cy = Math.round((y + 0.5) * ph);
          octx.fillStyle = sum > 0 ? "rgba(244,63,94,0.9)" : "rgba(59,130,246,0.9)"; // red/blue
          octx.fillRect(cx - Math.floor(markSize / 2), cy - Math.floor(markSize / 2), markSize, markSize);
        }
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

  if (showTransformHandles) {
    const selected = new Set(draw.selectedPlaneIds);
    if (selected.size > 0) {
      const hover = draw.planeTransformHover;
      const session = draw.planeTransformSession;
      const size = Math.max(4, pixelSize * 0.9);
      const buildHandles = (plane: typeof draw.layers[number] & { kind: "plane" }) => {
        const bounds = computePlaneBounds(plane.points);
        const { minX, maxX, minY, maxY, center } = bounds;
        const handles: Array<{ position: { x: number; y: number }; type: TransformHandleType }> = [];
        const push = (type: TransformHandleType, position: { x: number; y: number }) => {
          handles.push({ type, position });
        };
        push("scale-nw", { x: minX, y: minY });
        push("scale-n", { x: center.x, y: minY });
        push("scale-ne", { x: maxX, y: minY });
        push("scale-e", { x: maxX, y: center.y });
        push("scale-se", { x: maxX, y: maxY });
        push("scale-s", { x: center.x, y: maxY });
        push("scale-sw", { x: minX, y: maxY });
        push("scale-w", { x: minX, y: center.y });
        const rotationOffset = Math.max(1.5, Math.min(4, Math.max(maxX - minX, maxY - minY) * 0.25 + 1));
        push("rotate", { x: center.x, y: minY - rotationOffset });
        return handles;
      };
      octx.save();
      octx.lineWidth = 1.5;
      draw.layers.forEach((layer) => {
        if (layer.kind !== "plane") return;
        if (!selected.has(layer.planeId)) return;
        const handles = buildHandles(layer);
        handles.forEach((handle) => {
          const x = (handle.position.x + 0.5) * pixelSize;
          const y = (handle.position.y + 0.5) * pixelSize;
          const isHover = hover && hover.planeId === layer.planeId && hover.handle === handle.type;
          const isActive = session && session.planeId === layer.planeId && session.handle === handle.type;
          const baseColor = handle.type === "rotate" ? "rgba(250,204,21," : "rgba(244,244,245,";
          const emphasis = isActive ? 0.95 : isHover ? 0.75 : 0.55;
          if (handle.type === "rotate") {
            octx.beginPath();
            octx.strokeStyle = `${baseColor}${emphasis})`;
            octx.fillStyle = `${baseColor}${isActive ? 0.25 : 0.15})`;
            octx.arc(x, y, size * 0.6, 0, TAU);
            octx.fill();
            octx.stroke();
          } else {
            const half = size * 0.5;
            octx.beginPath();
            octx.strokeStyle = `${baseColor}${emphasis})`;
            octx.fillStyle = `${baseColor}${isActive ? 0.25 : 0.15})`;
            octx.rect(x - half, y - half, half * 2, half * 2);
            octx.fill();
            octx.stroke();
          }
        });
      });
      octx.restore();
    }
  }
}
