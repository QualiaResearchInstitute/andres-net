import { booleanCombine, flipPoints, rotatePoints90, skewPoints, smoothPoints, ensureWinding } from "./geometry";
import { clamp, hsvToRgb, pnpoly, polygonSignedArea } from "./math";
import { recomputePotential, markDagDirty, updateMasks } from "./simulation";
import {
  DrawingState,
  ImageLayer,
  Layer,
  PlaneLayer,
  PlaneMetadata,
  Point,
  SimulationState,
  StrokeLayer,
} from "./types";

export function depositLineSegmentToWall(
  sim: SimulationState,
  a: Point,
  b: Point,
  pixelSize: number,
  lineWidth: number,
  widthOverride?: number,
) {
  const { W, H, wall } = sim;
  const widthPx = widthOverride ?? lineWidth;
  const thick = Math.max(1, Math.floor(widthPx / (pixelSize || 1)));
  const dx = Math.abs(b.x - a.x);
  const dy = Math.abs(b.y - a.y);
  const sx = a.x < b.x ? 1 : -1;
  const sy = a.y < b.y ? 1 : -1;
  let x = a.x;
  let y = a.y;
  let err = dx - dy;
  while (true) {
    for (let oy = -thick; oy <= thick; oy++) {
      for (let ox = -thick; ox <= thick; ox++) {
        const jx = x + ox;
        const jy = y + oy;
        if (jx < 0 || jy < 0 || jx >= W || jy >= H) continue;
        wall[jy * W + jx] += 1;
      }
    }
    if (x === b.x && y === b.y) break;
    const e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x += sx;
    }
    if (e2 < dx) {
      err += dx;
      y += sy;
    }
  }
}

function depositImageOutline(sim: SimulationState, layer: ImageLayer, pixelSize: number, lineWidth: number) {
  if (!layer.loaded) return;
  const { position, width, height, outlineWidth } = layer;
  if (width <= 0 || height <= 0) return;
  const outline = [
    { x: position.x, y: position.y },
    { x: position.x + width - 1, y: position.y },
    { x: position.x + width - 1, y: position.y + height - 1 },
    { x: position.x, y: position.y + height - 1 },
    { x: position.x, y: position.y },
  ];
  for (let i = 1; i < outline.length; i++) {
    depositLineSegmentToWall(sim, outline[i - 1], outline[i], pixelSize, lineWidth, outlineWidth);
  }
}

function ensurePlaneMetadataDefaults(meta: PlaneMetadata) {
  if (meta.solo === undefined) meta.solo = false;
  if (meta.muted === undefined) meta.muted = false;
  if (meta.locked === undefined) meta.locked = false;
  return meta;
}

export function computePlaneMetadata(
  sim: SimulationState,
  planeId: number,
  points: Point[],
  existing?: PlaneMetadata,
): PlaneMetadata | null {
  const { W, H } = sim;
  if (points.length < 3) return null;
  let minx = Infinity;
  let miny = Infinity;
  let maxx = -Infinity;
  let maxy = -Infinity;
  for (const p of points) {
    minx = Math.min(minx, p.x);
    miny = Math.min(miny, p.y);
    maxx = Math.max(maxx, p.x);
    maxy = Math.max(maxy, p.y);
  }
  minx = Math.floor(clamp(minx, 0, W - 1));
  maxx = Math.floor(clamp(maxx, 0, W - 1));
  miny = Math.floor(clamp(miny, 0, H - 1));
  maxy = Math.floor(clamp(maxy, 0, H - 1));

  const cells: number[] = [];
  for (let y = miny; y <= maxy; y++) {
    for (let x = minx; x <= maxx; x++) {
      if (pnpoly(points, x + 0.5, y + 0.5)) {
        cells.push(y * W + x);
      }
    }
  }
  if (cells.length === 0) return null;

  const cx = points.reduce((a, p) => a + p.x, 0) / points.length;
  const cy = points.reduce((a, p) => a + p.y, 0) / points.length;
  const signedArea = polygonSignedArea(points);
  const orientationSign = signedArea === 0 ? 0 : signedArea > 0 ? 1 : -1;
  const orientation = orientationSign === 0 ? "flat" : orientationSign > 0 ? "ccw" : "cw";
  const color = existing?.color ?? hsvToRgb((planeId * 0.123) % 1, 0.7, 1.0);

  const meta: PlaneMetadata = ensurePlaneMetadataDefaults(
    existing ?? {
      cells: Int32Array.from(cells),
      R: 0,
      psi: 0,
      color,
      centroid: { x: cx, y: cy },
      orientation,
      orientationSign,
      solo: false,
      muted: false,
      locked: false,
    },
  );
  meta.cells = Int32Array.from(cells);
  meta.centroid = { x: cx, y: cy };
  meta.orientation = orientation;
  meta.orientationSign = orientationSign;
  return meta;
}

function getPlaneLayers(draw: DrawingState) {
  return draw.layers.filter((layer): layer is PlaneLayer => layer.kind === "plane");
}

export function getPlaneLayer(draw: DrawingState, planeId: number): PlaneLayer | null {
  for (let i = 0; i < draw.layers.length; i++) {
    const layer = draw.layers[i];
    if (layer.kind === "plane" && layer.planeId === planeId) {
      return layer;
    }
  }
  return null;
}

export function computePlaneBounds(points: Point[]) {
  if (points.length === 0) {
    return {
      minX: 0,
      maxX: 0,
      minY: 0,
      maxY: 0,
      width: 0,
      height: 0,
      center: { x: 0, y: 0 },
    };
  }
  let minX = points[0].x;
  let maxX = points[0].x;
  let minY = points[0].y;
  let maxY = points[0].y;
  for (let i = 1; i < points.length; i++) {
    const p = points[i];
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const width = maxX - minX;
  const height = maxY - minY;
  const center = { x: minX + width / 2, y: minY + height / 2 };
  return { minX, maxX, minY, maxY, width, height, center };
}

export function getPlaneOrder(draw: DrawingState): number[] {
  const order: number[] = [];
  for (let i = 0; i < draw.layers.length; i++) {
    const layer = draw.layers[i];
    if (layer.kind === "plane") {
      order.push(layer.planeId);
    }
  }
  return order;
}

export function applyPlaneOrder(
  sim: SimulationState,
  draw: DrawingState,
  order: number[],
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
): boolean {
  const current = getPlaneOrder(draw);
  if (order.length === 0 || current.length === 0) return false;
  let identical = current.length === order.length;
  if (identical) {
    for (let i = 0; i < current.length; i++) {
      if (current[i] !== order[i]) {
        identical = false;
        break;
      }
    }
  }
  if (identical) return false;
  const planeLayers = new Map<number, PlaneLayer>();
  const fallback: number[] = [];
  for (let i = 0; i < draw.layers.length; i++) {
    const layer = draw.layers[i];
    if (layer.kind === "plane") {
      planeLayers.set(layer.planeId, layer);
    }
  }
  const normalized: number[] = [];
  const seen = new Set<number>();
  order.forEach((id) => {
    if (!seen.has(id) && planeLayers.has(id)) {
      normalized.push(id);
      seen.add(id);
    }
  });
  current.forEach((id) => {
    if (!seen.has(id)) {
      normalized.push(id);
      seen.add(id);
    }
  });
  if (normalized.length !== current.length) return false;

  const planePositions: number[] = [];
  for (let i = 0; i < draw.layers.length; i++) {
    if (draw.layers[i].kind === "plane") {
      planePositions.push(i);
    }
  }
  if (planePositions.length !== normalized.length) return false;

  const updatedLayers = draw.layers.slice();
  for (let idx = 0; idx < planePositions.length; idx++) {
    const targetPlaneId = normalized[idx];
    const layer = planeLayers.get(targetPlaneId);
    if (!layer) continue;
    updatedLayers[planePositions[idx]] = layer;
  }
  draw.layers = updatedLayers;
  recomputePlaneDepthFromLayers(sim, draw);
  rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
  return true;
}

export function updatePlaneGeometry(
  sim: SimulationState,
  draw: DrawingState,
  planeId: number,
  newPoints: Point[],
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
): boolean {
  if (newPoints.length < 3) return false;
  const planeLayer = getPlaneLayer(draw, planeId);
  if (!planeLayer) return false;
  const existingMeta = sim.planeMeta.get(planeId);
  const metadata = computePlaneMetadata(sim, planeId, newPoints, existingMeta);
  if (!metadata) return false;
  planeLayer.points = newPoints.map((p) => ({ x: p.x, y: p.y }));
  planeLayer.centroid = { ...metadata.centroid };
  planeLayer.orientation = metadata.orientation;
  planeLayer.orientationSign = metadata.orientationSign;
  sim.planeMeta.set(planeId, metadata);
  recomputePlaneDepthFromLayers(sim, draw);
  rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
  return true;
}

function resolveActivePlaneIds(sim: SimulationState, draw: DrawingState): number[] {
  const planeLayers = getPlaneLayers(draw);
  const planeMeta = sim.planeMeta;
  const soloIds: number[] = [];
  planeLayers.forEach((layer) => {
    const meta = planeMeta.get(layer.planeId);
    if (!meta) return;
    ensurePlaneMetadataDefaults(meta);
    if (meta.solo && !meta.muted) {
      soloIds.push(layer.planeId);
    }
  });
  const restrictToSolo = soloIds.length > 0;
  const soloSet = new Set(soloIds);
  const active: number[] = [];
  planeLayers.forEach((layer) => {
    const meta = planeMeta.get(layer.planeId);
    if (!meta) return;
    ensurePlaneMetadataDefaults(meta);
    if (meta.muted) return;
    if (restrictToSolo && !soloSet.has(layer.planeId)) return;
    active.push(layer.planeId);
  });
  return active;
}

export function recomputePlaneDepthFromLayers(
  sim: SimulationState,
  draw: DrawingState,
) {
  const { planeDepth, planeMeta } = sim;
  planeDepth.fill(0);
  const planeLayers = getPlaneLayers(draw);
  const activeIds = resolveActivePlaneIds(sim, draw);
  const activeSet = new Set<number>();
  activeIds.forEach((planeId, order) => {
    const layer = planeLayers.find((pl) => pl.planeId === planeId);
    if (!layer) return;
    let meta = planeMeta.get(planeId);
    meta = computePlaneMetadata(sim, planeId, layer.points, meta) ?? meta;
    if (!meta) return;
    ensurePlaneMetadataDefaults(meta);
    meta.order = order;
    planeMeta.set(planeId, meta);
    const cells = meta.cells;
    for (let idx = 0; idx < cells.length; idx++) {
      const cell = cells[idx];
      planeDepth[cell] = clamp(planeDepth[cell] + 1, 0, 255);
    }
    activeSet.add(planeId);
  });
  sim.activePlaneIds = activeIds;
  sim.activePlaneSet = activeSet;
  updateMasks(sim);
  markDagDirty(sim);
}

export function rebuildWallFromLayers(
  sim: SimulationState,
  draw: DrawingState,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
) {
  sim.wall.fill(0);
  for (let i = 0; i < draw.layers.length; i++) {
    const layer = draw.layers[i];
    if (layer.kind === "stroke") {
      for (let t = 1; t < layer.points.length; t++) {
        depositLineSegmentToWall(sim, layer.points[t - 1], layer.points[t], pixelSize, lineWidth, layer.lineWidth);
      }
    } else if (layer.kind === "plane") {
      if (!sim.activePlaneSet.has(layer.planeId)) continue;
      const pts = layer.points;
      if (pts.length > 1) {
        for (let t = 1; t < pts.length; t++) {
          depositLineSegmentToWall(sim, pts[t - 1], pts[t], pixelSize, lineWidth, layer.outlineWidth);
        }
        depositLineSegmentToWall(sim, pts[pts.length - 1], pts[0], pixelSize, lineWidth, layer.outlineWidth);
      }
    } else if (layer.kind === "image") {
      depositImageOutline(sim, layer, pixelSize, lineWidth);
    }
  }
  recomputePotential(sim, emBlur);
}

export function addPlaneFromStroke(
  sim: SimulationState,
  draw: DrawingState,
  points: Point[],
  lineWidth: number,
  pixelSize: number,
  emBlur: number,
) {
  const { planeMeta } = sim;
  const id = draw.nextPlaneId++;
  const metadata = computePlaneMetadata(sim, id, points);
  if (!metadata) return;
  planeMeta.set(id, metadata);

  const planeLayer: PlaneLayer = {
    id: draw.nextLayerId++,
    kind: "plane",
    planeId: id,
    points: [...points],
    centroid: { ...metadata.centroid },
    orientation: metadata.orientation,
    orientationSign: metadata.orientationSign,
    outlineWidth: lineWidth,
  };
  draw.layers.push(planeLayer);
  draw.selectedPlaneIds = [id];
  recomputePlaneDepthFromLayers(sim, draw);
  rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
}

export function removeLastPlane(
  sim: SimulationState,
  draw: DrawingState,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
) {
  const { planeMeta } = sim;
  for (let i = draw.layers.length - 1; i >= 0; i--) {
    const layer = draw.layers[i];
    if (layer.kind !== "plane") continue;
    draw.layers.splice(i, 1);
    planeMeta.delete(layer.planeId);
    draw.selectedPlaneIds = draw.selectedPlaneIds.filter((id) => id !== layer.planeId);
    recomputePlaneDepthFromLayers(sim, draw);
    rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
    return;
  }
}

export function clearPlanes(
  sim: SimulationState,
  draw: DrawingState,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
) {
  const { planeMeta } = sim;
  planeMeta.clear();
  draw.layers = draw.layers.filter((layer) => layer.kind !== "plane");
  draw.nextPlaneId = 1;
  draw.selectedPlaneIds = [];
  recomputePlaneDepthFromLayers(sim, draw);
  rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
}

export function clearStrokes(
  sim: SimulationState,
  draw: DrawingState,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
) {
  draw.layers = draw.layers.filter((layer) => layer.kind !== "stroke");
  draw.currentStroke = [];
  rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
}

export function removeLastImage(
  sim: SimulationState,
  draw: DrawingState,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
) {
  for (let i = draw.layers.length - 1; i >= 0; i--) {
    const layer = draw.layers[i];
    if (layer.kind !== "image") continue;
    URL.revokeObjectURL(layer.src);
    draw.layers.splice(i, 1);
    if (draw.draggingImage?.layerId === layer.id) {
      draw.draggingImage = null;
    }
    rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
    return;
  }
}

export function clearImages(
  sim: SimulationState,
  draw: DrawingState,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
) {
  const kept: Layer[] = [];
  for (let i = 0; i < draw.layers.length; i++) {
    const layer = draw.layers[i];
    if (layer.kind === "image") {
      URL.revokeObjectURL(layer.src);
      if (draw.draggingImage?.layerId === layer.id) {
        draw.draggingImage = null;
      }
    } else {
      kept.push(layer);
    }
  }
  draw.layers = kept;
  rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
}

export function addImages(
  sim: SimulationState,
  draw: DrawingState,
  files: FileList | null,
  W: number,
  H: number,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
  imageOutlineWidth: number,
) {
  if (!files || files.length === 0) return;
  const pending: ImageLayer[] = [];
  Array.from(files).forEach((file) => {
    const url = URL.createObjectURL(file);
    const layer: ImageLayer = {
      id: draw.nextLayerId++,
      kind: "image",
      src: url,
      image: null,
      position: { x: Math.max(0, Math.floor((W - 12) / 2)), y: Math.max(0, Math.floor((H - 12) / 2)) },
      width: Math.max(4, Math.min(W, 16)),
      height: Math.max(4, Math.min(H, 16)),
      outlineWidth: imageOutlineWidth,
      loaded: false,
    };
    const img = new Image();
    img.onload = () => {
      layer.image = img;
      layer.loaded = true;
      const maxSpan = Math.max(4, Math.min(Math.min(W, H), 36));
      const aspect = img.width / Math.max(1, img.height);
      if (aspect >= 1) {
        layer.width = Math.max(1, Math.min(W, Math.round(maxSpan)));
        layer.height = Math.max(1, Math.min(H, Math.round(layer.width / aspect)));
      } else {
        layer.height = Math.max(1, Math.min(H, Math.round(maxSpan)));
        layer.width = Math.max(1, Math.min(W, Math.round(layer.height * aspect)));
      }
      layer.position = {
        x: clamp(Math.floor((W - layer.width) / 2), 0, Math.max(0, W - layer.width)),
        y: clamp(Math.floor((H - layer.height) / 2), 0, Math.max(0, H - layer.height)),
      };
      rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      const idx = draw.layers.indexOf(layer);
      if (idx >= 0) draw.layers.splice(idx, 1);
    };
    img.src = url;
    pending.push(layer);
  });
  draw.layers.push(...pending);
  rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
}

export function updateImageOutlineWidth(
  sim: SimulationState,
  draw: DrawingState,
  width: number,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
) {
  let needsRebuild = false;
  draw.layers.forEach((layer) => {
    if (layer.kind === "image" && layer.outlineWidth !== width) {
      layer.outlineWidth = width;
      needsRebuild = true;
    }
  });
  if (needsRebuild) {
    rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
  }
}

export function addStrokeLayer(
  sim: SimulationState,
  draw: DrawingState,
  points: Point[],
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
) {
  const strokeLayer: StrokeLayer = {
    id: draw.nextLayerId++,
    kind: "stroke",
    points: [...points],
    lineWidth,
  };
  draw.layers.push(strokeLayer);
  rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
}

export function setPlaneSelection(draw: DrawingState, planeIds: number[]) {
  draw.selectedPlaneIds = [...planeIds];
}

function getPlaneMetadata(sim: SimulationState, planeId: number): PlaneMetadata | null {
  const meta = sim.planeMeta.get(planeId);
  if (!meta) return null;
  return ensurePlaneMetadataDefaults(meta);
}

export function setPlaneSolo(sim: SimulationState, planeId: number, value: boolean): boolean {
  const meta = getPlaneMetadata(sim, planeId);
  if (!meta) return false;
  if (meta.locked && !meta.solo && value === true) {
    return false;
  }
  if (meta.solo === value) return false;
  meta.solo = value;
  if (value) meta.muted = false;
  return true;
}

export function setPlaneMuted(sim: SimulationState, planeId: number, value: boolean): boolean {
  const meta = getPlaneMetadata(sim, planeId);
  if (!meta) return false;
  if (meta.locked && value && !meta.muted) {
    return false;
  }
  if (meta.muted === value) return false;
  meta.muted = value;
  if (value) meta.solo = false;
  return true;
}

export function setPlaneLocked(sim: SimulationState, planeId: number, value: boolean): boolean {
  const meta = getPlaneMetadata(sim, planeId);
  if (!meta) return false;
  if (meta.locked === value) return false;
  meta.locked = value;
  return true;
}

export function reorderPlaneLayer(
  sim: SimulationState,
  draw: DrawingState,
  sourcePlaneId: number,
  targetPlaneId: number | null,
): boolean {
  const sourceIndex = draw.layers.findIndex(
    (layer) => layer.kind === "plane" && layer.planeId === sourcePlaneId,
  );
  if (sourceIndex < 0) return false;
  const sourceMeta = getPlaneMetadata(sim, sourcePlaneId);
  if (sourceMeta?.locked) return false;
  const targetIndex =
    targetPlaneId === null
      ? draw.layers.length
      : draw.layers.findIndex((layer) => layer.kind === "plane" && layer.planeId === targetPlaneId);
  if (targetPlaneId !== null && targetIndex < 0) return false;
  if (targetPlaneId !== null && sourcePlaneId === targetPlaneId) return false;
  const targetInsertIndex =
    targetPlaneId === null
      ? draw.layers.length
      : targetIndex;
  const [layer] = draw.layers.splice(sourceIndex, 1);
  let insertIndex = targetInsertIndex;
  if (targetPlaneId !== null && sourceIndex < targetInsertIndex) {
    insertIndex -= 1;
  }
  draw.layers.splice(insertIndex, 0, layer);
  return true;
}

function clonePoints(points: Point[]): Point[] {
  return points.map((p) => ({ x: p.x, y: p.y }));
}

function sanitizePoints(sim: SimulationState, points: Point[]): Point[] {
  const { W, H } = sim;
  const sanitized: Point[] = [];
  const eps = 1e-3;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const x = clamp(p.x, 0, W - 1e-3);
    const y = clamp(p.y, 0, H - 1e-3);
    const px = Number.parseFloat(x.toFixed(4));
    const py = Number.parseFloat(y.toFixed(4));
    const prev = sanitized[sanitized.length - 1];
    if (!prev || Math.hypot(prev.x - px, prev.y - py) > eps) {
      sanitized.push({ x: px, y: py });
    }
  }
  if (sanitized.length >= 3) {
    const first = sanitized[0];
    const last = sanitized[sanitized.length - 1];
    if (Math.hypot(first.x - last.x, first.y - last.y) <= eps) {
      sanitized.pop();
    }
  }
  return sanitized.length >= 3 ? sanitized : points;
}

export function deletePlanes(
  sim: SimulationState,
  draw: DrawingState,
  planeIds: number[],
  mode: "keep-wall" | "keep-energy" | "clean",
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
  energyBaseline: number,
) {
  if (planeIds.length === 0) return false;
  const planeSet = new Set(planeIds);
  let changed = false;
  const outlineStrokes: StrokeLayer[] = [];
  const affectedCells: number[] = [];

  for (let i = draw.layers.length - 1; i >= 0; i--) {
    const layer = draw.layers[i];
    if (layer.kind !== "plane") continue;
    if (!planeSet.has(layer.planeId)) continue;
    changed = true;
    const meta = sim.planeMeta.get(layer.planeId);
    if (meta) {
      for (let c = 0; c < meta.cells.length; c++) {
        affectedCells.push(meta.cells[c]);
      }
    }
    sim.planeMeta.delete(layer.planeId);
    draw.selectedPlaneIds = draw.selectedPlaneIds.filter((id) => id !== layer.planeId);
    draw.layers.splice(i, 1);
    if (mode === "keep-wall") {
      const outlinePoints = clonePoints(layer.points);
      if (outlinePoints.length > 0) {
        outlinePoints.push({ ...outlinePoints[0] });
      }
      const strokeLayer: StrokeLayer = {
        id: draw.nextLayerId++,
        kind: "stroke",
        points: outlinePoints,
        lineWidth: Math.max(1, layer.outlineWidth || lineWidth),
      };
      outlineStrokes.push(strokeLayer);
    }
  }

  if (!changed) return false;

  if (outlineStrokes.length > 0) {
    outlineStrokes.reverse().forEach((stroke) => {
      draw.layers.push(stroke);
    });
  }

  if (mode === "clean" && affectedCells.length > 0) {
    const { energy } = sim;
    for (let c = 0; c < affectedCells.length; c++) {
      energy[affectedCells[c]] = energyBaseline;
    }
  }

  recomputePlaneDepthFromLayers(sim, draw);
  rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
  return true;
}

export type PlaneTransformAction =
  | "flip-horizontal"
  | "flip-vertical"
  | "rotate-cw"
  | "rotate-ccw"
  | "skew-x-pos"
  | "skew-x-neg"
  | "skew-y-pos"
  | "skew-y-neg"
  | "smooth";

export type PlaneBooleanAction = "union" | "subtract" | "intersect";

export function transformPlaneShape(
  sim: SimulationState,
  draw: DrawingState,
  planeId: number,
  action: PlaneTransformAction,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
): boolean {
  const layer = getPlaneLayer(draw, planeId);
  if (!layer) return false;
  let nextPoints: Point[] = clonePoints(layer.points);
  switch (action) {
    case "flip-horizontal":
      nextPoints = flipPoints(nextPoints, "horizontal");
      break;
    case "flip-vertical":
      nextPoints = flipPoints(nextPoints, "vertical");
      break;
    case "rotate-cw":
      nextPoints = rotatePoints90(nextPoints, "cw");
      break;
    case "rotate-ccw":
      nextPoints = rotatePoints90(nextPoints, "ccw");
      break;
    case "skew-x-pos":
      nextPoints = skewPoints(nextPoints, "x", 0.35);
      break;
    case "skew-x-neg":
      nextPoints = skewPoints(nextPoints, "x", -0.35);
      break;
    case "skew-y-pos":
      nextPoints = skewPoints(nextPoints, "y", 0.35);
      break;
    case "skew-y-neg":
      nextPoints = skewPoints(nextPoints, "y", -0.35);
      break;
    case "smooth":
      nextPoints = smoothPoints(nextPoints, 2);
      break;
  }
  nextPoints = ensureWinding(sanitizePoints(sim, nextPoints));
  if (nextPoints.length < 3) return false;
  return updatePlaneGeometry(sim, draw, planeId, nextPoints, pixelSize, lineWidth, emBlur);
}

export function booleanCombinePlanes(
  sim: SimulationState,
  draw: DrawingState,
  basePlaneId: number,
  otherPlaneIds: number[],
  action: PlaneBooleanAction,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
): boolean {
  const baseLayer = getPlaneLayer(draw, basePlaneId);
  if (!baseLayer) return false;
  const others: Point[][] = [];
  const planeSet = new Set(otherPlaneIds);
  draw.layers.forEach((layer) => {
    if (layer.kind !== "plane") return;
    if (!planeSet.has(layer.planeId)) return;
    others.push(clonePoints(layer.points));
  });
  if (others.length === 0) return false;
  const mode = action === "union" ? "union" : action === "intersect" ? "intersect" : "subtract";
  const combined = booleanCombine(mode, clonePoints(baseLayer.points), others);
  if (combined.length < 3) return false;
  const sanitized = ensureWinding(sanitizePoints(sim, combined));
  if (sanitized.length < 3) return false;
  return updatePlaneGeometry(sim, draw, basePlaneId, sanitized, pixelSize, lineWidth, emBlur);
}

export function createDrawingState(): DrawingState {
  return {
    drawing: false,
    currentStroke: [],
    layers: [],
    nextPlaneId: 1,
    nextLayerId: 1,
    draggingImage: null,
    selectedPlaneIds: [],
    planeTransformSession: null,
    planeTransformHover: null,
    planeOrderUndo: [],
    planeOrderRedo: [],
  };
}
