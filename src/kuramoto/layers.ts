import { clamp, hsvToRgb, pnpoly, polygonSignedArea } from "./math";
import { recomputePotential, markDagDirty, updateMasks } from "./simulation";
import {
  DrawingState,
  ImageLayer,
  Layer,
  PlaneLayer,
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
) {
  const { W, H, planeDepth, planeMeta } = sim;
  const id = draw.nextPlaneId++;
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
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
        const k = y * W + x;
        cells.push(k);
        planeDepth[k] = clamp(planeDepth[k] + 1, 0, 255);
      }
    }
  }
  if (cells.length === 0) return;

  const cx = points.reduce((a, p) => a + p.x, 0) / points.length;
  const cy = points.reduce((a, p) => a + p.y, 0) / points.length;
  const signedArea = polygonSignedArea(points);
  const orientationSign = signedArea === 0 ? 0 : signedArea > 0 ? 1 : -1;
  const orientation = orientationSign === 0 ? "flat" : orientationSign > 0 ? "ccw" : "cw";
  const hue = (id * 0.123) % 1;
  const color = hsvToRgb(hue, 0.7, 1.0);
  const metadata = {
    cells: Int32Array.from(cells),
    R: 0,
    psi: 0,
    color,
    centroid: { x: cx, y: cy },
    orientation,
    orientationSign,
  };
  planeMeta.set(id, metadata);

  const planeLayer: PlaneLayer = {
    id: draw.nextLayerId++,
    kind: "plane",
    planeId: id,
    points: [...points],
    centroid: { x: cx, y: cy },
    orientation,
    orientationSign,
    outlineWidth: lineWidth,
  };
  draw.layers.push(planeLayer);
  updateMasks(sim);
  markDagDirty(sim);
}

export function removeLastPlane(
  sim: SimulationState,
  draw: DrawingState,
  pixelSize: number,
  lineWidth: number,
  emBlur: number,
) {
  const { planeDepth, planeMeta } = sim;
  for (let i = draw.layers.length - 1; i >= 0; i--) {
    const layer = draw.layers[i];
    if (layer.kind !== "plane") continue;
    draw.layers.splice(i, 1);
    const meta = planeMeta.get(layer.planeId);
    if (meta) {
      for (let idx = 0; idx < meta.cells.length; idx++) {
        const cell = meta.cells[idx];
        planeDepth[cell] = Math.max(0, planeDepth[cell] - 1);
      }
      planeMeta.delete(layer.planeId);
    }
    updateMasks(sim);
    markDagDirty(sim);
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
  const { planeDepth, planeMeta, surfMask, hypMask } = sim;
  planeDepth.fill(0);
  planeMeta.clear();
  surfMask.fill(0);
  hypMask.fill(0);
  draw.layers = draw.layers.filter((layer) => layer.kind !== "plane");
  draw.nextPlaneId = 1;
  markDagDirty(sim);
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
