import { clamp, distance } from "./math";
import {
  addPlaneFromStroke,
  addStrokeLayer,
  depositLineSegmentToWall,
  rebuildWallFromLayers,
} from "./layers";
import { recomputePotential } from "./simulation";
import { DrawingState, ImageLayer, Point, SimulationState } from "./types";

export type PointerHandlerConfig = {
  canvas: HTMLCanvasElement;
  sim: SimulationState;
  draw: DrawingState;
  W: number;
  H: number;
  pixelSize: number;
  lineWidth: number;
  emBlur: number;
  closeThreshold: number;
  autoClose: boolean;
  imageTool: boolean;
  onLayersChanged: () => void;
};

export function attachPointerHandlers(config: PointerHandlerConfig) {
  const {
    canvas,
    sim,
    draw,
    W,
    H,
    pixelSize,
    lineWidth,
    emBlur,
    closeThreshold,
    autoClose,
    imageTool,
    onLayersChanged,
  } = config;
  const rect = () => canvas.getBoundingClientRect();

  const toCell = (ev: PointerEvent): Point => {
    const r = rect();
    const scaleX = canvas.width / r.width;
    const scaleY = canvas.height / r.height;
    const x = Math.floor(((ev.clientX - r.left) * scaleX) / pixelSize);
    const y = Math.floor(((ev.clientY - r.top) * scaleY) / pixelSize);
    return { x: clamp(x, 0, W - 1), y: clamp(y, 0, H - 1) };
  };

  const findTopmostImageAt = (cell: Point): ImageLayer | undefined => {
    for (let i = draw.layers.length - 1; i >= 0; i--) {
      const layer = draw.layers[i];
      if (layer.kind !== "image" || !layer.loaded) continue;
      const { position, width, height } = layer;
      if (
        cell.x >= position.x &&
        cell.x < position.x + width &&
        cell.y >= position.y &&
        cell.y < position.y + height
      ) {
        return layer;
      }
    }
    return undefined;
  };

  const onDown = (ev: PointerEvent) => {
    if ((ev as any).button !== 0) return;
    ev.preventDefault();
    const cell = toCell(ev);
    if (imageTool) {
      const layer = findTopmostImageAt(cell);
      if (layer) {
        draw.draggingImage = {
          layerId: layer.id,
          offset: { x: cell.x - layer.position.x, y: cell.y - layer.position.y },
        };
      } else {
        draw.draggingImage = null;
      }
      return;
    }
    draw.drawing = true;
    draw.currentStroke = [cell];
  };

  const onMove = (ev: PointerEvent) => {
    if (imageTool) {
      const dragging = draw.draggingImage;
      if (!dragging) return;
      const targetLayer = draw.layers.find((l) => l.id === dragging.layerId);
      if (!targetLayer || targetLayer.kind !== "image") return;
      const cell = toCell(ev);
      const newX = clamp(cell.x - dragging.offset.x, 0, Math.max(0, W - targetLayer.width));
      const newY = clamp(cell.y - dragging.offset.y, 0, Math.max(0, H - targetLayer.height));
      if (targetLayer.position.x !== newX || targetLayer.position.y !== newY) {
        targetLayer.position = { x: newX, y: newY };
        rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
      }
      return;
    }
    if (!draw.drawing) return;
    const p = toCell(ev);
    const current = draw.currentStroke;
    const last = current[current.length - 1];
    if (!last || p.x !== last.x || p.y !== last.y) {
      current.push(p);
      depositLineSegmentToWall(sim, last || p, p, pixelSize, lineWidth);
      recomputePotential(sim, emBlur);
    }
  };

  const onUp = () => {
    if (imageTool) {
      if (draw.draggingImage) {
        draw.draggingImage = null;
        rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
      }
      return;
    }
    if (!draw.drawing) return;
    draw.drawing = false;
    const stroke = draw.currentStroke;
    draw.currentStroke = [];
    if (!stroke || stroke.length < 3) return;
    const first = stroke[0];
    const last = stroke[stroke.length - 1];
    const closed = autoClose && distance(first, last) * pixelSize <= closeThreshold;
    if (closed) {
      addPlaneFromStroke(sim, draw, stroke, lineWidth, pixelSize, emBlur);
      onLayersChanged();
    } else {
      addStrokeLayer(sim, draw, stroke, pixelSize, lineWidth, emBlur);
    }
  };

  canvas.addEventListener("pointerdown", onDown as any);
  window.addEventListener("pointermove", onMove as any);
  window.addEventListener("pointerup", onUp as any);

  return () => {
    canvas.removeEventListener("pointerdown", onDown as any);
    window.removeEventListener("pointermove", onMove as any);
    window.removeEventListener("pointerup", onUp as any);
  };
}
