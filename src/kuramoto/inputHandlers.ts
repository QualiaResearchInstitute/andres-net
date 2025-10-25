import { clamp, distance, pnpoly, polygonSignedArea } from "./math";
import {
  addPlaneFromStroke,
  addStrokeLayer,
  depositLineSegmentToWall,
  rebuildWallFromLayers,
  computePlaneBounds,
  getPlaneLayer,
  updatePlaneGeometry,
} from "./layers";
import { recomputePotential } from "./simulation";
import {
  DrawingState,
  ImageLayer,
  PlaneLayer,
  Point,
  SimulationState,
  TransformHandleType,
} from "./types";

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
  transformTool: boolean;
  openContextMenu?: (info: { planeId: number; screenX: number; screenY: number }) => void;
  closeContextMenu?: () => void;
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
    transformTool,
    openContextMenu,
    closeContextMenu,
    onLayersChanged,
  } = config;
  const rect = () => canvas.getBoundingClientRect();

  const toCanvasPoint = (ev: PointerEvent): Point => {
    const r = rect();
    const scaleX = canvas.width / r.width;
    const scaleY = canvas.height / r.height;
    const x = ((ev.clientX - r.left) * scaleX) / pixelSize;
    const y = ((ev.clientY - r.top) * scaleY) / pixelSize;
    return { x, y };
  };

  const toCell = (ev: PointerEvent): Point => {
    const p = toCanvasPoint(ev);
    const x = clamp(Math.floor(p.x), 0, W - 1);
    const y = clamp(Math.floor(p.y), 0, H - 1);
    return { x, y };
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

  type TransformHandleSpec = {
    plane: PlaneLayer;
    type: TransformHandleType;
    position: Point;
    bounds: ReturnType<typeof computePlaneBounds>;
  };

  const handleCursors: Record<TransformHandleType, string> = {
    move: "move",
    "scale-n": "ns-resize",
    "scale-ne": "nesw-resize",
    "scale-e": "ew-resize",
    "scale-se": "nwse-resize",
    "scale-s": "ns-resize",
    "scale-sw": "nesw-resize",
    "scale-w": "ew-resize",
  "scale-nw": "nwse-resize",
  rotate: "grab",
  };

  const EPS = 1e-6;

  const updatePointInPlace = (target: Point, x: number, y: number): boolean => {
    const dx = Math.abs(target.x - x);
    const dy = Math.abs(target.y - y);
    target.x = x;
    target.y = y;
    return dx > EPS || dy > EPS;
  };

  const refreshPlaneDerivedProps = (plane: PlaneLayer) => {
    const pts = plane.points;
    if (pts.length === 0) {
      plane.centroid.x = 0;
      plane.centroid.y = 0;
      plane.orientation = "flat";
      plane.orientationSign = 0;
      const metaZero = sim.planeMeta.get(plane.planeId);
      if (metaZero) {
        metaZero.centroid = metaZero.centroid ?? { x: 0, y: 0 };
        metaZero.centroid.x = 0;
        metaZero.centroid.y = 0;
        metaZero.orientation = plane.orientation;
        metaZero.orientationSign = plane.orientationSign;
      }
      return;
    }
    let sumX = 0;
    let sumY = 0;
    for (let i = 0; i < pts.length; i++) {
      sumX += pts[i].x;
      sumY += pts[i].y;
    }
    const centroidX = sumX / pts.length;
    const centroidY = sumY / pts.length;
    plane.centroid = plane.centroid ?? { x: centroidX, y: centroidY };
    plane.centroid.x = centroidX;
    plane.centroid.y = centroidY;
    const signedArea = polygonSignedArea(pts);
    const sign = signedArea === 0 ? 0 : signedArea > 0 ? 1 : -1;
    plane.orientationSign = sign;
    plane.orientation = sign === 0 ? "flat" : sign > 0 ? "ccw" : "cw";
    const meta = sim.planeMeta.get(plane.planeId);
    if (meta) {
      meta.centroid = meta.centroid ?? { x: centroidX, y: centroidY };
      meta.centroid.x = centroidX;
      meta.centroid.y = centroidY;
      meta.orientation = plane.orientation;
      meta.orientationSign = plane.orientationSign;
    }
  };

  const getSelectedPlaneLayers = () => {
    const selected = new Set(draw.selectedPlaneIds);
    if (selected.size === 0) return [] as PlaneLayer[];
    const result: PlaneLayer[] = [];
    for (let i = draw.layers.length - 1; i >= 0; i--) {
      const layer = draw.layers[i];
      if (layer.kind !== "plane") continue;
      if (!selected.has(layer.planeId)) continue;
      result.push(layer);
    }
    return result;
  };

  const buildTransformHandles = (plane: PlaneLayer): TransformHandleSpec[] => {
    const bounds = computePlaneBounds(plane.points);
    const { minX, maxX, minY, maxY, center } = bounds;
    const handles: TransformHandleSpec[] = [];
    const push = (type: TransformHandleType, position: Point) => {
      handles.push({ plane, type, position, bounds });
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

  const hitTestTransformHandles = (
    point: Point,
    selectedPlanes: PlaneLayer[],
  ): TransformHandleSpec | null => {
    const thresholdBase = Math.max(0.5, 6 / Math.max(1, pixelSize));
    for (let i = 0; i < selectedPlanes.length; i++) {
      const plane = selectedPlanes[i];
      const handles = buildTransformHandles(plane);
      for (let h = 0; h < handles.length; h++) {
        const handle = handles[h];
        const threshold =
          handle.type === "rotate" ? thresholdBase * 1.4 : thresholdBase;
        if (distance(point, handle.position) <= threshold) {
          return handle;
        }
      }
    }
    return null;
  };

  const hitTestPlaneInterior = (point: Point, planes: PlaneLayer[]) => {
    for (let i = 0; i < planes.length; i++) {
      const plane = planes[i];
      if (pnpoly(plane.points, point.x, point.y)) {
        return plane;
      }
    }
    return null;
  };

  const updateHoverHandle = (
    hover: TransformHandleSpec | null,
    fallbackPlaneMove: PlaneLayer | null,
  ) => {
    const nextHover =
      hover?.type && hover.type !== "move"
        ? { planeId: hover.plane.planeId, handle: hover.type }
        : fallbackPlaneMove
        ? { planeId: fallbackPlaneMove.planeId, handle: "move" as TransformHandleType }
        : null;
    if (
      (draw.planeTransformHover === null && nextHover === null) ||
      (draw.planeTransformHover &&
        nextHover &&
        draw.planeTransformHover.planeId === nextHover.planeId &&
        draw.planeTransformHover.handle === nextHover.handle)
    ) {
      // unchanged
    } else {
      draw.planeTransformHover = nextHover;
      onLayersChanged();
    }
    const cursorHandle = nextHover?.handle;
    if (cursorHandle) {
      const cursor =
        draw.planeTransformSession && cursorHandle === "move"
          ? "grabbing"
          : handleCursors[cursorHandle] ?? "default";
      canvas.style.cursor = cursor;
    } else {
      canvas.style.cursor = "default";
    }
  };

  const startTransformSession = (
    plane: PlaneLayer,
    handleSpec: TransformHandleSpec,
    pointer: Point,
  ) => {
    const bounds = handleSpec.bounds;
    const session = {
      planeId: plane.planeId,
      handle: handleSpec.type,
      initialPointer: { ...pointer },
      initialHandlePosition: { ...handleSpec.position },
      initialPoints: plane.points.map((p) => ({ x: p.x, y: p.y })),
      initialBounds: {
        minX: bounds.minX,
        maxX: bounds.maxX,
        minY: bounds.minY,
        maxY: bounds.maxY,
      },
      initialCenter: { ...bounds.center },
      initialRotationAngle:
        handleSpec.type === "rotate"
          ? Math.atan2(pointer.y - bounds.center.y, pointer.x - bounds.center.x)
          : undefined,
      dirty: false,
    };
    draw.planeTransformSession = session;
    canvas.style.cursor =
      handleSpec.type === "rotate" ? "grabbing" : handleCursors[handleSpec.type] ?? "move";
  };

  const applyTransformFromSession = (pointer: Point, ev: PointerEvent): boolean => {
    const session = draw.planeTransformSession;
    if (!session) return false;
    const plane = getPlaneLayer(draw, session.planeId);
    if (!plane) return false;
    const { handle, initialPointer, initialPoints, initialBounds, initialCenter, initialHandlePosition } =
      session;

    let planePoints = plane.points;
    if (planePoints.length !== initialPoints.length) {
      plane.points = initialPoints.map((p) => ({ x: p.x, y: p.y }));
      planePoints = plane.points;
    }

    if (handle === "move") {
      const dx = pointer.x - initialPointer.x;
      const dy = pointer.y - initialPointer.y;
      if (!Number.isFinite(dx) || !Number.isFinite(dy)) return false;
      let changed = false;
      for (let i = 0; i < initialPoints.length; i++) {
        const init = initialPoints[i];
        changed = updatePointInPlace(planePoints[i], init.x + dx, init.y + dy) || changed;
      }
      if (!changed) return false;
      refreshPlaneDerivedProps(plane);
      session.dirty = true;
      return true;
    }

    if (handle === "rotate") {
      const startAngle = session.initialRotationAngle ?? 0;
      let delta =
        Math.atan2(pointer.y - initialCenter.y, pointer.x - initialCenter.x) - startAngle;
      const snap = (15 * Math.PI) / 180;
      if (ev.shiftKey) {
        delta = Math.round(delta / snap) * snap;
      }
      if (!Number.isFinite(delta)) return false;
      const cos = Math.cos(delta);
      const sin = Math.sin(delta);
      let changed = false;
      for (let i = 0; i < initialPoints.length; i++) {
        const init = initialPoints[i];
        const dx = init.x - initialCenter.x;
        const dy = init.y - initialCenter.y;
        const nx = initialCenter.x + dx * cos - dy * sin;
        const ny = initialCenter.y + dx * sin + dy * cos;
        changed = updatePointInPlace(planePoints[i], nx, ny) || changed;
      }
      if (!changed) return false;
      refreshPlaneDerivedProps(plane);
      session.dirty = true;
      return true;
    }

    const affectsX = /e|w/i.test(handle);
    const affectsY = /n|s/i.test(handle);

    let pivotX = initialCenter.x;
    let pivotY = initialCenter.y;

    if (!ev.altKey) {
      if (/e/i.test(handle)) pivotX = initialBounds.minX;
      if (/w/i.test(handle)) pivotX = initialBounds.maxX;
      if (/n/i.test(handle)) pivotY = initialBounds.maxY;
      if (/s/i.test(handle)) pivotY = initialBounds.minY;
    }

    const pivot = { x: pivotX, y: pivotY };

    let scaleX = 1;
    let scaleY = 1;

    if (affectsX) {
      const denomX = initialHandlePosition.x - pivot.x;
      if (Math.abs(denomX) > EPS) {
        scaleX = (pointer.x - pivot.x) / denomX;
      }
    }

    if (affectsY) {
      const denomY = initialHandlePosition.y - pivot.y;
      if (Math.abs(denomY) > EPS) {
        scaleY = (pointer.y - pivot.y) / denomY;
      }
    }

    if (ev.shiftKey) {
      const ax = Math.abs(scaleX);
      const ay = Math.abs(scaleY);
      const dominant = !affectsX && affectsY ? ay : affectsX && !affectsY ? ax : Math.max(ax, ay);
      const signX = scaleX >= 0 ? 1 : -1;
      const signY = scaleY >= 0 ? 1 : -1;
      const uniform = dominant || 1;
      scaleX = affectsX ? uniform * signX : uniform * signY;
      scaleY = affectsY ? uniform * signY : uniform * signX;
    }

    if (!affectsX) scaleX = 1;
    if (!affectsY) scaleY = 1;

    if (!Number.isFinite(scaleX) || !Number.isFinite(scaleY)) {
      return false;
    }

    let changed = false;
    for (let i = 0; i < initialPoints.length; i++) {
      const init = initialPoints[i];
      const dx = init.x - pivot.x;
      const dy = init.y - pivot.y;
      const nx = pivot.x + dx * scaleX;
      const ny = pivot.y + dy * scaleY;
      changed = updatePointInPlace(planePoints[i], nx, ny) || changed;
    }

    if (!changed) return false;
    refreshPlaneDerivedProps(plane);
    session.dirty = true;
    return true;
  };

  const endTransformSession = (commit = true) => {
    const session = draw.planeTransformSession;
    if (session && commit && session.dirty) {
      const plane = getPlaneLayer(draw, session.planeId);
      if (plane) {
        updatePlaneGeometry(sim, draw, session.planeId, plane.points, pixelSize, lineWidth, emBlur);
      }
    }
    draw.planeTransformSession = null;
    draw.planeTransformHover = null;
    canvas.style.cursor = "default";
    onLayersChanged();
  };

  const onDown = (ev: PointerEvent) => {
    if ((ev as any).button !== 0) return;
    ev.preventDefault();
    if (closeContextMenu) closeContextMenu();
    const point = toCanvasPoint(ev);
    const cell = {
      x: clamp(Math.floor(point.x), 0, W - 1),
      y: clamp(Math.floor(point.y), 0, H - 1),
    };

    if (transformTool) {
      const selectedPlanes = getSelectedPlaneLayers();
      if (selectedPlanes.length > 0) {
        const handleHit = hitTestTransformHandles(point, selectedPlanes);
        if (handleHit) {
          startTransformSession(handleHit.plane, handleHit, point);
          draw.planeTransformHover = {
            planeId: handleHit.plane.planeId,
            handle: handleHit.type,
          };
          onLayersChanged();
          return;
        }
        const planeHit = hitTestPlaneInterior(point, selectedPlanes);
        if (planeHit) {
          const bounds = computePlaneBounds(planeHit.points);
          startTransformSession(planeHit, {
            plane: planeHit,
            type: "move",
            position: { ...point },
            bounds,
          }, point);
          draw.planeTransformHover = {
            planeId: planeHit.planeId,
            handle: "move",
          };
          onLayersChanged();
          return;
        }
      }
    }

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
    const pointer = toCanvasPoint(ev);

    if (transformTool) {
      if (ev.buttons !== 2 && closeContextMenu) {
        closeContextMenu();
      }
      if (draw.planeTransformSession) {
        const changed = applyTransformFromSession(pointer, ev);
        if (changed) {
          onLayersChanged();
        }
        return;
      }
      if (ev.buttons === 0) {
        const selectedPlanes = getSelectedPlaneLayers();
        if (selectedPlanes.length > 0) {
          const handleHit = hitTestTransformHandles(pointer, selectedPlanes);
          const planeHit = handleHit ? null : hitTestPlaneInterior(pointer, selectedPlanes);
          updateHoverHandle(handleHit, planeHit);
        } else if (draw.planeTransformHover) {
          draw.planeTransformHover = null;
          canvas.style.cursor = "default";
          onLayersChanged();
        }
      }
      return;
    }

    if (imageTool) {
      const dragging = draw.draggingImage;
      if (!dragging) return;
      const targetLayer = draw.layers.find((l) => l.id === dragging.layerId);
      if (!targetLayer || targetLayer.kind !== "image") return;
      const cell = {
        x: clamp(Math.floor(pointer.x), 0, Math.max(0, W - 1)),
        y: clamp(Math.floor(pointer.y), 0, Math.max(0, H - 1)),
      };
      const newX = clamp(cell.x - dragging.offset.x, 0, Math.max(0, W - targetLayer.width));
      const newY = clamp(cell.y - dragging.offset.y, 0, Math.max(0, H - targetLayer.height));
      if (targetLayer.position.x !== newX || targetLayer.position.y !== newY) {
        targetLayer.position = { x: newX, y: newY };
        rebuildWallFromLayers(sim, draw, pixelSize, lineWidth, emBlur);
      }
      return;
    }
    if (!draw.drawing) return;
    const p = {
      x: clamp(Math.floor(pointer.x), 0, W - 1),
      y: clamp(Math.floor(pointer.y), 0, H - 1),
    };
    const current = draw.currentStroke;
    const last = current[current.length - 1];
    if (!last || p.x !== last.x || p.y !== last.y) {
      current.push(p);
      depositLineSegmentToWall(sim, last || p, p, pixelSize, lineWidth);
      recomputePotential(sim, emBlur);
    }
  };

  const onUp = () => {
    if (closeContextMenu) closeContextMenu();
    if (draw.planeTransformSession) {
      endTransformSession(true);
      return;
    }
    if (transformTool) {
      return;
    }
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
  const onContextMenu = (ev: PointerEvent) => {
    if (!transformTool) return;
    ev.preventDefault();
    if (openContextMenu) {
      const pointer = toCanvasPoint(ev);
      const selectedPlanes = getSelectedPlaneLayers();
      if (selectedPlanes.length === 0) return;
      const planeHit = hitTestPlaneInterior(pointer, selectedPlanes);
      const target = planeHit ?? selectedPlanes[selectedPlanes.length - 1];
      if (!target) return;
      openContextMenu({
        planeId: target.planeId,
        screenX: ev.clientX,
        screenY: ev.clientY,
      });
    }
  };
  canvas.addEventListener("contextmenu", onContextMenu as any);

  return () => {
    canvas.removeEventListener("pointerdown", onDown as any);
    window.removeEventListener("pointermove", onMove as any);
    window.removeEventListener("pointerup", onUp as any);
    canvas.removeEventListener("contextmenu", onContextMenu as any);
  };
}
