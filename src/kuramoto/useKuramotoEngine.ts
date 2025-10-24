import { useCallback, useEffect, useRef } from "react";
import { attachPointerHandlers } from "./inputHandlers";
import {
  addImages as addImagesToLayers,
  clearImages as clearImagesFromLayers,
  clearPlanes as clearPlanesFromLayers,
  clearStrokes as clearStrokesFromLayers,
  createDrawingState,
  rebuildWallFromLayers,
  recomputePlaneDepthFromLayers,
  removeLastImage as removeLastImageFromLayers,
  removeLastPlane as removeLastPlaneFromLayers,
  reorderPlaneLayer,
  setPlaneLocked,
  setPlaneMuted,
  setPlaneSelection,
  setPlaneSolo,
  updateImageOutlineWidth as updateImageOutlineWidthLayers,
  applyPlaneOrder,
  getPlaneOrder,
  deletePlanes as deletePlanesFromLayers,
  transformPlaneShape,
  booleanCombinePlanes,
  PlaneTransformAction,
  PlaneBooleanAction,
} from "./layers";
import { renderFrame } from "./renderFrame";
import {
  clearWalls as clearWallsSim,
  createSimulation,
  setNoiseSeed,
  recomputePotential,
  resetPhases as resetPhasesSim,
  updateOmegaSpread,
} from "./simulation";
import { stepSimulation, StepConfig } from "./stepSimulation";
import type { AttentionMods } from "./stepSimulation";
import { updateAttentionFields } from "./attention";
import { DrawingState, PlaneLayer, SimulationState } from "./types";
import { ReflectorGraph } from "./graph/ReflectorGraph";

export type KuramotoEngineConfig = {
  W: number;
  H: number;
  pixelSize: number;
  lineWidth: number;
  closeThreshold: number;
  autoClose: boolean;
  emBlur: number;
  emGain: number;
  wallBarrier: number;
  imageTool: boolean;
  transformTool: boolean;
  showImages: boolean;
  imageOutlineWidth: number;
  showWalls: boolean;
  showLines: boolean;
  showOutlines: boolean;
  showDagDepth: boolean;
  showRhoOverlay: boolean;
  showDefectsOverlay: boolean;
  energyBaseline: number;
  energyLeak: number;
  energyDiff: number;
  sinkLine: number;
  sinkSurf: number;
  sinkHyp: number;
  trapSurf: number;
  trapHyp: number;
  minEnergySurf: number;
  minEnergyHyp: number;
  brightnessBase: number;
  energyGamma: number;
  alphaSurfToField: number;
  alphaFieldToSurf: number;
  KS1: number;
  KS2: number;
  KS3: number;
  alphaHypToField: number;
  alphaFieldToHyp: number;
  KH1: number;
  KH2: number;
  KH3: number;
  liftGain: number;
  hyperCurve: number;
  eventBarrier: number;
  dagSweeps: number;
  dagDepthOrdering: boolean;
  dagDepthFiltering: boolean;
  dagLogStats: boolean;
  stereo: boolean;
  ipd: number;
  stereoAlpha: number;
  wrap: boolean;
  paused: boolean;
  dt: number;
  stepsPerFrame: number;
  Kbase: number;
  K1: number;
  K2: number;
  K3: number;
  omegaSpread: number;
  noiseAmp: number;
  swProb: number;
  swEdgesPerNode: number;
  swMinDist: number;
  swMaxDist: number;
  swWeight: number;
  swNegFrac: number;
  reseedGraphKey: number;
  // Attention
  attentionEnabled: boolean;
  attentionParams: import("./attention").AttentionParams;
  attentionHeads: import("./attention").AttentionHead[];
  setImageOutlineWidth: (width: number) => void;
  onPlaneContextMenu?: (info: { planeId: number; screenX: number; screenY: number }) => void;
  onPlaneContextMenuClose?: () => void;
};

export type KuramotoEngineApi = {
  fieldCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  overlayCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  fileInputRef: React.MutableRefObject<HTMLInputElement | null>;
  step: (dt: number) => void;
  stepWithSeed: (dt: number, seed: number | null) => void;
  resetPhases: () => void;
  clearWalls: () => void;
  clearPlanes: () => void;
  removeLastPlane: () => void;
  clearStrokes: () => void;
  addImages: (files: FileList | null) => void;
  removeLastImage: () => void;
  clearImages: () => void;
  updateImageOutlineWidth: (width: number) => void;
  getPlaneLayersSnapshot: () => PlaneLayerSnapshot[];
  getSelectedPlaneIds: () => number[];
  setSelectedPlaneIds: (ids: number[]) => void;
  togglePlaneSolo: (planeId: number, value: boolean) => void;
  togglePlaneMuted: (planeId: number, value: boolean) => void;
  togglePlaneLocked: (planeId: number, value: boolean) => void;
  reorderPlane: (sourcePlaneId: number, targetPlaneId: number | null) => void;
  undoPlaneOrder: () => void;
  redoPlaneOrder: () => void;
  canUndoPlaneOrder: () => boolean;
  canRedoPlaneOrder: () => boolean;
  deletePlanes: (mode: "keep-wall" | "keep-energy" | "clean") => void;
  setTransportNoiseSeed: (seed: number) => void;
  getTransportNoiseSeed: () => number;
  transformPlane: (planeId: number, action: PlaneTransformAction) => void;
  combinePlanes: (basePlaneId: number, otherPlaneIds: number[], action: PlaneBooleanAction) => void;
  subscribeLayerChanges: (listener: () => void) => () => void;
  getSim: () => SimulationState | null;
};

export type PlaneLayerSnapshot = {
  planeId: number;
  layerId: number;
  orientation: string;
  orientationSign: number;
  centroid: { x: number; y: number };
  outlineWidth: number;
  color: [number, number, number];
  solo: boolean;
  muted: boolean;
  locked: boolean;
  order?: number;
  active: boolean;
};

function disposeImages(draw: DrawingState) {
  draw.layers.forEach((layer) => {
    if (layer.kind === "image") {
      URL.revokeObjectURL(layer.src);
    }
  });
}

function ordersEqual(a: number[] | undefined, b: number[]) {
  if (!a || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export function useKuramotoEngine(config: KuramotoEngineConfig): KuramotoEngineApi {
  const {
    W,
    H,
    pixelSize,
    lineWidth,
    closeThreshold,
    autoClose,
    emBlur,
    emGain,
    wallBarrier,
    imageTool,
    transformTool,
    showImages,
    imageOutlineWidth,
    showWalls,
    showLines,
    showOutlines,
    showDagDepth,
    showRhoOverlay,
    showDefectsOverlay,
    energyBaseline,
    energyLeak,
    energyDiff,
    sinkLine,
    sinkSurf,
    sinkHyp,
    trapSurf,
    trapHyp,
    minEnergySurf,
    minEnergyHyp,
    brightnessBase,
    energyGamma,
    alphaSurfToField,
    alphaFieldToSurf,
    KS1,
    KS2,
    KS3,
    alphaHypToField,
    alphaFieldToHyp,
    KH1,
    KH2,
    KH3,
    liftGain,
    hyperCurve,
    eventBarrier,
    dagSweeps,
    dagDepthOrdering,
    dagDepthFiltering,
    dagLogStats,
    stereo,
    ipd,
    stereoAlpha,
    wrap,
    paused,
    dt,
    stepsPerFrame,
    Kbase,
    K1,
    K2,
    K3,
    omegaSpread,
    noiseAmp,
    swProb,
    swEdgesPerNode,
    swMinDist,
    swMaxDist,
    swWeight,
    swNegFrac,
    reseedGraphKey,
    // Attention
    attentionEnabled,
    attentionParams,
    attentionHeads,
    setImageOutlineWidth,
    onPlaneContextMenu,
    onPlaneContextMenuClose,
  } = config;

  const fieldCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const rafRef = useRef<number>(0);
  const simRef = useRef<SimulationState | null>(null);
  const drawRef = useRef<DrawingState>(createDrawingState());
  const layerVersionRef = useRef(0);
  const layerListenersRef = useRef<Set<() => void>>(new Set());
  const attentionTimeRef = useRef(0);
  const attentionModsRef = useRef<AttentionMods | undefined>(undefined);
  const MAX_PLANE_ORDER_HISTORY = 64;

  const pushPlaneOrderSnapshot = useCallback(() => {
    const draw = drawRef.current;
    if (!draw) return;
    const order = getPlaneOrder(draw);
    if (order.length === 0) return;
    const undoStack = draw.planeOrderUndo;
    const last = undoStack[undoStack.length - 1];
    if (!ordersEqual(last, order)) {
      undoStack.push([...order]);
      if (undoStack.length > MAX_PLANE_ORDER_HISTORY) {
        undoStack.shift();
      }
    }
    draw.planeOrderRedo = [];
  }, []);

  const notifyLayersChanged = useCallback(() => {
    layerVersionRef.current += 1;
    layerListenersRef.current.forEach((fn) => {
      try {
        fn();
      } catch (err) {
        console.error("[KuramotoPainter] layer listener error", err);
      }
    });
  }, []);

  const subscribeLayerChanges = useCallback((listener: () => void) => {
    layerListenersRef.current.add(listener);
    return () => {
      layerListenersRef.current.delete(listener);
    };
  }, []);

  const horizonFactor = useCallback(
    (iInside: boolean, jInside: boolean, receiverInside: boolean) => {
      if (iInside === jInside) return 1.0;
      if (receiverInside && !jInside) return 1.0;
      if (!receiverInside && jInside) return 1.0 - eventBarrier;
      return 1.0;
    },
    [eventBarrier],
  );

  const buildStepConfig = useCallback(
    (dtValue: number): StepConfig => ({
      dt: dtValue,
      wrap,
      Kbase,
      K1,
      K2,
      K3,
      KS1,
      KS2,
      KS3,
      KH1,
      KH2,
      KH3,
      alphaSurfToField,
      alphaFieldToSurf,
      alphaHypToField,
      alphaFieldToHyp,
      swWeight,
      wallBarrier,
      emGain,
      energyBaseline,
      energyLeak,
      energyDiff,
      sinkLine,
      sinkSurf,
      sinkHyp,
      trapSurf,
      trapHyp,
      minEnergySurf,
      minEnergyHyp,
      noiseAmp,
      dagSweeps,
      dagDepthOrdering,
      dagDepthFiltering,
      dagLogStats,
      attentionMods: attentionModsRef.current,
      horizonFactor,
    }),
    [
      wrap,
      Kbase,
      K1,
      K2,
      K3,
      KS1,
      KS2,
      KS3,
      KH1,
      KH2,
      KH3,
      alphaSurfToField,
      alphaFieldToSurf,
      alphaHypToField,
      alphaFieldToHyp,
      swWeight,
      wallBarrier,
      emGain,
      energyBaseline,
      energyLeak,
      energyDiff,
      sinkLine,
      sinkSurf,
      sinkHyp,
      trapSurf,
      trapHyp,
      minEnergySurf,
      minEnergyHyp,
      noiseAmp,
      dagSweeps,
      dagDepthOrdering,
      dagDepthFiltering,
      dagLogStats,
      horizonFactor,
    ],
  );

  useEffect(() => {
    const prev = simRef.current;
    const { simulation, shouldResetDrawing } = createSimulation(prev, {
      W,
      H,
      wrap,
      omegaSpread,
      swProb,
      swEdgesPerNode,
      swMinDist,
      swMaxDist,
      swNegFrac,
      energyBaseline,
      reseedGraphKey,
    });
    simRef.current = simulation;
    // Instantiate ReflectorGraph once per simulation (opt-in source for A)
    (simRef.current as any)._reflectorGraph = (simRef.current as any)._reflectorGraph ?? new ReflectorGraph(W, H);
    if (shouldResetDrawing) {
      disposeImages(drawRef.current);
      drawRef.current = createDrawingState();
    }
    recomputePlaneDepthFromLayers(simulation, drawRef.current);
    notifyLayersChanged();
  }, [
    W,
    H,
    wrap,
    omegaSpread,
    swProb,
    swEdgesPerNode,
    swMinDist,
    swMaxDist,
    swNegFrac,
    energyBaseline,
    reseedGraphKey,
    notifyLayersChanged,
  ]);

  useEffect(() => {
    const canvas = fieldCanvasRef.current;
    if (canvas) {
      canvas.width = W * pixelSize;
      canvas.height = H * pixelSize;
    }
    const overlay = overlayCanvasRef.current;
    if (overlay) {
      overlay.width = W * pixelSize;
      overlay.height = H * pixelSize;
    }
  }, [W, H, pixelSize]);

  useEffect(() => {
    const sim = simRef.current;
    if (sim) {
      updateOmegaSpread(sim, omegaSpread);
    }
  }, [omegaSpread]);

  useEffect(() => {
    const sim = simRef.current;
    if (!sim || !overlayCanvasRef.current) return;
    const cleanup = attachPointerHandlers({
      canvas: overlayCanvasRef.current,
      sim,
      draw: drawRef.current,
      W,
      H,
      pixelSize,
      lineWidth,
      emBlur,
      closeThreshold,
      autoClose,
      imageTool,
      transformTool,
      openContextMenu: onPlaneContextMenu,
      closeContextMenu: onPlaneContextMenuClose,
      onLayersChanged: notifyLayersChanged,
    });
    return cleanup;
  }, [W, H, pixelSize, lineWidth, emBlur, closeThreshold, autoClose, imageTool, transformTool, notifyLayersChanged, onPlaneContextMenu, onPlaneContextMenuClose]);

  useEffect(() => {
    const sim = simRef.current;
    if (!sim) return;
    recomputePotential(sim, emBlur);
  }, [emBlur, W, H]);

  const renderCurrentFrame = useCallback(
    (sim: SimulationState) => {
      renderFrame(
        sim,
        drawRef.current,
        { field: fieldCanvasRef.current, overlay: overlayCanvasRef.current },
        {
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
          showTransformHandles: transformTool,
          showRhoOverlay,
          showDefectsOverlay,
        },
      );
    },
    [
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
      transformTool,
    ],
  );

  const runSimulationSteps = useCallback(
    (sim: SimulationState, cfg: StepConfig) => {
      for (let s = 0; s < stepsPerFrame; s++) {
        stepSimulation(sim, cfg);
      }
    },
    [stepsPerFrame],
  );

  useEffect(() => {
    const loop = () => {
      const sim = simRef.current;
      if (sim) {
        if (!paused) {
          if (attentionEnabled) {
            const out = updateAttentionFields(sim, attentionHeads, attentionParams, dt, attentionTimeRef.current, wrap);
            attentionModsRef.current = {
              Aact: out.Aact,
              Uact: out.Uact,
              lapA: out.lapA,
              divU: out.divU,
              gammaK: attentionParams.gammaK,
              betaK: attentionParams.betaK,
              gammaAlpha: attentionParams.gammaAlpha,
              betaAlpha: attentionParams.betaAlpha,
              gammaD: attentionParams.gammaD,
              deltaD: attentionParams.deltaD,
            };
            attentionTimeRef.current += dt * stepsPerFrame;
          } else {
            attentionModsRef.current = undefined;
          }
          const cfg = buildStepConfig(dt);
          runSimulationSteps(sim, cfg);
        }
        renderCurrentFrame(sim);
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [
    paused,
    buildStepConfig,
    dt,
    renderCurrentFrame,
    runSimulationSteps,
    attentionEnabled,
    attentionParams,
    attentionHeads,
    stepsPerFrame,
    wrap,
  ]);

  useEffect(() => {
    return () => disposeImages(drawRef.current);
  }, []);

  const stepOnce = useCallback(
    (dtOverride: number) => {
      const sim = simRef.current;
      if (!sim) return;
      const cfg = buildStepConfig(dtOverride);
      runSimulationSteps(sim, cfg);
      renderCurrentFrame(sim);
    },
    [
      buildStepConfig,
      renderCurrentFrame,
      runSimulationSteps,
    ],
  );

  const stepWithSeed = useCallback(
    (dtOverride: number, seed: number | null) => {
      const sim = simRef.current;
      if (!sim) return;
      if (seed !== null) {
        setNoiseSeed(sim, seed);
      }
      const cfg = buildStepConfig(dtOverride);
      runSimulationSteps(sim, cfg);
      renderCurrentFrame(sim);
    },
    [
      buildStepConfig,
      renderCurrentFrame,
      runSimulationSteps,
    ],
  );

  const resetPhases = useCallback(() => {
    const sim = simRef.current;
    if (!sim) return;
    resetPhasesSim(sim);
  }, []);

  const clearWalls = useCallback(() => {
    const sim = simRef.current;
    if (!sim) return;
    clearWallsSim(sim);
  }, []);

  const clearPlanes = useCallback(() => {
    const sim = simRef.current;
    if (!sim) return;
    clearPlanesFromLayers(sim, drawRef.current, pixelSize, lineWidth, emBlur);
    notifyLayersChanged();
  }, [emBlur, lineWidth, pixelSize, notifyLayersChanged]);

  const removeLastPlane = useCallback(() => {
    const sim = simRef.current;
    if (!sim) return;
    removeLastPlaneFromLayers(sim, drawRef.current, pixelSize, lineWidth, emBlur);
    notifyLayersChanged();
  }, [emBlur, lineWidth, pixelSize, notifyLayersChanged]);

  const clearStrokes = useCallback(() => {
    const sim = simRef.current;
    if (!sim) return;
    clearStrokesFromLayers(sim, drawRef.current, pixelSize, lineWidth, emBlur);
  }, [emBlur, lineWidth, pixelSize]);

  const addImages = useCallback(
    (files: FileList | null) => {
      const sim = simRef.current;
      if (!sim) return;
      addImagesToLayers(sim, drawRef.current, files, W, H, pixelSize, lineWidth, emBlur, imageOutlineWidth);
    },
    [W, H, pixelSize, lineWidth, emBlur, imageOutlineWidth],
  );

  const removeLastImage = useCallback(() => {
    const sim = simRef.current;
    if (!sim) return;
    removeLastImageFromLayers(sim, drawRef.current, pixelSize, lineWidth, emBlur);
  }, [emBlur, lineWidth, pixelSize]);

  const clearImages = useCallback(() => {
    const sim = simRef.current;
    if (!sim) return;
    clearImagesFromLayers(sim, drawRef.current, pixelSize, lineWidth, emBlur);
  }, [emBlur, lineWidth, pixelSize]);

  const updateImageOutlineWidth = useCallback(
    (width: number) => {
      setImageOutlineWidth(width);
      const sim = simRef.current;
      if (!sim) return;
      updateImageOutlineWidthLayers(sim, drawRef.current, width, pixelSize, lineWidth, emBlur);
    },
    [emBlur, lineWidth, pixelSize, setImageOutlineWidth],
  );

  const getPlaneLayersSnapshot = useCallback((): PlaneLayerSnapshot[] => {
    const sim = simRef.current;
    if (!sim) return [];
    const draw = drawRef.current;
    const activeSet = sim.activePlaneSet ?? new Set<number>();
    return draw.layers
      .filter((layer): layer is PlaneLayer => layer.kind === "plane")
      .map((layer) => {
        const meta = sim.planeMeta.get(layer.planeId);
        const orientation = meta?.orientation ?? layer.orientation;
        const orientationSign = meta?.orientationSign ?? layer.orientationSign;
        const centroid = meta?.centroid ?? layer.centroid;
        const color = meta?.color ?? ([255, 255, 255] as [number, number, number]);
        const solo = meta?.solo ?? false;
        const muted = meta?.muted ?? false;
        const locked = meta?.locked ?? false;
        const order = meta?.order;
        const active = activeSet.size === 0 ? !muted : activeSet.has(layer.planeId);
        const snapshot: PlaneLayerSnapshot = {
          planeId: layer.planeId,
          layerId: layer.id,
          orientation,
          orientationSign,
          centroid,
          outlineWidth: layer.outlineWidth,
          color,
          solo,
          muted,
          locked,
          order,
          active,
        };
        return snapshot;
      });
  }, []);

  const getSelectedPlaneIds = useCallback(() => {
    return [...drawRef.current.selectedPlaneIds];
  }, []);

  const setSelectedPlaneIdsApi = useCallback(
    (ids: number[]) => {
      const normalized = Array.from(new Set(ids));
      const current = drawRef.current.selectedPlaneIds;
      if (
        current.length === normalized.length &&
        current.every((value, idx) => value === normalized[idx])
      ) {
        return;
      }
      setPlaneSelection(drawRef.current, normalized);
      notifyLayersChanged();
    },
    [notifyLayersChanged],
  );

  const togglePlaneSolo = useCallback(
    (planeId: number, value: boolean) => {
      const sim = simRef.current;
      if (!sim) return;
      if (!setPlaneSolo(sim, planeId, value)) return;
      recomputePlaneDepthFromLayers(sim, drawRef.current);
      rebuildWallFromLayers(sim, drawRef.current, pixelSize, lineWidth, emBlur);
      notifyLayersChanged();
    },
    [emBlur, lineWidth, pixelSize, notifyLayersChanged],
  );

  const togglePlaneMuted = useCallback(
    (planeId: number, value: boolean) => {
      const sim = simRef.current;
      if (!sim) return;
      if (!setPlaneMuted(sim, planeId, value)) return;
      recomputePlaneDepthFromLayers(sim, drawRef.current);
      rebuildWallFromLayers(sim, drawRef.current, pixelSize, lineWidth, emBlur);
      notifyLayersChanged();
    },
    [emBlur, lineWidth, pixelSize, notifyLayersChanged],
  );

  const togglePlaneLocked = useCallback(
    (planeId: number, value: boolean) => {
      const sim = simRef.current;
      if (!sim) return;
      if (!setPlaneLocked(sim, planeId, value)) return;
      notifyLayersChanged();
    },
    [notifyLayersChanged],
  );

  const reorderPlane = useCallback(
    (sourcePlaneId: number, targetPlaneId: number | null) => {
      const sim = simRef.current;
      if (!sim) return;
      const draw = drawRef.current;
      const prevLength = draw.planeOrderUndo.length;
      pushPlaneOrderSnapshot();
      const snapshotAdded = draw.planeOrderUndo.length > prevLength;
      if (!reorderPlaneLayer(sim, drawRef.current, sourcePlaneId, targetPlaneId)) {
        if (snapshotAdded) {
          draw.planeOrderUndo.pop();
        }
        return;
      }
      recomputePlaneDepthFromLayers(sim, drawRef.current);
      rebuildWallFromLayers(sim, drawRef.current, pixelSize, lineWidth, emBlur);
      notifyLayersChanged();
    },
    [emBlur, lineWidth, pixelSize, notifyLayersChanged, pushPlaneOrderSnapshot],
  );

  const undoPlaneOrder = useCallback(() => {
    const sim = simRef.current;
    const draw = drawRef.current;
    if (!sim || !draw) return;
    if (draw.planeOrderUndo.length === 0) return;
    const target = draw.planeOrderUndo.pop();
    if (!target) return;
    const current = getPlaneOrder(draw);
    const applied = applyPlaneOrder(sim, draw, target, pixelSize, lineWidth, emBlur);
    if (!applied) {
      draw.planeOrderUndo.push(target);
      return;
    }
    draw.planeOrderRedo.push(current);
    notifyLayersChanged();
  }, [emBlur, lineWidth, pixelSize, notifyLayersChanged]);

  const redoPlaneOrder = useCallback(() => {
    const sim = simRef.current;
    const draw = drawRef.current;
    if (!sim || !draw) return;
    if (draw.planeOrderRedo.length === 0) return;
    const target = draw.planeOrderRedo.pop();
    if (!target) return;
    const current = getPlaneOrder(draw);
    const applied = applyPlaneOrder(sim, draw, target, pixelSize, lineWidth, emBlur);
    if (!applied) {
      draw.planeOrderRedo.push(target);
      return;
    }
    draw.planeOrderUndo.push(current);
    if (draw.planeOrderUndo.length > MAX_PLANE_ORDER_HISTORY) {
      draw.planeOrderUndo.shift();
    }
    notifyLayersChanged();
  }, [emBlur, lineWidth, pixelSize, notifyLayersChanged]);

  const canUndoPlaneOrder = useCallback(() => {
    const draw = drawRef.current;
    return draw ? draw.planeOrderUndo.length > 0 : false;
  }, []);

  const canRedoPlaneOrder = useCallback(() => {
    const draw = drawRef.current;
    return draw ? draw.planeOrderRedo.length > 0 : false;
  }, []);

  const setTransportNoiseSeed = useCallback((seed: number) => {
    const sim = simRef.current;
    if (!sim) return;
    setNoiseSeed(sim, seed);
  }, []);

  const getTransportNoiseSeed = useCallback(() => {
    const sim = simRef.current;
    return sim ? sim.noiseSeed : 0;
  }, []);

  const deletePlanes = useCallback(
    (mode: "keep-wall" | "keep-energy" | "clean") => {
      const sim = simRef.current;
      const draw = drawRef.current;
      if (!sim || !draw) return;
      if (draw.selectedPlaneIds.length === 0) return;
      pushPlaneOrderSnapshot();
      const changed = deletePlanesFromLayers(
        sim,
        draw,
        [...draw.selectedPlaneIds],
        mode,
        pixelSize,
        lineWidth,
        emBlur,
        energyBaseline,
      );
      if (!changed) return;
      notifyLayersChanged();
    },
    [pushPlaneOrderSnapshot, pixelSize, lineWidth, emBlur, energyBaseline, notifyLayersChanged],
  );

  const transformPlane = useCallback(
    (planeId: number, action: PlaneTransformAction) => {
      const sim = simRef.current;
      const draw = drawRef.current;
      if (!sim || !draw) return;
      const changed = transformPlaneShape(sim, draw, planeId, action, pixelSize, lineWidth, emBlur);
      if (!changed) return;
      notifyLayersChanged();
    },
    [pixelSize, lineWidth, emBlur, notifyLayersChanged],
  );

  const combinePlanes = useCallback(
    (basePlaneId: number, otherPlaneIds: number[], action: PlaneBooleanAction) => {
      const sim = simRef.current;
      const draw = drawRef.current;
      if (!sim || !draw) return;
      if (otherPlaneIds.length === 0) return;
      const changed = booleanCombinePlanes(
        sim,
        draw,
        basePlaneId,
        otherPlaneIds,
        action,
        pixelSize,
        lineWidth,
        emBlur,
      );
      if (!changed) return;
      notifyLayersChanged();
    },
    [pixelSize, lineWidth, emBlur, notifyLayersChanged],
  );

  return {
    fieldCanvasRef,
    overlayCanvasRef,
    fileInputRef,
    step: stepOnce,
    stepWithSeed,
    resetPhases,
    clearWalls,
    clearPlanes,
    removeLastPlane,
    clearStrokes,
    addImages,
    removeLastImage,
    clearImages,
    updateImageOutlineWidth,
    getPlaneLayersSnapshot,
    getSelectedPlaneIds,
    setSelectedPlaneIds: setSelectedPlaneIdsApi,
    togglePlaneSolo,
    togglePlaneMuted,
    togglePlaneLocked,
    reorderPlane,
    undoPlaneOrder,
    redoPlaneOrder,
    canUndoPlaneOrder,
    canRedoPlaneOrder,
    deletePlanes,
    setTransportNoiseSeed,
    getTransportNoiseSeed,
    transformPlane,
    combinePlanes,
    subscribeLayerChanges,
    getSim: () => simRef.current,
  };
}
