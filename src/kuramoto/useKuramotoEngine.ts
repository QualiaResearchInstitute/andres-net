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
} from "./layers";
import { renderFrame } from "./renderFrame";
import {
  clearWalls as clearWallsSim,
  createSimulation,
  recomputePotential,
  resetPhases as resetPhasesSim,
  updateOmegaSpread,
} from "./simulation";
import { stepSimulation, StepConfig } from "./stepSimulation";
import { DrawingState, PlaneLayer, SimulationState } from "./types";

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
  showImages: boolean;
  imageOutlineWidth: number;
  showWalls: boolean;
  showLines: boolean;
  showOutlines: boolean;
  showDagDepth: boolean;
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
  setImageOutlineWidth: (width: number) => void;
};

export type KuramotoEngineApi = {
  fieldCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  overlayCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  fileInputRef: React.MutableRefObject<HTMLInputElement | null>;
  step: (dt: number) => void;
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
  subscribeLayerChanges: (listener: () => void) => () => void;
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
    showImages,
    imageOutlineWidth,
    showWalls,
    showLines,
    showOutlines,
    showDagDepth,
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
    setImageOutlineWidth,
  } = config;

  const fieldCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const rafRef = useRef<number>(0);
  const simRef = useRef<SimulationState | null>(null);
  const drawRef = useRef<DrawingState>(createDrawingState());
  const layerVersionRef = useRef(0);
  const layerListenersRef = useRef<Set<() => void>>(new Set());

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
      onLayersChanged: notifyLayersChanged,
    });
    return cleanup;
  }, [W, H, pixelSize, lineWidth, emBlur, closeThreshold, autoClose, imageTool, notifyLayersChanged]);

  useEffect(() => {
    const sim = simRef.current;
    if (!sim) return;
    recomputePotential(sim, emBlur);
  }, [emBlur, W, H]);

  useEffect(() => {
    const loop = () => {
      const sim = simRef.current;
      if (sim) {
        if (!paused) {
          for (let s = 0; s < stepsPerFrame; s++) {
            stepSimulation(sim, buildStepConfig(dt));
          }
        }
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
          },
        );
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [
    paused,
    stepsPerFrame,
    buildStepConfig,
    dt,
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
  ]);

  useEffect(() => {
    return () => disposeImages(drawRef.current);
  }, []);

  const stepOnce = useCallback(
    (dtOverride: number) => {
      const sim = simRef.current;
      if (!sim) return;
      stepSimulation(sim, buildStepConfig(dtOverride));
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
        },
      );
    },
    [
      buildStepConfig,
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
      if (!reorderPlaneLayer(sim, drawRef.current, sourcePlaneId, targetPlaneId)) return;
      recomputePlaneDepthFromLayers(sim, drawRef.current);
      rebuildWallFromLayers(sim, drawRef.current, pixelSize, lineWidth, emBlur);
      notifyLayersChanged();
    },
    [emBlur, lineWidth, pixelSize, notifyLayersChanged],
  );

  return {
    fieldCanvasRef,
    overlayCanvasRef,
    fileInputRef,
    step: stepOnce,
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
    subscribeLayerChanges,
  };
}
