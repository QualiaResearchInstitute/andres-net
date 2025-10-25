import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useKuramotoEngine } from "./kuramoto/useKuramotoEngine";
import type { PlaneLayerSnapshot } from "./kuramoto/useKuramotoEngine";
import { Section, Row, Range, Toggle } from "./kuramoto/uiPrimitives";
import type { PlaneBooleanAction, PlaneTransformAction } from "./kuramoto/layers";
import { buildFeatureMatrix, quantizeToCodebook, makeToyPatCodebook, fitRidge, applyRidgeBatch, computeRGBTargets, mse, psnrFromMSE, patReconMSEFromDists } from "./kuramoto/reservoirReadout";
import type { RidgeHead } from "./kuramoto/reservoirReadout";
import { buildPatchesFromNeighbors } from "./kuramoto/reverseSampler";
import { computeAffect } from "./kuramoto/affect";
import { reverseStepInPlace } from "./kuramoto/reverseRunner";
import type { StepConfig } from "./kuramoto/stepSimulation";
import { h_phi } from "./kuramoto/controller";
import type { ControlSchedule } from "./kuramoto/controller";
import { computeMetrics } from "./kuramoto/eval";
import { runForward, runReverse, runFwdRev } from "./kuramoto/diffusion";
import { buildPATokens } from "./tokens/pat";
import { tokensToAttentionFields } from "./tokens/tokenOnly";
import { startPATStream } from "./tokens/stream";
import { updateAttentionFields } from "./kuramoto/attention";
import { startSceneRecorder, serializeSceneWithSchedules } from "./kuramoto/sceneRecorder";
import type { AttentionMods } from "./kuramoto/stepSimulation";
import { ReflectorGraphPanel } from "./kuramoto/graph/ReflectorGraphPanel";
import { stepValencePolicy } from "./kuramoto/policyValence";
import { TokenAtlas } from "./panels/TokenAtlas";
import type { PAToken } from "./tokens/pat";

function SchedulePlot({
  K,
  D,
  Kref,
  samples = 101,
  width = 240,
  height = 80,
  strokeK = "rgb(56,189,248)", // cyan-400
  strokeD = "rgb(244,114,182)", // pink-400
  strokeKref = "rgb(34,197,94)", // green-500
}: {
  K: (t: number) => number;
  D: (t: number) => number;
  Kref?: (t: number) => number;
  samples?: number;
  width?: number;
  height?: number;
  strokeK?: string;
  strokeD?: string;
  strokeKref?: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    // Sample
    const T = Math.max(2, samples | 0);
    const Ks: number[] = [];
    const Ds: number[] = [];
    const Krs: number[] = [];
    for (let i = 0; i < T; i++) {
      const t = i / (T - 1);
      Ks.push(K(t));
      Ds.push(D(t));
      if (Kref) Krs.push(Kref(t));
    }
    // Normalize to [0,1] for plotting (each series independently)
    const minK = Math.min(...Ks);
    const maxK = Math.max(...Ks);
    const minD = Math.min(...Ds);
    const maxD = Math.max(...Ds);
    const minKr = Krs.length > 0 ? Math.min(...Krs) : 0;
    const maxKr = Krs.length > 0 ? Math.max(...Krs) : 1;
    const nrm = (v: number, lo: number, hi: number) => {
      const d = hi - lo;
      return d > 1e-12 ? (v - lo) / d : 0.5;
    };
    // Axes
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 0.5);
    ctx.lineTo(width, height - 0.5);
    ctx.moveTo(0.5, 0);
    ctx.lineTo(0.5, height);
    ctx.stroke();
    // Plot K
    ctx.strokeStyle = strokeK;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < T; i++) {
      const x = (i / (T - 1)) * (width - 1);
      const y = (1 - nrm(Ks[i], minK, maxK)) * (height - 1);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // Plot D
    ctx.strokeStyle = strokeD;
    ctx.beginPath();
    for (let i = 0; i < T; i++) {
      const x = (i / (T - 1)) * (width - 1);
      const y = (1 - nrm(Ds[i], minD, maxD)) * (height - 1);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Plot Kref if provided
    if (Kref && Krs.length === T) {
      ctx.strokeStyle = strokeKref;
      ctx.beginPath();
      for (let i = 0; i < T; i++) {
        const x = (i / (T - 1)) * (width - 1);
        const y = (1 - nrm(Krs[i], minKr, maxKr)) * (height - 1);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }, [K, D, Kref, samples, width, height, strokeK, strokeD, strokeKref]);
  return <canvas ref={ref} className="rounded border border-zinc-800 bg-zinc-900" />;
}

export default function KuramotoPainter() {
  const [W, setW] = useState(96);
  const [H, setH] = useState(96);
  const [pixelSize, setPixelSize] = useState(7);

  const [paused, setPaused] = useState(false);
  const [dt, setDt] = useState(0.03);
  const [stepsPerFrame, setStepsPerFrame] = useState(1);
  const [wrap, setWrap] = useState(true);

  const [Kbase, setKbase] = useState(0.8);
  const [K1, setK1] = useState(1.0);
  const [K2, setK2] = useState(-0.3);
  const [K3, setK3] = useState(0.15);
  const [omegaSpread, setOmegaSpread] = useState(0.25);
  const [noiseAmp, setNoiseAmp] = useState(0.0);

  const [swProb, setSwProb] = useState(0.04);
  const [swEdgesPerNode, setSwEdgesPerNode] = useState(2);
  const [swMinDist, setSwMinDist] = useState(6);
  const [swMaxDist, setSwMaxDist] = useState(20);
  const [swWeight, setSwWeight] = useState(0.25);
  const [swNegFrac, setSwNegFrac] = useState(0.2);
  const [reseedGraphKey, setReseedGraphKey] = useState(0);

  const [lineWidth, setLineWidth] = useState(4);
  const [closeThreshold, setCloseThreshold] = useState(16);
  const [autoClose, setAutoClose] = useState(true);
  const [emBlur, setEmBlur] = useState(6);
  const [emGain, setEmGain] = useState(0.6);
  const [wallBarrier, setWallBarrier] = useState(0.5);
  const [imageTool, setImageTool] = useState(false);
  const [transformTool, setTransformTool] = useState(false);
  const [showImages, setShowImages] = useState(true);
  const [imageOutlineWidth, setImageOutlineWidth] = useState(4);

  const [showWalls, setShowWalls] = useState(true);
  const [showLines, setShowLines] = useState(true);
  const [showOutlines, setShowOutlines] = useState(true);
  const [showDagDepth, setShowDagDepth] = useState(false);
  const [showRhoOverlay, setShowRhoOverlay] = useState(false);
  const [showDefectsOverlay, setShowDefectsOverlay] = useState(false);
  // Affect headers (Phase-1 lightweight)
  const [arousal, setArousal] = useState(0);
  const [valence, setValence] = useState(0);
  const [targetArousal, setTargetArousal] = useState(0.5);
  const [targetValence, setTargetValence] = useState(0.5);
  const [valenceControl, setValenceControl] = useState(false);
  const [arousalBand, setArousalBand] = useState(0.1);
  const [policyEveryN, setPolicyEveryN] = useState(8);
  const policySeedRef = useRef(0);
  const policyFrameRef = useRef(0);
  const [rgbHead, setRgbHead] = useState<RidgeHead | null>(null);
  const [ctrlSched, setCtrlSched] = useState<ControlSchedule | null>(null);
  const [metrics, setMetrics] = useState<{ rhoMean:number; rhoStd:number; defect:number; entropy:number; aniso:number } | null>(null);
  const [determinismRMSE, setDeterminismRMSE] = useState<number | null>(null);
  const [rgbMSE, setRgbMSE] = useState<number | null>(null);
  const [rgbPSNR, setRgbPSNR] = useState<number | null>(null);
  const [patMSE, setPatMSE] = useState<number | null>(null);
  const [patUsage, setPatUsage] = useState<number | null>(null);
  const [patEntropy, setPatEntropy] = useState<number | null>(null);
  // PAT streaming controls
  const [patStreamUrl, setPatStreamUrl] = useState("ws://localhost:9090");
  const [patStreamFps, setPatStreamFps] = useState(5);
  const [patScale, setPatScale] = useState<32 | 64>(32);
  const [patStreaming, setPatStreaming] = useState(false);
  const patStreamRef = useRef<{ stop: () => void; ws: WebSocket } | null>(null);
  // Token-only attention and TokenAtlas
  const [tokenOnly, setTokenOnly] = useState(false);
  const [tokenAtlasLive, setTokenAtlasLive] = useState(false);
  const [tokenAtlasTokens, setTokenAtlasTokens] = useState<PAToken[] | null>(null);
  const [tokenAtlasEveryMs, setTokenAtlasEveryMs] = useState(1000);

  const [energyBaseline, setEnergyBaseline] = useState(0.35);
  const [energyLeak, setEnergyLeak] = useState(0.02);
  const [energyDiff, setEnergyDiff] = useState(0.12);
  const [sinkLine, setSinkLine] = useState(0.03);
  const [sinkSurf, setSinkSurf] = useState(0.02);
  const [sinkHyp, setSinkHyp] = useState(0.02);
  const [trapSurf, setTrapSurf] = useState(0.2);
  const [trapHyp, setTrapHyp] = useState(0.35);
  const [minEnergySurf, setMinEnergySurf] = useState(0.25);
  const [minEnergyHyp, setMinEnergyHyp] = useState(0.4);
  const [brightnessBase, setBrightnessBase] = useState(0.85);
  const [energyGamma, setEnergyGamma] = useState(1.1);

  const [alphaSurfToField, setAlphaSurfToField] = useState(0.6);
  const [alphaFieldToSurf, setAlphaFieldToSurf] = useState(0.5);
  const [KS1, setKS1] = useState(0.9);
  const [KS2, setKS2] = useState(0.0);
  const [KS3, setKS3] = useState(0.0);

  const [alphaHypToField, setAlphaHypToField] = useState(0.8);
  const [alphaFieldToHyp, setAlphaFieldToHyp] = useState(0.4);
  const [KH1, setKH1] = useState(0.7);
  const [KH2, setKH2] = useState(0.0);
  const [KH3, setKH3] = useState(0.0);

  const [liftGain, setLiftGain] = useState(0.8);
  const [hyperCurve, setHyperCurve] = useState(0.6);

  // Attention & Awareness states
  const [attentionEnabled, setAttentionEnabled] = useState(true);
  const [att_DA, setAtt_DA] = useState(0.6);
  const [att_DU, setAtt_DU] = useState(0.6);
  const [att_muA, setAtt_muA] = useState(0.12);
  const [att_muU, setAtt_muU] = useState(0.12);
  const [att_lambdaS, setAtt_lambdaS] = useState(1.0);
  const [att_lambdaC, setAtt_lambdaC] = useState(0.7);
  const [att_topo, setAtt_topo] = useState(0.8);
  const [att_gammaK, setAtt_gammaK] = useState(0.6);
  const [att_betaK, setAtt_betaK] = useState(0.2);
  const [att_gammaAlpha, setAtt_gammaAlpha] = useState(0.15);
  const [att_betaAlpha, setAtt_betaAlpha] = useState(0.1);
  const [att_gammaD, setAtt_gammaD] = useState(0.4);
  const [att_deltaD, setAtt_deltaD] = useState(0.15);
  const [att_wGrad, setAtt_wGrad] = useState(1.0);
  const [att_wLap, setAtt_wLap] = useState(0.6);
  const [att_wE, setAtt_wE] = useState(0.4);
  const [att_wDef, setAtt_wDef] = useState(1.0);
  const [att_contextR, setAtt_contextR] = useState(6);
  const [att_reflectorGain, setAtt_reflectorGain] = useState(0);
  const [head1, setHead1] = useState({ enabled: true, weight: 0.9, freq: 8, phase: 0, radius: 10, bindPrimaryPlane: true });
  const [head2, setHead2] = useState({ enabled: false, weight: 0.6, freq: 12, phase: 0, radius: 12, bindPrimaryPlane: true });
  const [head3, setHead3] = useState({ enabled: false, weight: 0.4, freq: 6, phase: 0, radius: 14, bindPrimaryPlane: false });

  const [eventBarrier, setEventBarrier] = useState(1.0);
  const [dagSweeps, setDagSweeps] = useState(0);
  const [dagDepthOrdering, setDagDepthOrdering] = useState(true);
  const [dagDepthFiltering, setDagDepthFiltering] = useState(true);
  const [dagLogStats, setDagLogStats] = useState(false);

  // Pockets (topological regions) controls
  const [pocketEnabled, setPocketEnabled] = useState(false);
  const [pocketWRho, setPocketWRho] = useState(1.0);
  const [pocketWAttn, setPocketWAttn] = useState(0.0);
  const [pocketWShear, setPocketWShear] = useState(1.0);
  const [pocketWDef, setPocketWDef] = useState(0.5);
  const [pocketThresh, setPocketThresh] = useState(0.6);
  const [pocketMinArea, setPocketMinArea] = useState(24);
  const [pocketHorizon, setPocketHorizon] = useState<"sealed" | "oneway" | "none">("sealed");
  const [pocketKboost, setPocketKboost] = useState(0.2);

  const [stereo, setStereo] = useState(false);
  const [ipd, setIpd] = useState(2.0);
  const [stereoAlpha, setStereoAlpha] = useState(0.5);
  const [transportSeed, setTransportSeed] = useState(1337);
  const [pinTransportSeed, setPinTransportSeed] = useState(true);
  const [stepBurst, setStepBurst] = useState(1);

  const [planeLayers, setPlaneLayers] = useState<PlaneLayerSnapshot[]>([]);
  const [selectedPlanes, setSelectedPlanesState] = useState<number[]>([]);
  const [layerVersion, setLayerVersion] = useState(0);
  const [draggingPlaneId, setDraggingPlaneId] = useState<number | null>(null);
  const [planeMenu, setPlaneMenu] = useState<{ planeId: number; x: number; y: number } | null>(null);
  const openPlaneMenu = useCallback((info: { planeId: number; screenX: number; screenY: number }) => {
    setPlaneMenu({ planeId: info.planeId, x: info.screenX, y: info.screenY });
  }, []);
  const closePlaneMenu = useCallback(() => {
    setPlaneMenu(null);
  }, []);

  const attentionParams = useMemo(
    () => ({
      DA: att_DA, DU: att_DU, muA: att_muA, muU: att_muU,
      lambdaS: att_lambdaS, lambdaC: att_lambdaC, topoGain: att_topo,
      gammaK: att_gammaK, betaK: att_betaK,
      gammaAlpha: att_gammaAlpha, betaAlpha: att_betaAlpha,
      gammaD: att_gammaD, deltaD: att_deltaD,
      wGrad: att_wGrad, wLap: att_wLap, wE: att_wE, wDef: att_wDef,
      contextRadius: att_contextR,
      reflectorGain: att_reflectorGain,
      etaV: 0,
      aClamp: 4.0, uClamp: 4.0,
    }),
    [att_DA, att_DU, att_muA, att_muU, att_lambdaS, att_lambdaC, att_topo, att_gammaK, att_betaK, att_gammaAlpha, att_betaAlpha, att_gammaD, att_deltaD, att_wGrad, att_wLap, att_wE, att_wDef, att_contextR, att_reflectorGain],
  );
  const attentionHeads = useMemo(() => [head1, head2, head3], [head1, head2, head3]);

  const engine = useKuramotoEngine({
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
    // Topological pockets
    pocketParams: {
      enabled: pocketEnabled,
      scoreWeights: { rho: pocketWRho, attn: pocketWAttn, shear: pocketWShear, defects: pocketWDef },
      scoreThresh: pocketThresh,
      minArea: Math.max(1, Math.floor(pocketMinArea)),
      horizon: pocketHorizon,
      Kboost: pocketKboost,
    },
    // Opaque substrate (PAT token-only attention)
    opaqueSubstrateMode: tokenOnly,
    patScale,
    // Attention
    attentionEnabled,
    attentionParams,
    attentionHeads,
    setImageOutlineWidth,
    onPlaneContextMenu: openPlaneMenu,
    onPlaneContextMenuClose: closePlaneMenu,
  });

  const {
    fieldCanvasRef,
    overlayCanvasRef,
    fileInputRef,
    step,
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
    getSelectedPlaneIds: readSelectedPlaneIds,
    setSelectedPlaneIds: syncSelectedPlaneIds,
    togglePlaneSolo,
    togglePlaneMuted,
    togglePlaneLocked,
    reorderPlane,
    subscribeLayerChanges,
    stepWithSeed,
    undoPlaneOrder,
    redoPlaneOrder,
    canUndoPlaneOrder,
    canRedoPlaneOrder,
    deletePlanes,
    setTransportNoiseSeed,
    getTransportNoiseSeed,
    transformPlane,
    combinePlanes,
    getSim,
  } = engine;

  const rebuildSmallWorld = useCallback(() => setReseedGraphKey((k) => k + 1), []);

  const undoAvailable = canUndoPlaneOrder();
  const redoAvailable = canRedoPlaneOrder();
  const hasSelection = selectedPlanes.length > 0;
  const menuPosition = useMemo(() => {
    if (!planeMenu) return null;
    if (typeof window === "undefined") {
      return { left: planeMenu.x, top: planeMenu.y };
    }
    return {
      left: Math.min(window.innerWidth - 220, planeMenu.x),
      top: Math.min(window.innerHeight - 260, planeMenu.y),
    };
  }, [planeMenu]);

  const transformActions = useMemo(
    () =>
      [
        { key: "flip-horizontal" as PlaneTransformAction, label: "Flip Horizontal" },
        { key: "flip-vertical" as PlaneTransformAction, label: "Flip Vertical" },
        { key: "rotate-cw" as PlaneTransformAction, label: "Rotate 90° CW" },
        { key: "rotate-ccw" as PlaneTransformAction, label: "Rotate 90° CCW" },
        { key: "skew-x-pos" as PlaneTransformAction, label: "Skew X +" },
        { key: "skew-x-neg" as PlaneTransformAction, label: "Skew X -" },
        { key: "skew-y-pos" as PlaneTransformAction, label: "Skew Y +" },
        { key: "skew-y-neg" as PlaneTransformAction, label: "Skew Y -" },
        { key: "smooth" as PlaneTransformAction, label: "Smooth Outline" },
      ],
    [],
  );

  const booleanActions = useMemo(
    () =>
      [
        { key: "union" as PlaneBooleanAction, label: "Union with Selection" },
        { key: "subtract" as PlaneBooleanAction, label: "Subtract Selection" },
        { key: "intersect" as PlaneBooleanAction, label: "Intersect Selection" },
      ],
    [],
  );
  const otherSelection = planeMenu ? selectedPlanes.filter((id) => id !== planeMenu.planeId) : [];

  const syncSeedFromSim = useCallback(() => {
    const raw = getTransportNoiseSeed();
    const nextSeed = Number.isFinite(raw) ? Math.floor(raw) : 0;
    setTransportSeed(nextSeed);
    if (pinTransportSeed) {
      setTransportNoiseSeed(nextSeed);
    }
    return nextSeed;
  }, [getTransportNoiseSeed, pinTransportSeed, setTransportNoiseSeed]);

  useEffect(() => {
    const unsubscribe = subscribeLayerChanges(() => {
      setLayerVersion((v) => v + 1);
    });
    return unsubscribe;
  }, [subscribeLayerChanges]);

  useEffect(() => {
    setPlaneLayers(getPlaneLayersSnapshot());
    setSelectedPlanesState(readSelectedPlaneIds());
  }, [getPlaneLayersSnapshot, readSelectedPlaneIds, layerVersion]);

  useEffect(() => {
    syncSeedFromSim();
  }, [syncSeedFromSim, W, H, reseedGraphKey]);

  useEffect(() => {
    if (pinTransportSeed) {
      setTransportNoiseSeed(Number.isFinite(transportSeed) ? Math.floor(transportSeed) : 0);
    }
  }, [pinTransportSeed, setTransportNoiseSeed, transportSeed]);

  useEffect(() => {
    if (selectedPlanes.length === 0 && planeMenu) {
      setPlaneMenu(null);
    }
  }, [selectedPlanes, planeMenu]);

  useEffect(() => {
    if (!planeMenu) return;
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPlaneMenu(null);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("keydown", handleKey);
    };
  }, [planeMenu]);

  useEffect(() => {
    if (!transformTool) {
      setPlaneMenu(null);
    }
  }, [transformTool]);

  const updateSelection = useCallback(
    (ids: number[]) => {
      const normalized = Array.from(new Set(ids));
      setSelectedPlanesState(normalized);
      syncSelectedPlaneIds(normalized);
    },
    [syncSelectedPlaneIds],
  );

  const handlePlaneClick = useCallback(
    (planeId: number, event: React.MouseEvent) => {
      event.preventDefault();
      const isSelected = selectedPlanes.includes(planeId);
      let nextSelection: number[];
      if (event.shiftKey) {
        nextSelection = isSelected
          ? selectedPlanes.filter((id) => id !== planeId)
          : [...selectedPlanes, planeId];
      } else {
        nextSelection = isSelected && selectedPlanes.length === 1 ? selectedPlanes : [planeId];
      }
      updateSelection(nextSelection);
    },
    [selectedPlanes, updateSelection],
  );

  const handleSoloToggle = useCallback(
    (planeId: number, value: boolean) => {
      togglePlaneSolo(planeId, value);
    },
    [togglePlaneSolo],
  );

  const handleMuteToggle = useCallback(
    (planeId: number, value: boolean) => {
      togglePlaneMuted(planeId, value);
    },
    [togglePlaneMuted],
  );

  const handleLockToggle = useCallback(
    (planeId: number, value: boolean) => {
      togglePlaneLocked(planeId, value);
    },
    [togglePlaneLocked],
  );

  const handleDragStart = useCallback(
    (planeId: number) => (event: React.DragEvent<HTMLDivElement>) => {
      const plane = planeLayers.find((p) => p.planeId === planeId);
      if (plane?.locked) {
        event.preventDefault();
        return;
      }
      setDraggingPlaneId(planeId);
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", String(planeId));
    },
    [planeLayers],
  );

  const handleDragEnd = useCallback(() => {
    setDraggingPlaneId(null);
  }, []);

  const handleDragOver = useCallback(
    (planeId: number | null) => (event: React.DragEvent<HTMLDivElement>) => {
      if (draggingPlaneId === null && planeId === null) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
    },
    [draggingPlaneId],
  );

  const handleDrop = useCallback(
    (targetPlaneId: number | null) => (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const data = event.dataTransfer.getData("text/plain");
      const source = draggingPlaneId ?? Number.parseInt(data, 10);
      if (!Number.isFinite(source)) {
        setDraggingPlaneId(null);
        return;
      }
      reorderPlane(source, targetPlaneId);
      setDraggingPlaneId(null);
    },
    [draggingPlaneId, reorderPlane],
  );

  const handleStepBurst = useCallback(() => {
    const frames = Math.max(1, Math.floor(stepBurst));
    if (pinTransportSeed) {
      let seed = Number.isFinite(transportSeed) ? Math.floor(transportSeed) : 0;
      setTransportNoiseSeed(seed);
      for (let i = 0; i < frames; i++) {
        stepWithSeed(dt, seed);
        seed = Number.isFinite(getTransportNoiseSeed()) ? Math.floor(getTransportNoiseSeed()) : seed;
        if (i < frames - 1) {
          setTransportNoiseSeed(seed);
        }
      }
      setTransportSeed(seed);
      setTransportNoiseSeed(seed);
    } else {
      for (let i = 0; i < frames; i++) {
        step(dt);
      }
      syncSeedFromSim();
    }
  }, [
    dt,
    getTransportNoiseSeed,
    pinTransportSeed,
    setTransportNoiseSeed,
    step,
    stepBurst,
    stepWithSeed,
    syncSeedFromSim,
    transportSeed,
  ]);

  const handleStart = useCallback(() => {
    if (pinTransportSeed) {
      const seed = Number.isFinite(transportSeed) ? Math.floor(transportSeed) : 0;
      setTransportNoiseSeed(seed);
    }
    setPaused(false);
  }, [pinTransportSeed, setPaused, setTransportNoiseSeed, transportSeed]);

  const handlePause = useCallback(() => {
    setPaused(true);
    syncSeedFromSim();
  }, [setPaused, syncSeedFromSim]);

  const handleStop = useCallback(() => {
    setPaused(true);
    syncSeedFromSim();
  }, [setPaused, syncSeedFromSim]);

  const handleTransformAction = useCallback(
    (planeId: number, action: PlaneTransformAction) => {
      transformPlane(planeId, action);
      closePlaneMenu();
    },
    [transformPlane, closePlaneMenu],
  );

  const handleBooleanAction = useCallback(
    (planeId: number, others: number[], action: PlaneBooleanAction) => {
      combinePlanes(planeId, others, action);
      closePlaneMenu();
    },
    [combinePlanes, closePlaneMenu],
  );

  // Compute PAT tokens + affect headers (arousal/valence)
  const handleComputePAT = useCallback(() => {
    const sim = getSim();
    if (!sim) return;
    const patches = buildPatchesFromNeighbors(sim);
    const { X, inDim } = buildFeatureMatrix(sim, { includeAE: true, includeRho: true, patches });
    // Toy codebook and quantization
    const codebook = makeToyPatCodebook(inDim, 4);
    const { tokens, dists } = quantizeToCodebook(X, inDim, codebook);
    // MSE proxy from quantization distances (feature space)
    const mPat = patReconMSEFromDists(dists, inDim);
    setPatMSE(mPat);
    // Usage coverage and entropy
    const hist = new Uint32Array(codebook.size);
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      if (t >= 0 && t < hist.length) hist[t]++;
    }
    let used = 0;
    let H = 0;
    const N = Math.max(1, tokens.length);
    for (let c = 0; c < hist.length; c++) {
      const cnt = hist[c];
      if (cnt > 0) used++;
      const p = cnt / N;
      if (p > 0) H += -p * Math.log2(p);
    }
    const coverage = used / Math.max(1, hist.length);
    const normEntropy = H / Math.max(1e-9, Math.log2(Math.max(2, hist.length)));
    setPatUsage(coverage);
    setPatEntropy(normEntropy);
    // Affect headers
    const aff = computeAffect(sim);
    setArousal(Number.isFinite(aff.arousal) ? aff.arousal : 0);
    setValence(Number.isFinite(aff.valence) ? aff.valence : 0);
  }, [getSim]);

  // Reverse config builder mirroring engine step config
  const buildReverseConfig = useCallback((): StepConfig => {
    const sim = getSim();
    // Build attention mods on demand so both forward (manual) and reverse honor gates
    let mods: AttentionMods | undefined = undefined;
    if (attentionEnabled && sim) {
      if (tokenOnly) {
        const tokens = buildPATokens(sim, (patScale as 32 | 64) ?? 32);
        const proxy = tokensToAttentionFields(tokens, W, H, ((patScale as 32 | 64) ?? 32));
        mods = {
          Aact: proxy.Aact,
          Uact: proxy.Uact,
          lapA: proxy.lapA,
          divU: proxy.divU,
          etaV: 0,
          gammaK: att_gammaK,
          betaK: att_betaK,
          gammaAlpha: att_gammaAlpha,
          betaAlpha: att_betaAlpha,
          gammaD: att_gammaD,
          deltaD: att_deltaD,
        };
      } else {
        // Use a simple deterministic time base for reverse utilities
        const out = updateAttentionFields(sim, [head1, head2, head3], attentionParams, dt, 0, wrap);
        mods = {
          Aact: out.Aact,
          Uact: out.Uact,
          lapA: out.lapA,
          divU: out.divU,
          A: out.A,
          lapAraw: out.lapAraw,
          advect: out.advect,
          etaV: attentionParams.etaV ?? 0,
          gammaK: att_gammaK,
          betaK: att_betaK,
          gammaAlpha: att_gammaAlpha,
          betaAlpha: att_betaAlpha,
          gammaD: att_gammaD,
          deltaD: att_deltaD,
        };
      }
    }
    return {
      dt,
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
      attentionMods: mods,
      horizonFactor: (iInside: boolean, jInside: boolean, receiverInside: boolean) => {
        if (iInside === jInside) return 1.0;
        if (receiverInside && !jInside) return 1.0;
        if (!receiverInside && jInside) return 1.0 - eventBarrier;
        return 1.0;
      },
    };
  }, [
    getSim,
    attentionEnabled,
    tokenOnly,
    W,
    H,
    patScale,
    head1,
    head2,
    head3,
    attentionParams,
    att_gammaK,
    att_betaK,
    att_gammaAlpha,
    att_betaAlpha,
    att_gammaD,
    att_deltaD,
    dt, wrap, Kbase, K1, K2, K3, KS1, KS2, KS3, KH1, KH2, KH3,
    alphaSurfToField, alphaFieldToSurf, alphaHypToField, alphaFieldToHyp,
    swWeight, wallBarrier, emGain, energyBaseline, energyLeak, energyDiff,
    sinkLine, sinkSurf, sinkHyp, trapSurf, trapHyp, minEnergySurf, minEnergyHyp,
    noiseAmp, dagSweeps, dagDepthOrdering, dagDepthFiltering, dagLogStats, eventBarrier
  ]);

  // Forward/Reverse multi-step helpers
  const handleForward = useCallback(() => {
    const sim = getSim();
    if (!sim) return;
    const frames = Math.max(1, Math.floor(stepBurst));
    // Simple "wide terminal" schedule: lower K, raise D
    const fSched = {
      K: (t: number) => Math.max(0.05, Kbase * (1 - t)),
      D: (t: number) => Math.max(0, noiseAmp + 0.2 * t),
    };
    runForward(sim, () => buildReverseConfig(), frames, fSched);
  }, [buildReverseConfig, getSim, stepBurst, Kbase, noiseAmp]);

  const handleReverse = useCallback(() => {
    const sim = getSim();
    if (!sim) return;
    const frames = Math.max(1, Math.floor(stepBurst));
    const rSched = {
      K: (t: number) => Math.max(0.05, Kbase * (1 - t)),
      D: (t: number) => Math.max(0, noiseAmp + 0.2 * t),
    };
    runReverse(sim, () => buildReverseConfig(), frames, 0.5, { sched: rSched, t0: 1, t1: 0 });
  }, [buildReverseConfig, getSim, stepBurst, Kbase, noiseAmp]);

  const handleReverseStep = useCallback(() => {
    const sim = getSim();
    if (!sim) return;
    runReverse(sim, () => buildReverseConfig(), 1, 0.5);
  }, [buildReverseConfig, getSim]);

  const handleFwdToRev = useCallback(() => {
    const sim = getSim();
    if (!sim) return;
    // Start from fresh phases for a clean demo
    resetPhases();
    const frames = Math.max(1, Math.floor(stepBurst));
    const fSched = {
      K: (t: number) => Math.max(0.05, Kbase * (1 - t)),
      D: (t: number) => Math.max(0, noiseAmp + 0.2 * t),
    };
    runFwdRev(
      sim,
      () => buildReverseConfig(),
      { steps: frames, sched: fSched },
      { steps: frames, t: 0.5, sched: fSched },
    );
  }, [buildReverseConfig, getSim, resetPhases, stepBurst, Kbase, noiseAmp]);

  // Schedules for plotting and recording
  const schedK = useCallback((t: number) => Math.max(0.05, Kbase * (1 - t)), [Kbase]);
  const schedKref = useCallback((t: number) => Math.max(0, 0.5 * Kbase * (1 - t)), [Kbase]);
  const schedPsi = useCallback((t: number) => 0, []);
  const schedD = useCallback((t: number) => Math.max(0, noiseAmp + 0.2 * t), [noiseAmp]);

  const handleRecordFwdToRev = useCallback(() => {
    const sim = getSim();
    if (!sim) return;
    // Sample schedules for serialization
    const T = 101;
    const tSeries: number[] = [];
    const kSeries: number[] = [];
    const dSeries: number[] = [];
    const krefSeries: number[] = [];
    const psiSeries: number[] = [];
    for (let i = 0; i < T; i++) {
      const t = i / (T - 1);
      tSeries.push(t);
      kSeries.push(schedK(t));
      dSeries.push(schedD(t));
      krefSeries.push(schedKref(t));
      psiSeries.push(schedPsi(t));
    }
    // Record snapshots at start, mid, end
    const rec = startSceneRecorder(getSim);
    rec.snapshot(0.0);
    const frames = Math.max(1, Math.floor(stepBurst));
    const fSched = { K: schedK, D: schedD };
    runForward(sim, () => buildReverseConfig(), frames, fSched, 0, 1);
    rec.snapshot(0.5);
    runReverse(sim, () => buildReverseConfig(), frames, 0.5, { sched: fSched, t0: 1, t1: 0 });
    rec.snapshot(1.0);
    const scene = rec.stop();
    const json = serializeSceneWithSchedules(scene, { t: tSeries, K: kSeries, D: dSeries, Kref: krefSeries, psi: psiSeries });
    const blob = new Blob([JSON.stringify(json)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "scene_fwd_rev.json";
    a.click();
    URL.revokeObjectURL(url);
  }, [getSim, stepBurst, buildReverseConfig, schedK, schedD]);

  const handleFitW = useCallback(() => {
    const sim = getSim();
    if (!sim) return;
    const N = sim.N;
    const { phases, energy } = sim;
    const { X, inDim } = buildFeatureMatrix(sim, { includeAE: true, includeRho: true });
    // Targets computed to match renderer's HSV mapping
    const Y = computeRGBTargets(phases, energy, brightnessBase, energyGamma);
    const outDim = 3;
    const head = fitRidge(X, Y, N, inDim, outDim, 1e-3);
    setRgbHead(head);
    // Baseline metrics
    const Ypred = applyRidgeBatch(head, X, N);
    const m = mse(Ypred, Y);
    const p = psnrFromMSE(m, 1.0);
    setRgbMSE(m);
    setRgbPSNR(p);
  }, [getSim, brightnessBase, energyGamma]);

  const handleFitController = useCallback(() => {
    const sim = getSim();
    const preset = h_phi("neutral", { smallWorldP: swProb, wallGain: wallBarrier }, { N: sim?.N ?? 0, seed: 1234, defaultSteps: 256 });
    setCtrlSched(preset.sched);
  }, [getSim, swProb, wallBarrier]);

  const handleEvalSnapshot = useCallback(() => {
    const sim = getSim();
    if (!sim) return;
    const m = computeMetrics(sim);
    setMetrics({
      rhoMean: m.rhoMean,
      rhoStd: m.rhoStd,
      defect: m.defectDensity,
      entropy: m.entropyGrad,
      aniso: m.anisotropyHV,
    });
    if (typeof console !== "undefined") {
      // also log for baseline capture
      // eslint-disable-next-line no-console
      console.table({
        rhoMean: m.rhoMean.toFixed(4),
        rhoStd: m.rhoStd.toFixed(4),
        defectDensity: m.defectDensity.toExponential(3),
        entropyGrad: m.entropyGrad.toFixed(4),
        anisotropyHV: m.anisotropyHV.toFixed(4),
      });
    }
  }, [getSim]);

  // PAT stream start/stop
  const makePATFrame = useCallback(() => {
    const sim = getSim();
    if (!sim) return null;
    return buildPATokens(sim, patScale);
  }, [getSim, patScale]);

  const handleStartPAT = useCallback(() => {
    try {
      patStreamRef.current?.stop();
    } catch {}
    const handle = startPATStream(makePATFrame, { url: patStreamUrl, fps: Math.max(1, Math.floor(patStreamFps)) });
    patStreamRef.current = handle;
    setPatStreaming(true);
  }, [makePATFrame, patStreamUrl, patStreamFps]);

  const handleStopPAT = useCallback(() => {
    try {
      patStreamRef.current?.stop();
    } catch {}
    patStreamRef.current = null;
    setPatStreaming(false);
  }, []);

  // TokenAtlas live updater
  useEffect(() => {
    if (!tokenAtlasLive) return;
    const id = setInterval(() => {
      const sim = getSim();
      if (!sim) return;
      const toks = buildPATokens(sim, patScale);
      setTokenAtlasTokens(toks);
    }, Math.max(100, tokenAtlasEveryMs));
    return () => clearInterval(id);
  }, [tokenAtlasLive, tokenAtlasEveryMs, getSim, patScale]);

  const handleDeterminismTest = useCallback(() => {
    const sim = getSim();
    if (!sim) return;
    // Use noise-free steps to test bit-identical determinism
    const frames = 8;
    const N = sim.N;
    const original = new Float32Array(sim.phases); // snapshot initial phases
    // Run A
    for (let i = 0; i < frames; i++) {
      step(dt);
    }
    const A = new Float32Array(sim.phases);
    // Reset to original
    sim.phases.set(original);
    // Run B
    for (let i = 0; i < frames; i++) {
      step(dt);
    }
    const B = sim.phases;
    // Compute RMSE
    let se = 0;
    for (let i = 0; i < N; i++) {
      const d = A[i] - B[i];
      se += d * d;
    }
    const rmse = Math.sqrt(se / Math.max(1, N));
    setDeterminismRMSE(rmse);
  }, [getSim, step, dt]);

  // Periodically update affect headers
  useEffect(() => {
    const id = setInterval(() => {
      const sim = getSim();
      if (!sim) return;
      const aff = computeAffect(sim);
      setArousal(Number.isFinite(aff.arousal) ? aff.arousal : 0);
      setValence(Number.isFinite(aff.valence) ? aff.valence : 0);
    }, 500);
    return () => clearInterval(id);
  }, [getSim]);

  // Sync deterministic policy seed when step seed changes or control toggles
  useEffect(() => {
    policySeedRef.current = Number.isFinite(transportSeed) ? Math.floor(transportSeed) : 0;
    policyFrameRef.current = 0;
  }, [transportSeed, valenceControl]);

  // Deterministic valence policy loop: every N frames adjust γK/γD/γα toward higher valence, arousal in band
  useEffect(() => {
    if (!valenceControl) return;
    let raf = 0;
    const tick = () => {
      const sim = getSim();
      if (sim && attentionEnabled && !paused) {
        const n = Math.max(1, Math.floor(policyEveryN));
        policyFrameRef.current += 1;
        if (policyFrameRef.current % n === 0) {
          const aff = computeAffect(sim);
          const cur = {
            gammaK: att_gammaK,
            gammaAlpha: att_gammaAlpha,
            gammaD: att_gammaD,
            deltaD: att_deltaD,
            heads: [head1, head2, head3].map((h) => ({ enabled: h.enabled, weight: h.weight })),
          };
          const cfg = {
            targetArousal: [Math.max(0, targetArousal - arousalBand), Math.min(1, targetArousal + arousalBand)] as [number, number],
            steps: 1,
            gammaK: [0, 1] as [number, number],
            gammaAlpha: [0, 0.5] as [number, number],
            gammaD: [0, 0.8] as [number, number],
            deltaD: [0, 0.3] as [number, number],
            headWeightBounds: [0, 1] as [number, number],
          };
          const res = stepValencePolicy(aff, cur, policySeedRef.current | 0, cfg);
          // Apply clamp-safe updates
          setAtt_gammaK(res.gammaK);
          setAtt_gammaAlpha(res.gammaAlpha);
          setAtt_gammaD(res.gammaD);
          // Keep δD stable for now; enable if policy starts returning tuned values
          // setAtt_deltaD(res.deltaD);

          // Deterministic seed advance (LCG)
          policySeedRef.current = ((policySeedRef.current * 1664525 + 1013904223) >>> 0);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [
    valenceControl,
    getSim,
    attentionEnabled,
    paused,
    policyEveryN,
    att_gammaK,
    att_gammaAlpha,
    att_gammaD,
    att_deltaD,
    head1,
    head2,
    head3,
    targetArousal,
    arousalBand,
  ]);

  useEffect(() => {
    if (typeof console === "undefined") return;
    const group = typeof console.groupCollapsed === "function" ? console.groupCollapsed : console.log;
    const end = typeof console.groupEnd === "function" ? console.groupEnd : (() => undefined);
    group("[KuramotoPainter] smoke tests");
    try {
      console.assert(typeof Section === "function", "Section component defined");
      console.assert(typeof Row === "function", "Row component defined");
      console.assert(typeof Range === "function", "Range component defined");
      console.assert(typeof Toggle === "function", "Toggle component defined");
      console.assert(
        fieldCanvasRef.current === null || fieldCanvasRef.current instanceof HTMLCanvasElement,
        "fieldCanvasRef is canvas or null",
      );
      console.assert(
        overlayCanvasRef.current === null || overlayCanvasRef.current instanceof HTMLCanvasElement,
        "overlayCanvasRef is canvas or null",
      );
    } finally {
      end();
    }
  }, [fieldCanvasRef, overlayCanvasRef]);

  return (
    <>
      <div className="w-full h-full flex gap-4 p-4 bg-black text-zinc-100">
        <div className="flex flex-col items-center">
        <div className="relative rounded-2xl overflow-hidden shadow-2xl" style={{ width: W * pixelSize, height: H * pixelSize }}>
          <canvas ref={fieldCanvasRef} className="block" width={W * pixelSize} height={H * pixelSize} />
          <canvas ref={overlayCanvasRef} className="absolute inset-0" width={W * pixelSize} height={H * pixelSize} />
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-zinc-400">
          <label className="flex items-center gap-2">
            <span>Step seed</span>
            <input
              type="number"
              value={transportSeed}
              onChange={(event) => {
                const next = Number.parseInt(event.target.value, 10);
                if (Number.isFinite(next)) {
                  setTransportSeed(next);
                  setTransportNoiseSeed(Math.floor(next));
                } else {
                  setTransportSeed(0);
                  setTransportNoiseSeed(0);
                }
              }}
              className="w-24 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
            />
          </label>
          <label className="flex items-center gap-2">
            <span>Step ×</span>
            <input
              type="number"
              min={1}
              max={240}
              value={stepBurst}
              onChange={(event) => {
                const next = Number.parseInt(event.target.value, 10);
                setStepBurst(Number.isFinite(next) && next > 0 ? next : 1);
              }}
              className="w-20 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
            />
          </label>
          <Toggle
            value={pinTransportSeed}
            onChange={(value) => {
              setPinTransportSeed(value);
              if (value) {
                setTransportNoiseSeed(Math.floor(transportSeed));
              }
            }}
            label="Pin seed"
          />
        </div>
        <div className="mt-3 flex gap-2 flex-wrap">
          <button onClick={handleStart} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Start
          </button>
          <button onClick={handlePause} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Pause
          </button>
          <button onClick={handleStop} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Stop
          </button>
          <button onClick={handleStepBurst} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Step
          </button>
          <button onClick={handleForward} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Forward
          </button>
          <button onClick={handleReverse} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Reverse
          </button>
          <button onClick={handleReverseStep} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Rev Step
          </button>
          <button onClick={handleFwdToRev} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Fwd→Rev
          </button>
          <button onClick={resetPhases} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Randomize
          </button>
          <button
            onClick={() => {
              clearWalls();
              clearPlanes();
            }}
            className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700"
          >
            Clear Walls+Planes
          </button>
          <button onClick={rebuildSmallWorld} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Reseed SW
          </button>
        </div>
        <p className="mt-2 text-xs text-zinc-400">
          Draw with left-click; close a loop to create a surface. Overlaps (≥2) act as hyperbolic regions. Stereo adds ghosted second eye.
        </p>
      </div>

      <div className="w-[520px] max-w-[52vw] grid grid-cols-2 gap-3 content-start">
        <Section title="Layer Stack">
          <div className="col-span-2 flex flex-col gap-2">
            {planeLayers.length === 0 ? (
              <div className="text-xs italic text-zinc-500">
                No planes yet. Close a stroke loop to create one.
              </div>
            ) : (
              planeLayers.map((plane) => {
                const isSelected = selectedPlanes.includes(plane.planeId);
                const isPrimary = selectedPlanes[0] === plane.planeId;
                const borderClass = isPrimary
                  ? "border-cyan-400 shadow-[0_0_0_1px_rgba(56,189,248,0.35)]"
                  : isSelected
                  ? "border-cyan-700"
                  : "border-zinc-800";
                const activityClass = !plane.active
                  ? "opacity-50"
                  : plane.muted
                  ? "opacity-70"
                  : "";
                const draggable = !plane.locked;
                return (
                  <div
                    key={plane.planeId}
                    className={`rounded-xl border px-3 py-2 bg-zinc-900/60 transition-colors ${borderClass} ${activityClass} ${
                      draggable ? "cursor-grab" : "cursor-not-allowed"
                    }`}
                    draggable={draggable}
                    onDragStart={handleDragStart(plane.planeId)}
                    onDragOver={handleDragOver(plane.planeId)}
                    onDrop={handleDrop(plane.planeId)}
                    onDragEnd={handleDragEnd}
                    onClick={(event) => handlePlaneClick(plane.planeId, event)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                        <span>Plane #{plane.planeId}</span>
                        <span
                          className="inline-block h-2.5 w-2.5 rounded-full"
                          style={{
                            backgroundColor: `rgb(${plane.color[0]},${plane.color[1]},${plane.color[2]})`,
                          }}
                        />
                      </div>
                      <div className="text-[11px] uppercase tracking-wide text-zinc-400">
                        {plane.orientation}
                      </div>
                    </div>
                    <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-zinc-500">
                      <span>depth {plane.order !== undefined ? plane.order + 1 : "-"}</span>
                      {plane.locked && <span className="text-amber-400">Locked</span>}
                      {!plane.active && <span className="text-rose-400">Inactive</span>}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-3 text-xs" onClick={(event) => event.stopPropagation()}>
                      <Toggle
                        value={plane.solo}
                        onChange={(value) => handleSoloToggle(plane.planeId, value)}
                        label="Solo"
                      />
                      <Toggle
                        value={plane.muted}
                        onChange={(value) => handleMuteToggle(plane.planeId, value)}
                        label="Mute"
                      />
                      <Toggle
                        value={plane.locked}
                        onChange={(value) => handleLockToggle(plane.planeId, value)}
                        label="Lock"
                      />
                    </div>
                  </div>
                );
              })
            )}
            {planeLayers.length > 0 && (
              <div
                className="rounded-lg border border-dashed border-zinc-700 px-3 py-2 text-center text-[11px] text-zinc-500"
                onDragOver={handleDragOver(null)}
                onDrop={handleDrop(null)}
              >
                Drop here to send to bottom
              </div>
            )}
          </div>
          <div className="col-span-2 flex flex-wrap items-center gap-2 text-xs mt-1">
            <button
              onClick={undoPlaneOrder}
              disabled={!undoAvailable}
              className={`px-3 py-1 rounded-lg border ${
                undoAvailable
                  ? "border-zinc-600 bg-zinc-900 hover:border-cyan-400 hover:text-cyan-300"
                  : "border-zinc-900 bg-zinc-900 text-zinc-600 cursor-not-allowed"
              }`}
            >
              Undo Order
            </button>
            <button
              onClick={redoPlaneOrder}
              disabled={!redoAvailable}
              className={`px-3 py-1 rounded-lg border ${
                redoAvailable
                  ? "border-zinc-600 bg-zinc-900 hover:border-cyan-400 hover:text-cyan-300"
                  : "border-zinc-900 bg-zinc-900 text-zinc-600 cursor-not-allowed"
              }`}
            >
              Redo Order
            </button>
            <Toggle value={transformTool} onChange={setTransformTool} label="Transform tool" />
          </div>
        </Section>

        <Section title="Schedules">
          <Row label="K(t) vs D(t)" full>
            <div className="flex gap-3 w-full items-center">
              <SchedulePlot K={schedK} D={schedD} Kref={schedKref} />
              <button onClick={handleRecordFwdToRev} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
                Record Fwd→Rev
              </button>
            </div>
          </Row>
        </Section>

        <Section title="Evaluation & Tests">
          <Row label="Snapshot" full>
            <div className="flex gap-2 w-full">
              <button onClick={handleEvalSnapshot} className="flex-1 px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
                Eval Metrics
              </button>
              <button onClick={handleDeterminismTest} className="flex-1 px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
                Determinism Test
              </button>
            </div>
          </Row>
          {metrics && (
            <>
              <Row label="ρ mean">
                <div className="text-xs font-mono">{metrics.rhoMean.toFixed(4)}</div>
              </Row>
              <Row label="ρ std">
                <div className="text-xs font-mono">{metrics.rhoStd.toFixed(4)}</div>
              </Row>
              <Row label="Defects">
                <div className="text-xs font-mono">{metrics.defect.toExponential(3)}</div>
              </Row>
              <Row label="Entropy">
                <div className="text-xs font-mono">{metrics.entropy.toFixed(4)}</div>
              </Row>
              <Row label="Anisotropy">
                <div className="text-xs font-mono">{metrics.aniso.toFixed(4)}</div>
              </Row>
            </>
          )}
          {determinismRMSE !== null && (
            <Row label="RMSE">
              <div className="text-xs font-mono">{determinismRMSE.toExponential(3)}</div>
            </Row>
          )}
        </Section>

        <Section title="Grid & Display">
          <Row label="Width">
            <Range value={W} min={32} max={160} step={2} onChange={setW} />
          </Row>
          <Row label="Height">
            <Range value={H} min={32} max={160} step={2} onChange={setH} />
          </Row>
          <Row label="Pixel">
            <Range value={pixelSize} min={5} max={12} step={1} onChange={setPixelSize} />
          </Row>
          <Row label="Wrap">
            <Toggle value={wrap} onChange={setWrap} />
          </Row>
          <Row label="Walls">
            <Toggle value={showWalls} onChange={setShowWalls} />
          </Row>
          <Row label="Lines">
            <Toggle value={showLines} onChange={setShowLines} />
          </Row>
          <Row label="Outlines">
            <Toggle value={showOutlines} onChange={setShowOutlines} />
          </Row>
          <Row label="Stereo">
            <Toggle value={stereo} onChange={setStereo} label="Stereo" />
          </Row>
        </Section>

        <Section title="Base Dynamics">
          <Row label="dt">
            <Range value={dt} min={0.005} max={0.1} step={0.005} onChange={setDt} />
          </Row>
          <Row label="Steps">
            <Range value={stepsPerFrame} min={1} max={10} step={1} onChange={setStepsPerFrame} />
          </Row>
          <Row label="K0">
            <Range value={Kbase} min={-2} max={2} step={0.05} onChange={setKbase} />
          </Row>
          <Row label="K1">
            <Range value={K1} min={-2} max={2} step={0.05} onChange={setK1} />
          </Row>
          <Row label="K2">
            <Range value={K2} min={-2} max={2} step={0.05} onChange={setK2} />
          </Row>
          <Row label="K3">
            <Range value={K3} min={-2} max={2} step={0.05} onChange={setK3} />
          </Row>
          <Row label="ω spread">
            <Range value={omegaSpread} min={0} max={1.5} step={0.05} onChange={setOmegaSpread} />
          </Row>
          <Row label="Noise">
            <Range value={noiseAmp} min={0} max={0.5} step={0.01} onChange={setNoiseAmp} />
          </Row>
        </Section>

        <Section title="Small-World">
          <Row label="Prob">
            <Range value={swProb} min={0} max={0.2} step={0.005} onChange={setSwProb} />
          </Row>
          <Row label="Edges/Node">
            <Range value={swEdgesPerNode} min={0} max={8} step={1} onChange={setSwEdgesPerNode} />
          </Row>
          <Row label="MinDist">
            <Range value={swMinDist} min={1} max={30} step={1} onChange={setSwMinDist} />
          </Row>
          <Row label="MaxDist">
            <Range value={swMaxDist} min={2} max={50} step={1} onChange={setSwMaxDist} />
          </Row>
          <Row label="Weight">
            <Range value={swWeight} min={-1} max={1} step={0.05} onChange={setSwWeight} />
          </Row>
          <Row label="NegFrac">
            <Range value={swNegFrac} min={0} max={1} step={0.05} onChange={setSwNegFrac} />
          </Row>
        </Section>

        <Section title="Brush & EM Field">
          <Row label="Line px">
            <Range value={lineWidth} min={1} max={16} step={1} onChange={setLineWidth} />
          </Row>
          <Row label="Close ≤px">
            <Range value={closeThreshold} min={4} max={40} step={1} onChange={setCloseThreshold} />
          </Row>
          <Row label="AutoClose">
            <Toggle value={autoClose} onChange={setAutoClose} label="AutoClose" />
          </Row>
          <Row label="Blur">
            <Range value={emBlur} min={0} max={20} step={1} onChange={setEmBlur} />
          </Row>
          <Row label="EM gain">
            <Range value={emGain} min={-2} max={2} step={0.05} onChange={setEmGain} />
          </Row>
          <Row label="Barrier">
            <Range value={wallBarrier} min={0} max={1} step={0.05} onChange={setWallBarrier} />
          </Row>
          <div className="col-span-2 flex gap-2">
            <button onClick={clearWalls} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700 w-full">
              Clear Walls
            </button>
          </div>
        </Section>

        <Section title="Images">
          <Row label="Visible">
            <Toggle value={showImages} onChange={setShowImages} label="Show" />
          </Row>
          <Row label="Move tool">
            <Toggle value={imageTool} onChange={setImageTool} label="Drag" />
          </Row>
          <Row label="Outline px">
            <Range value={imageOutlineWidth} min={1} max={24} step={1} onChange={updateImageOutlineWidth} />
          </Row>
          <Row label="Add image" full>
            <div className="flex gap-2 w-full">
              <button onClick={() => fileInputRef.current?.click()} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700 flex-1">
                Upload Image
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={(event) => {
                  addImages(event.target.files);
                  if (event.target) event.target.value = "";
                }}
              />
            </div>
          </Row>
          <Row label="Undo image" full>
            <button onClick={removeLastImage} className="w-full px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
              Remove Last Image
            </button>
          </Row>
          <Row label="Clear images" full>
            <button onClick={clearImages} className="w-full px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
              Clear Images
            </button>
          </Row>
        </Section>

        <Section title="Pockets">
          <Row label="Enable">
            <Toggle value={pocketEnabled} onChange={setPocketEnabled} label="Detect pockets" />
          </Row>
          <Row label="Horizon">
            <select
              value={pocketHorizon}
              onChange={(e) => setPocketHorizon(e.target.value as "sealed" | "oneway" | "none")}
              className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
            >
              <option value="sealed">sealed</option>
              <option value="oneway">oneway</option>
              <option value="none">none</option>
            </select>
          </Row>
          <Row label="K boost">
            <Range value={pocketKboost} min={0} max={2} step={0.05} onChange={setPocketKboost} />
          </Row>
          <Row label="Score w">
            <div className="flex gap-2 w-full">
              <Range value={pocketWRho} min={0} max={2} step={0.05} onChange={setPocketWRho} />
              <Range value={pocketWAttn} min={0} max={2} step={0.05} onChange={setPocketWAttn} />
              <Range value={pocketWShear} min={0} max={2} step={0.05} onChange={setPocketWShear} />
              <Range value={pocketWDef} min={0} max={2} step={0.05} onChange={setPocketWDef} />
            </div>
          </Row>
          <Row label="Thresh">
            <Range value={pocketThresh} min={0} max={2} step={0.01} onChange={setPocketThresh} />
          </Row>
          <Row label="Min area">
            <input
              type="number"
              min={1}
              max={W * H}
              value={pocketMinArea}
              onChange={(e) => setPocketMinArea(Number.parseInt(e.target.value, 10) || 1)}
              className="w-28 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
            />
          </Row>
        </Section>

        <Section title="DAG-Time & Horizon">
          <Row label="Event barrier">
            <Range value={eventBarrier} min={0} max={1} step={0.05} onChange={setEventBarrier} />
          </Row>
          <Row label="DAG sweeps">
            <Range value={dagSweeps} min={0} max={6} step={1} onChange={setDagSweeps} />
          </Row>
          <Row label="Depth order">
            <Toggle value={dagDepthOrdering} onChange={setDagDepthOrdering} />
          </Row>
          <Row label="Depth filter">
            <Toggle value={dagDepthFiltering} onChange={setDagDepthFiltering} />
          </Row>
          <Row label="Log DAG stats">
            <Toggle value={dagLogStats} onChange={setDagLogStats} label="Console" />
          </Row>
        </Section>

        <Section title="Energy & Brightness">
          <Row label="Baseline">
            <Range value={energyBaseline} min={0} max={1} step={0.01} onChange={setEnergyBaseline} />
          </Row>
          <Row label="Leak">
            <Range value={energyLeak} min={0} max={0.2} step={0.005} onChange={setEnergyLeak} />
          </Row>
          <Row label="Diffusion">
            <Range value={energyDiff} min={0} max={0.5} step={0.01} onChange={setEnergyDiff} />
          </Row>
          <Row label="Sink Line">
            <Range value={sinkLine} min={0} max={0.3} step={0.005} onChange={setSinkLine} />
          </Row>
          <Row label="Sink Surface">
            <Range value={sinkSurf} min={0} max={0.3} step={0.005} onChange={setSinkSurf} />
          </Row>
          <Row label="Sink Hyper">
            <Range value={sinkHyp} min={0} max={0.3} step={0.005} onChange={setSinkHyp} />
          </Row>
          <Row label="Trap Surface">
            <Range value={trapSurf} min={0} max={0.8} step={0.01} onChange={setTrapSurf} />
          </Row>
          <Row label="Trap Hyper">
            <Range value={trapHyp} min={0} max={1.0} step={0.01} onChange={setTrapHyp} />
          </Row>
          <Row label="Floor Surface">
            <Range value={minEnergySurf} min={0} max={1} step={0.01} onChange={setMinEnergySurf} />
          </Row>
          <Row label="Floor Hyper">
            <Range value={minEnergyHyp} min={0} max={1} step={0.01} onChange={setMinEnergyHyp} />
          </Row>
          <Row label="Brightness">
            <Range value={brightnessBase} min={0.2} max={1.2} step={0.05} onChange={setBrightnessBase} />
          </Row>
          <Row label="Energy γ">
            <Range value={energyGamma} min={0.5} max={2.5} step={0.05} onChange={setEnergyGamma} />
          </Row>
        </Section>

        <Section title="Surface Network">
          <Row label="α Surf→Field">
            <Range value={alphaSurfToField} min={0} max={2} step={0.05} onChange={setAlphaSurfToField} />
          </Row>
          <Row label="α Field→Surf">
            <Range value={alphaFieldToSurf} min={0} max={2} step={0.05} onChange={setAlphaFieldToSurf} />
          </Row>
          <Row label="KS1">
            <Range value={KS1} min={-2} max={2} step={0.05} onChange={setKS1} />
          </Row>
          <Row label="KS2">
            <Range value={KS2} min={-2} max={2} step={0.05} onChange={setKS2} />
          </Row>
          <Row label="KS3">
            <Range value={KS3} min={-2} max={2} step={0.05} onChange={setKS3} />
          </Row>
        </Section>

        <Section title="Hyper Network">
          <Row label="α Hyp→Field">
            <Range value={alphaHypToField} min={0} max={2} step={0.05} onChange={setAlphaHypToField} />
          </Row>
          <Row label="α Field→Hyp">
            <Range value={alphaFieldToHyp} min={0} max={2} step={0.05} onChange={setAlphaFieldToHyp} />
          </Row>
          <Row label="KH1">
            <Range value={KH1} min={-2} max={2} step={0.05} onChange={setKH1} />
          </Row>
          <Row label="KH2">
            <Range value={KH2} min={-2} max={2} step={0.05} onChange={setKH2} />
          </Row>
          <Row label="KH3">
            <Range value={KH3} min={-2} max={2} step={0.05} onChange={setKH3} />
          </Row>
        </Section>

        <Section title="Attention & Awareness">
          <Row label="Enable">
            <Toggle value={attentionEnabled} onChange={setAttentionEnabled} />
          </Row>
          <Row label="γK (bind)">
            <Range value={att_gammaK} min={0} max={1.0} step={0.02} onChange={setAtt_gammaK} />
          </Row>
          <Row label="γD / δD">
            <div className="flex gap-2 w-full">
              <Range value={att_gammaD} min={0} max={0.8} step={0.02} onChange={setAtt_gammaD} />
              <Range value={att_deltaD} min={0} max={0.3} step={0.01} onChange={setAtt_deltaD} />
            </div>
          </Row>
          <Row label="DA / μA">
            <div className="flex gap-2 w-full">
              <Range value={att_DA} min={0.1} max={2.0} step={0.05} onChange={setAtt_DA} />
              <Range value={att_muA} min={0.05} max={0.5} step={0.01} onChange={setAtt_muA} />
            </div>
          </Row>
          <Row label="DU / μU">
            <div className="flex gap-2 w-full">
              <Range value={att_DU} min={0.1} max={2.0} step={0.05} onChange={setAtt_DU} />
              <Range value={att_muU} min={0.05} max={0.5} step={0.01} onChange={setAtt_muU} />
            </div>
          </Row>
          <Row label="Salience">
            <div className="flex gap-2 w-full">
              <Range value={att_wGrad} min={0} max={2} step={0.05} onChange={setAtt_wGrad} />
              <Range value={att_wLap} min={0} max={2} step={0.05} onChange={setAtt_wLap} />
              <Range value={att_wE} min={0} max={2} step={0.05} onChange={setAtt_wE} />
              <Range value={att_wDef} min={0} max={2} step={0.05} onChange={setAtt_wDef} />
            </div>
          </Row>
          <Row label="λS / λC">
            <div className="flex gap-2 w-full">
              <Range value={att_lambdaS} min={0} max={2} step={0.05} onChange={setAtt_lambdaS} />
              <Range value={att_lambdaC} min={0} max={2} step={0.05} onChange={setAtt_lambdaC} />
            </div>
          </Row>
          <Row label="Topo gain">
            <Range value={att_topo} min={-1.0} max={1.0} step={0.05} onChange={setAtt_topo} />
          </Row>
          <Row label="Context R">
            <Range value={att_contextR} min={0} max={16} step={1} onChange={setAtt_contextR} />
          </Row>
          <Row label="Reflector gain">
            <Range value={att_reflectorGain} min={0} max={2.0} step={0.05} onChange={setAtt_reflectorGain} />
          </Row>
          <Row label="Head 1">
            <div className="flex gap-2 w-full">
              <Toggle value={head1.enabled} onChange={(v)=>setHead1({...head1, enabled:v})} label="On" />
              <Range value={head1.weight} min={0} max={1} step={0.05} onChange={(v)=>setHead1({...head1, weight:v})} />
              <Range value={head1.freq} min={1} max={20} step={0.5} onChange={(v)=>setHead1({...head1, freq:v})} />
              <Range value={head1.radius} min={2} max={Math.floor(Math.max(W,H)/3)} step={1} onChange={(v)=>setHead1({...head1, radius:v})} />
              <Toggle value={head1.bindPrimaryPlane} onChange={(v)=>setHead1({...head1, bindPrimaryPlane:v})} label="Bind" />
            </div>
          </Row>
          <Row label="Head 2">
            <div className="flex gap-2 w-full">
              <Toggle value={head2.enabled} onChange={(v)=>setHead2({...head2, enabled:v})} label="On" />
              <Range value={head2.weight} min={0} max={1} step={0.05} onChange={(v)=>setHead2({...head2, weight:v})} />
              <Range value={head2.freq} min={1} max={20} step={0.5} onChange={(v)=>setHead2({...head2, freq:v})} />
              <Range value={head2.radius} min={2} max={Math.floor(Math.max(W,H)/3)} step={1} onChange={(v)=>setHead2({...head2, radius:v})} />
              <Toggle value={head2.bindPrimaryPlane} onChange={(v)=>setHead2({...head2, bindPrimaryPlane:v})} label="Bind" />
            </div>
          </Row>
          <Row label="Head 3">
            <div className="flex gap-2 w-full">
              <Toggle value={head3.enabled} onChange={(v)=>setHead3({...head3, enabled:v})} label="On" />
              <Range value={head3.weight} min={0} max={1} step={0.05} onChange={(v)=>setHead3({...head3, weight:v})} />
              <Range value={head3.freq} min={1} max={20} step={0.5} onChange={(v)=>setHead3({...head3, freq:v})} />
              <Range value={head3.radius} min={2} max={Math.floor(Math.max(W,H)/3)} step={1} onChange={(v)=>setHead3({...head3, radius:v})} />
              <Toggle value={head3.bindPrimaryPlane} onChange={(v)=>setHead3({...head3, bindPrimaryPlane:v})} label="Bind" />
            </div>
          </Row>
        </Section>

        <ReflectorGraphPanel getSim={getSim} />

        <Section title="Lift & Stereo">
          <Row label="Lift gain">
            <Range value={liftGain} min={0} max={1.5} step={0.05} onChange={setLiftGain} />
          </Row>
          <Row label="Hyper curve">
            <Range value={hyperCurve} min={0.1} max={1.5} step={0.05} onChange={setHyperCurve} />
          </Row>
          <Row label="Stereo α">
            <Range value={stereoAlpha} min={0} max={1} step={0.05} onChange={setStereoAlpha} />
          </Row>
          <Row label="IPD">
            <Range value={ipd} min={0} max={6} step={0.1} onChange={setIpd} />
          </Row>
        </Section>

        <Section title="Reservoir">
          <Row label="Fit (W_RGB)" full>
            <button onClick={handleFitW} className="w-full px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
              Fit Readout W_RGB
            </button>
          </Row>
          <Row label="Fit (hφ)" full>
            <button onClick={handleFitController} className="w-full px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
              Fit Controller hφ
            </button>
          </Row>
          {rgbHead && (
            <div className="col-span-2 text-[11px] text-zinc-400">
              W_RGB fitted: inDim={rgbHead.inDim}, outDim={rgbHead.outDim}
            </div>
          )}
          {typeof rgbMSE === "number" && (
            <div className="col-span-2 text-[11px] text-zinc-400">
              RGB MSE={rgbMSE.toExponential(3)} | PSNR={rgbPSNR?.toFixed(2)} dB
            </div>
          )}
          {ctrlSched && (
            <div className="col-span-2 text-[11px] text-zinc-400">
              hφ sched: steps={ctrlSched.steps}, seed={ctrlSched.seed}, K_gain={ctrlSched.K_gain.toFixed(2)}
            </div>
          )}
        </Section>

        <Section title="Tokens & Affect">
          <Row label="Arousal">
            <div className="text-sm font-mono px-2 py-1 rounded bg-zinc-900 border border-zinc-800">{arousal.toFixed(3)}</div>
          </Row>
          <Row label="Valence">
            <div className="text-sm font-mono px-2 py-1 rounded bg-zinc-900 border border-zinc-800">{valence.toFixed(3)}</div>
          </Row>
          <Row label="Target arousal">
            <Range value={targetArousal} min={0} max={1} step={0.01} onChange={setTargetArousal} />
          </Row>
          <Row label="Target valence">
            <Range value={targetValence} min={0} max={1} step={0.01} onChange={setTargetValence} />
          </Row>
          <Row label="Valence control">
            <Toggle value={valenceControl} onChange={setValenceControl} label="Enable" />
          </Row>
          <Row label="Arousal band ±">
            <Range value={arousalBand} min={0} max={0.5} step={0.01} onChange={setArousalBand} />
          </Row>
          <Row label="Policy N-frames">
            <input
              type="number"
              min={1}
              max={60}
              value={policyEveryN}
              onChange={(e) => setPolicyEveryN(Number.parseInt(e.target.value, 10) || 1)}
              className="w-24 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
            />
          </Row>
          <Row label="Token-only (PAT)">
            <Toggle value={tokenOnly} onChange={setTokenOnly} label="Opaque substrate" />
          </Row>
          <Row label="TokenAtlas live">
            <div className="flex gap-2 w-full">
              <Toggle value={tokenAtlasLive} onChange={setTokenAtlasLive} label="Enable" />
              <input
                type="number"
                min={100}
                max={5000}
                value={tokenAtlasEveryMs}
                onChange={(e) => setTokenAtlasEveryMs(Number.parseInt(e.target.value, 10) || 1000)}
                className="w-28 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
              />
              <span className="text-[11px] text-zinc-500 self-center">ms</span>
            </div>
          </Row>
          <Row label="PAT stream URL">
            <input
              type="text"
              value={patStreamUrl}
              onChange={(e) => setPatStreamUrl(e.target.value)}
              className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
            />
          </Row>
          <Row label="FPS / Scale">
            <div className="flex gap-2 w-full">
              <input
                type="number"
                min={1}
                max={60}
                value={patStreamFps}
                onChange={(e) => setPatStreamFps(Number.parseInt(e.target.value, 10) || 1)}
                className="w-24 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
              />
              <select
                value={patScale}
                onChange={(e) => setPatScale(Number(e.target.value) as 32 | 64)}
                className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
              >
                <option value={32}>32</option>
                <option value={64}>64</option>
              </select>
              <button
                onClick={patStreaming ? handleStopPAT : handleStartPAT}
                className={`px-3 py-1 rounded-xl ${patStreaming ? "bg-rose-800 hover:bg-rose-700" : "bg-zinc-800 hover:bg-zinc-700"}`}
              >
                {patStreaming ? "Stop PAT Stream" : "Start PAT Stream"}
              </button>
            </div>
          </Row>
          <Row label="PAT" full>
            <button onClick={handleComputePAT} className="w-full px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
              Compute PAT Snapshot
            </button>
          </Row>
          <Row label="TokenAtlas" full>
            <TokenAtlas tokens={tokenAtlasTokens} />
          </Row>
          {typeof patMSE === "number" && (
            <>
              <Row label="PAT MSE">
                <div className="text-xs font-mono">{patMSE.toExponential(3)}</div>
              </Row>
              <Row label="Codebook use">
                <div className="text-xs font-mono">{(100 * (patUsage ?? 0)).toFixed(1)}%</div>
              </Row>
              <Row label="Usage entropy">
                <div className="text-xs font-mono">{(100 * (patEntropy ?? 0)).toFixed(1)}%</div>
              </Row>
            </>
          )}
        </Section>

        <Section title="Debug & Planes">
          <Row label="Depth overlay">
            <Toggle value={showDagDepth} onChange={setShowDagDepth} label="DAG depth" />
          </Row>
          <Row label="ρ overlay">
            <Toggle value={showRhoOverlay} onChange={setShowRhoOverlay} label="Local order" />
          </Row>
          <Row label="Defects overlay">
            <Toggle value={showDefectsOverlay} onChange={setShowDefectsOverlay} label="±1 vortices" />
          </Row>
          <Row label="Reseed" full>
            <button onClick={rebuildSmallWorld} className="w-full px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
              Reseed Graph
            </button>
          </Row>
          <Row label="Delete sel" full>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => deletePlanes("keep-wall")}
                disabled={!hasSelection}
                className={`px-3 py-1 rounded-xl border ${
                  hasSelection
                    ? "border-zinc-600 bg-zinc-900 hover:border-cyan-400 hover:text-cyan-300"
                    : "border-zinc-900 bg-zinc-900 text-zinc-600 cursor-not-allowed"
                }`}
              >
                Keep Wall
              </button>
              <button
                onClick={() => deletePlanes("keep-energy")}
                disabled={!hasSelection}
                className={`px-3 py-1 rounded-xl border ${
                  hasSelection
                    ? "border-zinc-600 bg-zinc-900 hover:border-cyan-400 hover:text-cyan-300"
                    : "border-zinc-900 bg-zinc-900 text-zinc-600 cursor-not-allowed"
                }`}
              >
                Keep Energy
              </button>
              <button
                onClick={() => deletePlanes("clean")}
                disabled={!hasSelection}
                className={`px-3 py-1 rounded-xl border ${
                  hasSelection
                    ? "border-rose-600 bg-rose-900/30 text-rose-200 hover:border-rose-400"
                    : "border-zinc-900 bg-zinc-900 text-zinc-600 cursor-not-allowed"
                }`}
              >
                Clean
              </button>
            </div>
          </Row>
          <Row label="Clear planes" full>
            <button onClick={clearPlanes} className="w-full px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
              Clear Planes
            </button>
          </Row>
          <Row label="Undo plane" full>
            <button onClick={removeLastPlane} className="w-full px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
              Remove Last Plane
            </button>
          </Row>
          <Row label="Clear strokes" full>
            <button onClick={clearStrokes} className="w-full px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
              Clear Strokes
            </button>
          </Row>
        </Section>
      </div>
    </div>
      {planeMenu && menuPosition && (
        <div className="fixed inset-0 z-50" onClick={closePlaneMenu}>
          <div
            className="absolute min-w-[200px] rounded-xl border border-zinc-700 bg-zinc-900/95 text-xs shadow-2xl backdrop-blur-sm"
            style={{ left: menuPosition.left, top: menuPosition.top }}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="px-3 py-2 text-[10px] uppercase tracking-wide text-zinc-500">Transform</div>
            <div className="flex flex-col gap-1 px-2 pb-2">
              {transformActions.map((item) => (
                <button
                  key={item.key}
                  onClick={() => handleTransformAction(planeMenu.planeId, item.key)}
                  className="rounded-lg px-3 py-1 text-left hover:bg-zinc-800"
                >
                  {item.label}
                </button>
              ))}
            </div>
            <div className="mx-2 border-t border-zinc-800" />
            <div className="px-3 pt-2 pb-1 text-[10px] uppercase tracking-wide text-zinc-500">Boolean</div>
            <div className="flex flex-col gap-1 px-2 pb-2">
              {booleanActions.map((item) => {
                const disabled = otherSelection.length === 0;
                return (
                  <button
                    key={item.key}
                    disabled={disabled}
                    onClick={() => handleBooleanAction(planeMenu.planeId, otherSelection, item.key)}
                    className={`rounded-lg px-3 py-1 text-left ${
                      disabled
                        ? "cursor-not-allowed text-zinc-600"
                        : "hover:bg-zinc-800 text-zinc-100"
                    }`}
                  >
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
