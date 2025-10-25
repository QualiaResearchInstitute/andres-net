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
import { buildPATokens } from "../tokens/pat";
import { tokensToAttentionFields } from "../tokens/tokenOnly";
import { computeAffect } from "./affect";
import { stepValencePolicy, type PolicyCfg } from "./policyValence";
import { computePocketScore, extractPockets, type Pocket } from "./topology";

// Local helper for angle wrapping (keeps determinism consistent with other modules)
function wrapAngle(x: number) {
  let y = (x + Math.PI) % (2 * Math.PI);
  if (y < 0) y += 2 * Math.PI;
  return y - Math.PI;
}

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
  // Topological pockets (optional)
  pocketParams?: {
    enabled?: boolean;
    scoreWeights: { rho: number; attn: number; shear: number; defects: number };
    scoreThresh: number;
    minArea: number;
    horizon: "sealed" | "oneway" | "none";
    Kboost: number;
  };
  // Opaque substrate toggle and PAT scale for token-only mode
  opaqueSubstrateMode?: boolean;
  patScale?: 32 | 64;
  // Attention
  attentionEnabled: boolean;
  attentionParams: import("./attention").AttentionParams;
  attentionHeads: import("./attention").AttentionHead[];
  // Deterministic valence policy (optional)
  valencePolicyEnabled?: boolean;
  valencePolicy?: PolicyCfg;
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
    pocketParams,
    // Opaque substrate
    opaqueSubstrateMode,
    patScale,
    // Attention
    attentionEnabled,
    attentionParams,
    attentionHeads,
    setImageOutlineWidth,
    onPlaneContextMenu,
    onPlaneContextMenuClose,
    // Valence policy (optional)
    valencePolicyEnabled,
    valencePolicy,
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
  // Policy state (effective gains) and deterministic frame counter
  const policyGainsRef = useRef<{ gammaK: number; gammaAlpha: number; gammaD: number; deltaD: number }>({
    gammaK: attentionParams.gammaK,
    gammaAlpha: attentionParams.gammaAlpha,
    gammaD: attentionParams.gammaD,
    deltaD: attentionParams.deltaD,
  });
  const policyFrameRef = useRef(0);
  // Pockets closures (set per-frame if enabled)
  const horizonPairRef = useRef<((i: number, j: number) => number) | undefined>(undefined);
  const KpairBoostRef = useRef<((i: number, j: number) => number) | undefined>(undefined);

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
      horizonPair: horizonPairRef.current,
      KpairBoost: KpairBoostRef.current,
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
          // Determine effective gains (may be adjusted by valence policy)
          let effGammaK = attentionParams.gammaK;
          let effGammaAlpha = attentionParams.gammaAlpha;
          let effGammaD = attentionParams.gammaD;
          let effDeltaD = attentionParams.deltaD;

          if (valencePolicyEnabled) {
            try {
              const affect = computeAffect(sim);
              const cur = {
                gammaK: policyGainsRef.current.gammaK,
                gammaAlpha: policyGainsRef.current.gammaAlpha,
                gammaD: policyGainsRef.current.gammaD,
                deltaD: policyGainsRef.current.deltaD,
                heads: (attentionHeads ?? []).map((h) => ({ enabled: h.enabled, weight: h.weight })),
              };
              const cfgLocal: PolicyCfg = {
                targetArousal: valencePolicy?.targetArousal ?? [0.3, 0.6],
                steps: valencePolicy?.steps ?? 1,
                gammaK: valencePolicy?.gammaK ?? [-1.5, 1.5],
                gammaAlpha: valencePolicy?.gammaAlpha ?? [-1.5, 1.5],
                gammaD: valencePolicy?.gammaD ?? [-1.0, 1.0],
                deltaD: valencePolicy?.deltaD ?? [0, 1.0],
                headWeightBounds: valencePolicy?.headWeightBounds ?? [0, 1],
              };
              const seedBase = sim.noiseSeed | 0;
              const rngSeed = (seedBase ^ ((policyFrameRef.current << 1) >>> 0)) >>> 0;
              const next = stepValencePolicy(affect, cur, rngSeed, cfgLocal);
              policyGainsRef.current = {
                gammaK: next.gammaK,
                gammaAlpha: next.gammaAlpha,
                gammaD: next.gammaD,
                deltaD: next.deltaD,
              };
              effGammaK = next.gammaK;
              effGammaAlpha = next.gammaAlpha;
              effGammaD = next.gammaD;
              effDeltaD = next.deltaD;
              policyFrameRef.current++;
            } catch {
              // keep previous gains if any error
            }
          } else {
            policyGainsRef.current = {
              gammaK: attentionParams.gammaK,
              gammaAlpha: attentionParams.gammaAlpha,
              gammaD: attentionParams.gammaD,
              deltaD: attentionParams.deltaD,
            };
          }

          if (attentionEnabled) {
            if (opaqueSubstrateMode) {
              const tokens = buildPATokens(sim, (patScale as 32 | 64) ?? 32);
              const proxy = tokensToAttentionFields(tokens, W, H, ((patScale as 32 | 64) ?? 32));
              attentionModsRef.current = {
                Aact: proxy.Aact,
                Uact: proxy.Uact,
                lapA: proxy.lapA,
                divU: proxy.divU,
                // keep A2 disabled in opaque-substrate shortcut
                etaV: 0,
                gammaK: effGammaK,
                betaK: attentionParams.betaK,
                gammaAlpha: effGammaAlpha,
                betaAlpha: attentionParams.betaAlpha,
                gammaD: effGammaD,
                deltaD: effDeltaD,
              };
            } else {
              const out = updateAttentionFields(sim, attentionHeads, attentionParams, dt, attentionTimeRef.current, wrap);
              attentionModsRef.current = {
                Aact: out.Aact,
                Uact: out.Uact,
                lapA: out.lapA,
                divU: out.divU,
                // A2 optional fields
                A: out.A,
                lapAraw: out.lapAraw,
                advect: out.advect,
                etaV: attentionParams.etaV ?? 0,
                // Effective gains (policy-adjusted if enabled)
                gammaK: effGammaK,
                betaK: attentionParams.betaK,
                gammaAlpha: effGammaAlpha,
                betaAlpha: attentionParams.betaAlpha,
                gammaD: effGammaD,
                deltaD: effDeltaD,
              };
            }
            attentionTimeRef.current += dt * stepsPerFrame;
          } else {
            attentionModsRef.current = undefined;
          }

          // Pockets (optional, deterministic)
          if (pocketParams?.enabled) {
            const W = sim.W | 0;
            const H = sim.H | 0;
            const N = (W * H) | 0;
            const phases = sim.phases;

            // Precompute cos/sin
            const cosTh = new Float32Array(N);
            const sinTh = new Float32Array(N);
            for (let i = 0; i < N; i++) {
              const th = phases[i];
              cosTh[i] = Math.cos(th);
              sinTh[i] = Math.sin(th);
            }

            // grad|phi| central differences (wrap-aware)
            const gradPhiMag = new Float32Array(N);
            for (let y = 0; y < H; y++) {
              const ym = (y - 1 + H) % H;
              const yp = (y + 1) % H;
              for (let x = 0; x < W; x++) {
                const xm = (x - 1 + W) % W;
                const xp = (x + 1) % W;
                const i = y * W + x;
                const iL = y * W + xm, iR = y * W + xp, iU = ym * W + x, iD = yp * W + x;
                const c = cosTh[i], s = sinTh[i];
                const dcosdx = 0.5 * (cosTh[iR] - cosTh[iL]);
                const dsindx = 0.5 * (sinTh[iR] - sinTh[iL]);
                const dcosdy = 0.5 * (cosTh[iD] - cosTh[iU]);
                const dsindy = 0.5 * (sinTh[iD] - sinTh[iU]);
                const gx = -s * dcosdx + c * dsindx;
                const gy = -s * dcosdy + c * dsindy;
                gradPhiMag[i] = Math.hypot(gx, gy);
              }
            }

            // local coherence rho via 3x3 complex mean
            const rho = new Float32Array(N);
            for (let y = 0; y < H; y++) {
              const ym = (y - 1 + H) % H;
              const yp = (y + 1) % H;
              for (let x = 0; x < W; x++) {
                const xm = (x - 1 + W) % W;
                const xp = (x + 1) % W;
                let cx = 0, sx = 0, cnt = 0;
                // 3x3 neighborhood
                const y0 = ym, y1 = y, y2 = yp;
                const x0 = xm, x1 = x, x2 = xp;
                const idxs = [
                  y0 * W + x0, y0 * W + x1, y0 * W + x2,
                  y1 * W + x0, y1 * W + x1, y1 * W + x2,
                  y2 * W + x0, y2 * W + x1, y2 * W + x2,
                ];
                for (let k = 0; k < 9; k++) {
                  const ii = idxs[k];
                  cx += cosTh[ii];
                  sx += sinTh[ii];
                  cnt++;
                }
                rho[y * W + x] = cnt > 0 ? Math.hypot(cx, sx) / cnt : 0;
              }
            }

            // vortex indicator via plaquette winding
            const vort = new Uint8Array(N);
            for (let y = 0; y < H - 1; y++) {
              for (let x = 0; x < W - 1; x++) {
                const i00 = y * W + x;
                const i10 = y * W + (x + 1);
                const i11 = (y + 1) * W + (x + 1);
                const i01 = (y + 1) * W + x;
                const d1 = wrapAngle(Math.atan2(sinTh[i10], cosTh[i10]) - Math.atan2(sinTh[i00], cosTh[i00]));
                const d2 = wrapAngle(Math.atan2(sinTh[i11], cosTh[i11]) - Math.atan2(sinTh[i10], cosTh[i10]));
                const d3 = wrapAngle(Math.atan2(sinTh[i01], cosTh[i01]) - Math.atan2(sinTh[i11], cosTh[i11]));
                const d4 = wrapAngle(Math.atan2(sinTh[i00], cosTh[i00]) - Math.atan2(sinTh[i01], cosTh[i01]));
                const sum = d1 + d2 + d3 + d4;
                if (sum > Math.PI || sum < -Math.PI) {
                  vort[i00] = 1;
                }
              }
            }

            const Aact = (attentionModsRef.current?.Aact) ?? new Float32Array(N);
            const score = computePocketScore(
              rho,
              Aact,
              gradPhiMag,
              vort,
              W,
              H,
              pocketParams.scoreWeights,
            );
            const pockets: Pocket[] = extractPockets(
              score,
              W,
              H,
              pocketParams.scoreThresh,
              pocketParams.minArea,
            );

            const pocketId = new Int32Array(N);
            for (let i = 0; i < N; i++) pocketId[i] = -1;
            for (let p = 0; p < pockets.length; p++) {
              const pk = pockets[p];
              const m = pk.mask;
              for (let i = 0; i < N; i++) {
                if (m[i]) pocketId[i] = pk.id;
              }
            }

            // Build closures
            const mode = pocketParams.horizon;
            const boostVal = pocketParams.Kboost ?? 0;
            horizonPairRef.current = (i: number, j: number) => {
              const pi = pocketId[i], pj = pocketId[j];
              if (pi < 0 && pj < 0) return 1.0;
              if (mode === "none") return 1.0;
              if (mode === "sealed") {
                return (pi === pj) ? 1.0 : 0.0;
              }
              // oneway: allow into pocket, block out of pocket
              // receiver is i (phase update target), source is j
              if (mode === "oneway") {
                // block if source is inside and receiver is outside (leak out)
                if (pj >= 0 && pi < 0) return 0.0;
                return 1.0;
              }
              return 1.0;
            };
            KpairBoostRef.current = (i: number, j: number) => {
              const pi = pocketId[i], pj = pocketId[j];
              return (pi >= 0 && pj >= 0 && pi === pj) ? (1 + boostVal) : 1.0;
            };
          } else {
            horizonPairRef.current = undefined;
            KpairBoostRef.current = undefined;
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
    pocketParams,
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
