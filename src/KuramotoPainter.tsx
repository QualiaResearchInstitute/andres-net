import React, { useCallback, useEffect, useState } from "react";
import { useKuramotoEngine } from "./kuramoto/useKuramotoEngine";
import type { PlaneLayerSnapshot } from "./kuramoto/useKuramotoEngine";
import { Section, Row, Range, Toggle } from "./kuramoto/uiPrimitives";

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
  const [showImages, setShowImages] = useState(true);
  const [imageOutlineWidth, setImageOutlineWidth] = useState(4);

  const [showWalls, setShowWalls] = useState(true);
  const [showLines, setShowLines] = useState(true);
  const [showOutlines, setShowOutlines] = useState(true);
  const [showDagDepth, setShowDagDepth] = useState(false);

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

  const [eventBarrier, setEventBarrier] = useState(1.0);
  const [dagSweeps, setDagSweeps] = useState(0);
  const [dagDepthOrdering, setDagDepthOrdering] = useState(true);
  const [dagDepthFiltering, setDagDepthFiltering] = useState(true);
  const [dagLogStats, setDagLogStats] = useState(false);

  const [stereo, setStereo] = useState(false);
  const [ipd, setIpd] = useState(2.0);
  const [stereoAlpha, setStereoAlpha] = useState(0.5);

  const [planeLayers, setPlaneLayers] = useState<PlaneLayerSnapshot[]>([]);
  const [selectedPlanes, setSelectedPlanesState] = useState<number[]>([]);
  const [layerVersion, setLayerVersion] = useState(0);
  const [draggingPlaneId, setDraggingPlaneId] = useState<number | null>(null);

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
  } = engine;

  const rebuildSmallWorld = useCallback(() => setReseedGraphKey((k) => k + 1), []);

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
    <div className="w-full h-full flex gap-4 p-4 bg-black text-zinc-100">
      <div className="flex flex-col items-center">
        <div className="relative rounded-2xl overflow-hidden shadow-2xl" style={{ width: W * pixelSize, height: H * pixelSize }}>
          <canvas ref={fieldCanvasRef} className="block" width={W * pixelSize} height={H * pixelSize} />
          <canvas ref={overlayCanvasRef} className="absolute inset-0" width={W * pixelSize} height={H * pixelSize} />
        </div>
        <div className="mt-3 flex gap-2 flex-wrap">
          <button onClick={() => setPaused((p) => !p)} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            {paused ? "Play" : "Pause"}
          </button>
          <button onClick={() => step(dt)} className="px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
            Step
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

        <Section title="Debug & Planes">
          <Row label="Depth overlay">
            <Toggle value={showDagDepth} onChange={setShowDagDepth} label="DAG depth" />
          </Row>
          <Row label="Reseed" full>
            <button onClick={rebuildSmallWorld} className="w-full px-3 py-1 rounded-xl bg-zinc-800 hover:bg-zinc-700">
              Reseed Graph
            </button>
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
  );
}
