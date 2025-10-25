import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { SimulationState } from "../types";
import { Section, Row, Toggle } from "../uiPrimitives";

type Props = {
  // Provided by KuramotoPainter/useKuramotoEngine to access the running sim
  getSim: () => SimulationState | null;
};

type RG = {
  getNodes: () => ReadonlyArray<{ x: number; y: number; a: number; w: number }>;
  updateCandidates: (S: Float32Array, K?: number, minDist?: number) => void;
  step: (lambda?: number, sigma?: number, k?: number) => void;
};

function drawReflectorNodes(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  nodes: ReadonlyArray<{ x: number; y: number; a: number; w: number }>,
) {
  const cw = ctx.canvas.width;
  const ch = ctx.canvas.height;
  ctx.clearRect(0, 0, cw, ch);

  // Grid background
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(0, 0, cw, ch);

  // Scale from sim grid to panel canvas
  const sx = cw / Math.max(1, W);
  const sy = ch / Math.max(1, H);

  // Draw nodes as circles with radius from weight and color from activity
  for (const n of nodes) {
    const cx = (n.x + 0.5) * sx;
    const cy = (n.y + 0.5) * sy;
    const r = Math.max(2, 2 + 6 * Math.tanh(Math.max(0, n.w) / 2));
    const a = Math.max(0, n.a);
    const hue = 200; // cyan-ish
    const sat = 60 + Math.min(40, 100 * a);
    const light = 40 + Math.min(40, 100 * a);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${hue}deg ${sat}% ${light}%)`;
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

export function ReflectorGraphPanel({ getSim }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [panelSize, setPanelSize] = useState<{ w: number; h: number }>({ w: 240, h: 240 });

  // Accessor helpers
  const sim = getSim();
  const W = sim?.W ?? 0;
  const H = sim?.H ?? 0;

  // Derive nodes safely (may be missing if not instantiated)
  const nodes: ReadonlyArray<{ x: number; y: number; a: number; w: number }> = useMemo(() => {
    if (!sim) return [];
    const rg: RG | undefined = (sim as any)._reflectorGraph;
    try {
      return rg ? rg.getNodes() : [];
    } catch {
      return [];
    }
  }, [sim, sim?.W, sim?.H, (sim as any)?._reflectorGraph, sim?.energy]); // include energy to redraw occasionally

  const refresh = useCallback(() => {
    const c = canvasRef.current;
    if (!c || !sim) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const rg: RG | undefined = (sim as any)._reflectorGraph;
    const list = rg ? rg.getNodes() : [];
    drawReflectorNodes(ctx, sim.W, sim.H, list);
  }, [sim]);

  useEffect(() => {
    // Resize canvas to device pixel ratio for crisp rendering
    const c = canvasRef.current;
    if (!c) return;
    const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
    c.width = Math.floor(panelSize.w * dpr);
    c.height = Math.floor(panelSize.h * dpr);
    c.style.width = `${panelSize.w}px`;
    c.style.height = `${panelSize.h}px`;
    const ctx = c.getContext("2d");
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    refresh();
  }, [panelSize, refresh]);

  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(() => {
      refresh();
    }, 500);
    return () => clearInterval(id);
  }, [autoRefresh, refresh]);

  const handleManualStep = useCallback(() => {
    const s = getSim();
    if (!s) return;
    const rg: RG | undefined = (s as any)._reflectorGraph;
    if (!rg) return;
    try {
      // One power iteration step with defaults
      rg.step(0.9, 8, 4);
    } catch {}
    refresh();
  }, [getSim, refresh]);

  return (
    <Section title="Reflector Graph">
      <Row label="Auto refresh">
        <Toggle value={autoRefresh} onChange={setAutoRefresh} label="Auto" />
      </Row>
      <Row label="Preview" full>
        <div className="flex flex-col gap-2 w-full">
          <canvas ref={canvasRef} className="rounded-xl border border-zinc-800 bg-black" />
          <div className="flex items-center justify-between text-[11px] text-zinc-400">
            <div>
              Nodes: <span className="text-zinc-200">{nodes.length}</span>
            </div>
            <div>
              Grid: <span className="text-zinc-200">{W}×{H}</span>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={refresh} className="px-2 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700">
                Refresh
              </button>
              <button onClick={handleManualStep} className="px-2 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700">
                Step
              </button>
            </div>
          </div>
        </div>
      </Row>
    </Section>
  );
}

export default ReflectorGraphPanel;
