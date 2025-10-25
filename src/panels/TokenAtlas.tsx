import React, { useMemo } from "react";
import type { PAToken } from "../tokens/pat";

export type TokenAtlasProps = {
  tokens: PAToken[] | null;
  bins?: number; // default 4 (matches current LFQ bins)
};

type AtlasStats = {
  count: number;
  dims: number;
  perDimHists: number[][];
  perDimUsage: number[];       // fraction of non-empty bins per dim
  perDimPerplexity: number[];  // 2^H for each dim
  attnMean: number;
  attnStd: number;
  arousal: number;
  valence: number;
  // Optional extras if present in tokens
  reconMSE?: number;           // mean reconstruction MSE across tokens (if provided)
  headCount?: number;          // number of heads if tokens provide phases
  headMeans?: number[];        // per-head circular mean phase (radians)
};

function computeAtlas(tokens: PAToken[], bins: number): AtlasStats {
  const count = tokens.length;
  const dims = Math.max(0, tokens[0]?.code?.rvq?.length ?? 0);
  const perDimHists: number[][] = Array.from({ length: dims }, () => new Array(bins).fill(0));
  let sumArousal = 0, sumValence = 0, sumAtt = 0, sumAtt2 = 0;

  // Recon MSE aggregation (optional)
  let reconSum = 0;
  let reconCnt = 0;

  // Head phases aggregation (optional)
  let maxHeads = 0;
  for (const t of tokens) {
    if (t.heads && t.heads.length > maxHeads) maxHeads = t.heads.length;
  }
  const headCos: number[] = new Array(maxHeads).fill(0);
  const headSin: number[] = new Array(maxHeads).fill(0);
  const headN: number[] = new Array(maxHeads).fill(0);

  for (const t of tokens) {
    const code = t.code?.rvq ?? [];
    for (let d = 0; d < dims; d++) {
      const idx = code[d] ?? 0;
      if (idx >= 0 && idx < bins) perDimHists[d][idx] += 1;
    }
    const a = t.stats.attnMean ?? 0;
    sumAtt += a;
    sumAtt2 += a * a;
    sumArousal += t.stats.arousal ?? 0;
    sumValence += t.stats.valence ?? 0;

    if (typeof t.stats.reconMSE === "number") {
      reconSum += t.stats.reconMSE;
      reconCnt++;
    }
    if (t.heads && maxHeads > 0) {
      const L = Math.min(maxHeads, t.heads.length);
      for (let h = 0; h < L; h++) {
        const th = t.heads[h] ?? 0;
        headCos[h] += Math.cos(th);
        headSin[h] += Math.sin(th);
        headN[h] += 1;
      }
    }
  }

  const perDimUsage: number[] = new Array(dims);
  const perDimPerplexity: number[] = new Array(dims);
  for (let d = 0; d < dims; d++) {
    let used = 0;
    let H = 0;
    for (let b = 0; b < bins; b++) {
      const c = perDimHists[d][b] | 0;
      if (c > 0) used++;
      const p = c / Math.max(1, count);
      if (p > 0) H += -p * Math.log2(p);
    }
    perDimUsage[d] = used / Math.max(1, bins);
    perDimPerplexity[d] = Math.pow(2, H);
  }

  const mAtt = count > 0 ? sumAtt / count : 0;
  const vAtt = count > 0 ? Math.max(0, sumAtt2 / count - mAtt * mAtt) : 0;

  const headMeans =
    maxHeads > 0
      ? headCos.map((c, i) => (headN[i] > 0 ? Math.atan2(headSin[i], c) : 0))
      : undefined;

  return {
    count,
    dims,
    perDimHists,
    perDimUsage,
    perDimPerplexity,
    attnMean: mAtt,
    attnStd: Math.sqrt(vAtt),
    arousal: count > 0 ? sumArousal / count : 0,
    valence: count > 0 ? sumValence / count : 0,
    reconMSE: reconCnt > 0 ? reconSum / reconCnt : undefined,
    headCount: maxHeads > 0 ? maxHeads : undefined,
    headMeans,
  };
}

export function TokenAtlas({ tokens, bins = 4 }: TokenAtlasProps) {
  const stats = useMemo(() => {
    if (!tokens || tokens.length === 0) {
      return {
        count: 0,
        dims: 0,
        perDimHists: [],
        perDimUsage: [],
        perDimPerplexity: [],
        attnMean: 0,
        attnStd: 0,
        arousal: 0,
        valence: 0,
        reconMSE: undefined,
        headCount: 0,
        headMeans: [],
      } as AtlasStats;
    }
    return computeAtlas(tokens, bins);
  }, [tokens, bins]);

  if (!tokens || tokens.length === 0) {
    return (
      <div className="rounded-xl border border-zinc-800 p-3 text-xs text-zinc-400 bg-zinc-900/50">
        TokenAtlas: no tokens
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-zinc-800 p-3 text-xs text-zinc-300 bg-zinc-900/50">
      <div className="flex flex-wrap gap-3">
        <div>tokens: <span className="font-mono">{stats.count}</span></div>
        <div>dims: <span className="font-mono">{stats.dims}</span></div>
        <div>attn μ/σ: <span className="font-mono">{stats.attnMean.toFixed(3)} / {stats.attnStd.toFixed(3)}</span></div>
        <div>arousal μ: <span className="font-mono">{stats.arousal.toFixed(3)}</span></div>
        <div>valence μ: <span className="font-mono">{stats.valence.toFixed(3)}</span></div>
        {typeof stats.reconMSE === "number" && (
          <div>recon MSE μ: <span className="font-mono">{stats.reconMSE.toFixed(4)}</span></div>
        )}
        {stats.headCount && stats.headCount > 0 && (
          <div>heads: <span className="font-mono">{stats.headCount}</span></div>
        )}
      </div>
      {stats.headCount && stats.headCount > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <div className="text-[11px] text-zinc-400">head means (deg):</div>
          <div className="flex flex-wrap gap-2">
            {stats.headMeans?.map((a, i) => (
              <span key={i} className="font-mono">{((a * 180) / Math.PI).toFixed(0)}</span>
            ))}
          </div>
        </div>
      )}
      <div className="mt-2 grid grid-cols-1 gap-2">
        {stats.perDimHists.map((hist, d) => {
          const total = hist.reduce((a, b) => a + b, 0) || 1;
          return (
            <div key={d} className="rounded-lg border border-zinc-800 p-2">
              <div className="flex items-center justify-between text-[11px] text-zinc-400">
                <div>dim {d}</div>
                <div>usage {(100 * stats.perDimUsage[d]).toFixed(1)}%</div>
                <div>perplexity {stats.perDimPerplexity[d].toFixed(2)}</div>
              </div>
              <div className="mt-1 flex gap-1 items-end h-12">
                {hist.map((c, b) => {
                  const h = Math.round((c / total) * 100);
                  return (
                    <div key={b} className="flex flex-col items-center">
                      <div
                        className="w-6 bg-cyan-600/60 border border-cyan-400/30"
                        style={{ height: `${Math.max(2, h / 2)}px` }}
                        title={`bin ${b}: ${c}`}
                      />
                      <div className="text-[10px] text-zinc-500 mt-1">{b}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
