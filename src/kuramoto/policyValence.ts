/**
 * Deterministic valence-seeking policy.
 * Adjusts attention/physics knobs to increase valence while keeping arousal within a target band.
 *
 * Notes:
 * - Strictly deterministic: fixed candidate grid, fixed iteration order, seed-based head index.
 * - Lightweight linear proxy for scoring to avoid running full sim branches.
 * - Leaves head weights unchanged by default for stability; can be enabled later.
 */

export type AffectLike = {
  arousal: number; // [0,1]
  valence: number; // [0,1]
};

export type PolicyCurParams = {
  gammaK: number;
  gammaAlpha: number;
  gammaD: number;
  deltaD: number;
  // Optional, reserved for future head targeting (top-1 head weight nudging)
  heads?: Array<{ enabled?: boolean; weight?: number }>;
};

export interface PolicyCfg {
  targetArousal: [number, number]; // inclusive band
  steps: number; // reserved for future multi-step lookahead
  gammaK: [number, number];
  gammaAlpha: [number, number];
  gammaD: [number, number];
  deltaD: [number, number];
  headWeightBounds: [number, number];
}

export type PolicyResult = {
  gammaK: number;
  gammaAlpha: number;
  gammaD: number;
  deltaD: number;
  heads?: Array<{ weight?: number }>;
  picked?: { dGammaK: number; dGammaD: number; dGammaAlpha: number; headIdx: number };
  score?: number;
};

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/**
 * Deterministic beam/grid search over small deltas around the current params.
 * Score = proxyValence - lambda * arousalPenalty, where proxyValence is linear in deltas.
 */
export function stepValencePolicy(
  affect: AffectLike,
  params: PolicyCurParams,
  rngSeed: number,
  cfg: PolicyCfg
): PolicyResult {
  // Deterministic candidate deltas
  const d = (a: number) => [0, +a, -a] as const;
  const candidates: Array<{ dγK: number; dγD: number; dγα: number }> = [];
  for (const dγK of d(0.05)) for (const dγD of d(0.05)) for (const dγα of d(0.02)) {
    candidates.push({ dγK, dγD, dγα });
  }

  // Deterministic head selection (if needed later)
  const headsLen = Math.max(1, params.heads?.length ?? 1);
  const headIdx = ((rngSeed * 1103515245 + 12345) >>> 0) % headsLen;

  // Penalty if outside arousal band
  const [aLo, aHi] = cfg.targetArousal;
  const inBand = affect.arousal >= aLo && affect.arousal <= aHi;
  const arousalPenalty = inBand ? 0 : 0.2 * Math.min(Math.abs(affect.arousal - (affect.arousal > aHi ? aHi : aLo)), 1);

  // Simple linear proxy for valence change (domain knowledge heuristic)
  // ↑γK tends to increase coherence and reduce arousal; ↑γD tends to increase chaos (decrease valence, increase arousal)
  let bestIdx = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const proxyValence = affect.valence + 0.30 * c.dγK - 0.20 * c.dγD + 0.05 * c.dγα;
    const score = proxyValence - arousalPenalty;
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  const best = candidates[bestIdx];

  // Apply deltas with clamping to bounds
  const next: PolicyResult = {
    gammaK: clamp(params.gammaK + best.dγK, cfg.gammaK[0], cfg.gammaK[1]),
    gammaD: clamp(params.gammaD + best.dγD, cfg.gammaD[0], cfg.gammaD[1]),
    gammaAlpha: clamp(params.gammaAlpha + best.dγα, cfg.gammaAlpha[0], cfg.gammaAlpha[1]),
    deltaD: clamp(params.deltaD, cfg.deltaD[0], cfg.deltaD[1]), // keep δD stable for now (optionally tune later)
    picked: { dGammaK: best.dγK, dGammaD: best.dγD, dGammaAlpha: best.dγα, headIdx },
    score: bestScore,
  };

  // Optional: top-1 head weight nudging (disabled by default for stability)
  // If enabling, uncomment the block below.
  /*
  if (params.heads && params.heads.length > 0) {
    const [lo, hi] = cfg.headWeightBounds;
    const headsOut = params.heads.map((h, idx) => {
      if (idx !== headIdx) return {};
      const w = clamp((h.weight ?? 0) * 1.05, lo, hi);
      return { weight: w };
    });
    next.heads = headsOut;
  }
  */

  return next;
}
