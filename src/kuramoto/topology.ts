export interface Pocket {
  id: number;
  mask: Uint8Array;         // 1 for in-pocket, 0 otherwise
  rimIdx: Uint32Array;      // linear indices along rim (unordered set)
  flux: number;             // placeholder (computed elsewhere if needed)
  area: number;
}

/**
 * Compute pocket score field:
 * P = w.rho * rho + w.attn * tanh(A) - w.shear * |gradPhi| - w.defects * (vort?1:0)
 */
export function computePocketScore(
  rho: Float32Array,
  A: Float32Array,
  gradPhiMag: Float32Array,
  vort: Uint8Array,
  W: number,
  H: number,
  w: { rho: number; attn: number; shear: number; defects: number }
): Float32Array {
  const N = W * H;
  const out = new Float32Array(N);
  const wr = w.rho ?? 0;
  const wa = w.attn ?? 0;
  const ws = w.shear ?? 0;
  const wd = w.defects ?? 0;
  for (let i = 0; i < N; i++) {
    const att = A[i] ?? 0;
    out[i] = wr * (rho[i] ?? 0) + wa * Math.tanh(att) - ws * (gradPhiMag[i] ?? 0) - wd * (vort[i] ? 1 : 0);
  }
  return out;
}

/**
 * Extract pockets from a score map:
 * - threshold
 * - morphological close (4-neigh, 1 dilate + 1 erode) deterministically
 * - 4-connected components
 * - rim detection (any 4-neigh outside)
 */
export function extractPockets(
  score: Float32Array,
  W: number,
  H: number,
  thresh: number,
  minArea: number
): Pocket[] {
  const N = W * H;
  const mask = new Uint8Array(N);
  for (let i = 0; i < N; i++) mask[i] = score[i] >= thresh ? 1 : 0;

  // Morphological close: 1 dilate then 1 erode (4-neighborhood), no wrap
  const dil = new Uint8Array(N);
  // Dilate
  for (let y = 0; y < H; y++) {
    const ym = y - 1;
    const yp = y + 1;
    for (let x = 0; x < W; x++) {
      const xm = x - 1;
      const xp = x + 1;
      const i = y * W + x;
      let on = mask[i];
      if (!on) {
        if (x > 0 && mask[y * W + xm]) on = 1;
        else if (x + 1 < W && mask[y * W + xp]) on = 1;
        else if (y > 0 && mask[ym * W + x]) on = 1;
        else if (y + 1 < H && mask[yp * W + x]) on = 1;
      }
      dil[i] = on ? 1 : 0;
    }
  }
  const cls = new Uint8Array(N);
  // Erode
  for (let y = 0; y < H; y++) {
    const ym = y - 1;
    const yp = y + 1;
    for (let x = 0; x < W; x++) {
      const xm = x - 1;
      const xp = x + 1;
      const i = y * W + x;
      let on =
        dil[i] &&
        (x > 0 ? dil[y * W + xm] : 0) &&
        (x + 1 < W ? dil[y * W + xp] : 0) &&
        (y > 0 ? dil[ym * W + x] : 0) &&
        (y + 1 < H ? dil[yp * W + x] : 0);
      cls[i] = on ? 1 : 0;
    }
  }

  // Connected components (4-neigh), deterministic scan order
  const labels = new Int32Array(N);
  for (let i = 0; i < N; i++) labels[i] = -1;

  const pockets: Pocket[] = [];
  let nextId = 0;
  const stack = new Uint32Array(N);
  for (let y0 = 0; y0 < H; y0++) {
    for (let x0 = 0; x0 < W; x0++) {
      const i0 = y0 * W + x0;
      if (!cls[i0] || labels[i0] !== -1) continue;

      // Flood fill
      let sp = 0;
      stack[sp++] = i0;
      labels[i0] = nextId;
      let area = 0;

      while (sp > 0) {
        const i = stack[--sp] >>> 0;
        area++;
        const y = (i / W) | 0;
        const x = i - y * W;
        // 4-neighbors (no wrap)
        if (x > 0) {
          const j = i - 1;
          if (cls[j] && labels[j] === -1) {
            labels[j] = nextId;
            stack[sp++] = j;
          }
        }
        if (x + 1 < W) {
          const j = i + 1;
          if (cls[j] && labels[j] === -1) {
            labels[j] = nextId;
            stack[sp++] = j;
          }
        }
        if (y > 0) {
          const j = i - W;
          if (cls[j] && labels[j] === -1) {
            labels[j] = nextId;
            stack[sp++] = j;
          }
        }
        if (y + 1 < H) {
          const j = i + W;
          if (cls[j] && labels[j] === -1) {
            labels[j] = nextId;
            stack[sp++] = j;
          }
        }
      }

      if (area >= minArea) {
        // Build per-pocket mask and rim
        const pmask = new Uint8Array(N);
        for (let i = 0; i < N; i++) {
          if (labels[i] === nextId) pmask[i] = 1;
        }
        const rim: number[] = [];
        for (let i = 0; i < N; i++) {
          if (!pmask[i]) continue;
          const y = (i / W) | 0;
          const x = i - y * W;
          let boundary = false;
          if (x === 0 || !pmask[i - 1]) boundary = true;
          else if (x + 1 >= W || !pmask[i + 1]) boundary = true;
          else if (y === 0 || !pmask[i - W]) boundary = true;
          else if (y + 1 >= H || !pmask[i + W]) boundary = true;
          if (boundary) rim.push(i);
        }
        pockets.push({
          id: nextId,
          mask: pmask,
          rimIdx: Uint32Array.from(rim),
          flux: 0,
          area,
        });
      }
      nextId++;
    }
  }
  return pockets;
}

/**
 * Sum wrapped differences of phase along rim indices (placeholder for tightness check).
 * Deterministic scan; consumers may normalize by rim length.
 */
export function rimCirculation(
  phi: Float32Array,
  rim: Uint32Array,
  W: number,
  H: number
): number {
  // Simple local gradient magnitude sum over rim cells
  let acc = 0;
  for (let k = 0; k < rim.length; k++) {
    const i = rim[k] >>> 0;
    const y = (i / W) | 0;
    const x = i - y * W;
    const xm = x > 0 ? i - 1 : i;
    const xp = x + 1 < W ? i + 1 : i;
    const ym = y > 0 ? i - W : i;
    const yp = y + 1 < H ? i + W : i;
    const gx = wrapAngle(phi[xp] - phi[xm]) * 0.5;
    const gy = wrapAngle(phi[yp] - phi[ym]) * 0.5;
    acc += Math.hypot(gx, gy);
  }
  return acc;
}

function wrapAngle(x: number) {
  let y = (x + Math.PI) % (2 * Math.PI);
  if (y < 0) y += 2 * Math.PI;
  return y - Math.PI;
}
