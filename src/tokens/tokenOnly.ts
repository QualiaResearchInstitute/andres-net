import type { PAToken } from "./pat";
import calib from "./calib.json";

/**
 * Token-only proxy for attention modulators.
 * Builds A and U proxies from PAT tokens without reading the substrate (phases/energy).
 * - A proxy: splat per-patch Gaussian with amplitude ~ stats.attnMean (fallback: entropyRate)
 * - U proxy: a blurred copy of A (context echo)
 * - Aact/Uact: tanh-bounded activities
 * - lapA/divU: simple 5-point Laplacians of A and U respectively
 */
export function tokensToAttentionFields(
  tokens: PAToken[],
  W: number,
  H: number,
  scale: 32 | 64 = 32
): { Aact: Float32Array; Uact: Float32Array; lapA: Float32Array; divU: Float32Array } {
  const A = new Float32Array(W * H);
  const U = new Float32Array(W * H);

  // Splat tokens into A using a Gaussian kernel centered at patch midpoints.
  // Use attnMean as amplitude; fallback to entropyRate if attnMean is 0/NaN.
  const sigma = Math.max(2, Math.round(0.5 * scale));
  const twoSig2 = 2 * sigma * sigma;

  for (const t of tokens) {
    // patchId looks like "x0,y0,scale"
    const hdr = t.hdr;
    const pid = hdr.patchId || "0,0,32";
    const parts = pid.split(",");
    let x0 = 0, y0 = 0, sc = scale;
    if (parts.length >= 2) {
      x0 = Math.max(0, Math.min(W - 1, parseInt(parts[0] || "0", 10) | 0));
      y0 = Math.max(0, Math.min(H - 1, parseInt(parts[1] || "0", 10) | 0));
    }
    if (parts.length >= 3) {
      const scParsed = parseInt(parts[2] || String(scale), 10) | 0;
      if (scParsed === 32 || scParsed === 64) sc = scParsed;
    }
    const cx = Math.min(W - 1, x0 + Math.floor(sc / 2));
    const cy = Math.min(H - 1, y0 + Math.floor(sc / 2));

    const amp = sanitizeAmplitude(t);
    if (amp === 0) continue;

    // Local window bounds to speed up splat
    const rx = Math.max(1, sigma * 2);
    const ry = Math.max(1, sigma * 2);
    const xmin = Math.max(0, cx - rx);
    const xmax = Math.min(W - 1, cx + rx);
    const ymin = Math.max(0, cy - ry);
    const ymax = Math.min(H - 1, cy + ry);

    for (let y = ymin; y <= ymax; y++) {
      const dy = y - cy;
      const row = y * W;
      for (let x = xmin; x <= xmax; x++) {
        const dx = x - cx;
        const g = Math.exp(-(dx * dx + dy * dy) / twoSig2);
        A[row + x] += amp * g;
      }
    }
  }

  // U is a context echo (blurred A)
  const Ublur = boxBlurSeparable(A, W, H, Math.max(2, Math.round(0.75 * sigma)));

  // Assign U
  U.set(Ublur);

  // Optional deterministic calibration (fixed scalars from JSON)
  const aScale = (calib as any)?.Aact?.scale ?? 1;
  const aBias = (calib as any)?.Aact?.bias ?? 0;
  const uScale = (calib as any)?.Uact?.scale ?? 1;
  const uBias = (calib as any)?.Uact?.bias ?? 0;

  // Activities (bounded)
  const Aact = new Float32Array(W * H);
  const Uact = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const Ai = aScale * A[i] + aBias;
    const Ui = uScale * U[i] + uBias;
    Aact[i] = tanh(Ai);
    Uact[i] = tanh(Ui);
  }

  // Laplacians as proxies
  const lapA = laplacian5(A, W, H, true);
  const divU = laplacian5(U, W, H, true);

  return { Aact, Uact, lapA, divU };
}

function sanitizeAmplitude(t: PAToken) {
  const m = t.stats?.attnMean ?? 0;
  const e = t.stats?.entropyRate ?? 0;
  let amp = Number.isFinite(m) && m > 0 ? m : (Number.isFinite(e) ? e : 0);
  // light compression
  amp = Math.max(0, Math.min(1, amp));
  return amp;
}

const tanh = (x: number) => Math.tanh(x);

function clamp(x: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, x));
}

function boxBlurSeparable(src: Float32Array, W: number, H: number, radius: number, wrap = true) {
  const N = W * H;
  if (radius <= 0) return new Float32Array(src);
  const tmp = new Float32Array(N);
  const out = new Float32Array(N);
  const r = Math.floor(radius);
  const win = 2 * r + 1;

  // horizontal
  for (let y = 0; y < H; y++) {
    let acc = 0;
    for (let dx = -r; dx <= r; dx++) {
      const x0 = clamp(dx, 0, W - 1);
      const i0 = y * W + (wrap ? (dx + W) % W : x0);
      acc += src[i0];
    }
    for (let x = 0; x < W; x++) {
      const xl = x - r - 1;
      const xr = x + r;
      const il = y * W + (wrap ? (xl + W) % W : clamp(xl, 0, W - 1));
      const ir = y * W + (wrap ? (xr + W) % W : clamp(xr, 0, W - 1));
      if (x === 0) {
        tmp[y * W + x] = acc / win;
      } else {
        acc += src[ir] - src[il];
        tmp[y * W + x] = acc / win;
      }
    }
  }

  // vertical
  for (let x = 0; x < W; x++) {
    let acc = 0;
    for (let dy = -r; dy <= r; dy++) {
      const y0 = clamp(dy, 0, H - 1);
      const i0 = (wrap ? (dy + H) % H : y0) * W + x;
      acc += tmp[i0];
    }
    for (let y = 0; y < H; y++) {
      const yu = y - r - 1;
      const yd = y + r;
      const iu = (wrap ? (yu + H) % H : clamp(yu, 0, H - 1)) * W + x;
      const id = (wrap ? (yd + H) % H : clamp(yd, 0, H - 1)) * W + x;
      if (y === 0) {
        out[y * W + x] = acc / win;
      } else {
        acc += tmp[id] - tmp[iu];
        out[y * W + x] = acc / win;
      }
    }
  }

  return out;
}

function laplacian5(field: Float32Array, W: number, H: number, wrap = true) {
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const ym = wrap ? (y - 1 + H) % H : Math.max(0, y - 1);
    const yp = wrap ? (y + 1) % H : Math.min(H - 1, y + 1);
    for (let x = 0; x < W; x++) {
      const xm = wrap ? (x - 1 + W) % W : Math.max(0, x - 1);
      const xp = wrap ? (x + 1) % W : Math.min(W - 1, x + 1);
      const i = y * W + x;
      const l = y * W + xm;
      const r = y * W + xp;
      const u = ym * W + x;
      const d = yp * W + x;
      out[i] = field[l] + field[r] + field[u] + field[d] - 4 * field[i];
    }
  }
  return out;
}
