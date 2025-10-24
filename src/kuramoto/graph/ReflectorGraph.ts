export type ReflectorNode = {
  x: number;
  y: number;
  a: number; // PageRank-like activity
  w: number; // node weight (persistence/strength)
};

export class ReflectorGraph {
  private W: number;
  private H: number;
  private nodes: ReflectorNode[] = [];
  private a0: Float32Array; // preferred activities (external prompt/policy)
  private tmp: Float32Array; // scratch

  constructor(W: number, H: number) {
    this.W = W;
    this.H = H;
    this.a0 = new Float32Array(0);
    this.tmp = new Float32Array(0);
  }

  // Refresh candidates from salience or persistent features
  // Keep top-K nodes, update positions and weights deterministically.
  updateCandidates(S: Float32Array, K = 16, minDist = 6) {
    const W = this.W, H = this.H, N = W * H;
    const idxs = Array.from({ length: N }, (_, i) => i);
    // Deterministic sort given deterministic S
    idxs.sort((i, j) => S[j] - S[i]);

    const nodes: ReflectorNode[] = [];
    const dmin2 = minDist * minDist;

    const take = Math.min(K, N);
    outer: for (let t = 0; t < take; t++) {
      const i = idxs[t];
      const x = i % W;
      const y = (i / W) | 0;
      // Greedy spacing with min distance
      for (let k = 0; k < nodes.length; k++) {
        const dx = nodes[k].x - x;
        const dy = nodes[k].y - y;
        if (dx * dx + dy * dy < dmin2) continue outer;
      }
      nodes.push({ x, y, a: 0, w: Math.max(0, S[i]) });
      if (nodes.length >= K) break;
    }
    this.nodes = nodes;
    if (this.a0.length !== nodes.length) {
      this.a0 = new Float32Array(nodes.length);
      this.tmp = new Float32Array(nodes.length);
    }
  }

  // One power-iteration step: a = (1-λ) a0 + λ P^T a
  // Build k-NN symmetric weights with Gaussian falloff each step.
  step(lambda = 0.9, sigma = 8, k = 4) {
    const n = this.nodes.length;
    if (n === 0) return;
    const nn = k;

    // Build row-stochastic P and compute pull = P^T a
    for (let i = 0; i < n; i++) this.tmp[i] = 0;
    for (let i = 0; i < n; i++) {
      // find k nearest j
      let best: { j: number; w: number }[] = [];
      const xi = this.nodes[i].x, yi = this.nodes[i].y;
      for (let j = 0; j < n; j++) if (j !== i) {
        const xj = this.nodes[j].x, yj = this.nodes[j].y;
        const dx = xi - xj, dy = yi - yj;
        const w = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        best.push({ j, w });
      }
      best.sort((a, b) => b.w - a.w);
      best = best.slice(0, Math.min(nn, best.length));
      let sumW = 0;
      for (const e of best) sumW += e.w;
      const inv = sumW > 0 ? 1 / sumW : 0;
      // P^T a accumulation (pull form)
      let pull = 0;
      for (const e of best) pull += (e.w * inv) * this.nodes[e.j].a;
      this.tmp[i] = (1 - lambda) * (this.a0[i] || 0) + lambda * pull;
    }
    // swap into nodes
    for (let i = 0; i < n; i++) this.nodes[i].a = this.tmp[i];
  }

  // Splat node activity back into a W×H field for A sources
  splatToField(gain = 1.0, sigma = 6): Float32Array {
    const out = new Float32Array(this.W * this.H);
    const twoSig2 = 2 * sigma * sigma;
    for (const node of this.nodes) {
      const ax = gain * Math.max(0, node.a) * Math.max(1e-6, node.w);
      if (ax === 0) continue;
      for (let y = 0; y < this.H; y++) {
        for (let x = 0; x < this.W; x++) {
          const dx = x - node.x;
          const dy = y - node.y;
          const g = Math.exp(-(dx * dx + dy * dy) / twoSig2);
          out[y * this.W + x] += ax * g;
        }
      }
    }
    return out;
  }

  // Optional: external prior a0 (prompt-driven)
  setPrior(prior: Float32Array) {
    if (prior.length !== this.nodes.length) return;
    this.a0.set(prior);
  }

  // Expose nodes for debugging/visualization (immutable copy)
  getNodes(): ReadonlyArray<ReflectorNode> {
    return this.nodes;
  }
}
