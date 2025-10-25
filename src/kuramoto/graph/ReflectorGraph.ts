export type ReflectorNode = {
  x: number;
  y: number;
  a: number; // PageRank-like activity (current)
  w: number; // node weight (persistence/strength)
};

export class ReflectorGraph {
  private W: number;
  private H: number;
  private nodes: ReflectorNode[] = [];

  // Deterministic seeds and buffers
  private a0: Float32Array; // preferred activities (external prompt/policy)
  private a: Float32Array; // working activity vector
  private next: Float32Array; // scratch/next activities

  // Pre-sorted adjacency and outdegree (deterministic traversal)
  private adj: number[][] = [];
  private outdeg: number[] = [];

  constructor(W: number, H: number) {
    this.W = W;
    this.H = H;
    this.a0 = new Float32Array(0);
    this.a = new Float32Array(0);
    this.next = new Float32Array(0);
  }

  // Refresh candidates from salience or persistent features
  // Keep top-K nodes, update positions and weights deterministically.
  updateCandidates(S: Float32Array, K = 16, minDist = 6) {
    const W = this.W, H = this.H, N = W * H;
    const idxs = Array.from({ length: N }, (_, i) => i);
    // Deterministic sort given deterministic S (tie-breaker by index)
    idxs.sort((i, j) => {
      const dj = S[j] - S[i];
      if (dj !== 0) return dj > 0 ? 1 : -1;
      return i - j;
    });

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
      const amp = Math.max(0, S[i]);
      nodes.push({ x, y, a: amp, w: amp });
      if (nodes.length >= K) break;
    }
    this.nodes = nodes;

    // Resize buffers
    if (this.a0.length !== nodes.length) {
      this.a0 = new Float32Array(nodes.length);
      this.a = new Float32Array(nodes.length);
      this.next = new Float32Array(nodes.length);
    } else {
      this.a.fill(0);
      this.next.fill(0);
    }

    // Initialize current activities from node.a (deterministic seed)
    for (let i = 0; i < nodes.length; i++) {
      this.a[i] = nodes[i].a;
    }

    // Rebuild deterministic adjacency once per candidate refresh
    this.buildAdjacency();
  }

  private buildAdjacency() {
    const n = this.nodes.length;
    this.adj = new Array(n);
    this.outdeg = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const xi = this.nodes[i].x, yi = this.nodes[i].y;
      const list: { j: number; d2: number }[] = [];
      for (let j = 0; j < n; j++) if (j !== i) {
        const xj = this.nodes[j].x, yj = this.nodes[j].y;
        const dx = xi - xj, dy = yi - yj;
        const d2 = dx * dx + dy * dy;
        list.push({ j, d2 });
      }
      // Deterministic ordering: sort by distance, tie-break by index
      list.sort((A, B) => (A.d2 - B.d2) || (A.j - B.j));
      this.adj[i] = list.map(e => e.j);
      this.outdeg[i] = this.adj[i].length;
    }
  }

  /**
   * Deterministic PageRank-like update:
   * For iters:
   *   next[v] = (1-damp)*a0[v]
   *   For each v in [0..N):
   *     as = damp * a[v] / max(1, outdeg[v])
   *     For nbr in first topK of adj[v] (pre-sorted):
   *       next[nbr] += as
   *   Swap a,next
   * Copies a back into nodes[].a
   */
  step(damp = 0.9, iters = 1, topK = 4) {
    const n = this.nodes.length;
    if (n === 0) return;

    const K = Math.max(0, Math.min(topK | 0, n - 1));

    for (let iter = 0; iter < Math.max(1, iters | 0); iter++) {
      // Seed
      for (let v = 0; v < n; v++) {
        this.next[v] = (1 - damp) * (this.a0[v] || 0);
      }
      // Deterministic neighbor iteration (pre-sorted adjacency)
      for (let v = 0; v < n; v++) {
        const deg = Math.max(1, this.outdeg[v]);
        const share = (damp * this.a[v]) / deg;
        const nbrs = this.adj[v];
        const limit = K > 0 ? Math.min(K, nbrs.length) : nbrs.length;
        for (let k = 0; k < limit; k++) {
          const j = nbrs[k];
          this.next[j] += share;
        }
      }
      // swap
      const tmp = this.a;
      this.a = this.next;
      this.next = tmp;
    }

    // Write back to nodes
    for (let i = 0; i < n; i++) {
      this.nodes[i].a = this.a[i];
    }
  }

  // Splat node activity back into a W×H field for A sources
  splatToField(gain = 1.0, sigma = 6): Float32Array {
    const out = new Float32Array(this.W * this.H);
    const twoSig2 = 2 * sigma * sigma;
    for (const node of this.nodes) {
      const ax = gain * Math.max(0, node.a) * Math.max(1e-6, node.w);
      if (ax === 0) continue;
      for (let y = 0; y < this.H; y++) {
        const row = y * this.W;
        for (let x = 0; x < this.W; x++) {
          const dx = x - node.x;
          const dy = y - node.y;
          const g = Math.exp(-(dx * dx + dy * dy) / twoSig2);
          out[row + x] += ax * g;
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
