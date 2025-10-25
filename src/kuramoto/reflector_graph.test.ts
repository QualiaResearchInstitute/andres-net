import { describe, it, expect } from "vitest";
import { ReflectorGraph } from "./graph/ReflectorGraph";

function makeFlatSalience(W: number, H: number, spike?: { x: number; y: number; v: number }) {
  const S = new Float32Array(W * H);
  if (spike) {
    const idx = spike.y * W + spike.x;
    S[idx] = spike.v;
  }
  return S;
}

describe("ReflectorGraph determinism and damping", () => {
  it("produces identical node activities across identical runs", () => {
    const W = 16, H = 16;
    const S = makeFlatSalience(W, H); // all ties -> deterministic by index and spacing

    const A = new ReflectorGraph(W, H);
    const B = new ReflectorGraph(W, H);

    A.updateCandidates(S, 8, 4);
    B.updateCandidates(S, 8, 4);

    // Step with damping and fixed traversal order
    A.step(0.9, 6, 4);
    B.step(0.9, 6, 4);

    const aNodes = A.getNodes();
    const bNodes = B.getNodes();
    expect(aNodes.length).toBe(bNodes.length);

    for (let i = 0; i < aNodes.length; i++) {
      expect(aNodes[i].x).toBe(bNodes[i].x);
      expect(aNodes[i].y).toBe(bNodes[i].y);
      expect(aNodes[i].w).toBeCloseTo(bNodes[i].w, 12);
      expect(aNodes[i].a).toBeCloseTo(bNodes[i].a, 12);
    }
  });

  it("remains stable when all salience entries are equal (no oscillatory breathing)", () => {
    const W = 12, H = 12;
    const S = makeFlatSalience(W, H);

    const G = new ReflectorGraph(W, H);
    G.updateCandidates(S, 10, 3);

    // Run multiple batches of steps and ensure no NaN/Inf and boundedness
    for (let r = 0; r < 5; r++) {
      G.step(0.85, 4, 3);
      const nodes = G.getNodes();
      for (const n of nodes) {
        expect(Number.isFinite(n.a)).toBe(true);
        expect(n.a).toBeGreaterThanOrEqual(0);
        // activities are seeded from a0/a and damped; should not explode
        expect(n.a).toBeLessThanOrEqual(10);
      }
    }
  });

  it("deterministic tie-breaking by index yields stable candidate set under equal S", () => {
    const W = 10, H = 10;
    const S = makeFlatSalience(W, H);

    const G1 = new ReflectorGraph(W, H);
    const G2 = new ReflectorGraph(W, H);
    G1.updateCandidates(S, 6, 2);
    G2.updateCandidates(S, 6, 2);

    const n1 = G1.getNodes();
    const n2 = G2.getNodes();
    expect(n1.length).toBe(n2.length);
    for (let i = 0; i < n1.length; i++) {
      expect(n1[i].x).toBe(n2[i].x);
      expect(n1[i].y).toBe(n2[i].y);
    }
  });
});
