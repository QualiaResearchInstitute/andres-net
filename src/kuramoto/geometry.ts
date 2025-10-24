import * as martinez from "martinez-polygon-clipping";
import { polygonSignedArea } from "./math";
import type { Point } from "./types";

type MartinezPolygon = number[][][];

function ensureClosedRing(points: Point[]): Array<[number, number]> {
  const ring: Array<[number, number]> = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    ring.push([p.x, p.y] as [number, number]);
  }
  if (ring.length === 0) return ring;
  const first = ring[0];
  const last = ring[ring.length - 1];
  if (!first || !last) return ring;
  const fx = first[0];
  const fy = first[1];
  const lx = last[0];
  const ly = last[1];
  if (fx !== lx || fy !== ly) {
    ring.push([fx, fy]);
  }
  return ring;
}

export function toMartinez(points: Point[]): MartinezPolygon {
  if (points.length < 3) return [];
  const ring = ensureClosedRing(points);
  const ringCoords: number[][] = [];
  for (let i = 0; i < ring.length; i++) {
    const coord = ring[i];
    if (!coord) continue;
    const x = coord[0];
    const y = coord[1];
    ringCoords.push([x, y]);
  }
  return [ringCoords] as MartinezPolygon;
}

function fromMartinez(poly: MartinezPolygon): Point[] {
  if (!poly || poly.length === 0) return [];
  let bestRing: number[][] | null = null;
  let bestArea = -Infinity;
  for (let i = 0; i < poly.length; i++) {
    const polygon = poly[i] as number[][];
    if (!polygon || polygon.length === 0) continue;
    const ringCandidate = polygon;
    if (ringCandidate.length < 4) continue;
    let area = 0;
    for (let k = 0; k < ringCandidate.length - 1; k++) {
      const currentPoint = ringCandidate[k] as number[];
      const nextPoint = ringCandidate[k + 1] as number[];
      if (!currentPoint || !nextPoint) continue;
      const x1 = currentPoint[0] ?? 0;
      const y1 = currentPoint[1] ?? 0;
      const x2 = nextPoint[0] ?? 0;
      const y2 = nextPoint[1] ?? 0;
      area += x1 * y2 - x2 * y1;
    }
    area = Math.abs(area * 0.5);
    if (area > bestArea) {
      bestArea = area;
      bestRing = ringCandidate as number[][];
    }
  }
  if (!bestRing) return [];
  const result: Point[] = [];
  for (let i = 0; i < bestRing.length - 1; i++) {
    const point = bestRing[i] as number[];
    if (!point) continue;
    const x = point[0] ?? 0;
    const y = point[1] ?? 0;
    result.push({ x, y });
  }
  return result;
}

export function flipPoints(points: Point[], axis: "horizontal" | "vertical"): Point[] {
  if (points.length === 0) return points;
  const cx = points.reduce((acc, p) => acc + p.x, 0) / points.length;
  const cy = points.reduce((acc, p) => acc + p.y, 0) / points.length;
  return points.map((p) =>
    axis === "horizontal"
      ? { x: cx * 2 - p.x, y: p.y }
      : { x: p.x, y: cy * 2 - p.y },
  );
}

export function rotatePoints90(points: Point[], direction: "cw" | "ccw"): Point[] {
  if (points.length === 0) return points;
  const cx = points.reduce((acc, p) => acc + p.x, 0) / points.length;
  const cy = points.reduce((acc, p) => acc + p.y, 0) / points.length;
  const sign = direction === "cw" ? -1 : 1;
  return points.map((p) => {
    const dx = p.x - cx;
    const dy = p.y - cy;
    return {
      x: cx + sign * dy,
      y: cy - sign * dx,
    };
  });
}

export function skewPoints(points: Point[], axis: "x" | "y", factor: number): Point[] {
  if (points.length === 0) return points;
  const cx = points.reduce((acc, p) => acc + p.x, 0) / points.length;
  const cy = points.reduce((acc, p) => acc + p.y, 0) / points.length;
  return points.map((p) => {
    const dx = p.x - cx;
    const dy = p.y - cy;
    return axis === "x"
      ? { x: p.x + dy * factor, y: p.y }
      : { x: p.x, y: p.y + dx * factor };
  });
}

export function smoothPoints(points: Point[], iterations = 1): Point[] {
  if (points.length < 3) return points;
  let current = points.map((p) => ({ x: p.x, y: p.y }));
  for (let iter = 0; iter < iterations; iter++) {
    const next: Point[] = [];
    for (let i = 0; i < current.length; i++) {
      const prev = current[(i - 1 + current.length) % current.length];
      const cur = current[i];
      const nextPoint = current[(i + 1) % current.length];
      next.push({
        x: (prev.x + cur.x * 2 + nextPoint.x) / 4,
        y: (prev.y + cur.y * 2 + nextPoint.y) / 4,
      });
    }
    current = next;
  }
  return current;
}

export function ensureWinding(points: Point[]): Point[] {
  if (points.length < 3) return points;
  const area = polygonSignedArea(points);
  if (area < 0) {
    const reversed = points.slice().reverse();
    return reversed;
  }
  return points;
}

export function booleanCombine(
  mode: "union" | "intersect" | "subtract",
  base: Point[],
  others: Point[][],
): Point[] {
  if (base.length < 3 || others.length === 0) return base;
  const basePoly = toMartinez(base);
  let result = basePoly;
  for (let i = 0; i < others.length; i++) {
    const other = toMartinez(others[i]);
    if (other.length === 0) continue;
    switch (mode) {
      case "union":
        result = martinez.union(result, other) as MartinezPolygon;
        break;
      case "intersect":
        result = martinez.intersection(result, other) as MartinezPolygon;
        break;
      case "subtract":
        result = martinez.diff(result, other) as MartinezPolygon;
        break;
    }
  }
  const polygon = fromMartinez(result);
  return ensureWinding(polygon);
}
