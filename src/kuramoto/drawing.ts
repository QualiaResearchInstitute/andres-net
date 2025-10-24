import { ImageLayer, Point } from "./types";

type PolygonMeta = {
  color?: [number, number, number];
  centroid?: Point;
  orientation?: string;
  orientationSign?: number;
};

export function drawPolyline(
  ctx: CanvasRenderingContext2D,
  pts: Point[],
  pixelSize: number,
  color = "#fff",
  lineWidth = 2,
) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo((pts[0].x + 0.5) * pixelSize, (pts[0].y + 0.5) * pixelSize);
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo((pts[i].x + 0.5) * pixelSize, (pts[i].y + 0.5) * pixelSize);
  }
  ctx.stroke();
  ctx.restore();
}

export function drawPolygon(
  ctx: CanvasRenderingContext2D,
  pts: Point[],
  pixelSize: number,
  meta?: PolygonMeta,
  showOrientation = false,
  options?: { highlight?: boolean; primary?: boolean; muted?: boolean; active?: boolean },
) {
  if (pts.length < 3) return;
  const rgb = meta?.color ?? [0, 255, 255];
  const highlight = options?.highlight ?? false;
  const primary = options?.primary ?? false;
  const muted = options?.muted ?? false;
  const active = options?.active ?? true;
  let strokeStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  if (!active) {
    strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.25)`;
  } else if (muted) {
    strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.35)`;
  }
  if (highlight) {
    strokeStyle = primary ? "rgba(56,189,248,0.95)" : "rgba(56,189,248,0.65)";
  }
  const lineWidth = highlight ? (primary ? 4 : 3) : 2;
  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = strokeStyle;
  if ((meta?.orientationSign ?? 0) < 0) {
    ctx.setLineDash([6, 3]);
  }
  ctx.beginPath();
  ctx.moveTo((pts[0].x + 0.5) * pixelSize, (pts[0].y + 0.5) * pixelSize);
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo((pts[i].x + 0.5) * pixelSize, (pts[i].y + 0.5) * pixelSize);
  }
  ctx.closePath();
  ctx.stroke();
  if (showOrientation && meta?.centroid) {
    const orientationAlpha = highlight ? 0.95 : 0.8;
    ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${orientationAlpha})`;
    ctx.font = `${Math.max(10, pixelSize * 1.6)}px monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const label = (meta.orientation ?? "flat").toUpperCase();
    ctx.fillText(label, (meta.centroid.x + 0.5) * pixelSize, (meta.centroid.y + 0.5) * pixelSize);
  }
  ctx.restore();
}

export function drawImageOutline(
  ctx: CanvasRenderingContext2D,
  layer: ImageLayer,
  pixelSize: number,
  strokeStyle: string,
) {
  const { position, width, height } = layer;
  if (width <= 0 || height <= 0) return;
  ctx.save();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  const x = position.x * pixelSize;
  const y = position.y * pixelSize;
  const w = Math.max(0, width * pixelSize);
  const h = Math.max(0, height * pixelSize);
  ctx.strokeRect(x + 0.5, y + 0.5, Math.max(0, w - 1), Math.max(0, h - 1));
  ctx.restore();
}
