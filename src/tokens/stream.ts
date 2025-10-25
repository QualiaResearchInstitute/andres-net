import { encode } from "cbor-x";
import type { PAToken } from "./pat";

/**
 * Send a PAT frame over an open WebSocket using CBOR encoding.
 * Safe to call even if the socket isn't open; it will no-op until OPEN.
 */
export function streamPAT(ws: WebSocket, tokens: PAToken[]) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  // Minimal envelope for downstream consumers
  const payload = { type: "PAT", t: performance.now?.() ?? Date.now(), tokens };
  try {
    const buf = encode(payload);
    ws.send(buf);
  } catch {
    // As a fallback, try JSON (keeps demo usable if CBOR fails unexpectedly)
    try {
      ws.send(JSON.stringify(payload));
    } catch {
      // swallow
    }
  }
}

/** Open a WebSocket for PAT streaming */
export function openPATStream(url: string): WebSocket {
  const ws = new WebSocket(url);
  // binary frames preferred
  try {
    // @ts-ignore (not in TS lib dom types)
    ws.binaryType = "arraybuffer";
  } catch {
    // ignore
  }
  return ws;
}

/** Close a WebSocket safely */
export function closePATStream(ws: WebSocket | null | undefined) {
  if (!ws) return;
  try {
    ws.close();
  } catch {
    // ignore
  }
}

/**
 * Start a throttled PAT stream given a frame supplier.
 * Returns a stop() handle to teardown timer + socket.
 */
export function startPATStream(
  makeFrame: () => PAToken[] | null,
  opts: { url: string; fps: number },
): { stop: () => void; ws: WebSocket } {
  const ws = openPATStream(opts.url);
  const interval = Math.max(50, Math.floor(1000 / Math.max(1, opts.fps)));
  const timer = setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const frame = makeFrame();
    if (frame && frame.length > 0) {
      streamPAT(ws, frame);
    }
  }, interval);

  const stop = () => {
    try {
      clearInterval(timer);
    } catch {
      // ignore
    }
    closePATStream(ws);
  };

  return { stop, ws };
}
