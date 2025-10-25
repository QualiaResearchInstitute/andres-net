export type Point = { x: number; y: number };

export type StrokeLayer = {
  id: number;
  kind: "stroke";
  points: Point[];
  lineWidth: number;
};

export type PlaneLayer = {
  id: number;
  kind: "plane";
  planeId: number;
  points: Point[];
  centroid: Point;
  orientation: string;
  orientationSign: number;
  outlineWidth: number;
};

export type ImageLayer = {
  id: number;
  kind: "image";
  src: string;
  image: HTMLImageElement | null;
  position: Point;
  width: number;
  height: number;
  outlineWidth: number;
  loaded: boolean;
};

export type Layer = StrokeLayer | PlaneLayer | ImageLayer;

export type DragImageState =
  | {
      layerId: number;
      offset: Point;
    }
  | null;

export type TransformHandleType =
  | "move"
  | "scale-n"
  | "scale-ne"
  | "scale-e"
  | "scale-se"
  | "scale-s"
  | "scale-sw"
  | "scale-w"
  | "scale-nw"
  | "rotate";

export type PlaneTransformHover = {
  planeId: number;
  handle: TransformHandleType;
} | null;

export type PlaneTransformSession = {
  planeId: number;
  handle: TransformHandleType;
  initialPointer: Point;
  initialHandlePosition: Point;
  initialPoints: Point[];
  initialBounds: { minX: number; maxX: number; minY: number; maxY: number };
  initialCenter: Point;
  initialRotationAngle?: number;
  dirty: boolean;
};

export type PlaneMetadata = {
  cells: Int32Array;
  R: number;
  psi: number;
  color: [number, number, number];
  centroid: Point;
  orientation: string;
  orientationSign: number;
  solo: boolean;
  muted: boolean;
  locked: boolean;
  order?: number;
};

export type DagStats = {
  step: number;
  filtered: number;
  used: number;
  visited: number;
  sweeps: number;
};

export type SimulationDag = {
  dirty: boolean;
  layers: Int32Array[];
  maxDepth: number;
  snapshot: Float32Array;
  stats: DagStats;
};

export type DrawingState = {
  drawing: boolean;
  currentStroke: Point[];
  layers: Layer[];
  nextPlaneId: number;
  nextLayerId: number;
  draggingImage: DragImageState;
  selectedPlaneIds: number[];
  planeTransformSession: PlaneTransformSession | null;
  planeTransformHover: PlaneTransformHover;
  planeOrderUndo: number[][];
  planeOrderRedo: number[][];
};

export type SimulationState = {
  W: number;
  H: number;
  N: number;
  phases: Float32Array;
  omegaSeeds: Float32Array;
  omegas: Float32Array;
  neighbors: Array<Int32Array>;
  neighborBands: Array<Uint8Array>;
  swEdges: Array<Array<[number, number]>>;
  wall: Float32Array;
  pot: Float32Array;
  planeDepth: Uint8Array;
  planeMeta: Map<number, PlaneMetadata>;
  nextPhases: Float32Array;
  rng: () => number;
  ringOffsets: Array<Array<[number, number]>>;
  surfPhi: Float32Array;
  surfOmega: Float32Array;
  hypPhi: Float32Array;
  hypOmega: Float32Array;
  surfMask: Uint8Array;
  hypMask: Uint8Array;
  energy: Float32Array;
  nextEnergy: Float32Array;
  dag: SimulationDag;
  reseedKey: number;
  activePlaneIds: number[];
  activePlaneSet: Set<number>;
  noiseSeed: number;
};
