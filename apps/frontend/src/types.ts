// Mirrors the layer contract in docs/layers/README.md.

export type LayerType = 'grid' | 'points' | 'tracks' | 'polygons'

export interface LayerMeta {
  layer: string
  type: LayerType
  label: string
  units?: string
  default_visible?: boolean
  style?: { render?: string; field?: string; u?: string; v?: string }
}

/** Straight down camera facts (clip.json "camera"); with them flow is in m/s. */
export interface Camera {
  model?: string
  altitude_m: number
  hfov_deg: number
  dfov_deg?: number
  sensor_aspect?: string
  altitude_source: 'assumed' | 'stated' | 'telemetry'
}

export interface Clip {
  clip_id: string
  name?: string
  fps: number
  frame_size: [number, number]
  n_frames: number
  video_url?: string
  camera?: Camera
  layers: LayerMeta[]
}

export interface ClipSummary {
  clip_id: string
  name: string
  fps: number
  frame_size: [number, number]
  n_frames: number
  layers: string[]
}

export type Matrix = number[][]
export type PointRow = [x: number, y: number, score: number]
export type TrackRow = [id: number, x: number, y: number, score: number]
/** One outlined object; `box` is [x0, y0, x1, y1] and `polygon` closes on its first point. */
export interface PolygonObject {
  cls: string
  score: number
  box: [number, number, number, number]
  polygon: [number, number][]
}

interface LayerBase {
  schema_version: number
  clip_id: string
  layer: string
  frame_size: [number, number]
  max_age_frames: number
}

export interface GridLayer extends LayerBase {
  type: 'grid'
  grid: { rows: number; cols: number }
  units: string
  fields: string[]
  /** Per frame: a `valid` mask plus one matrix per field. */
  frames: Record<string, Record<string, Matrix>>
}

export interface PointsLayer extends LayerBase {
  type: 'points'
  frames: Record<string, PointRow[]>
}

export interface TracksLayer extends LayerBase {
  type: 'tracks'
  altitude_m?: number
  frames: Record<string, TrackRow[]>
}

export interface PolygonsLayer extends LayerBase {
  type: 'polygons'
  /** Every class the layer can hold, in model order; colours follow this order. */
  classes: string[]
  frames: Record<string, PolygonObject[]>
}

export type Layer = GridLayer | PointsLayer | TracksLayer | PolygonsLayer

/** A layer plus its sorted frame numbers, so lookups can binary-search. */
export interface IndexedLayer {
  layer: Layer
  keys: number[]
}

/** A model run started from the browser (GET /clips/<id>/runs/<run_id>). */
export interface Run {
  run_id: string
  clip_id: string
  runner: string
  state: 'queued' | 'running' | 'done' | 'failed'
  /** What the worker is doing now, e.g. "Kaggle: waiting for a GPU". */
  stage: string
  created: string
  updated: string
  finished: string | null
  /** Frames decoded so far; only the local runner reports it. */
  progress: { done: number; total: number } | null
  layers: string[]
  /** Modules that failed, with the reason; the other layers were still added. */
  errors: Record<string, string>
  /** Why the whole run failed. */
  error: string | null
}

/** A live stream (GET /streams/<id>): a source read at real time with the models running on it. */
export interface Stream {
  stream_id: string
  name: string
  source_clip: string | null
  modules: string[]
  state: 'queued' | 'running' | 'done' | 'failed'
  stage: string
  fps: number | null
  frame_size: [number, number] | null
  frames_read: number
  samples: number
  /** Modules that failed, with the reason. */
  errors: Record<string, string>
  error: string | null
  /** The clip the stream was saved as, once it has ended. */
  clip_id: string | null
}

/** One layer sample of a stream, as soon as its model made it (GET /streams/<id>/samples). */
export interface LiveSample {
  seq: number
  layer: string
  frame: number
  /** Unix seconds, on the backend's clock. */
  t_capture: number | null
  t_ready: number
  /** That frame's value in the stored layer format. */
  data: unknown
  /** On a layer's first sample only: the layer without its frames, and its clip.json entry. */
  header?: Omit<Layer, 'frames'>
  meta?: LayerMeta
}
