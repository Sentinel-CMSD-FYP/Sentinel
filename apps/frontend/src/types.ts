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

export interface Clip {
  clip_id: string
  name?: string
  fps: number
  frame_size: [number, number]
  n_frames: number
  video_url?: string
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
