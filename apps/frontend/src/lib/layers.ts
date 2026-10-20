import type {
  GridLayer,
  IndexedLayer,
  Layer,
  LayerMeta,
  TrackRow,
  TracksLayer,
} from '../types'
import { keysInWindow, sortedKeys } from './frames'

export function indexLayer(layer: Layer): IndexedLayer {
  return { layer, keys: sortedKeys(layer.frames) }
}

export type GridRender =
  | { kind: 'heatmap'; field: string }
  | { kind: 'arrows'; u: string; v: string }

/** How to draw a grid layer: the clip's style hint if it is usable, else a sensible guess. */
export function gridRender(meta: LayerMeta, layer: GridLayer): GridRender {
  const has = (f: string) => layer.fields.includes(f)
  const s = meta.style ?? {}
  const u = s.u ?? 'u'
  const v = s.v ?? 'v'
  if (s.render === 'heatmap' && s.field && has(s.field)) return { kind: 'heatmap', field: s.field }
  if (s.render === 'arrows' && has(u) && has(v)) return { kind: 'arrows', u, v }
  if (has('u') && has('v')) return { kind: 'arrows', u: 'u', v: 'v' }
  return { kind: 'heatmap', field: layer.fields[0] }
}

function* validCells(layer: GridLayer): Generator<[entry: Record<string, number[][]>, r: number, c: number]> {
  for (const entry of Object.values(layer.frames)) {
    for (let r = 0; r < layer.grid.rows; r++) {
      for (let c = 0; c < layer.grid.cols; c++) {
        if (entry.valid?.[r]?.[c] === 1) yield [entry, r, c]
      }
    }
  }
}

/** Largest value of `field` over every valid cell of every frame (the heatmap's top). */
export function fieldMax(layer: GridLayer, field: string): number {
  let max = 0
  for (const [entry, r, c] of validCells(layer)) {
    const x = entry[field]?.[r]?.[c]
    if (typeof x === 'number' && x > max) max = x
  }
  return max
}

/** Largest vector length over every valid cell of every frame (the longest arrow). */
export function maxSpeed(layer: GridLayer, uField: string, vField: string): number {
  let max = 0
  for (const [entry, r, c] of validCells(layer)) {
    const u = entry[uField]?.[r]?.[c]
    const v = entry[vField]?.[r]?.[c]
    if (typeof u === 'number' && typeof v === 'number') max = Math.max(max, Math.hypot(u, v))
  }
  return max
}

/** Recent positions per track id, oldest first, for drawing trails. */
export function buildTrails(
  layer: TracksLayer,
  keys: number[],
  frame: number,
  trailFrames: number,
): Map<number, [number, number][]> {
  const trails = new Map<number, [number, number][]>()
  for (const k of keysInWindow(keys, frame, trailFrames)) {
    for (const [id, x, y] of layer.frames[String(k)] as TrackRow[]) {
      const t = trails.get(id)
      if (t) t.push([x, y])
      else trails.set(id, [[x, y]])
    }
  }
  return trails
}
