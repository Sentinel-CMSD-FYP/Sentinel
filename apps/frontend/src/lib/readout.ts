import type { GridLayer } from '../types'
import type { GridRender } from './layers'

type Entry = Record<string, number[][]>

export interface Reading {
  /** Speed for arrow layers, the field value for heatmaps. */
  value: number
  /** Direction of travel as an arrow glyph (arrow layers only). */
  heading?: string
}

const GLYPHS = ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗']

/** Eight-way arrow for a vector; image y points down, so +v is "down". */
export function heading(u: number, v: number): string {
  const sector = Math.round(Math.atan2(v, u) / (Math.PI / 4))
  return GLYPHS[((sector % 8) + 8) % 8]
}

/** The grid cell under a point given in original-frame pixels, or null if outside the frame. */
export function cellAt(layer: GridLayer, x: number, y: number): { row: number; col: number } | null {
  const [w, h] = layer.frame_size
  if (x < 0 || y < 0 || x >= w || y >= h) return null
  return {
    row: Math.floor((y / h) * layer.grid.rows),
    col: Math.floor((x / w) * layer.grid.cols),
  }
}

/** What one cell holds right now; null for a gap, because gaps stay gaps. */
export function readCell(entry: Entry, mode: GridRender, row: number, col: number): Reading | null {
  if (entry.valid?.[row]?.[col] !== 1) return null
  if (mode.kind === 'heatmap') return { value: entry[mode.field]?.[row]?.[col] ?? 0 }
  const u = entry[mode.u]?.[row]?.[col] ?? 0
  const v = entry[mode.v]?.[row]?.[col] ?? 0
  const value = Math.hypot(u, v)
  return { value, heading: value > 0 ? heading(u, v) : undefined }
}

export interface Summary {
  mean: number
  peak: number
  cells: number
  total: number
}

/** Mean and peak over the valid cells of one sample. */
export function summarise(layer: GridLayer, entry: Entry, mode: GridRender): Summary {
  let sum = 0
  let peak = 0
  let cells = 0
  for (let r = 0; r < layer.grid.rows; r++) {
    for (let c = 0; c < layer.grid.cols; c++) {
      const reading = readCell(entry, mode, r, c)
      if (!reading) continue
      cells++
      sum += reading.value
      peak = Math.max(peak, reading.value)
    }
  }
  return { mean: cells ? sum / cells : 0, peak, cells, total: layer.grid.rows * layer.grid.cols }
}

/** Short number: whole above 100, otherwise two decimals without trailing zeros. */
export function fmt(n: number): string {
  return n >= 100 ? n.toFixed(0) : n.toFixed(2).replace(/\.?0+$/, '') || '0'
}

/** Unit as people write it: `m_per_s` becomes `m/s`. */
export function unitLabel(units: string): string {
  return units.replace('_per_', '/')
}
