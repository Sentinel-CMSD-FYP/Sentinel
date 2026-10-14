import type { GridLayer, PointRow, TrackRow } from '../types'
import { rampColour, rgba, trackColour } from './colour'

/** Canvas CSS pixels per original-frame pixel. */
export interface Scale {
  sx: number
  sy: number
}

type Entry = Record<string, number[][]>

/** Fill each valid cell with a colour for its value; zero stays transparent. */
export function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  layer: GridLayer,
  entry: Entry,
  field: string,
  top: number,
  k: Scale,
): void {
  if (top <= 0) return
  const cw = (layer.frame_size[0] / layer.grid.cols) * k.sx
  const ch = (layer.frame_size[1] / layer.grid.rows) * k.sy
  for (let r = 0; r < layer.grid.rows; r++) {
    for (let c = 0; c < layer.grid.cols; c++) {
      if (entry.valid?.[r]?.[c] !== 1) continue // gaps stay gaps: draw nothing
      const t = (entry[field]?.[r]?.[c] ?? 0) / top
      if (t <= 0) continue
      ctx.fillStyle = rgba(rampColour(t), 0.25 + 0.45 * Math.min(1, t))
      ctx.fillRect(c * cw, r * ch, cw, ch)
    }
  }
}

/** One arrow per valid cell; its length is proportional to the vector, the longest filling ~45% of a cell. */
export function drawArrows(
  ctx: CanvasRenderingContext2D,
  layer: GridLayer,
  entry: Entry,
  uField: string,
  vField: string,
  top: number,
  k: Scale,
): void {
  const cw = (layer.frame_size[0] / layer.grid.cols) * k.sx
  const ch = (layer.frame_size[1] / layer.grid.rows) * k.sy
  const reach = 0.45 * Math.min(cw, ch)
  ctx.lineCap = 'round'
  for (let r = 0; r < layer.grid.rows; r++) {
    for (let c = 0; c < layer.grid.cols; c++) {
      if (entry.valid?.[r]?.[c] !== 1) continue
      const u = entry[uField]?.[r]?.[c] ?? 0
      const v = entry[vField]?.[r]?.[c] ?? 0
      const x0 = (c + 0.5) * cw
      const y0 = (r + 0.5) * ch
      const speed = Math.hypot(u, v)
      if (top <= 0 || speed < top * 0.02) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)' // valid but (nearly) still
        ctx.beginPath()
        ctx.arc(x0, y0, 2, 0, Math.PI * 2)
        ctx.fill()
        continue
      }
      const len = (speed / top) * reach
      const ang = Math.atan2(v, u)
      const x1 = x0 + Math.cos(ang) * len
      const y1 = y0 + Math.sin(ang) * len
      const head = Math.min(9, Math.max(4, len * 0.4))
      for (const [width, colour] of [
        [4.5, 'rgba(0, 0, 0, 0.55)'],
        [2, '#ffffff'],
      ] as const) {
        ctx.strokeStyle = colour
        ctx.lineWidth = width
        ctx.beginPath()
        ctx.moveTo(x0, y0)
        ctx.lineTo(x1, y1)
        ctx.moveTo(x1 - Math.cos(ang - 0.45) * head, y1 - Math.sin(ang - 0.45) * head)
        ctx.lineTo(x1, y1)
        ctx.lineTo(x1 - Math.cos(ang + 0.45) * head, y1 - Math.sin(ang + 0.45) * head)
        ctx.stroke()
      }
    }
  }
}

export function drawPoints(ctx: CanvasRenderingContext2D, rows: PointRow[], k: Scale): void {
  ctx.lineWidth = 1.5
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)'
  for (const [x, y, score] of rows) {
    ctx.fillStyle = `rgba(251, 191, 36, ${0.45 + 0.55 * score})`
    ctx.beginPath()
    ctx.arc(x * k.sx, y * k.sy, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
  }
}

/** Faded trail per track id, then a dot at each track's latest position. */
export function drawTracks(
  ctx: CanvasRenderingContext2D,
  trails: Map<number, [number, number][]>,
  current: TrackRow[],
  k: Scale,
): void {
  ctx.lineWidth = 2
  ctx.lineCap = 'round'
  for (const [id, pts] of trails) {
    const colour = trackColour(id)
    for (let i = 1; i < pts.length; i++) {
      ctx.globalAlpha = 0.25 + 0.75 * (i / (pts.length - 1))
      ctx.strokeStyle = colour
      ctx.beginPath()
      ctx.moveTo(pts[i - 1][0] * k.sx, pts[i - 1][1] * k.sy)
      ctx.lineTo(pts[i][0] * k.sx, pts[i][1] * k.sy)
      ctx.stroke()
    }
  }
  ctx.globalAlpha = 1
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)'
  ctx.lineWidth = 1.5
  for (const [id, x, y] of current) {
    ctx.fillStyle = trackColour(id)
    ctx.beginPath()
    ctx.arc(x * k.sx, y * k.sy, 4.5, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
  }
}
