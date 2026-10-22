import { describe, expect, it } from 'vitest'
import type { IndexedLayer, Layer, LayerMeta } from '../types'
import { drawLayer, gridTop } from './overlay'

/** Records the canvas calls drawLayer makes; enough of the 2D context for the polygon renderer. */
function fakeCtx() {
  const calls: string[] = []
  const ctx = new Proxy({} as Record<string, unknown>, {
    get: (target, prop: string) => (prop in target ? target[prop] : (...args: unknown[]) => calls.push(`${prop}(${args.join(',')})`)),
    set: (target, prop: string, value) => ((target[prop] = value), true),
  })
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls }
}

const k = { sx: 0.5, sy: 0.5 }
const indexed = (layer: Layer): IndexedLayer => ({ layer, keys: [0] })
const meta = (type: string): LayerMeta => ({ layer: 'l', type, label: 'L' }) as LayerMeta

describe('drawLayer', () => {
  it('outlines polygons in canvas pixels', () => {
    const { ctx, calls } = fakeCtx()
    const layer = {
      schema_version: 1, clip_id: 'c', layer: 'infra', type: 'polygons', frame_size: [100, 100],
      max_age_frames: 125, classes: ['wall'],
      frames: { '0': [{ cls: 'wall', score: 0.9, box: [10, 10, 30, 30], polygon: [[10, 10], [30, 10], [30, 30]] }] },
    } as Layer
    drawLayer(ctx, meta('polygons'), indexed(layer), 0, 0, k)
    expect(calls).toEqual(['beginPath()', 'moveTo(5,5)', 'lineTo(15,5)', 'lineTo(15,15)', 'closePath()', 'fill()', 'stroke()'])
  })

  it('draws nothing, and does not throw, for a type this viewer does not know', () => {
    const { ctx, calls } = fakeCtx()
    const layer = { type: 'heights', frames: { '0': { anything: 1 } } } as unknown as Layer
    expect(() => drawLayer(ctx, meta('heights'), indexed(layer), 0, 0, k)).not.toThrow()
    expect(calls).toEqual([])
  })

  it('draws a heat map at half opacity when faint', () => {
    const layer = {
      schema_version: 1, clip_id: 'c', layer: 'density', type: 'grid', frame_size: [10, 10],
      max_age_frames: 1, grid: { rows: 1, cols: 1 }, units: 'people_per_cell', fields: ['density'],
      frames: { '0': { valid: [[1]], density: [[2]] } },
    } as Layer
    const heat = { layer: 'density', type: 'grid', label: 'D', style: { render: 'heatmap', field: 'density' } } as LayerMeta
    const alpha = (faint: boolean) => {
      const { ctx } = fakeCtx()
      drawLayer(ctx, heat, indexed(layer), 0, 0, k, faint)
      return Number(String(ctx.fillStyle).split(',').at(-1)?.replace(')', ''))
    }
    expect(alpha(true)).toBeCloseTo(alpha(false) / 2)
  })
})

describe('gridTop', () => {
  const flow = (u: number) =>
    ({
      schema_version: 1, clip_id: 'c', layer: 'flow', type: 'grid', frame_size: [10, 10],
      max_age_frames: 1, grid: { rows: 1, cols: 1 }, units: 'px_per_frame', fields: ['u', 'v'],
      frames: { '0': { valid: [[1]], u: [[u]], v: [[0]] } },
    }) as Layer & { type: 'grid' }
  const arrows = { layer: 'flow', type: 'grid', label: 'F', style: { render: 'arrows', u: 'u', v: 'v' } } as LayerMeta

  it('scales arrows to at least 0.5 px per frame, so noise stays short', () => {
    expect(gridTop(flow(0.1), arrows)).toBe(0.5)
    expect(gridTop(flow(2), arrows)).toBe(2)
  })
})
