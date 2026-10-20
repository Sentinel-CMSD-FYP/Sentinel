import { describe, expect, it } from 'vitest'
import type { IndexedLayer, Layer, LayerMeta } from '../types'
import { drawLayer } from './overlay'

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
})
