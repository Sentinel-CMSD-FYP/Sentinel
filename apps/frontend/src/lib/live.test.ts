import { describe, expect, it } from 'vitest'
import type { GridLayer, LiveSample } from '../types'
import { addSamples, emptyLive, KEEP_FRAMES, latencyStats, LIVE_BUDGET_S, liveKey } from './live'

const header = {
  schema_version: 1, clip_id: 's', layer: 'density', type: 'grid', frame_size: [10, 10],
  grid: { rows: 1, cols: 1 }, units: 'people_per_cell', fields: ['density'], max_age_frames: 25,
} as const
const meta = { layer: 'density', type: 'grid', label: 'Density', default_visible: true } as const

function sample(seq: number, frame: number, first = false): LiveSample {
  return {
    seq, layer: 'density', frame, t_capture: 100 + frame / 25, t_ready: 101 + frame / 25,
    data: { valid: [[1]], density: [[frame]] },
    ...(first ? { header: header as unknown as LiveSample['header'], meta } : {}),
  }
}

describe('addSamples', () => {
  it('builds the layer from its header, then grows it with a new object each time', () => {
    const live = emptyLive()
    addSamples(live, [sample(0, 0, true)])
    const before = live.layers.density.layer
    addSamples(live, [sample(1, 25)])
    const after = live.layers.density
    expect(live.metas).toEqual([meta])
    expect(after.keys).toEqual([0, 25])
    expect(after.layer).not.toBe(before) // so the cached heat map scale is computed again
    expect((after.layer as GridLayer).frames['25'].density).toEqual([[25]])
    expect(live.captured.density.get(25)).toBe(101)
  })

  it(`keeps only the last ${KEEP_FRAMES} frames of samples`, () => {
    const live = emptyLive()
    addSamples(live, [sample(0, 0, true), sample(1, 100), sample(2, KEEP_FRAMES + 50)])
    expect(live.layers.density.keys).toEqual([100, KEEP_FRAMES + 50])
    expect(Object.keys(live.layers.density.layer.frames)).toEqual(['100', String(KEEP_FRAMES + 50)])
    expect([...live.captured.density.keys()]).toEqual([100, KEEP_FRAMES + 50])
  })

  it('ignores a sample whose layer header it never saw', () => {
    const live = emptyLive()
    addSamples(live, [sample(5, 50)])
    expect(live.layers).toEqual({})
  })
})

describe('liveKey', () => {
  it('shows the newest sample until max_age_frames (as seconds) plus the budget after its capture', () => {
    const live = emptyLive()
    addSamples(live, [sample(0, 0, true), sample(1, 25)])
    const { density } = live.layers
    const captured = live.captured.density // frame 25 captured at 101 s
    const limit = 101 + 25 / 25 + LIVE_BUDGET_S // max age 25 frames at 25 fps = 1 s
    expect(liveKey(density, captured, 101.2, 25)).toBe(25)
    expect(liveKey(density, captured, limit - 0.01, 25)).toBe(25)
    expect(liveKey(density, captured, limit, 25)).toBeNull()
    expect(liveKey(density, new Map(), 101.2, 25)).toBeNull() // no capture time: not shown
    expect(liveKey({ ...density, keys: [] }, captured, 101.2, 25)).toBeNull()
  })
})

describe('latencyStats', () => {
  it('summarises capture to drawn per layer', () => {
    const rec = (layer: string, d: number) => ({ stream: 's', layer, frame: 0, tCapture: 10, tDrawn: 10 + d })
    const stats = latencyStats([rec('density', 3), rec('density', 1), rec('density', 2), rec('picture', 0.1)])
    expect(stats.density).toEqual({ n: 3, median: 2, p90: 3, max: 3, last: 2 })
    expect(stats.picture.n).toBe(1)
  })
})
