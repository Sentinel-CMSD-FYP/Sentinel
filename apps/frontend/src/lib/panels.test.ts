import { describe, expect, it } from 'vitest'
import type { Clip, LayerMeta } from '../types'
import { buildPanels, gridColumns } from './panels'

const meta = (layer: string, type: LayerMeta['type'] = 'grid'): LayerMeta => ({
  layer,
  type,
  label: layer.toUpperCase(),
})
const clip = {
  clip_id: 'c',
  fps: 25,
  frame_size: [100, 50],
  n_frames: 10,
  layers: [meta('flow'), meta('points', 'points'), meta('tracks', 'tracks')],
} as Clip

describe('buildPanels', () => {
  it('single mode is one combined panel with only the switched-on layers', () => {
    const p = buildPanels(clip, { flow: true, points: false, tracks: true }, 'single')
    expect(p).toHaveLength(1)
    expect(p[0].metas.map((m) => m.layer)).toEqual(['flow', 'tracks'])
  })

  it('split mode adds one panel per layer after the combined one', () => {
    const p = buildPanels(clip, { flow: true }, 'split')
    expect(p.map((x) => x.id)).toEqual(['all', 'flow', 'points', 'tracks'])
    expect(p[2].title).toBe('POINTS')
  })

  it('a layer panel shows its layer even when its box is off', () => {
    const p = buildPanels(clip, {}, 'split')
    expect(p[0].metas).toHaveLength(0)
    expect(p[1].metas.map((m) => m.layer)).toEqual(['flow'])
  })

  it('has no layer panels for a clip with no layers', () => {
    const empty = { ...clip, layers: [] } as Clip
    expect(buildPanels(empty, {}, 'split').map((x) => x.id)).toEqual(['all'])
  })
})

describe('gridColumns', () => {
  it('grows from one to three columns', () => {
    expect([1, 2, 4, 5, 6].map(gridColumns)).toEqual([1, 2, 2, 3, 3])
  })
})
