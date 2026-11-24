import { describe, expect, it } from 'vitest'
import type { GridLayer, LayerMeta, PolygonsLayer, TracksLayer } from '../types'
import { buildTrails, countByClass, fieldMax, gridRender, indexLayer, maxSpeed } from './layers'

function grid(fields: string[], frames: GridLayer['frames']): GridLayer {
  return {
    schema_version: 1, clip_id: 'c', layer: 'g', type: 'grid', frame_size: [100, 100],
    max_age_frames: 25, grid: { rows: 1, cols: 2 }, units: 'x', fields, frames,
  }
}

const meta = (style?: LayerMeta['style']): LayerMeta => ({ layer: 'g', type: 'grid', label: 'G', style })

describe('indexLayer', () => {
  it('sorts frame keys numerically', () => {
    const l = grid(['a'], { '100': { valid: [[1, 1]] }, '25': { valid: [[1, 1]] } })
    expect(indexLayer(l).keys).toEqual([25, 100])
  })
})

describe('gridRender', () => {
  const flow = grid(['u', 'v', 'speed'], {})
  const dens = grid(['density'], {})

  it('follows an explicit style hint', () => {
    expect(gridRender(meta({ render: 'heatmap', field: 'speed' }), flow)).toEqual({
      kind: 'heatmap', field: 'speed',
    })
    expect(gridRender(meta({ render: 'arrows' }), flow)).toEqual({ kind: 'arrows', u: 'u', v: 'v' })
  })
  it('guesses arrows when a layer has u and v, otherwise a heatmap of the first field', () => {
    expect(gridRender(meta(), flow).kind).toBe('arrows')
    expect(gridRender(meta(), dens)).toEqual({ kind: 'heatmap', field: 'density' })
  })
  it('ignores a hint that names a missing field', () => {
    expect(gridRender(meta({ render: 'heatmap', field: 'nope' }), dens)).toEqual({
      kind: 'heatmap', field: 'density',
    })
    expect(gridRender(meta({ render: 'arrows' }), dens)).toEqual({ kind: 'heatmap', field: 'density' })
  })
})

describe('fieldMax and maxSpeed', () => {
  const l = grid(['d', 'u', 'v'], {
    '0': { valid: [[1, 0]], d: [[3, 99]], u: [[3, 99]], v: [[4, 99]] },
    '25': { valid: [[1, 1]], d: [[2, 7]], u: [[0, 6]], v: [[1, 8]] },
  })
  it('takes the maximum over valid cells only', () => {
    expect(fieldMax(l, 'd')).toBe(7) // the 99 sits in an invalid cell
  })
  it('measures the longest vector over valid cells only', () => {
    expect(maxSpeed(l, 'u', 'v')).toBe(10) // (6, 8), not the invalid (99, 99)
  })
  it('is zero when nothing is valid', () => {
    expect(fieldMax(grid(['d'], { '0': { valid: [[0, 0]], d: [[5, 5]] } }), 'd')).toBe(0)
  })
})

describe('buildTrails', () => {
  const layer: TracksLayer = {
    schema_version: 1, clip_id: 'c', layer: 't', type: 'tracks', frame_size: [100, 100],
    max_age_frames: 5,
    frames: {
      '0': [[1, 10, 10, 1]],
      '5': [[1, 12, 10, 1], [2, 50, 50, 1]],
      '10': [[1, 14, 10, 1], [2, 52, 50, 1]],
      '15': [[1, 99, 99, 1]],
    },
  }
  const keys = [0, 5, 10, 15]

  it('collects each id\'s recent positions oldest first, up to the current frame', () => {
    const t = buildTrails(layer, keys, 10, 11)
    expect(t.get(1)).toEqual([[10, 10], [12, 10], [14, 10]])
    expect(t.get(2)).toEqual([[50, 50], [52, 50]])
  })
  it('drops positions older than the trail window and ignores the future', () => {
    const t = buildTrails(layer, keys, 10, 6)
    expect(t.get(1)).toEqual([[12, 10], [14, 10]])
    expect(t.has(99)).toBe(false)
    expect([...t.values()].flat().some(([x]) => x === 99)).toBe(false)
  })
})

describe('countByClass', () => {
  const obj = (cls: string) => ({ cls, score: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], polygon: [] })
  const layer: PolygonsLayer = {
    schema_version: 1, clip_id: 'c', layer: 'infra', type: 'polygons', frame_size: [100, 100],
    max_age_frames: 125, classes: ['wall', 'obstruction'], frames: {},
  }

  it('counts per class in class order, keeping classes with none', () => {
    expect(countByClass(layer, [obj('obstruction'), obj('obstruction')])).toEqual([
      ['wall', 0],
      ['obstruction', 2],
    ])
  })
  it('adds a class the layer does not list at the end', () => {
    expect(countByClass(layer, [obj('tree')])).toEqual([['wall', 0], ['obstruction', 0], ['tree', 1]])
  })
})
