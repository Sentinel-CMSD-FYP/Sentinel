import { describe, expect, it } from 'vitest'
import type { GridLayer } from '../types'
import { cellAt, fmt, heading, readCell, summarise, unitLabel } from './readout'

const layer: GridLayer = {
  schema_version: 1,
  clip_id: 'x',
  layer: 'flow',
  type: 'grid',
  frame_size: [100, 50],
  max_age_frames: 10,
  grid: { rows: 2, cols: 4 },
  units: 'px_per_frame',
  fields: ['u', 'v'],
  frames: {},
}
const arrows = { kind: 'arrows', u: 'u', v: 'v' } as const
const entry = {
  valid: [
    [1, 0, 1, 1],
    [1, 1, 1, 1],
  ],
  u: [
    [3, 9, 0, 0],
    [0, 0, 0, 0],
  ],
  v: [
    [4, 9, 0, 0],
    [0, 0, 0, 0],
  ],
}

describe('heading', () => {
  it('names the eight directions, with +v pointing down', () => {
    expect(heading(1, 0)).toBe('→')
    expect(heading(0, 1)).toBe('↓')
    expect(heading(-1, 0)).toBe('←')
    expect(heading(0, -1)).toBe('↑')
    expect(heading(1, -1)).toBe('↗')
  })
})

describe('cellAt', () => {
  it('maps frame pixels to a cell and rejects points outside', () => {
    expect(cellAt(layer, 0, 0)).toEqual({ row: 0, col: 0 })
    expect(cellAt(layer, 99, 49)).toEqual({ row: 1, col: 3 })
    expect(cellAt(layer, 100, 10)).toBeNull()
    expect(cellAt(layer, -1, 10)).toBeNull()
  })
})

describe('readCell', () => {
  it('returns speed and heading for a valid cell', () => {
    const r = readCell(entry, arrows, 0, 0)
    expect(r?.value).toBe(5)
    expect(r?.heading).toBe('↘')
  })
  it('returns null for a gap', () => {
    expect(readCell(entry, arrows, 0, 1)).toBeNull()
  })
  it('has no heading for a still cell', () => {
    expect(readCell(entry, arrows, 1, 0)?.heading).toBeUndefined()
  })
})

describe('summarise', () => {
  it('averages over valid cells only', () => {
    const s = summarise(layer, entry, arrows)
    expect(s.cells).toBe(7)
    expect(s.total).toBe(8)
    expect(s.peak).toBe(5)
    expect(s.mean).toBeCloseTo(5 / 7)
  })
})

describe('fmt', () => {
  it('trims zeros', () => {
    expect(fmt(2)).toBe('2')
    expect(fmt(1.5)).toBe('1.5')
    expect(fmt(0)).toBe('0')
    expect(fmt(250.4)).toBe('250')
  })
})

describe('unitLabel', () => {
  it('writes per as a slash', () => {
    expect(unitLabel('m_per_s')).toBe('m/s')
    expect(unitLabel('px_per_frame')).toBe('px/frame')
    expect(unitLabel('people_per_m2')).toBe('people/m2')
  })
})
