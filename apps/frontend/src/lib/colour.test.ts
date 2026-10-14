import { describe, expect, it } from 'vitest'
import { rampColour, trackColour } from './colour'

describe('rampColour', () => {
  it('starts dark and ends bright', () => {
    expect(rampColour(0)).toEqual([68, 1, 84])
    expect(rampColour(1)).toEqual([253, 231, 37])
  })
  it('clamps values outside 0..1 and ignores NaN', () => {
    expect(rampColour(-3)).toEqual(rampColour(0))
    expect(rampColour(7)).toEqual(rampColour(1))
    expect(rampColour(Number.NaN)).toEqual(rampColour(0))
  })
  it('blends between stops', () => {
    // halfway between the first two stops
    expect(rampColour(0.125)).toEqual([64, 42, 112])
  })
})

describe('trackColour', () => {
  it('is stable per id and differs between neighbouring ids', () => {
    expect(trackColour(3)).toBe(trackColour(3))
    expect(trackColour(3)).not.toBe(trackColour(4))
  })
})
