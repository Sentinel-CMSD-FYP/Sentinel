import { describe, expect, it } from 'vitest'
import { frameForTime, keysInWindow, pickSample, sortedKeys } from './frames'

describe('frameForTime', () => {
  it('rounds time * fps to the nearest frame', () => {
    expect(frameForTime(0, 25)).toBe(0)
    expect(frameForTime(1, 25)).toBe(25)
    expect(frameForTime(0.519, 25)).toBe(13)
  })
  it('never goes negative', () => {
    expect(frameForTime(-0.2, 25)).toBe(0)
  })
})

describe('sortedKeys', () => {
  it('sorts numerically, not as strings', () => {
    expect(sortedKeys({ '100': 1, '25': 1, '0': 1, '5': 1 })).toEqual([0, 5, 25, 100])
  })
})

describe('pickSample', () => {
  const keys = [0, 25, 50, 150]
  const maxAge = 25

  it('returns the exact sample on its own frame', () => {
    expect(pickSample(keys, 25, maxAge)).toBe(25)
  })
  it('keeps showing a sample until it is maxAge frames old', () => {
    expect(pickSample(keys, 49, maxAge)).toBe(25)
    expect(pickSample(keys, 50, maxAge)).toBe(50)
  })
  it('returns null once a sample is too old: gaps stay gaps', () => {
    expect(pickSample(keys, 75, maxAge)).toBeNull() // 50 + 25
    expect(pickSample(keys, 100, maxAge)).toBeNull()
    expect(pickSample(keys, 149, maxAge)).toBeNull()
  })
  it('shows again as soon as data resumes after a gap', () => {
    expect(pickSample(keys, 150, maxAge)).toBe(150)
  })
  it('returns null before the first sample and for an empty layer', () => {
    expect(pickSample([10, 20], 5, maxAge)).toBeNull()
    expect(pickSample([], 5, maxAge)).toBeNull()
  })
  it('returns the last sample when the frame is beyond all of them but still young', () => {
    expect(pickSample(keys, 160, maxAge)).toBe(150)
    expect(pickSample(keys, 175, maxAge)).toBeNull()
  })
})

describe('keysInWindow', () => {
  it('keeps keys within the window up to and including the frame', () => {
    expect(keysInWindow([0, 5, 10, 15, 20, 25], 20, 11)).toEqual([10, 15, 20])
  })
  it('excludes future keys', () => {
    expect(keysInWindow([0, 5, 10], 5, 100)).toEqual([0, 5])
  })
})
