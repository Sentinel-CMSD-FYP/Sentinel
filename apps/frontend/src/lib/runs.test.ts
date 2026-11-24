import { describe, expect, it } from 'vitest'
import type { Run } from '../types'
import { isActive, runSummary } from './runs'

const base: Run = {
  run_id: 'a_1', clip_id: 'a', runner: 'kaggle', state: 'running', stage: 'Kaggle: waiting for a GPU',
  created: '', updated: '', finished: null, progress: null, layers: [], errors: {}, error: null,
}

describe('runs', () => {
  it('is active while queued or running', () => {
    expect(isActive({ ...base, state: 'queued' })).toBe(true)
    expect(isActive(base)).toBe(true)
    expect(isActive({ ...base, state: 'done' })).toBe(false)
    expect(isActive({ ...base, state: 'failed' })).toBe(false)
  })
  it('shows the stage, with frames when the runner reports them', () => {
    expect(runSummary(base)).toBe('Kaggle: waiting for a GPU')
    expect(runSummary({ ...base, stage: 'running the models', progress: { done: 100, total: 300 } }))
      .toBe('running the models (100 / 300 frames)')
  })
  it('names the failed modules of a finished run', () => {
    expect(runSummary({ ...base, state: 'done', layers: ['density'] })).toBe('Done: 1 layer added')
    expect(runSummary({ ...base, state: 'done', layers: ['density', 'flow'], errors: { points: 'x' } }))
      .toBe('Done: 2 layers added; failed: points')
  })
  it('says why a run failed', () => {
    expect(runSummary({ ...base, state: 'failed', error: 'set KAGGLE_USERNAME' })).toBe('Failed: set KAGGLE_USERNAME')
  })
})
