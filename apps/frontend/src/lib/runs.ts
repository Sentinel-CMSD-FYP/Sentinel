import type { Run } from '../types'

/** How often the upload panel asks the backend about a running job. */
export const POLL_MS = 3000

export const isActive = (run: Run) => run.state === 'queued' || run.state === 'running'

/** One line for the upload panel: what the run is doing, or how it ended. */
export function runSummary(run: Run): string {
  if (run.state === 'failed') return `Failed: ${run.error ?? 'unknown reason'}`
  if (run.state === 'done') {
    const added = `Done: ${run.layers.length} layer${run.layers.length === 1 ? '' : 's'} added`
    const failed = Object.keys(run.errors)
    return failed.length ? `${added}; failed: ${failed.join(', ')}` : added
  }
  const p = run.progress
  return p && p.total ? `${run.stage} (${p.done} / ${p.total} frames)` : run.stage
}
