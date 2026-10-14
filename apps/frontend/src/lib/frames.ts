/** Video time to frame number, as the layer contract defines it: round(time * fps). */
export function frameForTime(timeSeconds: number, fps: number): number {
  return Math.max(0, Math.round(timeSeconds * fps))
}

export function sortedKeys(frames: Record<string, unknown>): number[] {
  return Object.keys(frames)
    .map(Number)
    .sort((a, b) => a - b)
}

/**
 * The sample to show at `frame`: the latest one at or before it, but only while it is
 * younger than `maxAge` frames. Past that, there is nothing to show. Gaps stay gaps:
 * an old value is never carried forward and nothing is interpolated.
 */
export function pickSample(keys: number[], frame: number, maxAge: number): number | null {
  let lo = 0
  let hi = keys.length - 1
  let found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (keys[mid] <= frame) {
      found = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  if (found < 0) return null
  return frame - keys[found] < maxAge ? keys[found] : null
}

/** Sample frames in (frame - window, frame], oldest first. */
export function keysInWindow(keys: number[], frame: number, window: number): number[] {
  return keys.filter((k) => k <= frame && frame - k < window)
}
