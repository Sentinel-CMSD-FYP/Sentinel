import type { IndexedLayer, Layer, LayerMeta, LiveSample, Stream } from '../types'

/** How often a live view asks for new samples. */
export const SAMPLES_POLL_MS = 250
/** How often the live view reads the stream list. */
export const STREAMS_POLL_MS = 2000
/** Lateness a live overlay may have on top of its max_age_frames: the Tier 1 latency target. */
export const LIVE_BUDGET_S = 3
/** Samples older than this many frames behind a layer's newest are dropped (tracks need 30). */
export const KEEP_FRAMES = 250

export const streamActive = (s: Stream) => s.state === 'queued' || s.state === 'running'

/** A stream's layers as they grow: the samples kept so far, and when each one's frame was captured. */
export interface LiveLayers {
  metas: LayerMeta[]
  layers: Record<string, IndexedLayer>
  captured: Record<string, Map<number, number>>
}

export const emptyLive = (): LiveLayers => ({ metas: [], layers: {}, captured: {} })

/**
 * Add samples (in arrival order) to `live`, in place. A changed layer gets a new layer object, so
 * the heat map and arrow scales (cached per layer object) follow the values seen so far.
 */
export function addSamples(live: LiveLayers, samples: LiveSample[]): void {
  for (const s of samples) {
    if (s.header && s.meta) {
      if (!live.metas.some((m) => m.layer === s.layer)) live.metas = [...live.metas, s.meta]
      live.layers[s.layer] = { layer: { ...s.header, frames: {} } as Layer, keys: [] }
      live.captured[s.layer] = new Map()
    }
    const cur = live.layers[s.layer]
    if (!cur) continue // its header was never seen (cannot happen when reading from offset 0)
    const keys = [...cur.keys, s.frame].filter((k) => k > s.frame - KEEP_FRAMES)
    const old = cur.layer.frames as Record<string, unknown>
    const frames: Record<string, unknown> = {}
    for (const k of keys) frames[k] = k === s.frame ? s.data : old[k]
    live.layers[s.layer] = { layer: { ...cur.layer, frames } as Layer, keys }
    const captured = live.captured[s.layer]
    if (s.t_capture !== null) captured.set(s.frame, s.t_capture)
    for (const k of captured.keys()) if (k <= s.frame - KEEP_FRAMES) captured.delete(k)
  }
}

/**
 * The sample a live panel shows: the layer's newest, while its frame was captured less than
 * `max_age_frames` (as seconds) plus the latency budget ago, on the wall clock. Past that the layer
 * is a gap, as in a stored clip. Wall clock, not frames: a reader that falls behind its source
 * advances frames slowly, and a sample many seconds old must still go.
 */
export function liveKey(indexed: IndexedLayer, captured: Map<number, number> | undefined, now: number, fps: number): number | null {
  const key = indexed.keys.at(-1)
  const t = key === undefined ? undefined : captured?.get(key)
  if (key === undefined || t === undefined) return null
  return now - t < indexed.layer.max_age_frames / fps + LIVE_BUDGET_S ? key : null
}

/** One sample (or picture, layer "picture") from capture to drawn on screen. */
export interface LatencyRecord {
  stream: string
  layer: string
  frame: number
  tCapture: number
  tDrawn: number
}

export interface LatencyStats {
  n: number
  median: number
  p90: number
  max: number
  last: number
}

function quantile(sorted: number[], q: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]
}

/** Seconds from capture to drawn, per layer. */
export function latencyStats(records: LatencyRecord[]): Record<string, LatencyStats> {
  const by: Record<string, number[]> = {}
  for (const r of records) (by[r.layer] ??= []).push(r.tDrawn - r.tCapture)
  const out: Record<string, LatencyStats> = {}
  for (const [layer, xs] of Object.entries(by)) {
    const sorted = [...xs].sort((a, b) => a - b)
    out[layer] = { n: xs.length, median: quantile(sorted, 0.5), p90: quantile(sorted, 0.9), max: sorted.at(-1)!, last: xs.at(-1)! }
  }
  return out
}
