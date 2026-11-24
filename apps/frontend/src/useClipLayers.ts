import { useEffect, useState } from 'react'
import { getLayer } from './api'
import { indexLayer } from './lib/layers'
import type { Clip, IndexedLayer } from './types'

export interface LayerState {
  layers: Record<string, IndexedLayer>
  /** Layers the clip lists but the backend could not serve, with the reason. */
  failed: Record<string, string>
  loading: boolean
}

const EMPTY: LayerState = { layers: {}, failed: {}, loading: false }
const LOADING: LayerState = { layers: {}, failed: {}, loading: true }

/** Downloads every layer of a clip once; the stage then looks samples up locally. */
export function useClipLayers(clip: Clip | null): LayerState {
  // Results are tagged with the clip they belong to, so switching clips reads as
  // "loading" until the new results arrive, with no state set inside the effect.
  const [done, setDone] = useState<{ clipId: string; state: LayerState } | null>(null)

  useEffect(() => {
    if (!clip) return
    const ctl = new AbortController()
    Promise.allSettled(clip.layers.map((m) => getLayer(clip.clip_id, m.layer, ctl.signal))).then(
      (results) => {
        if (ctl.signal.aborted) return
        const layers: Record<string, IndexedLayer> = {}
        const failed: Record<string, string> = {}
        results.forEach((r, i) => {
          const name = clip.layers[i].layer
          if (r.status === 'fulfilled') layers[name] = indexLayer(r.value)
          else failed[name] = r.reason instanceof Error ? r.reason.message : 'failed to load'
        })
        setDone({ clipId: clip.clip_id, state: { layers, failed, loading: false } })
      },
    )
    return () => ctl.abort()
  }, [clip])

  if (!clip) return EMPTY
  return done?.clipId === clip.clip_id ? done.state : LOADING
}
