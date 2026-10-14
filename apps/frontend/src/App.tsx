import { useCallback, useEffect, useState } from 'react'
import { getClip, listClips, videoSrc } from './api'
import { LayerPanel } from './components/LayerPanel'
import { VideoStage, type StageStatus } from './components/VideoStage'
import type { Clip, ClipSummary } from './types'
import { useClipLayers } from './useClipLayers'

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))

export default function App() {
  const [clips, setClips] = useState<ClipSummary[] | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [clip, setClip] = useState<Clip | null>(null)
  const [visible, setVisible] = useState<Record<string, boolean>>({})
  const [status, setStatus] = useState<StageStatus>({ frame: 0, live: {} })
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const ctl = new AbortController()
    listClips(ctl.signal)
      .then((list) => {
        setClips(list)
        setSelectedId((cur) => cur ?? list[0]?.clip_id ?? null)
      })
      .catch((e) => !ctl.signal.aborted && setError(message(e)))
    return () => ctl.abort()
  }, [])

  useEffect(() => {
    if (!selectedId) return
    const ctl = new AbortController()
    getClip(selectedId, ctl.signal)
      .then((c) => {
        setClip(c)
        setVisible(Object.fromEntries(c.layers.map((l) => [l.layer, l.default_visible ?? true])))
        setStatus({ frame: 0, live: {} })
        setError(null)
      })
      .catch((e) => !ctl.signal.aborted && setError(message(e)))
    return () => ctl.abort()
  }, [selectedId])

  const layerState = useClipLayers(clip)
  const toggle = useCallback((layer: string) => setVisible((v) => ({ ...v, [layer]: !v[layer] })), [])

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <header className="mb-5 flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold tracking-tight text-slate-50">Sentinel</h1>
        <span className="text-sm text-slate-400">Tier 1 viewer</span>
        {clips && clips.length > 0 && (
          <label className="ml-auto flex items-center gap-2 text-sm text-slate-300">
            Clip
            <select
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100"
              value={selectedId ?? ''}
              onChange={(e) => setSelectedId(e.target.value)}
            >
              {clips.map((c) => (
                <option key={c.clip_id} value={c.clip_id}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>
        )}
      </header>

      {error && (
        <p role="alert" className="mb-4 rounded-lg border border-rose-800 bg-rose-950/50 p-3 text-rose-200">
          {error}
        </p>
      )}

      {clips?.length === 0 && !error && (
        <p className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-slate-300">
          The backend has no clips yet. Generate a fake one with{' '}
          <code className="rounded bg-slate-800 px-1">python docs/layers/make_sample_clip.py data/samples/demo_fake</code>.
        </p>
      )}

      {clip && (
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <div>
            <VideoStage
              clip={clip}
              src={videoSrc(clip)}
              layers={layerState.layers}
              visible={visible}
              onStatus={setStatus}
            />
            <p className="mt-2 text-sm text-slate-400">
              {clip.name ?? clip.clip_id} · {clip.frame_size[0]}×{clip.frame_size[1]} · {clip.fps} fps · frame{' '}
              {status.frame} of {clip.n_frames} · {(status.frame / clip.fps).toFixed(2)} s
            </p>
          </div>
          <LayerPanel clip={clip} state={layerState} visible={visible} live={status.live} onToggle={toggle} />
        </div>
      )}
    </div>
  )
}
