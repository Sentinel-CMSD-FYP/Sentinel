import { useCallback, useEffect, useState } from 'react'
import { getClip, listClips, videoSrc } from './api'
import { LayerPanel } from './components/LayerPanel'
import { LiveView } from './components/LiveView'
import { ModeSwitch } from './components/ModeSwitch'
import { Stage, type StageStatus } from './components/Stage'
import { UploadPanel } from './components/UploadPanel'
import { cameraLabel } from './lib/camera'
import type { ViewMode } from './lib/panels'
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
  const [mode, setMode] = useState<ViewMode>('single')
  const [page, setPage] = useState<'clips' | 'live'>('clips')
  // Bumped after an upload or a finished run, so the clip list and the open clip are read again.
  const [reload, setReload] = useState(0)

  useEffect(() => {
    const ctl = new AbortController()
    listClips(ctl.signal)
      .then((list) => {
        setClips(list)
        setSelectedId((cur) => cur ?? list[0]?.clip_id ?? null)
      })
      .catch((e) => !ctl.signal.aborted && setError(message(e)))
    return () => ctl.abort()
  }, [reload])

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
  }, [selectedId, reload])

  const layerState = useClipLayers(clip)
  const toggle = useCallback((layer: string) => setVisible((v) => ({ ...v, [layer]: !v[layer] })), [])
  const showUploaded = useCallback((clipId: string) => {
    setSelectedId(clipId)
    setReload((n) => n + 1)
  }, [])
  const refresh = useCallback(() => setReload((n) => n + 1), [])

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <header className="card mb-5 flex flex-wrap items-center gap-3 px-4 py-3">
        <div className="grid size-9 place-items-center rounded-xl bg-gradient-to-br from-sky-400 to-indigo-500 shadow-lg shadow-sky-500/20">
          <svg viewBox="0 0 24 24" className="size-5 text-white" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 3v3M12 18v3M3 12h3M18 12h3" />
          </svg>
        </div>
        <div className="leading-tight">
          <h1 className="text-lg font-semibold tracking-tight text-slate-50">Sentinel</h1>
          <span className="text-xs text-slate-400">Crowd early-warning · Tier 1 viewer</span>
        </div>
        <div role="group" aria-label="Page" className="ml-auto inline-flex rounded-xl border border-white/10 bg-slate-950/60 p-1">
          {(['clips', 'live'] as const).map((p) => (
            <button
              key={p}
              type="button"
              aria-pressed={page === p}
              onClick={() => setPage(p)}
              className={`rounded-lg px-3 py-1.5 text-sm transition ${
                page === p ? 'bg-sky-500 font-medium text-slate-950' : 'text-slate-300 hover:bg-white/10'
              }`}
            >
              {p === 'clips' ? 'Clips' : 'Live'}
            </button>
          ))}
        </div>
        {clips && clips.length > 0 && (
          <label className="flex items-center gap-2 text-sm text-slate-300">
            Clip
            <select
              className="rounded-lg border border-white/10 bg-slate-950/70 px-3 py-1.5 text-slate-100 outline-none focus:border-sky-400"
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

      {page === 'clips' && <UploadPanel onUploaded={showUploaded} onRunDone={refresh} />}

      {error && (
        <p role="alert" className="mb-4 rounded-lg border border-rose-800 bg-rose-950/50 p-3 text-rose-200">
          {error}
        </p>
      )}

      {clips?.length === 0 && !error && (
        <p className="card p-4 text-slate-300">
          The backend has no clips yet. Generate a fake one with{' '}
          <code className="rounded bg-slate-800 px-1">python docs/layers/make_sample_clip.py data/samples/demo_fake</code>.
        </p>
      )}

      {page === 'live' && <LiveView clipId={selectedId} clipName={clip?.name ?? selectedId} onStreamEnded={refresh} />}

      {page === 'clips' && clip && (
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <div>
            <ModeSwitch mode={mode} onChange={setMode} />
            <Stage
              clip={clip}
              src={videoSrc(clip)}
              layers={layerState.layers}
              visible={visible}
              mode={mode}
              onStatus={setStatus}
            />
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
              <span className="mr-1 text-sm font-medium text-slate-200">{clip.name ?? clip.clip_id}</span>
              {[
                `${clip.frame_size[0]}×${clip.frame_size[1]}`,
                `${clip.fps} fps`,
                `frame ${status.frame} / ${clip.n_frames}`,
                `${(status.frame / clip.fps).toFixed(2)} s`,
                ...(clip.camera ? [cameraLabel(clip.camera)] : []),
              ].map((t) => (
                <span key={t} className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 tabular-nums text-slate-300">
                  {t}
                </span>
              ))}
            </div>
          </div>
          <LayerPanel clip={clip} state={layerState} visible={visible} live={status.live} frame={status.frame} onToggle={toggle} />
        </div>
      )}
    </div>
  )
}
