import { useCallback, useEffect, useRef, useState } from 'react'
import { listStreams, startStream, stopStream } from '../api'
import { STREAMS_POLL_MS, streamActive } from '../lib/live'
import type { ViewMode } from '../lib/panels'
import type { LayerMeta, Stream } from '../types'
import { LiveStage } from './LiveStage'
import { ModeSwitch } from './ModeSwitch'

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))

interface Props {
  /** The clip picked in the header; "Go live" replays it as a stream. */
  clipId: string | null
  clipName: string | null
  /** A watched stream ended, so it is now a clip: the clip list should be read again. */
  onStreamEnded: () => void
}

/**
 * Every running stream side by side (1 to 4, each with its combined overlay), or one of them on
 * its own with the single and split views. Streams that end stay until closed, with their clip id.
 */
export function LiveView({ clipId, clipName, onStreamEnded }: Props) {
  const [streams, setStreams] = useState<Stream[]>([])
  const [shown, setShown] = useState<string[]>([])
  const [focus, setFocus] = useState<string | null>(null)
  const [mode, setMode] = useState<ViewMode>('single')
  const [visible, setVisible] = useState<Record<string, boolean>>({})
  const [labels, setLabels] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)
  const wasActive = useRef(new Set<string>())

  useEffect(() => {
    const ctl = new AbortController()
    let timer = 0
    const poll = () => {
      listStreams(ctl.signal)
        .then((list) => {
          setStreams(list)
          const active = list.filter(streamActive).map((s) => s.stream_id)
          setShown((cur) => [...cur, ...active.filter((s) => !cur.includes(s))])
          for (const s of list) {
            if (streamActive(s)) wasActive.current.add(s.stream_id)
            else if (wasActive.current.delete(s.stream_id)) onStreamEnded()
          }
          setError(null)
        })
        .catch((e) => !ctl.signal.aborted && setError(message(e)))
        .finally(() => {
          if (!ctl.signal.aborted) timer = window.setTimeout(poll, STREAMS_POLL_MS)
        })
    }
    poll()
    return () => {
      ctl.abort()
      clearTimeout(timer)
    }
  }, [onStreamEnded])

  // Each layer's box starts as the clip.json default; the boxes apply to every stream.
  const addMetas = useCallback((metas: LayerMeta[]) => {
    setVisible((v) => {
      const fresh = metas.filter((m) => !(m.layer in v))
      return fresh.length ? { ...v, ...Object.fromEntries(fresh.map((m) => [m.layer, m.default_visible ?? true])) } : v
    })
    setLabels((l) => (metas.every((m) => l[m.layer]) ? l : { ...l, ...Object.fromEntries(metas.map((m) => [m.layer, m.label])) }))
  }, [])

  async function goLive() {
    if (!clipId) return
    try {
      const s = await startStream(clipId)
      setStreams((cur) => [s, ...cur])
      setShown((cur) => [...cur, s.stream_id])
      setError(null)
    } catch (e) {
      setError(message(e))
    }
  }

  async function stop(id: string) {
    try {
      await stopStream(id)
    } catch (e) {
      setError(message(e))
    }
  }

  const byId = new Map(streams.map((s) => [s.stream_id, s]))
  const tiles = (focus ? [focus] : shown).map((id) => byId.get(id)).filter((s): s is Stream => !!s)
  const close = (id: string) => {
    setShown((cur) => cur.filter((s) => s !== id))
    if (focus === id) setFocus(null)
  }

  return (
    <div>
      <div className="card mb-4 flex flex-wrap items-center gap-3 px-4 py-3 text-sm">
        <button
          type="button"
          disabled={!clipId}
          onClick={goLive}
          className="rounded-lg bg-rose-500 px-4 py-1.5 font-medium text-white transition hover:bg-rose-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Go live with this clip
        </button>
        <span className="text-slate-400">
          {clipName ? `Replays "${clipName}" at real time as if a drone sent it, with every model.` : 'Pick a clip first.'}
        </span>
        <span className="ml-auto flex flex-wrap gap-3 text-slate-300">
          {Object.entries(labels).map(([layer, label]) => (
            <label key={layer} className="flex items-center gap-1.5">
              <input type="checkbox" checked={!!visible[layer]} onChange={() => setVisible((v) => ({ ...v, [layer]: !v[layer] }))} />
              {label}
            </label>
          ))}
        </span>
      </div>
      {error && (
        <p role="alert" className="mb-4 rounded-lg border border-rose-800 bg-rose-950/50 p-3 text-rose-200">
          {error}
        </p>
      )}
      {focus && <ModeSwitch mode={mode} onChange={setMode} />}
      {tiles.length === 0 && <p className="card p-4 text-slate-300">No live streams. Pick a clip and press Go live.</p>}
      <div className={`grid gap-5 ${focus || tiles.length === 1 ? '' : 'lg:grid-cols-2'}`}>
        {tiles.map((s) => (
          <section key={s.stream_id} className="card p-3">
            <header className="mb-2 flex flex-wrap items-center gap-2 text-xs">
              <span className={`size-2 rounded-full ${streamActive(s) ? 'bg-rose-500' : 'bg-slate-500'}`} />
              <span className="text-sm font-medium text-slate-100">{s.name}</span>
              <span className="text-slate-400">
                {s.state === 'running' && s.stage === 'live' ? `frame ${s.frames_read}` : s.stage}
                {s.clip_id && `, saved as clip ${s.clip_id}`}
                {s.error && `: ${s.error}`}
              </span>
              <span className="ml-auto flex gap-2">
                {streamActive(s) && <TileButton onClick={() => stop(s.stream_id)}>Stop</TileButton>}
                <TileButton onClick={() => setFocus(focus ? null : s.stream_id)}>{focus ? 'All streams' : 'Open'}</TileButton>
                {!streamActive(s) && <TileButton onClick={() => close(s.stream_id)}>Close</TileButton>}
              </span>
            </header>
            {Object.entries(s.errors).map(([module, why]) => (
              <p key={module} className="mb-1 text-xs text-amber-200">
                {module}: {why}
              </p>
            ))}
            {s.fps && s.frame_size ? (
              <LiveStage
                stream={{ ...s, fps: s.fps, frame_size: s.frame_size }}
                mode={focus ? mode : 'single'}
                visible={visible}
                onMetas={addMetas}
              />
            ) : (
              <div className="flex aspect-video items-center justify-center rounded-2xl bg-black text-sm text-slate-400">{s.stage}</div>
            )}
          </section>
        ))}
      </div>
    </div>
  )
}

function TileButton({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" onClick={onClick} className="rounded-lg border border-white/10 px-2.5 py-1 text-slate-200 hover:bg-white/10">
      {children}
    </button>
  )
}
