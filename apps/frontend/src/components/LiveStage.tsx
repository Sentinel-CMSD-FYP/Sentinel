import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getSamples, streamVideoUrl } from '../api'
import {
  addSamples,
  emptyLive,
  latencyStats,
  LIVE_BUDGET_S,
  liveKey,
  SAMPLES_POLL_MS,
  type LatencyRecord,
  type LatencyStats,
  type LiveLayers,
} from '../lib/live'
import { readMjpeg, type MjpegPart } from '../lib/mjpeg'
import { paintPanel, type Pointer } from '../lib/paint'
import { buildPanels, gridColumns, type ViewMode } from '../lib/panels'
import type { Clip, IndexedLayer, LayerMeta, Stream } from '../types'
import { Panel } from './Stage'

/**
 * Every capture to drawn record of this page, per stream, and every sample as it arrived (drawn or
 * not: a sample later than the budget is never drawn). On `window` so a latency measurement can be
 * read out of the browser (runs/step29/measure.js).
 */
const records: Record<string, LatencyRecord[]> = {}
const arrivals: Record<string, { layer: string; frame: number; tCapture: number | null; tReady: number; tReceived: number }[]> = {}
Object.assign(window, { sentinelLatency: records, sentinelArrivals: arrivals })
/** Per stream, the last picture and sample key recorded, so a remounted stage does not record them again. */
const recorded: Record<string, { picture: number; layers: Record<string, number> }> = {}

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))

interface Props {
  /** A stream whose worker has opened its source, so `fps` and `frame_size` are known. */
  stream: Stream & { fps: number; frame_size: [number, number] }
  mode: ViewMode
  visible: Record<string, boolean>
  onMetas: (metas: LayerMeta[]) => void
}

/**
 * One live stream: the newest picture with each layer's newest sample over it, drawn by the same
 * renderers as a stored clip. A sample is drawn as of its own frame (tracks trails end there) and
 * hidden once it is too far behind the picture (`liveKey`).
 */
export function LiveStage({ stream, mode, visible, onMetas }: Props) {
  const id = stream.stream_id
  const live = useRef<LiveLayers>(emptyLive())
  const picture = useRef<{ bitmap: ImageBitmap; frame: number; tCapture: number } | null>(null)
  const canvases = useRef(new Map<string, HTMLCanvasElement>())
  const hostRef = useRef<HTMLDivElement>(null)
  const pointer = useRef<Pointer | null>(null)
  const dirty = useRef(true)
  const lastSig = useRef('')
  const [metas, setMetas] = useState<LayerMeta[]>([])
  const [panelLive, setPanelLive] = useState<Record<string, Record<string, boolean>>>({})
  const [stats, setStats] = useState<Record<string, LatencyStats>>({})
  const [pictureError, setPictureError] = useState<string | null>(null)

  const { fps, frame_size } = stream
  const clip = useMemo<Clip>(
    () => ({ clip_id: id, name: stream.name, fps, frame_size, n_frames: 0, layers: metas }),
    [id, stream.name, fps, frame_size, metas],
  )
  const panels = useMemo(() => buildPanels(clip, visible, mode), [clip, visible, mode])
  const latest = useRef({ clip, panels })
  useEffect(() => {
    latest.current = { clip, panels }
    dirty.current = true
  }, [clip, panels])
  useEffect(() => onMetas(metas), [metas, onMetas])

  // The picture: MJPEG parts decoded one at a time; while one decodes only the newest waits.
  useEffect(() => {
    const ctl = new AbortController()
    let pending: MjpegPart | null = null
    let decoding = false
    const decode = async () => {
      decoding = true
      while (pending) {
        const part: MjpegPart = pending
        pending = null
        try {
          const bitmap = await createImageBitmap(new Blob([part.jpeg], { type: 'image/jpeg' }))
          if (ctl.signal.aborted) {
            bitmap.close()
            break
          }
          picture.current?.bitmap.close()
          picture.current = { bitmap, frame: part.frame, tCapture: part.tCapture }
          dirty.current = true
        } catch {
          /* a broken JPEG: the next one replaces it */
        }
      }
      decoding = false
    }
    readMjpeg(
      streamVideoUrl(id),
      (part) => {
        pending = part
        if (!decoding) void decode()
      },
      ctl.signal,
    ).catch((e) => !ctl.signal.aborted && setPictureError(message(e)))
    return () => {
      ctl.abort()
      picture.current?.bitmap.close()
      picture.current = null
    }
  }, [id])

  // The samples: polled from byte offset 0, so every layer header is seen.
  useEffect(() => {
    const ctl = new AbortController()
    let after = 0
    let timer = 0
    const poll = async () => {
      try {
        const res = await getSamples(id, after, ctl.signal)
        after = res.next
        const now = Date.now() / 1000
        const got = (arrivals[id] ??= [])
        for (const x of res.samples) got.push({ layer: x.layer, frame: x.frame, tCapture: x.t_capture, tReady: x.t_ready, tReceived: now })
        if (res.samples.length) {
          addSamples(live.current, res.samples)
          setMetas(live.current.metas) // the same array unless a layer was new
          dirty.current = true
        } else if (res.state === 'done' || res.state === 'failed') {
          return // ended, and read to the end
        }
      } catch {
        if (ctl.signal.aborted) return // else a missed poll: try again
      }
      timer = window.setTimeout(poll, SAMPLES_POLL_MS)
    }
    void poll()
    return () => {
      ctl.abort()
      clearTimeout(timer)
    }
  }, [id])

  // Paint whenever something changed, and note when each new picture and sample reaches the screen.
  useEffect(() => {
    const log = (records[id] ??= [])
    const done = (recorded[id] ??= { picture: -1, layers: {} })
    let handle = 0
    const paint = () => {
      handle = requestAnimationFrame(paint)
      if (!dirty.current) return
      dirty.current = false
      const { clip, panels } = latest.current
      const pic = picture.current
      const frame = pic?.frame ?? 0
      const now = Date.now() / 1000
      const pick = (ix: IndexedLayer) => liveKey(ix, live.current.captured[ix.layer.layer], now, clip.fps)
      const dpr = window.devicePixelRatio || 1
      const lives: Record<string, Record<string, boolean>> = {}
      for (const panel of panels) {
        const canvas = canvases.current.get(panel.id)
        const ctx = canvas?.getContext('2d')
        if (!canvas || !ctx) continue
        const w = canvas.clientWidth
        const h = canvas.clientHeight
        if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
          canvas.width = Math.round(w * dpr)
          canvas.height = Math.round(h * dpr)
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        lives[panel.id] = paintPanel({
          ctx, w, h, image: pic?.bitmap ?? null, clip, metas: panel.metas, layers: live.current.layers, frame,
          pointer: pointer.current, pick,
        })
      }

      if (pic && pic.frame > done.picture) {
        done.picture = pic.frame
        log.push({ stream: id, layer: 'picture', frame: pic.frame, tCapture: pic.tCapture, tDrawn: now })
      }
      for (const layer of new Set(panels.flatMap((p) => p.metas.map((m) => m.layer)))) {
        const ix = live.current.layers[layer]
        const key = ix ? pick(ix) : null
        if (key === null || key <= (done.layers[layer] ?? -1)) continue
        done.layers[layer] = key
        const t = live.current.captured[layer]?.get(key)
        if (t !== undefined) log.push({ stream: id, layer, frame: key, tCapture: t, tDrawn: now })
      }
      const sig = JSON.stringify(lives)
      if (sig !== lastSig.current) {
        lastSig.current = sig
        setPanelLive(lives)
      }
    }
    handle = requestAnimationFrame(paint)
    const resize = new ResizeObserver(() => (dirty.current = true))
    if (hostRef.current) resize.observe(hostRef.current)
    const statsTimer = window.setInterval(() => {
      setStats(latencyStats(log))
      dirty.current = true // samples age out even when no new picture comes (the stream ended)
    }, 1000)
    return () => {
      cancelAnimationFrame(handle)
      resize.disconnect()
      clearInterval(statsTimer)
    }
  }, [id])

  const register = useCallback((panel: string, el: HTMLCanvasElement | null) => {
    if (el) canvases.current.set(panel, el)
    else canvases.current.delete(panel)
  }, [])
  const track = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    pointer.current = { nx: (e.clientX - r.left) / r.width, ny: (e.clientY - r.top) / r.height }
    dirty.current = true
  }
  const untrack = () => {
    pointer.current = null
    dirty.current = true
  }

  const label = (layer: string) => metas.find((m) => m.layer === layer)?.label ?? layer
  return (
    <div ref={hostRef}>
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${gridColumns(panels.length)}, minmax(0, 1fr))` }}>
        {panels.map((p) => (
          <Panel
            key={p.id}
            spec={p}
            aspect={`${frame_size[0]} / ${frame_size[1]}`}
            live={panelLive[p.id]}
            register={register}
            onMove={track}
            onLeave={untrack}
            onClick={() => {}}
          />
        ))}
      </div>
      {pictureError && <p className="mt-2 text-xs text-rose-300">{pictureError}</p>}
      <p className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-400" title="Capture to drawn on screen: last (median over the stream)">
        {Object.entries(stats).map(([layer, s]) => (
          <span key={layer} className="tabular-nums">
            {layer === 'picture' ? 'Picture' : label(layer)}{' '}
            <span className={s.last > LIVE_BUDGET_S ? 'text-amber-300' : 'text-slate-200'}>{s.last.toFixed(1)} s</span>{' '}
            ({s.median.toFixed(1)} s)
          </span>
        ))}
      </p>
    </div>
  )
}
