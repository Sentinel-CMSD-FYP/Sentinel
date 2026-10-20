import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { frameForTime } from '../lib/frames'
import { paintPanel, type Pointer } from '../lib/paint'
import { ALL_PANEL, buildPanels, gridColumns, type PanelSpec, type ViewMode } from '../lib/panels'
import type { Clip, IndexedLayer } from '../types'
import { Controls } from './Controls'

export interface StageStatus {
  frame: number
  /** Per visible layer: does it have a sample to show at this frame? */
  live: Record<string, boolean>
}

interface Props {
  clip: Clip
  src: string | null
  layers: Record<string, IndexedLayer>
  visible: Record<string, boolean>
  mode: ViewMode
  onStatus: (s: StageStatus) => void
}

/**
 * One video, decoded once, drawn into every panel. The panels all read the same clock, so they
 * cannot drift apart, and one pointer is shared so a cell can be compared across models.
 */
export function Stage({ clip, src, layers, visible, mode, onStatus }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const hostRef = useRef<HTMLDivElement>(null)
  const canvases = useRef(new Map<string, HTMLCanvasElement>())
  const redraw = useRef<() => void>(() => {})
  const pointer = useRef<Pointer | null>(null)
  const lastSig = useRef('')
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [loop, setLoop] = useState(false)
  const [panelLive, setPanelLive] = useState<Record<string, Record<string, boolean>>>({})

  const panels = useMemo(() => buildPanels(clip, visible, mode), [clip, visible, mode])
  const latest = useRef({ clip, layers, panels, onStatus })

  useEffect(() => {
    latest.current = { clip, layers, panels, onStatus }
    redraw.current()
  }, [clip, layers, panels, onStatus])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const draw = () => {
      const { clip, layers, panels, onStatus } = latest.current
      const now = frameForTime(video.currentTime, clip.fps)
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
          ctx, w, h, video, clip, metas: panel.metas, layers, frame: now, pointer: pointer.current,
        })
      }

      const sig = `${now}|${JSON.stringify(lives)}`
      if (sig !== lastSig.current) {
        lastSig.current = sig
        setFrame(now)
        setPanelLive(lives)
        onStatus({ frame: now, live: lives[ALL_PANEL] ?? {} })
      }
    }
    redraw.current = draw

    // requestVideoFrameCallback fires whenever a new frame is shown, including after a seek
    // while paused; browsers without it fall back to polling every animation frame.
    let handle = 0
    const hasRvfc = typeof video.requestVideoFrameCallback === 'function'
    const loop = () => {
      draw()
      handle = hasRvfc ? video.requestVideoFrameCallback(loop) : requestAnimationFrame(loop)
    }
    handle = hasRvfc ? video.requestVideoFrameCallback(loop) : requestAnimationFrame(loop)

    const sync = () => {
      setPlaying(!video.paused)
      draw()
    }
    const events = ['seeked', 'loadedmetadata', 'play', 'pause', 'ended'] as const
    events.forEach((e) => video.addEventListener(e, sync))
    const resize = new ResizeObserver(draw)
    if (hostRef.current) resize.observe(hostRef.current)
    draw()

    return () => {
      if (hasRvfc) video.cancelVideoFrameCallback(handle)
      else cancelAnimationFrame(handle)
      events.forEach((e) => video.removeEventListener(e, sync))
      resize.disconnect()
      redraw.current = () => {}
    }
  }, [src])

  const register = useCallback((id: string, el: HTMLCanvasElement | null) => {
    if (el) canvases.current.set(id, el)
    else canvases.current.delete(id)
  }, [])
  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    v.playbackRate = speed
    v.loop = loop
  }, [speed, loop, src])

  const toggle = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) void v.play()
    else v.pause()
  }, [])
  const seek = useCallback(
    (f: number) => {
      const v = videoRef.current
      if (v) v.currentTime = f / clip.fps
    },
    [clip.fps],
  )
  const step = useCallback(
    (delta: number) => {
      const v = videoRef.current
      if (!v) return
      v.pause()
      const target = Math.min(clip.n_frames - 1, Math.max(0, frameForTime(v.currentTime, clip.fps) + delta))
      v.currentTime = target / clip.fps
    },
    [clip.fps, clip.n_frames],
  )
  const fullscreen = useCallback(() => {
    const el = hostRef.current
    if (!el) return
    if (document.fullscreenElement) void document.exitFullscreen()
    else void el.requestFullscreen()
  }, [])

  const track = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    pointer.current = { nx: (e.clientX - r.left) / r.width, ny: (e.clientY - r.top) / r.height }
    redraw.current()
  }
  const untrack = () => {
    pointer.current = null
    redraw.current()
  }

  if (!src) {
    return (
      <div className="card flex aspect-video items-center justify-center text-slate-400">
        This clip has no video.
      </div>
    )
  }

  const controls = (variant: 'bar' | 'overlay') => (
    <Controls
      variant={variant}
      frame={frame}
      nFrames={clip.n_frames}
      fps={clip.fps}
      playing={playing}
      speed={speed}
      loop={loop}
      onToggle={toggle}
      onSeek={seek}
      onStep={step}
      onSpeed={setSpeed}
      onLoop={() => setLoop((l) => !l)}
      onFullscreen={fullscreen}
    />
  )
  const cols = gridColumns(panels.length)
  return (
    <div ref={hostRef} className="[&:fullscreen]:overflow-auto [&:fullscreen]:bg-slate-950 [&:fullscreen]:p-4">
      <video
        ref={videoRef}
        src={src}
        muted
        playsInline
        preload="auto"
        className="pointer-events-none absolute size-px opacity-0"
      />
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
        {panels.map((p) => (
          <Panel
            key={p.id}
            spec={p}
            aspect={`${clip.frame_size[0]} / ${clip.frame_size[1]}`}
            live={panelLive[p.id]}
            register={register}
            onMove={track}
            onLeave={untrack}
            onClick={toggle}
          >
            {p.id === ALL_PANEL && controls('overlay')}
          </Panel>
        ))}
      </div>
      {mode === 'split' && controls('bar')}
    </div>
  )
}

interface PanelProps {
  spec: PanelSpec
  aspect: string
  live: Record<string, boolean> | undefined
  register: (id: string, el: HTMLCanvasElement | null) => void
  onMove: (e: React.PointerEvent<HTMLDivElement>) => void
  onLeave: () => void
  onClick: () => void
  children?: React.ReactNode
}

function Panel({ spec, aspect, live, register, onMove, onLeave, onClick, children }: PanelProps) {
  const empty = spec.metas.length === 0
  const gap = !empty && live && spec.metas.every((m) => live[m.layer] === false)
  return (
    <figure className="m-0 overflow-hidden rounded-2xl bg-black shadow-xl shadow-black/40 ring-1 ring-white/10">
      <figcaption className="flex items-center gap-2 bg-slate-950/80 px-3 py-1.5 text-xs">
        <span className="font-medium text-slate-100">{spec.title}</span>
        {spec.id === ALL_PANEL && <span className="text-slate-500">follows the layer boxes</span>}
        {gap && <span className="ml-auto text-amber-300">no data at this moment</span>}
      </figcaption>
      <div
        className="group relative w-full cursor-pointer"
        style={{ aspectRatio: aspect }}
        onPointerMove={onMove}
        onPointerLeave={onLeave}
        onClick={onClick}
      >
        <canvas
          ref={(el) => register(spec.id, el)}
          className="absolute inset-0 h-full w-full"
        />
        {children}
      </div>
    </figure>
  )
}
