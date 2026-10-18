import { useEffect, useRef } from 'react'
import { drawHover } from '../lib/draw'
import { frameForTime, pickSample } from '../lib/frames'
import { gridRender } from '../lib/layers'
import { drawLayer } from '../lib/overlay'
import { cellAt, fmt, readCell, unitLabel } from '../lib/readout'
import type { Clip, IndexedLayer } from '../types'

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
  onStatus: (s: StageStatus) => void
}

/** The video with a canvas laid exactly over it; the canvas redraws whenever a frame is shown. */
export function VideoStage({ clip, src, layers, visible, onStatus }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const redraw = useRef<() => void>(() => {})
  const latest = useRef({ clip, layers, visible, onStatus })
  const lastSig = useRef('')
  const pointer = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    latest.current = { clip, layers, visible, onStatus }
    redraw.current()
  }, [clip, layers, visible, onStatus])

  useEffect(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    const draw = () => {
      const { clip, layers, visible, onStatus } = latest.current
      const frame = frameForTime(video.currentTime, clip.fps)
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      const dpr = window.devicePixelRatio || 1
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr)
        canvas.height = Math.round(h * dpr)
      }
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      const scale = { sx: w / clip.frame_size[0], sy: h / clip.frame_size[1] }
      const live: Record<string, boolean> = {}
      for (const meta of clip.layers) {
        const indexed = layers[meta.layer]
        if (!visible[meta.layer] || !indexed) continue
        const key = pickSample(indexed.keys, frame, indexed.layer.max_age_frames)
        live[meta.layer] = key !== null
        if (key !== null) drawLayer(ctx, meta, indexed, key, frame, scale)
      }

      const p = pointer.current
      if (p) {
        const fx = p.x / scale.sx
        const fy = p.y / scale.sy
        const lines: string[] = []
        let box: { x: number; y: number; w: number; h: number } | null = null
        for (const meta of clip.layers) {
          const indexed = layers[meta.layer]
          if (!visible[meta.layer] || !indexed || indexed.layer.type !== 'grid') continue
          const cell = cellAt(indexed.layer, fx, fy)
          if (!cell) continue
          const cw = (indexed.layer.frame_size[0] / indexed.layer.grid.cols) * scale.sx
          const ch = (indexed.layer.frame_size[1] / indexed.layer.grid.rows) * scale.sy
          box ??= { x: cell.col * cw, y: cell.row * ch, w: cw, h: ch }
          const key = pickSample(indexed.keys, frame, indexed.layer.max_age_frames)
          const reading =
            key === null
              ? null
              : readCell(indexed.layer.frames[String(key)], gridRender(meta, indexed.layer), cell.row, cell.col)
          lines.push(
            reading
              ? `${meta.label}: ${fmt(reading.value)} ${unitLabel(indexed.layer.units)}${reading.heading ? ' ' + reading.heading : ''}`
              : `${meta.label}: no data`,
          )
        }
        if (box) drawHover(ctx, box, lines, p, { w, h })
      }

      const sig = `${frame}|${JSON.stringify(live)}`
      if (sig !== lastSig.current) {
        lastSig.current = sig
        onStatus({ frame, live })
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

    const events = ['seeked', 'loadedmetadata', 'play', 'pause'] as const
    events.forEach((e) => video.addEventListener(e, draw))
    const move = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect()
      pointer.current = { x: e.clientX - r.left, y: e.clientY - r.top }
      draw()
    }
    const leave = () => {
      pointer.current = null
      draw()
    }
    const host = canvas.parentElement
    host?.addEventListener('pointermove', move)
    host?.addEventListener('pointerleave', leave)
    const resize = new ResizeObserver(draw)
    resize.observe(canvas)
    draw()

    return () => {
      if (hasRvfc) video.cancelVideoFrameCallback(handle)
      else cancelAnimationFrame(handle)
      events.forEach((e) => video.removeEventListener(e, draw))
      host?.removeEventListener('pointermove', move)
      host?.removeEventListener('pointerleave', leave)
      resize.disconnect()
      redraw.current = () => {}
    }
  }, [src])

  if (!src) {
    return (
      <div className="flex aspect-video items-center justify-center rounded-lg bg-slate-900 text-slate-400">
        This clip has no video.
      </div>
    )
  }

  return (
    <div
      className="relative w-full overflow-hidden rounded-2xl bg-black shadow-2xl shadow-black/50 ring-1 ring-white/10"
      style={{ aspectRatio: `${clip.frame_size[0]} / ${clip.frame_size[1]}` }}
    >
      <video ref={videoRef} src={src} controls playsInline preload="auto" className="absolute inset-0 h-full w-full" />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
    </div>
  )
}
