import { useEffect, useRef } from 'react'
import { frameForTime, pickSample } from '../lib/frames'
import { drawLayer } from '../lib/overlay'
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
    const resize = new ResizeObserver(draw)
    resize.observe(canvas)
    draw()

    return () => {
      if (hasRvfc) video.cancelVideoFrameCallback(handle)
      else cancelAnimationFrame(handle)
      events.forEach((e) => video.removeEventListener(e, draw))
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
      className="relative w-full overflow-hidden rounded-lg bg-black"
      style={{ aspectRatio: `${clip.frame_size[0]} / ${clip.frame_size[1]}` }}
    >
      <video ref={videoRef} src={src} controls playsInline preload="auto" className="absolute inset-0 h-full w-full" />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
    </div>
  )
}
