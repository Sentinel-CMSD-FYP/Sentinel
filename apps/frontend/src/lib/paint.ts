import type { Clip, IndexedLayer, LayerMeta } from '../types'
import { drawHover } from './draw'
import { pickSample } from './frames'
import { gridRender } from './layers'
import { drawLayer } from './overlay'
import { cellAt, fmt, readCell, unitLabel } from './readout'

/** Pointer position as a fraction of the panel, so every panel can follow the same pointer. */
export interface Pointer {
  nx: number
  ny: number
}

export interface PaintInput {
  ctx: CanvasRenderingContext2D
  /** Panel size in CSS pixels. */
  w: number
  h: number
  video: HTMLVideoElement | null
  clip: Clip
  metas: LayerMeta[]
  layers: Record<string, IndexedLayer>
  frame: number
  pointer: Pointer | null
}

/** Draw the video frame, the given layers and the hover readout; returns which layers had data. */
export function paintPanel(i: PaintInput): Record<string, boolean> {
  const { ctx, w, h, video, clip, metas, layers, frame, pointer } = i
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, w, h)
  if (video && video.readyState >= 2) ctx.drawImage(video, 0, 0, w, h)

  const scale = { sx: w / clip.frame_size[0], sy: h / clip.frame_size[1] }
  const live: Record<string, boolean> = {}
  const samples: Record<string, number | null> = {}
  for (const meta of metas) {
    const indexed = layers[meta.layer]
    if (!indexed) continue
    const key = pickSample(indexed.keys, frame, indexed.layer.max_age_frames)
    samples[meta.layer] = key
    live[meta.layer] = key !== null
    if (key !== null) drawLayer(ctx, meta, indexed, key, frame, scale)
  }

  if (pointer) paintHover(ctx, w, h, metas, layers, samples, pointer, scale)
  return live
}

function paintHover(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  metas: LayerMeta[],
  layers: Record<string, IndexedLayer>,
  samples: Record<string, number | null>,
  pointer: Pointer,
  scale: { sx: number; sy: number },
): void {
  const at = { x: pointer.nx * w, y: pointer.ny * h }
  const lines: string[] = []
  let box: { x: number; y: number; w: number; h: number } | null = null
  for (const meta of metas) {
    const indexed = layers[meta.layer]
    if (!indexed || indexed.layer.type !== 'grid') continue
    const cell = cellAt(indexed.layer, at.x / scale.sx, at.y / scale.sy)
    if (!cell) continue
    const cw = (indexed.layer.frame_size[0] / indexed.layer.grid.cols) * scale.sx
    const ch = (indexed.layer.frame_size[1] / indexed.layer.grid.rows) * scale.sy
    box ??= { x: cell.col * cw, y: cell.row * ch, w: cw, h: ch }
    const key = samples[meta.layer]
    const reading =
      key == null
        ? null
        : readCell(indexed.layer.frames[String(key)], gridRender(meta, indexed.layer), cell.row, cell.col)
    lines.push(
      reading
        ? `${meta.label}: ${fmt(reading.value)} ${unitLabel(indexed.layer.units)}${reading.heading ? ' ' + reading.heading : ''}`
        : `${meta.label}: no data`,
    )
  }
  if (box) {
    drawHover(ctx, box, lines, at, { w, h })
    return
  }
  // No grid layer here (for example a panel of dots): mark where the pointer is in the other panels.
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.arc(at.x, at.y, 7, 0, Math.PI * 2)
  ctx.stroke()
}
