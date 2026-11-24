import type { Clip, IndexedLayer, LayerMeta } from '../types'
import { drawHover, drawOutline } from './draw'
import { pickSample } from './frames'
import { gridRender } from './layers'
import { drawLayer } from './overlay'
import { cellAt, fmt, objectAt, readCell, unitLabel } from './readout'

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
  /** The picture under the layers (a playing video, or a live frame); null draws black. */
  image: CanvasImageSource | null
  clip: Clip
  metas: LayerMeta[]
  layers: Record<string, IndexedLayer>
  frame: number
  pointer: Pointer | null
  /**
   * The sample to draw per layer. Default: the stored clip rule (pickSample with max_age_frames).
   * A live view passes its own rule, and its samples are drawn as of their own frame.
   */
  pick?: (indexed: IndexedLayer) => number | null
}

/** Draw the video frame, the given layers and the hover readout; returns which layers had data. */
export function paintPanel(i: PaintInput): Record<string, boolean> {
  const { ctx, w, h, image, clip, metas, layers, frame, pointer, pick } = i
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, w, h)
  if (image) ctx.drawImage(image, 0, 0, w, h)

  // Dots are 4 px in a panel ~960 px wide and shrink to 2 px in small split panels.
  const scale = { sx: w / clip.frame_size[0], sy: h / clip.frame_size[1], dot: Math.min(4, Math.max(2, w / 240)) }
  const faint = metas.length > 1
  const live: Record<string, boolean> = {}
  const samples: Record<string, number | null> = {}
  for (const meta of metas) {
    const indexed = layers[meta.layer]
    if (!indexed) continue
    const key = pick ? pick(indexed) : pickSample(indexed.keys, frame, indexed.layer.max_age_frames)
    samples[meta.layer] = key
    live[meta.layer] = key !== null
    if (key !== null) drawLayer(ctx, meta, indexed, key, pick ? key : frame, scale, faint)
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
    if (indexed?.layer.type === 'polygons') {
      const key = samples[meta.layer]
      if (key == null) continue
      const hit = objectAt(indexed.layer.frames[String(key)], at.x / scale.sx, at.y / scale.sy)
      if (!hit) continue
      drawOutline(ctx, hit.polygon, scale)
      lines.push(`${meta.label}: ${hit.cls} ${hit.score.toFixed(2)}`)
      continue
    }
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
  if (box || lines.length) drawHover(ctx, box, lines, at, { w, h })
  if (box) return
  // No grid layer here (for example a panel of dots): mark where the pointer is in the other panels.
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.arc(at.x, at.y, 7, 0, Math.PI * 2)
  ctx.stroke()
}
