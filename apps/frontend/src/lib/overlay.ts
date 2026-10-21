import type { GridLayer, IndexedLayer, LayerMeta, PointRow, TrackRow, TracksLayer } from '../types'
import { drawArrows, drawHeatmap, drawPoints, drawPolygons, drawTracks, type Scale } from './draw'
import { buildTrails, fieldMax, gridRender, maxSpeed } from './layers'

const TRAIL_FRAMES = 30

// Smallest speed a full length arrow stands for, per unit. Without it a near still crowd draws
// its noise at full length, because arrows are scaled to the clip's fastest cell.
const ARROW_FLOOR: Record<string, number> = { px_per_frame: 0.5, m_per_s: 0.5 }

// The colour scale and arrow length are fixed per layer, so values stay comparable over time.
// Scanning every frame is not free, so each is computed once.
const scales = new WeakMap<GridLayer, number>()

export function gridTop(layer: GridLayer, meta: LayerMeta): number {
  let top = scales.get(layer)
  if (top === undefined) {
    const mode = gridRender(meta, layer)
    top =
      mode.kind === 'arrows'
        ? Math.max(maxSpeed(layer, mode.u, mode.v), ARROW_FLOOR[layer.units] ?? 0)
        : fieldMax(layer, mode.field)
    scales.set(layer, top)
  }
  return top
}

/**
 * Draw one layer's sample for `key` (a frame number that exists in the layer). `faint` is set when
 * other layers share the panel, so a heat map does not hide them.
 */
export function drawLayer(
  ctx: CanvasRenderingContext2D,
  meta: LayerMeta,
  indexed: IndexedLayer,
  key: number,
  frame: number,
  k: Scale,
  faint = false,
): void {
  const { layer, keys } = indexed
  const at = String(key)
  if (layer.type === 'grid') {
    const entry = layer.frames[at]
    const mode = gridRender(meta, layer)
    const top = gridTop(layer, meta)
    if (mode.kind === 'arrows') drawArrows(ctx, layer, entry, mode.u, mode.v, top, k)
    else drawHeatmap(ctx, layer, entry, mode.field, top, k, faint)
  } else if (layer.type === 'points') {
    drawPoints(ctx, layer.frames[at] as PointRow[], k)
  } else if (layer.type === 'tracks') {
    const tracks = layer as TracksLayer
    drawTracks(ctx, buildTrails(tracks, keys, frame, TRAIL_FRAMES), tracks.frames[at] as TrackRow[], k)
  } else if (layer.type === 'polygons') {
    drawPolygons(ctx, layer.classes, layer.frames[at], k)
  }
  // Any other type (a newer contract than this viewer) draws nothing rather than breaking the page.
}
