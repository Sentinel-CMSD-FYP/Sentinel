import type { GridLayer, IndexedLayer, LayerMeta, PointRow, TrackRow, TracksLayer } from '../types'
import { drawArrows, drawHeatmap, drawPoints, drawTracks, type Scale } from './draw'
import { buildTrails, fieldMax, gridRender, maxSpeed } from './layers'

const TRAIL_FRAMES = 30

// The colour scale and arrow length are fixed per layer, so values stay comparable over time.
// Scanning every frame is not free, so each is computed once.
const scales = new WeakMap<GridLayer, number>()

export function gridTop(layer: GridLayer, meta: LayerMeta): number {
  let top = scales.get(layer)
  if (top === undefined) {
    const mode = gridRender(meta, layer)
    top = mode.kind === 'arrows' ? maxSpeed(layer, mode.u, mode.v) : fieldMax(layer, mode.field)
    scales.set(layer, top)
  }
  return top
}

/** Draw one layer's sample for `key` (a frame number that exists in the layer). */
export function drawLayer(
  ctx: CanvasRenderingContext2D,
  meta: LayerMeta,
  indexed: IndexedLayer,
  key: number,
  frame: number,
  k: Scale,
): void {
  const { layer, keys } = indexed
  const at = String(key)
  if (layer.type === 'grid') {
    const entry = layer.frames[at]
    const mode = gridRender(meta, layer)
    const top = gridTop(layer, meta)
    if (mode.kind === 'arrows') drawArrows(ctx, layer, entry, mode.u, mode.v, top, k)
    else drawHeatmap(ctx, layer, entry, mode.field, top, k)
  } else if (layer.type === 'points') {
    drawPoints(ctx, layer.frames[at] as PointRow[], k)
  } else {
    const tracks = layer as TracksLayer
    drawTracks(ctx, buildTrails(tracks, keys, frame, TRAIL_FRAMES), tracks.frames[at] as TrackRow[], k)
  }
}
