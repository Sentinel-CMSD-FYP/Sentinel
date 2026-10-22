import type { Clip, LayerMeta } from '../types'

export type ViewMode = 'single' | 'split'

export interface PanelSpec {
  id: string
  title: string
  /** The layers this panel draws, in clip order. */
  metas: LayerMeta[]
}

export const ALL_PANEL = 'all'

/**
 * The panels to show. The combined panel follows the on/off boxes; in split mode every layer also
 * gets a panel of its own, which ignores the boxes so each model can be read on its own.
 */
export function buildPanels(clip: Clip, visible: Record<string, boolean>, mode: ViewMode): PanelSpec[] {
  const all: PanelSpec = {
    id: ALL_PANEL,
    title: 'All layers',
    metas: clip.layers.filter((m) => visible[m.layer]),
  }
  if (mode === 'single') return [all]
  return [all, ...clip.layers.map((m) => ({ id: m.layer, title: m.label, metas: [m] }))]
}

/** Columns for a grid of `n` equal panels: one big panel, then two across, then three. */
export function gridColumns(n: number): number {
  if (n <= 1) return 1
  return n <= 4 ? 2 : 3
}
