import { RAMP_GRADIENT } from '../lib/colour'
import { gridRender } from '../lib/layers'
import { gridTop } from '../lib/overlay'
import type { Clip, LayerMeta } from '../types'
import type { LayerState } from '../useClipLayers'

interface Props {
  clip: Clip
  state: LayerState
  visible: Record<string, boolean>
  /** Per visible layer: has a sample to show right now. A missing entry means not drawn. */
  live: Record<string, boolean>
  onToggle: (layer: string) => void
}

const TYPE_LABEL = { grid: 'grid', points: 'points', tracks: 'tracks' } as const

function status(meta: LayerMeta, p: Props): { text: string; tone: string; title?: string } {
  const failed = p.state.failed[meta.layer]
  if (failed) return { text: 'no data file', tone: 'text-rose-300', title: failed }
  if (p.state.loading) return { text: 'loading…', tone: 'text-slate-400' }
  if (!p.visible[meta.layer]) return { text: 'hidden', tone: 'text-slate-500' }
  if (p.live[meta.layer] === false) {
    return { text: 'no data at this moment', tone: 'text-amber-300', title: 'A gap: nothing is drawn.' }
  }
  return { text: 'showing', tone: 'text-emerald-300' }
}

function Legend({ meta, state }: { meta: LayerMeta; state: LayerState }) {
  const indexed = state.layers[meta.layer]
  if (!indexed || indexed.layer.type !== 'grid') return null
  const mode = gridRender(meta, indexed.layer)
  const top = gridTop(indexed.layer, meta)
  const units = indexed.layer.units
  const fmt = (n: number) => (n >= 100 ? n.toFixed(0) : n.toFixed(2).replace(/\.?0+$/, ''))

  if (mode.kind === 'arrows') {
    return (
      <p className="mt-1 text-xs text-slate-400">
        Arrow length is proportional to speed; the longest is {fmt(top)} {units}.
      </p>
    )
  }
  return (
    <div className="mt-1">
      <div className="h-2 w-full rounded" style={{ background: RAMP_GRADIENT }} />
      <div className="mt-0.5 flex justify-between text-xs text-slate-400">
        <span>0</span>
        <span>
          {fmt(top)} {units}
        </span>
      </div>
    </div>
  )
}

export function LayerPanel(p: Props) {
  return (
    <section aria-label="Layers" className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Layers</h2>
      <ul className="mt-3 space-y-3">
        {p.clip.layers.map((meta) => {
          const s = status(meta, p)
          return (
            <li key={meta.layer}>
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  className="size-4 accent-sky-400"
                  checked={p.visible[meta.layer] ?? false}
                  onChange={() => p.onToggle(meta.layer)}
                />
                <span className="font-medium text-slate-100">{meta.label}</span>
                <span className="rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-400">
                  {TYPE_LABEL[meta.type]}
                </span>
                <span className={`ml-auto text-xs ${s.tone}`} title={s.title}>
                  {s.text}
                </span>
              </label>
              {p.visible[meta.layer] && <Legend meta={meta} state={p.state} />}
            </li>
          )
        })}
      </ul>
    </section>
  )
}
