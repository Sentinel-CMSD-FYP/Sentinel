import { RAMP_GRADIENT } from '../lib/colour'
import { pickSample } from '../lib/frames'
import { gridRender } from '../lib/layers'
import { gridTop } from '../lib/overlay'
import { fmt, summarise, unitLabel } from '../lib/readout'
import type { Clip, LayerMeta } from '../types'
import type { LayerState } from '../useClipLayers'

interface Props {
  clip: Clip
  state: LayerState
  visible: Record<string, boolean>
  /** Per visible layer: has a sample to show right now. A missing entry means not drawn. */
  live: Record<string, boolean>
  /** Current video frame, for the live numbers. */
  frame: number
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

/** Speed (or value) right now: mean and peak over the cells that have data. */
function Now({ meta, state, frame }: { meta: LayerMeta; state: LayerState; frame: number }) {
  const indexed = state.layers[meta.layer]
  if (!indexed || indexed.layer.type !== 'grid') return null
  const key = pickSample(indexed.keys, frame, indexed.layer.max_age_frames)
  if (key === null) return null
  const mode = gridRender(meta, indexed.layer)
  const s = summarise(indexed.layer, indexed.layer.frames[String(key)], mode)
  const label = mode.kind === 'arrows' ? 'speed' : mode.field
  const stat = (name: string, value: string) => (
    <div className="rounded-md bg-slate-950/60 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{name}</div>
      <div className="text-sm font-semibold tabular-nums text-slate-100">{value}</div>
    </div>
  )
  return (
    <div className="mt-2">
      <div className="grid grid-cols-3 gap-1.5">
        {stat(`mean ${label}`, fmt(s.mean))}
        {stat(`peak ${label}`, fmt(s.peak))}
        {stat('cells', `${s.cells}/${s.total}`)}
      </div>
      <p className="mt-1 text-[11px] text-slate-500">{unitLabel(indexed.layer.units)}. Hover the video for one cell.</p>
    </div>
  )
}

function Legend({ meta, state }: { meta: LayerMeta; state: LayerState }) {
  const indexed = state.layers[meta.layer]
  if (!indexed || indexed.layer.type !== 'grid') return null
  const mode = gridRender(meta, indexed.layer)
  const top = gridTop(indexed.layer, meta)
  const units = unitLabel(indexed.layer.units)

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
    <section aria-label="Layers" className="card p-4">
      <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-300/80">Layers</h2>
      <ul className="mt-3 space-y-2.5">
        {p.clip.layers.map((meta) => {
          const s = status(meta, p)
          return (
            <li key={meta.layer} className="rounded-lg border border-white/5 bg-white/[0.03] p-3">
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
              {p.visible[meta.layer] && (
                <>
                  <Legend meta={meta} state={p.state} />
                  <Now meta={meta} state={p.state} frame={p.frame} />
                </>
              )}
            </li>
          )
        })}
      </ul>
    </section>
  )
}
