import type { ViewMode } from '../lib/panels'

/** Single (one combined panel) or split (one panel per model too). */
export function ModeSwitch({ mode, onChange }: { mode: ViewMode; onChange: (m: ViewMode) => void }) {
  const options: [ViewMode, string][] = [
    ['single', 'Single'],
    ['split', 'Split by model'],
  ]
  return (
    <div className="mb-3 flex flex-wrap items-center gap-3">
      <div role="group" aria-label="View" className="inline-flex rounded-xl border border-white/10 bg-slate-950/60 p-1">
        {options.map(([m, label]) => (
          <button
            key={m}
            type="button"
            aria-pressed={mode === m}
            onClick={() => onChange(m)}
            className={`rounded-lg px-3 py-1.5 text-sm transition ${
              mode === m ? 'bg-sky-500 font-medium text-slate-950' : 'text-slate-300 hover:bg-white/10'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {mode === 'split' && (
        <span className="text-xs text-slate-500">
          One panel per model, all on the same clock. The boxes on the right control the combined panel only.
        </span>
      )}
    </div>
  )
}
