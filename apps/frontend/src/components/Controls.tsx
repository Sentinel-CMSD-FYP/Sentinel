const SPEEDS = [0.25, 0.5, 1, 2] as const

interface Props {
  frame: number
  nFrames: number
  fps: number
  playing: boolean
  speed: number
  loop: boolean
  onToggle: () => void
  onSeek: (frame: number) => void
  onStep: (delta: number) => void
  onSpeed: (speed: number) => void
  onLoop: () => void
  onFullscreen: () => void
  /** `overlay` sits on the bottom edge of a panel; `bar` is a card of its own. */
  variant?: 'bar' | 'overlay'
}

const clock = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`

const icon = 'size-4'
const button =
  'grid size-8 shrink-0 place-items-center rounded-lg text-slate-200 transition hover:bg-white/15'

/**
 * Playback for the one video behind every panel. It can be shown in several places (the combined
 * panel and the shared bar); each drives the same clock, so every panel always shows the same frame.
 */
export function Controls(p: Props) {
  const last = Math.max(0, p.nFrames - 1)
  const overlay = p.variant === 'overlay'
  return (
    <div
      onClick={(e) => e.stopPropagation()}
      onPointerMove={(e) => e.stopPropagation()}
      className={
        overlay
          ? 'absolute inset-x-0 bottom-0 flex flex-wrap items-center gap-x-1.5 gap-y-1 bg-gradient-to-t from-black/85 to-transparent px-2.5 pb-2 pt-6'
          : 'card mt-3 flex flex-wrap items-center gap-1.5 px-3 py-2'
      }
    >
      <button
        type="button"
        onClick={p.onToggle}
        aria-label={p.playing ? 'Pause' : 'Play'}
        className="grid size-9 shrink-0 place-items-center rounded-full bg-sky-500 text-slate-950 transition hover:bg-sky-400"
      >
        {p.playing ? (
          <svg viewBox="0 0 24 24" className={icon} fill="currentColor" aria-hidden="true">
            <rect x="6" y="5" width="4" height="14" rx="1" />
            <rect x="14" y="5" width="4" height="14" rx="1" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" className={icon} fill="currentColor" aria-hidden="true">
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
      </button>
      <button type="button" onClick={() => p.onStep(-1)} aria-label="Back one frame" className={button}>
        <svg viewBox="0 0 24 24" className={icon} fill="currentColor" aria-hidden="true">
          <path d="M6 5h2v14H6zM9 12l10 7V5z" />
        </svg>
      </button>
      <button type="button" onClick={() => p.onStep(1)} aria-label="Forward one frame" className={button}>
        <svg viewBox="0 0 24 24" className={icon} fill="currentColor" aria-hidden="true">
          <path d="M16 5h2v14h-2zM15 12 5 5v14z" />
        </svg>
      </button>
      <input
        type="range"
        aria-label="Position in the clip"
        min={0}
        max={last}
        step={1}
        value={Math.min(p.frame, last)}
        onChange={(e) => p.onSeek(Number(e.target.value))}
        className={`h-1.5 cursor-pointer accent-sky-400 ${overlay ? 'order-first basis-full' : 'mx-1 min-w-24 flex-1'}`}
      />
      <span className="mr-auto shrink-0 pl-1 text-xs tabular-nums text-slate-200">
        {clock(p.frame / p.fps)} / {clock(last / p.fps)}
      </span>
      <select
        aria-label="Playback speed"
        value={p.speed}
        onChange={(e) => p.onSpeed(Number(e.target.value))}
        className="h-8 shrink-0 rounded-lg border border-white/10 bg-slate-950/70 px-1.5 text-xs text-slate-100"
      >
        {SPEEDS.map((s) => (
          <option key={s} value={s}>
            {s}×
          </option>
        ))}
      </select>
      <button
        type="button"
        onClick={p.onLoop}
        aria-label="Loop"
        aria-pressed={p.loop}
        className={`${button} ${p.loop ? 'bg-sky-500/30 text-sky-200' : ''}`}
      >
        <svg viewBox="0 0 24 24" className={icon} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M17 2l4 4-4 4M3 11V9a3 3 0 0 1 3-3h15M7 22l-4-4 4-4M21 13v2a3 3 0 0 1-3 3H3" />
        </svg>
      </button>
      <button type="button" onClick={p.onFullscreen} aria-label="Full screen" className={button}>
        <svg viewBox="0 0 24 24" className={icon} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
          <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
        </svg>
      </button>
    </div>
  )
}
