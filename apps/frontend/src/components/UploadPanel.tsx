import { useEffect, useState, type FormEvent } from 'react'
import { getActiveRun, getRun, startRun, uploadClip } from '../api'
import { cameraForm, EMPTY_CAMERA, type CameraFields } from '../lib/camera'
import { isActive, POLL_MS, runSummary } from '../lib/runs'
import type { Run } from '../types'

type Phase =
  | { kind: 'idle' }
  | { kind: 'uploading'; fraction: number }
  | { kind: 'run'; run: Run }
  | { kind: 'error'; message: string }

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))

/**
 * Upload a video, then run every model on it in the backend (on Kaggle by default).
 * `onUploaded` fires once the clip exists (it plays without layers), `onRunDone` when the run ends.
 */
export function UploadPanel({ onUploaded, onRunDone }: {
  onUploaded: (clipId: string) => void
  onRunDone: (clipId: string) => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [camera, setCamera] = useState<CameraFields>(EMPTY_CAMERA)
  const setCam = (patch: Partial<CameraFields>) => setCamera((c) => ({ ...c, ...patch }))
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' })
  const [open, setOpen] = useState(false)

  const run = phase.kind === 'run' ? phase.run : null
  const busy = phase.kind === 'uploading' || (run !== null && isActive(run))

  // After a reload (or in another tab) follow the run that is still going.
  useEffect(() => {
    const ctl = new AbortController()
    getActiveRun(ctl.signal)
      .then((active) => {
        if (!active) return
        setPhase((p) => (p.kind === 'idle' ? { kind: 'run', run: active } : p))
        setOpen(true)
        onUploaded(active.clip_id) // show the clip whose layers are on their way
      })
      .catch(() => {}) // no backend yet: the clip list shows that error already
    return () => ctl.abort()
  }, [onUploaded])

  useEffect(() => {
    if (!run || !isActive(run)) return
    const ctl = new AbortController()
    const timer = setTimeout(() => {
      getRun(run.clip_id, run.run_id, ctl.signal)
        .then((next) => {
          setPhase({ kind: 'run', run: next })
          if (!isActive(next)) onRunDone(next.clip_id)
        })
        // A missed poll (backend restarting) is retried with the same run.
        .catch(() => !ctl.signal.aborted && setPhase({ kind: 'run', run: { ...run } }))
    }, POLL_MS)
    return () => {
      clearTimeout(timer)
      ctl.abort()
    }
  }, [run, onRunDone])

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!file) return
    setPhase({ kind: 'uploading', fraction: 0 })
    try {
      const clip = await uploadClip(
        file,
        name.trim(),
        (fraction) => setPhase({ kind: 'uploading', fraction }),
        cameraForm(camera),
      )
      onUploaded(clip.clip_id)
      setPhase({ kind: 'run', run: await startRun(clip.clip_id) })
    } catch (err) {
      setPhase({ kind: 'error', message: message(err) })
    }
  }

  return (
    <details className="card mb-5 px-4 py-3" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="cursor-pointer text-sm font-medium text-slate-200">Upload a video and run the models</summary>
      <form onSubmit={submit} className="mt-3 flex flex-wrap items-end gap-3 text-sm">
        <label className="flex flex-col gap-1 text-slate-300">
          Video (mp4, mov, avi, mkv)
          <input
            type="file"
            accept=".mp4,.mov,.avi,.mkv,video/*"
            disabled={busy}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="text-slate-200 file:mr-3 file:rounded-lg file:border-0 file:bg-white/10 file:px-3 file:py-1.5 file:text-slate-100"
          />
        </label>
        <label className="flex flex-col gap-1 text-slate-300">
          Name shown in the viewer (optional)
          <input
            type="text"
            value={name}
            maxLength={100}
            disabled={busy}
            onChange={(e) => setName(e.target.value)}
            placeholder="the file name"
            className="rounded-lg border border-white/10 bg-slate-950/70 px-3 py-1.5 text-slate-100 outline-none focus:border-sky-400"
          />
        </label>
        <fieldset disabled={busy} className="flex basis-full flex-wrap items-end gap-3 border-t border-white/10 pt-3">
          <legend className="basis-full pb-2 text-xs text-slate-400">
            Camera (optional). With the altitude and a field of view, flow is shown in m/s instead of px per frame.
          </legend>
          <label className="flex flex-col gap-1 text-slate-300">
            Altitude (m)
            <input
              type="number"
              min="1"
              max="1000"
              step="any"
              value={camera.altitude_m}
              onChange={(e) => setCam({ altitude_m: e.target.value })}
              className="w-24 rounded-lg border border-white/10 bg-slate-950/70 px-3 py-1.5 text-slate-100 outline-none focus:border-sky-400"
            />
          </label>
          <label className="flex flex-col gap-1 text-slate-300">
            Altitude is
            <select
              value={camera.altitude_source}
              onChange={(e) => setCam({ altitude_source: e.target.value as CameraFields['altitude_source'] })}
              className="rounded-lg border border-white/10 bg-slate-950/70 px-3 py-1.5 text-slate-100 outline-none focus:border-sky-400"
            >
              <option value="assumed">assumed</option>
              <option value="stated">stated by the pilot</option>
              <option value="telemetry">from the flight log</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-slate-300">
            Field of view (°)
            <input
              type="number"
              min="1"
              max="179"
              step="any"
              value={camera.fov_deg}
              onChange={(e) => setCam({ fov_deg: e.target.value })}
              className="w-24 rounded-lg border border-white/10 bg-slate-950/70 px-3 py-1.5 text-slate-100 outline-none focus:border-sky-400"
            />
          </label>
          <label className="flex flex-col gap-1 text-slate-300">
            Measured
            <select
              value={camera.fov_kind}
              onChange={(e) => setCam({ fov_kind: e.target.value as CameraFields['fov_kind'] })}
              className="rounded-lg border border-white/10 bg-slate-950/70 px-3 py-1.5 text-slate-100 outline-none focus:border-sky-400"
            >
              <option value="horizontal">horizontally</option>
              <option value="diagonal">diagonally (spec sheet)</option>
            </select>
          </label>
          {camera.fov_kind === 'diagonal' && (
            <label className="flex flex-col gap-1 text-slate-300">
              Sensor shape
              <select
                value={camera.sensor_aspect}
                onChange={(e) => setCam({ sensor_aspect: e.target.value })}
                className="rounded-lg border border-white/10 bg-slate-950/70 px-3 py-1.5 text-slate-100 outline-none focus:border-sky-400"
              >
                <option value="4:3">4:3 (most drone cameras)</option>
                <option value="3:2">3:2</option>
                <option value="16:9">16:9</option>
              </select>
            </label>
          )}
          <label className="flex flex-col gap-1 text-slate-300">
            Drone or camera
            <input
              type="text"
              value={camera.camera_model}
              maxLength={100}
              onChange={(e) => setCam({ camera_model: e.target.value })}
              placeholder="e.g. DJI Mini 3"
              className="rounded-lg border border-white/10 bg-slate-950/70 px-3 py-1.5 text-slate-100 outline-none focus:border-sky-400"
            />
          </label>
        </fieldset>
        <button
          type="submit"
          disabled={!file || busy}
          className="rounded-lg bg-sky-500 px-4 py-1.5 font-medium text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Upload and run
        </button>
      </form>

      <div aria-live="polite" className="mt-3 text-sm">
        {phase.kind === 'uploading' &&
          (phase.fraction < 1 ? (
            <p className="text-slate-300">Uploading: {Math.round(phase.fraction * 100)}%</p>
          ) : (
            <p className="text-slate-300">Preparing the video on the server…</p>
          ))}
        {run && (
          <div className={run.state === 'failed' ? 'text-rose-200' : 'text-slate-300'}>
            <p>
              <span className="text-slate-500">{run.clip_id}: </span>
              {runSummary(run)}
            </p>
            {Object.entries(run.errors).map(([module, why]) => (
              <p key={module} className="text-amber-200">
                {module}: {why}
              </p>
            ))}
            {isActive(run) && (
              <p className="text-xs text-slate-500">
                The clip already plays; its layers appear when the run ends. On Kaggle a run waits for a free GPU first.
              </p>
            )}
          </div>
        )}
        {phase.kind === 'error' && <p role="alert" className="text-rose-200">{phase.message}</p>}
      </div>
    </details>
  )
}
