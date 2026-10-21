import type { Clip, ClipSummary, Layer, LiveSample, Run, Stream } from './types'

// Empty by default: in dev the Vite proxy forwards /api to the backend.
const BASE: string = import.meta.env.VITE_API_URL ?? ''

export class ApiError extends Error {}

export const UNREACHABLE =
  'Cannot reach the backend. Is it running? (in apps/backend: python manage.py runserver)'

/** The error to show for a failed response: the backend's own `detail` when it sent one. */
export function errorFor(status: number, statusText: string, body: unknown): ApiError {
  // The dev proxy answers 502 when nothing is listening behind it.
  if (status === 0 || [502, 503, 504].includes(status)) return new ApiError(UNREACHABLE)
  const detail = (body as { detail?: string } | null)?.detail ?? statusText
  return new ApiError(`${status}: ${detail}`)
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, init)
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    throw new ApiError(UNREACHABLE)
  }
  if (!res.ok) {
    let body: unknown = null
    try {
      body = await res.json()
    } catch {
      /* body was not JSON */
    }
    throw errorFor(res.status, res.statusText, body)
  }
  return (await res.json()) as T
}

const getJson = <T>(path: string, signal?: AbortSignal) => request<T>(path, { signal })

export const listClips = (signal?: AbortSignal) => getJson<ClipSummary[]>('/api/v1/clips', signal)

export const getClip = (id: string, signal?: AbortSignal) =>
  getJson<Clip>(`/api/v1/clips/${encodeURIComponent(id)}`, signal)

export const getLayer = (clipId: string, layer: string, signal?: AbortSignal) =>
  getJson<Layer>(`/api/v1/clips/${encodeURIComponent(clipId)}/layers/${encodeURIComponent(layer)}`, signal)

export const videoSrc = (clip: Clip): string | null => (clip.video_url ? `${BASE}${clip.video_url}` : null)

const clipPath = (id: string) => `/api/v1/clips/${encodeURIComponent(id)}`

/**
 * Upload a video as a new clip (the backend turns it into a clip before it answers).
 * XMLHttpRequest, not fetch, because only it reports upload progress: `onProgress(0..1)`.
 */
export function uploadClip(
  file: File,
  name: string,
  onProgress: (fraction: number) => void,
  camera: Record<string, string> = {},
): Promise<Clip> {
  const form = new FormData()
  form.append('file', file)
  if (name) form.append('name', name)
  for (const [k, v] of Object.entries(camera)) form.append(k, v)
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/api/v1/clips`)
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
    xhr.onerror = () => reject(new ApiError(UNREACHABLE))
    xhr.onload = () => {
      let body: unknown = null
      try {
        body = JSON.parse(xhr.responseText)
      } catch {
        /* body was not JSON */
      }
      if (xhr.status === 201) resolve(body as Clip)
      else reject(errorFor(xhr.status, xhr.statusText, body))
    }
    xhr.send(form)
  })
}

/** Start every model on a clip in the background. */
export const startRun = (clipId: string) => request<Run>(`${clipPath(clipId)}/runs`, { method: 'POST' })

/** The run going now on the server (any clip), so a reloaded page can follow it again. */
export const getActiveRun = (signal?: AbortSignal) =>
  getJson<{ run: Run | null }>('/api/v1/runs/active', signal).then((r) => r.run)

export const getRun = (clipId: string, runId: string, signal?: AbortSignal) =>
  getJson<Run>(`${clipPath(clipId)}/runs/${encodeURIComponent(runId)}`, signal)

const streamPath = (id: string) => `/api/v1/streams/${encodeURIComponent(id)}`
const postJson = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const listStreams = (signal?: AbortSignal) =>
  getJson<{ streams: Stream[] }>('/api/v1/streams', signal).then((r) => r.streams)

/** Replay a stored clip as a live stream (as if a drone sent it), with every model. */
export const startStream = (clipId: string) => postJson<Stream>('/api/v1/streams', { clip_id: clipId })

export const stopStream = (id: string) => postJson<Stream>(`${streamPath(id)}/stop`, {})

/** Samples written after byte offset `after`; ask again with `next`. */
export const getSamples = (id: string, after: number, signal?: AbortSignal) =>
  getJson<{ samples: LiveSample[]; next: number; state: Stream['state'] }>(`${streamPath(id)}/samples?after=${after}`, signal)

/** The live picture: MJPEG with X-Frame-Id and X-Capture-Time on every part. */
export const streamVideoUrl = (id: string) => `${BASE}${streamPath(id)}/video`
