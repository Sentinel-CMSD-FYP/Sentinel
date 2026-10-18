import type { Clip, ClipSummary, Layer } from './types'

// Empty by default: in dev the Vite proxy forwards /api to the backend.
const BASE: string = import.meta.env.VITE_API_URL ?? ''

export class ApiError extends Error {}

export const UNREACHABLE =
  'Cannot reach the backend. Is it running? (in apps/backend: python manage.py runserver)'

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, { signal })
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    throw new ApiError(UNREACHABLE)
  }
  if (!res.ok) {
    // The dev proxy answers 502 when nothing is listening behind it.
    if ([502, 503, 504].includes(res.status)) throw new ApiError(UNREACHABLE)
    let detail = res.statusText
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? detail
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(`${res.status}: ${detail}`)
  }
  return (await res.json()) as T
}

export const listClips = (signal?: AbortSignal) => getJson<ClipSummary[]>('/api/v1/clips', signal)

export const getClip = (id: string, signal?: AbortSignal) =>
  getJson<Clip>(`/api/v1/clips/${encodeURIComponent(id)}`, signal)

export const getLayer = (clipId: string, layer: string, signal?: AbortSignal) =>
  getJson<Layer>(`/api/v1/clips/${encodeURIComponent(clipId)}/layers/${encodeURIComponent(layer)}`, signal)

export const videoSrc = (clip: Clip): string | null => (clip.video_url ? `${BASE}${clip.video_url}` : null)
