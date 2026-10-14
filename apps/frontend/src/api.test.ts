import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getClip, listClips, UNREACHABLE, videoSrc } from './api'

function respond(status: number, body: unknown, statusText = '') {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: status >= 200 && status < 300,
      status,
      statusText,
      json: async () => {
        if (body === undefined) throw new Error('not json')
        return body
      },
    })),
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('api errors', () => {
  it('returns parsed JSON on success', async () => {
    respond(200, [{ clip_id: 'a' }])
    expect(await listClips()).toEqual([{ clip_id: 'a' }])
  })
  it('shows the backend\'s own detail message for a 404', async () => {
    respond(404, { detail: "no clip 'x'" }, 'Not Found')
    await expect(getClip('x')).rejects.toThrow("404: no clip 'x'")
  })
  it('falls back to the status text when the body is not JSON', async () => {
    respond(500, undefined, 'Server Error')
    await expect(listClips()).rejects.toThrow('500: Server Error')
  })
  it('explains a 502 from the dev proxy as "backend not running"', async () => {
    respond(502, undefined, 'Bad Gateway')
    await expect(listClips()).rejects.toThrow(UNREACHABLE)
  })
  it('explains a network failure the same way', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('failed'))))
    await expect(listClips()).rejects.toBeInstanceOf(ApiError)
    await expect(listClips()).rejects.toThrow(UNREACHABLE)
  })
  it('passes an abort through untouched so callers can ignore it', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new DOMException('x', 'AbortError'))))
    await expect(listClips()).rejects.toMatchObject({ name: 'AbortError' })
  })
  it('encodes ids in URLs', async () => {
    respond(200, {})
    await getClip('a b')
    expect((fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0]).toBe('/api/v1/clips/a%20b')
  })
})

describe('videoSrc', () => {
  const base = { clip_id: 'a', fps: 25, frame_size: [10, 10] as [number, number], n_frames: 1, layers: [] }
  it('is null when the clip has no video', () => {
    expect(videoSrc(base)).toBeNull()
  })
  it('uses the clip\'s video_url', () => {
    expect(videoSrc({ ...base, video_url: '/api/v1/clips/a/video' })).toBe('/api/v1/clips/a/video')
  })
})
