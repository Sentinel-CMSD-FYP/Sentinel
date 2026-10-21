import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, errorFor, getActiveRun, getClip, getRun, listClips, startRun, UNREACHABLE, uploadClip, videoSrc } from './api'

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

/** Stands in for XMLHttpRequest: answers `status` with `body` once sent, after one progress event. */
function fakeXhr(status: number, body: string, statusText = '') {
  const sent: { method?: string; url?: string; form?: FormData } = {}
  class FakeXhr {
    status = 0
    statusText = ''
    responseText = ''
    upload: { onprogress?: (e: { lengthComputable: boolean; loaded: number; total: number }) => void } = {}
    onload?: () => void
    onerror?: () => void
    open(method: string, url: string) {
      Object.assign(sent, { method, url })
    }
    send(form: FormData) {
      sent.form = form
      this.upload.onprogress?.({ lengthComputable: true, loaded: 5, total: 10 })
      if (status === 0) return this.onerror?.()
      Object.assign(this, { status, statusText, responseText: body })
      this.onload?.()
    }
  }
  vi.stubGlobal('XMLHttpRequest', FakeXhr)
  return sent
}

describe('upload and runs', () => {
  const file = new File(['x'], 'my clip.mp4')

  it('uploads the file and name as a form, reporting progress', async () => {
    const sent = fakeXhr(201, JSON.stringify({ clip_id: 'my_clip' }))
    const seen: number[] = []
    expect(await uploadClip(file, 'Field test', (f) => seen.push(f))).toEqual({ clip_id: 'my_clip' })
    expect(sent.method).toBe('POST')
    expect(sent.url).toBe('/api/v1/clips')
    expect((sent.form!.get('file') as File).name).toBe('my clip.mp4')
    expect(sent.form?.get('name')).toBe('Field test')
    expect(seen).toEqual([0.5])
  })
  it('adds the camera fields to the form', async () => {
    const sent = fakeXhr(201, '{}')
    await uploadClip(file, '', () => {}, { altitude_m: '60', fov_deg: '71.5' })
    expect(sent.form?.get('altitude_m')).toBe('60')
    expect(sent.form?.get('fov_deg')).toBe('71.5')
  })
  it('leaves the name out when it is empty', async () => {
    const sent = fakeXhr(201, '{}')
    await uploadClip(file, '', () => {})
    expect(sent.form?.has('name')).toBe(false)
  })
  it("shows the backend's reason for a refused upload", async () => {
    fakeXhr(413, JSON.stringify({ detail: 'the file is over the 1024 MB limit' }), 'Too Large')
    await expect(uploadClip(file, '', () => {})).rejects.toThrow('413: the file is over the 1024 MB limit')
  })
  it('explains a network failure during an upload as "backend not running"', async () => {
    fakeXhr(0, '')
    await expect(uploadClip(file, '', () => {})).rejects.toThrow(UNREACHABLE)
  })
  it('starts a run with a POST and reads it back by id', async () => {
    respond(202, { run_id: 'r 1', state: 'queued' })
    await startRun('my_clip')
    await getRun('my_clip', 'r 1')
    const calls = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls
    expect(calls[0][0]).toBe('/api/v1/clips/my_clip/runs')
    expect(calls[0][1]).toMatchObject({ method: 'POST' })
    expect(calls[1][0]).toBe('/api/v1/clips/my_clip/runs/r%201')
  })
  it('reads the active run, or null when nothing runs', async () => {
    respond(200, { run: null })
    expect(await getActiveRun()).toBeNull()
    respond(200, { run: { run_id: 'r' } })
    expect(await getActiveRun()).toEqual({ run_id: 'r' })
  })
  it("shows the backend's reason when a run cannot start", async () => {
    respond(409, { detail: 'run x is still going; one run at a time' }, 'Conflict')
    await expect(startRun('a')).rejects.toThrow('409: run x is still going; one run at a time')
  })
  it('builds errors from the status, the detail or the status text', () => {
    expect(errorFor(502, 'Bad Gateway', null).message).toBe(UNREACHABLE)
    expect(errorFor(400, 'Bad Request', { detail: 'no' }).message).toBe('400: no')
    expect(errorFor(500, 'Server Error', null).message).toBe('500: Server Error')
  })
})
