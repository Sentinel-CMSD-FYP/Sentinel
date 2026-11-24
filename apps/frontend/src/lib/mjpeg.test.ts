import { describe, expect, it } from 'vitest'
import { MjpegParser } from './mjpeg'

const enc = new TextEncoder()

/** One part as the backend writes it (views.stream_video). */
function part(frame: number, jpeg: number[]): Uint8Array {
  const head = enc.encode(
    `--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.length}\r\nX-Frame-Id: ${frame}\r\nX-Capture-Time: ${1000 + frame}.250\r\n\r\n`,
  )
  const out = new Uint8Array(head.length + jpeg.length + 2)
  out.set(head)
  out.set(jpeg, head.length)
  out.set([13, 10], head.length + jpeg.length)
  return out
}

describe('MjpegParser', () => {
  it('cuts parts by their length, even when the JPEG holds the head end bytes', () => {
    const body = new Uint8Array([...part(0, [0xff, 0xd8, 13, 10, 13, 10, 0xff, 0xd9]), ...part(1, [1, 2, 3])])
    const parts = new MjpegParser().push(body)
    expect(parts.map((p) => [p.frame, p.tCapture, [...p.jpeg]])).toEqual([
      [0, 1000.25, [0xff, 0xd8, 13, 10, 13, 10, 0xff, 0xd9]],
      [1, 1001.25, [1, 2, 3]],
    ])
  })

  it('waits for the rest of a part split across chunks', () => {
    const body = new Uint8Array([...part(7, [9, 9, 9, 9]), ...part(8, [5])])
    const parser = new MjpegParser()
    const got = []
    for (let i = 0; i < body.length; i += 5) got.push(...parser.push(body.subarray(i, i + 5)))
    expect(got.map((p) => [p.frame, [...p.jpeg]])).toEqual([[7, [9, 9, 9, 9]], [8, [5]]])
  })

  it('refuses a part without a length', () => {
    expect(() => new MjpegParser().push(enc.encode('--frame\r\nContent-Type: image/jpeg\r\n\r\nxx'))).toThrow(/Content-Length/)
  })
})
