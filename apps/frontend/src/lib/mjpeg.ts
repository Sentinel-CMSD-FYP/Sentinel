/** One picture of the live MJPEG stream, with the frame id and capture time the backend sent. */
export interface MjpegPart {
  frame: number
  /** Unix seconds, on the backend's clock. */
  tCapture: number
  jpeg: Uint8Array<ArrayBuffer>
}

const HEAD_END = [13, 10, 13, 10] // \r\n\r\n

function indexOf(buf: Uint8Array, seq: number[]): number {
  outer: for (let i = 0; i + seq.length <= buf.length; i++) {
    for (let j = 0; j < seq.length; j++) if (buf[i + j] !== seq[j]) continue outer
    return i
  }
  return -1
}

/**
 * Splits `multipart/x-mixed-replace` bytes into JPEGs. The backend sends a Content-Length per
 * part, so a part is cut by its length, never by searching the JPEG for the boundary.
 */
export class MjpegParser {
  private buf = new Uint8Array(0)
  private readonly decoder = new TextDecoder()

  push(chunk: Uint8Array): MjpegPart[] {
    const joined = new Uint8Array(this.buf.length + chunk.length)
    joined.set(this.buf)
    joined.set(chunk, this.buf.length)
    this.buf = joined
    const parts: MjpegPart[] = []
    for (;;) {
      const end = indexOf(this.buf, HEAD_END)
      if (end < 0) break
      const headers: Record<string, string> = {}
      for (const line of this.decoder.decode(this.buf.subarray(0, end)).split('\r\n')) {
        const at = line.indexOf(':')
        if (at > 0) headers[line.slice(0, at).trim().toLowerCase()] = line.slice(at + 1).trim()
      }
      const n = Number(headers['content-length'])
      if (!Number.isInteger(n) || n < 0) throw new Error('MJPEG part without a Content-Length')
      const start = end + HEAD_END.length
      if (this.buf.length < start + n) break // the rest of this JPEG is still on its way
      parts.push({
        frame: Number(headers['x-frame-id']),
        tCapture: Number(headers['x-capture-time']),
        jpeg: this.buf.slice(start, start + n),
      })
      this.buf = this.buf.slice(start + n) // the \r\n after the JPEG starts the next head
    }
    return parts
  }
}

/** Read an MJPEG response to its end, handing over each picture as it arrives. */
export async function readMjpeg(url: string, onPart: (p: MjpegPart) => void, signal: AbortSignal): Promise<void> {
  const res = await fetch(url, { signal })
  if (!res.ok || !res.body) throw new Error(`live picture: ${res.status} ${res.statusText}`)
  const reader = res.body.getReader()
  const parser = new MjpegParser()
  for (;;) {
    const { done, value } = await reader.read()
    if (done) return
    for (const part of parser.push(value)) onPart(part)
  }
}
