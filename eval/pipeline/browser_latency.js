// End to end live latency, measured in the browser (steps 29 and 31). Paste into the DevTools
// console of the Sentinel page with the Live page open (Chrome window in front, not minimised:
// a hidden tab draws only a frame or two per second). Do not switch Open / All streams during a run.
//   await latency.run(3)            -> 3 streams of seq011 at once, every layer box ticked, waits
//                                      for the end, returns the summary over the time all were live
//   await latency.run(4, 'seq023')  -> another stored clip
// Records window.sentinelLatency (capture to drawn) and window.sentinelArrivals (each sample as it
// arrived: ready = model done, received = in the browser). Times are Unix seconds of the backend
// machine, so run the browser on the same machine as the backend.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const q = (xs, p) => xs[Math.min(xs.length - 1, Math.floor(p * xs.length))]
const stats = (v) => {
  v.sort((a, b) => a - b)
  return v.length ? { n: v.length, median: +q(v, 0.5).toFixed(2), p90: +q(v, 0.9).toFixed(2), max: +v.at(-1).toFixed(2) } : { n: 0 }
}
const streams = async () => (await (await fetch('/api/v1/streams')).json()).streams

window.latency = {
  async run(n, clip = 'seq011') {
    const ids = []
    for (let i = 0; i < n; i++) {
      const res = await fetch('/api/v1/streams', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clip_id: clip }),
      })
      ids.push((await res.json()).stream_id)
    }
    for (;;) {
      for (const c of document.querySelectorAll('input[type=checkbox]')) if (!c.checked) c.click()
      const s = (await streams()).filter((x) => ids.includes(x.stream_id))
      if (s.every((x) => x.state === 'done' || x.state === 'failed')) break
      await sleep(300)
    }
    await sleep(2000) // the last samples
    return this.summary(ids)
  },
  async summary(ids) {
    const L = window.sentinelLatency
    const A = window.sentinelArrivals
    const win = ids.map((id) => {
      const t = (L[id] || []).filter((r) => r.layer === 'picture').map((r) => r.tCapture)
      return [Math.min(...t), Math.max(...t)]
    })
    const lo = Math.max(...win.map((w) => w[0]))
    const hi = Math.min(...win.map((w) => w[1]))
    const inside = (t) => t !== null && t >= lo && t <= hi
    const drawn = {}
    for (const r of ids.flatMap((id) => L[id] || [])) if (inside(r.tCapture)) (drawn[r.layer] ??= []).push(r.tDrawn - r.tCapture)
    const ready = {}
    const received = {}
    for (const id of ids) {
      const seen = new Set()
      for (const a of A[id] || []) {
        const key = `${a.layer}|${a.frame}`
        if (seen.has(key) || !inside(a.tCapture)) continue
        seen.add(key)
        ;(ready[a.layer] ??= []).push(a.tReady - a.tCapture)
        ;(received[a.layer] ??= []).push(a.tReceived - a.tCapture)
      }
    }
    const layers = {}
    for (const k of new Set([...Object.keys(drawn), ...Object.keys(received)])) {
      layers[k] = { drawn: stats(drawn[k] || []), received: stats(received[k] || []), ready: stats(ready[k] || []) }
    }
    const status = (await streams()).filter((s) => ids.includes(s.stream_id))
    return JSON.stringify({
      streams: ids.length,
      overlap_s: +(hi - lo).toFixed(1),
      pictures_per_s_per_stream: +((drawn.picture?.length ?? 0) / ids.length / (hi - lo)).toFixed(1),
      layers,
      workers: status.map((s) => ({ id: s.stream_id, ran: s.ran, skipped: s.skipped, late_s: s.late_s, errors: s.errors })),
    })
  },
}
