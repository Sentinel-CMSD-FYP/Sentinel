# Sentinel frontend

React + Vite + TypeScript + Tailwind. A viewer that plays a clip's video and draws the model layers
on top, each with an on/off switch. It only reads from the backend; it has no model logic.

## Run it

Two terminals, backend first:

```powershell
# 1. backend (from the repo root)
cd apps/backend
.\.venv\Scripts\python.exe manage.py runserver

# 2. frontend
cd apps/frontend
npm install
npm run dev
```

Open http://localhost:5173. If no clips show up, make the fake one:

```powershell
python docs/layers/make_sample_clip.py data/samples/demo_fake
```

In dev, Vite forwards `/api` to `http://127.0.0.1:8000` (override with `VITE_BACKEND_URL`), so the
browser sees one origin. For a build served from elsewhere, set `VITE_API_URL` to the backend's URL.

## What it does

1. Lists clips from `GET /api/v1/clips` and loads the one you pick. The "Upload a video and run
   the models" panel uploads a video (`POST /api/v1/clips`, with upload progress), selects the new
   clip at once, starts a run (`POST .../runs`) and polls its status every 3 s; when it ends, the clip
   is read again so its layers appear. Module failures and run errors are shown in the panel.
   Optional camera fields (altitude and its source, field of view horizontal or diagonal with the
   sensor shape, drone model; `lib/camera.ts`) go with the upload; with them flow is in m/s, and
   the clip's info row shows e.g. "60 m (assumed), HFOV 71.5°".
2. Downloads every layer the clip declares, once.
3. Plays the video. Whenever a frame is shown it works out the frame number
   (`round(time * fps)`) and draws each visible layer's sample for that frame on a canvas laid
   exactly over the video.

Four renderers cover every layer, as in `docs/layers/README.md`:

| layer type | drawn as |
|---|---|
| `grid` with `u` and `v` fields | one arrow per valid cell, length proportional to speed (flow); a full length arrow is the clip's peak speed but at least 0.5 px/frame (or 0.5 m/s), so a near still crowd does not draw noise at full length |
| `grid` otherwise | heatmap, colour scaled to the layer's maximum (density); half opacity when other layers share the panel |
| `points` | dots, more opaque for higher score; the layer panel shows the count at this frame |
| `tracks` | coloured dots with a short trail per track id; the layer panel shows the count at this frame |
| `polygons` | lightly filled outlines, one colour per class (order of the layer's `classes`); the layer panel lists objects per class; hovering an object outlines it and shows its class and score |

Dots of points and tracks are 4 px in a full size panel and shrink to 2 px in small split panels.

A layer of any other type is listed but draws nothing, so a newer contract cannot break the page.

The `style` hint in `clip.json` can override the grid choice (`{"render": "heatmap", "field": "..."}`).

**Gaps stay gaps.** A sample is shown only for `max_age_frames` frames. After that nothing is drawn,
and the layer panel says "no data at this moment". Cells with `valid: 0` are never drawn, and values
are never interpolated or carried forward.

## Live page

The "Live" button in the header switches to running streams (`components/LiveView.tsx`,
`LiveStage.tsx`, `lib/live.ts`, `lib/mjpeg.ts`):

* "Go live with this clip" replays the clip picked in the header as a stream (`POST /streams`,
  every model). Every running stream shows as a tile with its combined panel (1 to 4 side by side);
  "Open" shows one stream alone with the single and split views, "Stop" ends it. A stream that ends
  stays until "Close" and names the clip it was saved as; the clip list is read again.
* Picture: the MJPEG stream (`GET /streams/<id>/video`) is read with `fetch` and cut into JPEGs by
  their Content-Length, so each picture keeps its `X-Frame-Id` and `X-Capture-Time`. JPEGs are
  decoded one at a time; while one decodes only the newest waits.
* Samples: polled every 250 ms from byte offset 0 (`GET /streams/<id>/samples?after=`); a layer's
  first sample carries its header and its clip.json entry (`meta`: label, style, default
  visibility). The last 250 frames of samples are kept per layer.
* Drawing: the same renderers as a stored clip. Each layer shows its newest sample (drawn as of its
  own frame) while it was captured less than `max_age_frames` (as seconds) plus 3 s ago on the
  wall clock; after that it is a gap ("no data at this moment"). The layer boxes apply to every
  stream.
* Latency: under each stream, capture to drawn per layer (last, median). Every record is on
  `window.sentinelLatency` (drawn) and `window.sentinelArrivals` (every sample as it arrived), which
  `runs/step29/measure.js` summarises. The tab must be in front: Chrome throttles a background or
  minimised tab to about 1 frame/s.

## Checks (what CI runs)

```powershell
npm ci
npm run lint
npm test
npm run build
```

`src/lib/` holds the logic worth testing (frame lookup and the gap rule, colour ramp, grid scales,
trails). The canvas drawing itself is checked by looking at it.

The video must be H.264; browsers refuse anything else (see `docs/layers/README.md`).
