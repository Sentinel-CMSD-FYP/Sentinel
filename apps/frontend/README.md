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

1. Lists clips from `GET /api/v1/clips` and loads the one you pick.
2. Downloads every layer the clip declares, once.
3. Plays the video. Whenever a frame is shown it works out the frame number
   (`round(time * fps)`) and draws each visible layer's sample for that frame on a canvas laid
   exactly over the video.

Four renderers cover every layer, as in `docs/layers/README.md`:

| layer type | drawn as |
|---|---|
| `grid` with `u` and `v` fields | one arrow per valid cell, length proportional to speed (flow) |
| `grid` otherwise | heatmap, colour scaled to the layer's maximum (density) |
| `points` | dots, more opaque for higher score |
| `tracks` | coloured dots with a short trail per track id |
| `polygons` | lightly filled outlines, one colour per class (order of the layer's `classes`); the layer panel lists objects per class |

A layer of any other type is listed but draws nothing, so a newer contract cannot break the page.

The `style` hint in `clip.json` can override the grid choice (`{"render": "heatmap", "field": "..."}`).

**Gaps stay gaps.** A sample is shown only for `max_age_frames` frames. After that nothing is drawn,
and the layer panel says "no data at this moment". Cells with `valid: 0` are never drawn, and values
are never interpolated or carried forward.

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
