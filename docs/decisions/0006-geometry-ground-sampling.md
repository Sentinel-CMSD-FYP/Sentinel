# 0006 Geometry: ground sampling distance and flow in metres per second

Date: 2026-10-17   Owner: Mahid
Context: sprint tasks 4, 7 and 11. Flow is in pixels per frame, but a pixel covers more ground the
higher the drone is, and the risk thresholds are physical (normal walking is about 1.2 m/s). The
proposal says flow is stored in metres per second so views from different altitudes compare.

Decision:
- `sentinel_ml.geometry` converts pixels to metres for a **straight-down camera over flat ground**:
  ground width = 2 * altitude * tan(horizontal FOV / 2), GSD = ground width / image width in pixels.
- Altitude and horizontal field of view are **explicit inputs with no hidden defaults**. No dataset
  used so far (DroneCrowd, the YouTube clip) carries either, so any m/s figure from them rests on
  assumed values and says so: the exporter records `altitude_m`, `hfov_deg` and
  `ground_sampling_m_per_px` under `produced_by` in `clip.json`.
- The exporter writes `units: "m_per_s"` only when it is given both `--altitude` and `--hfov`;
  otherwise it stays in `px_per_frame`. The GSD uses the width of the video it wrote, which keeps
  the field of view when the video is shrunk.
- `flow.ground_flow.get_flow(frame_a, frame_b, altitude_m, hfov_deg, fps)` is the one-call form:
  a `rows x cols` grid of vx, vy, speed and speed variance in m/s. Camera-motion subtraction is
  **off** by default there (frames are assumed registered) and **on** by default in the exporter.

Evidence: 32 new tests, 79 in total in `ml/`, all passing. The GSD formula is checked against an
independent route (pinhole focal length in pixels, `altitude / fx`) and agrees to rounding. A car of
4.5 m x 1.8 m drawn as a camera at 60 m would see it measures back to within 2 to 3% (the limit is
whole-pixel rounding). A 1.2 m/s walker comes out at 0.57 px/frame at 25 fps, 60 m, 84 degrees,
1280 px. Doubling altitude doubles every m/s value, as it should. On the 15 s market clip with
assumed 60 m and 84 degrees the longest cell-mean arrow is 0.18 m/s.

Alternatives & why not: a ground homography for an angled camera (needed eventually, since the GSD
then varies across the image; not built); a per-frame field of view for zooming clips (the market
clip zooms 4.6x over its 53 s, so a single field of view is wrong for it; its first 15 s are within
1.3x); default altitude and field of view values (they would make assumptions look like data).

Open issues (need the team's decision):
1. Real altitude and field of view for any clip we show. Ahmed's tracker assumes 60 m; use the same
   number unless a clip has telemetry.
2. Sprint task 10 (validate on own 40 to 80 m footage) is not possible without such footage; mark
   N/A with this note unless footage is collected.
3. Angled cameras and zoom need the homography and per-frame field of view above.
4. Decision numbers clash: Ahmed's branch also has a 0003 (`env-setup`) next to
   `0003-layer-contract-and-replay-first`; one needs renumbering before the merge.
