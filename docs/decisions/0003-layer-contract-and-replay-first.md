# 0003 Layer contract, replay first

Date: 2026-10-11   Owner: Mahid
Context: the Tier 1 demo needs a frontend that plays video with toggleable model layers. No
backend or frontend exists yet and no teammate has started them, so the shape of the data between
models and frontend needs agreeing before anything is built.

Decision:
- Every model output is a **layer** of one of three types: `grid`, `points`, `tracks`. The frontend has
  three renderers; adding a model needs no frontend work.
- **Replay first, live later.** The demo plays recorded clips with precomputed layers. Clips and
  live drone streams share one data shape, so live is added with a WebSocket later.
- **Image-space grids for now,** ground cells once geometry exists.
- **Storage:** JSON files plus a small Django model for clip metadata. No Postgres need for the demo.
- **Stack:** stay with Django + DRF + Channels as already chosen.
- Reuse the conventions of `docs/contracts.md` (original pixels, string frame keys); extend with the
  grid layer and API in `docs/layers/`.

Evidence: design discussion on 2026-10-11. `docs/layers/validate.py` accepts a generated clip and
rejects eight kinds of malformed data (wrong matrix size, missing mask, frame beyond the clip,
non-numeric frame key, point outside the frame, score above 1, duplicate track id, unknown type).

Alternatives & why not: one bespoke format per model (every new model means new frontend code);
live streaming first (more moving parts than the deadline allows); ground cells now (needs the
geometry that does not exist yet).

Open issues (need the team's decision): Ahmed to approve the extension of `docs/contracts.md`; the
fifth model's output; flow grid size; how the video is served.
