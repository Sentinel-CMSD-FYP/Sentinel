# Project Sentinel

Multi-drone crowd crush and stampede early-warning system (decision support, no physical intervention).
FYP, FAST-NUCES Islamabad.

> Validated with public datasets and simulation, not real Hajj operational data.

## Layout

| Path | Purpose |
|---|---|
| `apps/frontend` | React + Vite + Tailwind dashboard |
| `apps/backend` | Django + DRF + Channels API |
| `ml` | Perception, risk scoring, LLM explainer (`sentinel_ml`) |
| `simulation` | Gazebo (gz-sim) + PX4 SITL + pedsim_ros |
| `eval` | Numeric targets and metric scripts |
| `data` | Gitignored; manifests only |
| `docs/decisions` | Decisions log (one file per decision) |
| `infra` | Docker Compose for dev services |

## Quick start

```bash
docker compose -f infra/docker-compose.yml up -d   # postgres + redis

# backend
cd apps/backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python manage.py migrate && python manage.py runserver

# frontend
cd apps/frontend && npm ci && npm run dev
```

## Workflow

- `main` is protected: PR + green CI + 1 review.
- Branches: `feat/<area>-<name>`, `fix/<area>-<name>`.
- Never commit data, weights or `.env` files.
- Hard Rule: no Tier 3 work until Tier 1 and 2 meet `eval/targets.yaml`.