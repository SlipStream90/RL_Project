# NEXUS — Multi-Agent Negotiation & Explainability Platform

NEXUS simulates a five-agent marketplace (2 buyers, 2 sellers, 1 market manager)
where agents negotiate via learned RL policies (MAPPO / IPPO), with a
DistilBERT natural-language layer and an evidence-grounded explainability (XAI)
engine. No Docker — everything runs as local processes on Windows.

- **Backend:** FastAPI + uvicorn (Python 3.12, PyTorch, PettingZoo/Gymnasium) — port 8000
- **Frontend:** Next.js 15 + React 19 — port 3000
- **Live demo policies:** rule-based baselines (sellers concede toward their floor,
  buyers raise bids over time; opposite-role accepts only). Trained MAPPO/IPPO
  experiment numbers live in the Experiment Lab, read from `results/summary.json`.

## Quick start (live demonstration)

Double-click `launch_demo.bat`, or run it from a terminal:

```powershell
.\launch_demo.bat
```

It builds the frontend if needed, starts the backend (:8000) and frontend
(:3000) in minimized windows, opens http://localhost:3000/marketplace in your
browser, and runs a real 100-step episode (seed 42) against the live API,
saving the recording to `results/live_demo_100.json`.

## Manual setup and run

Prerequisites: Python 3.12 (`py -3.12`), Node 20+ (`npm`).

```powershell
# One-time setup (venv + pip install + npm install + build)
.\scripts\setup_demo.ps1

# Start backend + frontend and open the dashboard
.\scripts\run_demo.ps1

# Backend-only smoke demo (no browser)
.\.venv\Scripts\python.exe scripts\demo_script.py
```

Endpoints:

- `GET  http://localhost:8000/health`
- `POST /api/v1/simulation/start` — `{seed, max_steps}` → `{episode_id, state}`
- `POST /api/v1/simulation/{id}/step` — step the live episode
- `POST /api/v1/query` — `{question, episode_id, agent_id}` → evidence-backed XAI answer
- Dashboard: http://localhost:3000

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest          # backend (env, RL, NLP, XAI, API)
cd frontend; npx playwright test --project=chromium   # frontend E2E (Playwright)
```

## Layout

- `CONTRACT.md` — frozen integration contract (API / env / RL / NLP / XAI schemas)
- `NEXUS_PRD.md`, `NEXUS_ENV_SPEC.md` — product and environment specifications
- `src/nexus/` — `api`, `env`, `experiments`, `nlp`, `rl`, `xai`
- `frontend/` — Next.js dashboard (Marketplace, Agent Profiles, Negotiation
  Replay, Explainability, Experiment Lab, How it works)
- `results/` — JSON evidence from experiment runs (logs and checkpoints are gitignored)
- `scripts/` — setup, demo launchers, and the live-episode runner
