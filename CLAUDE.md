# NEXUS — Multi-Agent Negotiation & Explainability Platform

NEXUS is a software-only research platform that simulates a 5-agent marketplace (2 buyers, 2 sellers, 1 market manager) where agents negotiate via learned RL policies (MAPPO / IPPO), with a BERT-based natural-language interface and an explainability (XAI) engine. **No Docker required** — everything runs as local processes.

## Stack
- **Backend:** FastAPI + uvicorn (Python 3.12, PyTorch, PettingZoo/Gymnasium) — port 8000
- **Frontend:** Next.js 15 + React 19 — port 3000
- **RL:** PyTorch MAPPO (CTDE) + Independent PPO baselines
- **NLP:** DistilBERT intent classification / slot extraction
- **XAI:** Trajectory recording, reward attribution, counterfactuals, policy diffs

## Repo layout (post-cleanup)

```
CLAUDE.md              # this file
CONTRACT.md            # frozen integration contract (API/env/RL/NLP/XAI schemas)
NEXUS_PRD.md           # main product PRD (architecture, roadmap, API design)
NEXUS_ENV_SPEC.md      # math environment spec (state/action/reward, MAPPO)
pyproject.toml         # nexus package (src layout, pytest config)
requirements.txt       # Python deps
config/default.yaml    # default env config
scripts/
  setup_demo.ps1       # one-time: venv + pip install + npm install + build
  run_demo.ps1         # start FastAPI (:8000) + Next.js (:3000), open dashboard
  demo_script.py       # backend-only smoke demo (TestClient)
  run_full_gates.py    # full experiment gates (5000-ep runs)
src/nexus/             # api, env, experiments, nlp, rl, xai
tests/                 # 7 pytest files
frontend/              # Next.js app (built; node_modules gitignored)
results/               # JSON evidence + checkpoints + DistilBERT model (gitignored logs/binaries)
.venv/                 # Python venv (gitignored)
```

## How to run (Windows, no Docker)

Prerequisites: Python 3.12 (`py -3.12`), Node 20+ (`npm`).

```powershell
# One-time setup
.\scripts\setup_demo.ps1

# Start the demo (backend :8000 + frontend :3000, opens dashboard)
.\scripts\run_demo.ps1

# Backend-only smoke demo (no browser)
.\.venv\Scripts\python.exe scripts\demo_script.py
```

Endpoints:
- `GET http://localhost:8000/health`
- `POST /api/v1/simulation/start` (json: `{seed, max_steps}`)
- `POST /api/v1/simulation/{id}/step`
- `POST /api/v1/query` (json: `{question, episode_id, agent_id}`) — evidence-backed XAI
- Dashboard: `http://localhost:3000`

## Guidance for Claude (do not change)

- **Do not re-add Docker** (`docker-compose.yml` was removed intentionally). Local processes are the verified equivalent.
- **Do not re-add junk stubs** — the weekly `PRD_Week*.md` files, `RESULTS.md`, LaTeX report artifacts (`NEXUS_Report.*`), and runtime logs were removed. Keep `results/*.json` evidence files; `results/*.log` and `results/checkpoints/` are gitignored.
- `.gitignore` is already correct (Python, venv, Node, results logs/binaries, LaTeX, editors, OS files).
- The single source of truth for API/env/RL/NLP/XAI schemas is `CONTRACT.md` — changes require a contract bump.
- `demo_script.py` still references the deleted `RESULTS.md` in a comment (line 3) — harmless, do not need to fix unless touching that file.
- The 4 `results/*.log` files may be locked by a running server process; they are gitignored via `results/*.log` and harmless locally.

## Live-eval demo script (3 min)
1. Open dashboard at `localhost:3000` (5 sections: Marketplace, Agent Profiles, Negotiation Replay, Explainability, Experiment Lab).
2. Start a live negotiation: `POST /api/v1/simulation/start` with `seed: 42`, then step through turns.
3. Ask an explainability question: `"Why did Buyer A reject Seller A's first offer?"` — answer is grounded in trajectory data, not LLM generation.
4. Show results evidence: `results/summary.json` (MAPPO wins 4/5 metrics over 5,000 eps) and `results/bert_metrics_full.json` (100% intent accuracy, 21 ms/utterance).