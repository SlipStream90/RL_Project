"""Run one live 100-step NEXUS episode against the running backend (:8000).

Saves the full recorded run to results/live_demo_100.json.
Every number comes from the API (Contract: price/qty from state, never
decoded from the action array). Safe to re-run; overwrites the file.
"""
import json
import urllib.request
from pathlib import Path

API = "http://localhost:8000"
SEED = 42
MAX_STEPS = 100
OUT = Path(__file__).resolve().parent.parent / "results" / "live_demo_100.json"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    health = call("GET", "/health")
    assert health.get("status") == "ok", f"backend unhealthy: {health}"

    start = call("POST", "/api/v1/simulation/start",
                 {"seed": SEED, "max_steps": MAX_STEPS})
    episode_id = start["episode_id"]
    state = start["state"]
    print(f"episode {episode_id} started (seed {SEED}, max_steps {MAX_STEPS})")

    turns = []
    totals = {}
    done = False
    while not done and state["step"] < MAX_STEPS:
        r = call("POST", f"/api/v1/simulation/{episode_id}/step", {})
        state = r["state"]
        done = bool(r.get("done"))
        if r.get("agent") is not None:
            totals[r["agent"]] = totals.get(r["agent"], 0.0) + float(r.get("reward") or 0.0)
            turns.append({
                "step": state["step"],
                "agent": r["agent"],
                "action": r.get("action"),
                "reward": r.get("reward"),
                "active_offer": state.get("active_offer"),
                "deals_so_far": len(state.get("deals") or []),
                "next": state.get("agent_selection"),
            })
        if state["step"] % 20 == 0:
            print(f"  step {state['step']}/{MAX_STEPS} deals={len(state.get('deals') or [])}")

    result = {
        "source": "live API run via scripts/run_live_episode.py",
        "api": API, "seed": SEED, "max_steps": MAX_STEPS,
        "episode_id": episode_id, "done": done,
        "final_step": state["step"], "turns_recorded": len(turns),
        "deals": state.get("deals") or [],
        "total_reward_per_agent": {k: round(v, 4) for k, v in sorted(totals.items())},
        "final_budgets": state.get("budgets"),
        "final_inventories": state.get("inventories"),
        "turns": turns,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"done={done} final_step={state['step']} deals={len(result['deals'])}")
    print(f"total_reward_per_agent={result['total_reward_per_agent']}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
