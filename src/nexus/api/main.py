"""NEXUS FastAPI backend. Explanations format recorded evidence only."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nexus.env import NexusMarketplaceEnv
from nexus.env.baselines import baseline_policy
from nexus.env.types import AGENTS, band_to_price, default_agent_configs
from nexus.xai import TrajectoryRecorder, detect_exploits, reward_attribution

app = FastAPI(title="NEXUS", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"])

_sessions: dict[str, NexusMarketplaceEnv] = {}
_recorder = TrajectoryRecorder()
_policies: dict[str, dict] = {}


class StartReq(BaseModel):
    seed: int = 42
    max_steps: int = 100


class StepReq(BaseModel):
    action: list[int] | None = None  # None -> baseline policy acts


class QueryReq(BaseModel):
    question: str
    episode_id: str | None = None
    agent_id: str | None = None


class CounterReq(BaseModel):
    episode_id: str
    step_id: int
    alternate_action: list[int]


def _get_env(episode_id: str) -> NexusMarketplaceEnv:
    env = _sessions.get(episode_id)
    if env is None:
        raise HTTPException(404, f"episode {episode_id} not found")
    return env


def _state_payload(env: NexusMarketplaceEnv) -> dict[str, Any]:
    offer = None
    if env._active_offer:
        o, p, q = env._active_offer
        offer = {"offerer": o, "price": p, "quantity": q}
    return {"episode_id": env.current_episode_id, "step": env._step_count,
            "agents": list(env.agents), "agent_selection": env.agent_selection if env.agents else None,
            "active_offer": offer, "deals": env._deals,
            "budgets": dict(env._budgets), "inventories": dict(env._inventories)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/simulation/start")
def start(req: StartReq):
    env = NexusMarketplaceEnv({"max_steps": req.max_steps})
    env.reset(seed=req.seed)
    _sessions[env.current_episode_id] = env
    _policies[env.current_episode_id] = {a: baseline_policy(a, seed=req.seed) for a in AGENTS}
    return {"episode_id": env.current_episode_id, "state": _state_payload(env)}


@app.post("/api/v1/simulation/{episode_id}/step")
def step(episode_id: str, req: StepReq):
    env = _get_env(episode_id)
    if not env.agents:
        return {"done": True, "state": _state_payload(env)}
    agent = env.agent_selection
    if env.terminations[agent] or env.truncations[agent]:
        env.step(None)
        return {"done": not bool(env.agents), "agent": agent, "state": _state_payload(env)}
    if req.action is None:
        # Use each policy's own seeded RNG/state so behaviour evolves with the
        # negotiation instead of being re-derived from the step count alone.
        action = _policies[episode_id][agent].act(env.observe(agent), env.action_masks(agent), None)
    else:
        action = np.asarray(req.action, dtype=np.int64)
    env.step(action)
    _recorder.record_many([t for t in env.get_trajectory() if _recorder.load(episode_id) == [] or True][-1:])
    return {"done": not bool(env.agents), "agent": agent, "action": [int(x) for x in np.asarray(action).reshape(-1)],
            "reward": float(env.rewards.get(agent, 0.0)), "state": _state_payload(env)}


@app.post("/api/v1/simulation/{episode_id}/reset")
def reset(episode_id: str, req: StartReq):
    env = _get_env(episode_id)
    env.reset(seed=req.seed)
    return {"episode_id": env.current_episode_id, "state": _state_payload(env)}


@app.get("/api/v1/simulation/{episode_id}/state")
def sim_state(episode_id: str):
    return _state_payload(_get_env(episode_id))


@app.get("/api/v1/episodes/{episode_id}/state")
def episode_state(episode_id: str):
    return _state_payload(_get_env(episode_id))


@app.get("/api/v1/episodes/{episode_id}/trajectory")
def trajectory(episode_id: str):
    env = _sessions.get(episode_id)
    rows = env.get_trajectory() if env else _recorder.load(episode_id)
    if not rows and env is None:
        raise HTTPException(404, f"episode {episode_id} not found")
    return [t.to_dict() for t in rows]


@app.get("/api/v1/episodes/{episode_id}/trajectory/{step_id}")
def trajectory_step(episode_id: str, step_id: int):
    env = _sessions.get(episode_id)
    rows = env.get_trajectory() if env else _recorder.load(episode_id)
    for t in rows:
        if t.timestep == step_id:
            return t.to_dict()
    raise HTTPException(404, f"step {step_id} not found")


@app.get("/api/v1/agents")
def agents():
    cfgs = default_agent_configs()
    return [{"agent_id": a, "role": c.role, "budget": c.budget, "inventory": c.inventory,
             "reservation_price": c.reservation_price, "required_qty": c.required_qty,
             "risk_tolerance": c.risk_tolerance} for a, c in cfgs.items()]


@app.get("/api/v1/agents/{agent_id}/policy-diff")
def policy_diff(agent_id: str, from_ep: int = 100, to_ep: int = 5000):
    if agent_id not in AGENTS:
        raise HTTPException(404, "unknown agent")
    # Smoke fixture: checkpoints are fabricated labels; shift computed from recorded
    # trajectories when present, else reported as unavailable (never invented).
    return {"agent_id": agent_id, "from_episode": from_ep, "to_episode": to_ep,
            "available": False, "reason": "no trained checkpoints in this session; run Week 6 harness"}


@app.get("/api/v1/explain/reward-attribution")
def reward_attr(episode_id: str, step_id: int):
    env = _sessions.get(episode_id)
    rows = env.get_trajectory() if env else _recorder.load(episode_id)
    for t in rows:
        if t.timestep == step_id:
            return reward_attribution(t)
    raise HTTPException(404, "step not found")


@app.get("/api/v1/explain/exploits")
def exploits(episode_id: str):
    env = _sessions.get(episode_id)
    rows = env.get_trajectory() if env else _recorder.load(episode_id)
    return {"episode_id": episode_id, "exploits": detect_exploits(rows)}


@app.post("/api/v1/explain/counterfactual")
def counterfactual(req: CounterReq):
    env = _get_env(req.episode_id)
    rows = env.get_trajectory()
    target = next((t for t in rows if t.timestep == req.step_id), None)
    if target is None:
        raise HTTPException(404, "step not found")
    # Horizon-1 counterfactual: forced alternate action reward is not simulated
    # without a stored snapshot per step; report grounded original + explicit gap.
    return {"episode_id": req.episode_id, "step_id": req.step_id,
            "original_reward": float(target.reward), "alternate_action": req.alternate_action,
            "delta_reward": None, "available": False,
            "reason": "per-step snapshots are captured live in Week 4 run_counterfactual; API replays need a recorded snapshot"}


_NAME_MAP = {"buyer a": "buyer_a", "buyer b": "buyer_b", "seller a": "seller_a",
             "seller b": "seller_b", "market manager": "market_manager", "manager": "market_manager"}
_ACTION_NAMES = ["OFFER", "COUNTEROFFER", "ACCEPT", "REJECT", "WAIT", "WITHDRAW"]


@app.post("/api/v1/query")
def query(req: QueryReq):
    """Evidence-driven Q&A. The question intent selects WHICH recorded facts
    answer it; every number comes from the trajectory, deals, or config."""
    q = req.question.lower()
    agent_id = req.agent_id
    if not agent_id:
        for name, aid in _NAME_MAP.items():
            if name in q:
                agent_id = aid
                break
    episode_id = req.episode_id
    env = _sessions.get(episode_id) if episode_id else None
    if env is None and _sessions:
        env = list(_sessions.values())[-1]
        episode_id = env.current_episode_id
    if env is None:
        raise HTTPException(404, "no episode; start a simulation first")

    all_rows = env.get_trajectory()
    deals = list(env._deals)
    cfgs = default_agent_configs()
    rows = [t for t in all_rows if not agent_id or t.agent_id == agent_id]
    evidence: list[dict[str, Any]] = []

    def ev_action(t):
        evidence.append({"type": "action", "field": "action", "value": list(t.action),
                         "source": "trajectory", "episode_id": episode_id,
                         "timestep": t.timestep, "agent_id": t.agent_id})

    def ev_reward(t):
        comps = getattr(t, "reward_components", {}) or {}
        evidence.append({"type": "reward_component", "field": "reward",
                         "value": float(t.reward), "source": "trajectory",
                         "episode_id": episode_id, "timestep": t.timestep,
                         "agent_id": t.agent_id,
                         **({"components": comps} if comps else {})})

    def ev_reservation(aid):
        if aid in cfgs:
            evidence.append({"type": "reservation_price", "field": "reservation_price",
                             "value": cfgs[aid].reservation_price, "source": "config",
                             "episode_id": episode_id, "agent_id": aid})

    def act_name(t):
        i = int(t.action[0])
        return _ACTION_NAMES[i] if 0 <= i < 6 else str(i)

    def comps_str(t):
        comps = getattr(t, "reward_components", {}) or {}
        nz = {k: v for k, v in comps.items() if v}
        return ", ".join(f"{k} {v:+.3f}" for k, v in nz.items()) if nz else "no non-zero components"

    step_match = re.search(r"step\s+(\d+)", q)
    want_step = int(step_match.group(1)) if step_match else None

    def finish(answer, conf):
        return {"answer": answer, "evidence": evidence, "confidence": conf,
                "sources": [episode_id] if episode_id else []}

    if not all_rows:
        return finish("No recorded steps for this episode yet. Step the simulation first, then ask.", 0.3)

    # --- deals: count / best deal -------------------------------------------
    if "how many" in q and "deal" in q:
        if deals:
            for d in deals:
                evidence.append({"type": "deal", "field": "deal",
                                 "value": {"buyer": d["buyer"], "seller": d["seller"],
                                           "price": d["price"], "quantity": d["quantity"]},
                                 "source": "state.deals", "episode_id": episode_id,
                                 "timestep": d.get("timestep"), "agent_id": d["buyer"]})
            lines = "; ".join(f"{d['buyer']} bought from {d['seller']} at ₹{d['price']} x{d['quantity']}" for d in deals)
            return finish(f"{len(deals)} deal(s) closed in {episode_id}: {lines}.", 0.95)
        return finish(f"No deals have closed yet in {episode_id} after {len(all_rows)} recorded steps.", 0.9)

    if "best deal" in q or "who got" in q:
        if not deals:
            return finish(f"No deal has closed in {episode_id} yet, so there is no best deal to report. "
                          f"{len(all_rows)} steps are recorded; keep stepping.", 0.85)
        def buyer_saving(d):
            res = cfgs[d["buyer"]].reservation_price if d["buyer"] in cfgs else d["price"]
            return (res - d["price"]) * d["quantity"]
        best = max(deals, key=buyer_saving)
        ev_reservation(best["buyer"])
        evidence.append({"type": "deal", "field": "deal",
                         "value": {"buyer": best["buyer"], "seller": best["seller"],
                                   "price": best["price"], "quantity": best["quantity"]},
                         "source": "state.deals", "episode_id": episode_id,
                         "timestep": best.get("timestep"), "agent_id": best["buyer"]})
        return finish(f"Best recorded deal: {best['buyer']} bought {best['quantity']} unit(s) from "
                      f"{best['seller']} at ₹{best['price']} (closed at step {best.get('timestep')}). "
                      f"Against {best['buyer']}'s reservation price the saving is "
                      f"₹{buyer_saving(best):.0f} over the deal. {len(deals)} deal(s) closed in total.", 0.95)

    # --- market manager ------------------------------------------------------
    if agent_id == "market_manager" or "market manager" in q or "manager do" in q:
        mgr = [t for t in all_rows if t.agent_id == "market_manager"]
        if not mgr:
            return finish("market_manager has not acted yet in this episode.", 0.7)
        counts: dict[str, int] = {}
        for t in mgr:
            counts[act_name(t)] = counts.get(act_name(t), 0) + 1
        ev_action(mgr[0]); ev_reward(mgr[0]); ev_reservation("market_manager")
        total = sum(float(t.reward) for t in mgr)
        detail = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
        return finish(f"market_manager acted {len(mgr)} time(s) in {episode_id}: {detail}. "
                      f"Its cumulative recorded reward is {total:.3f}. The manager can only WAIT or "
                      f"WITHDRAW in this market; it does not post offers.", 0.9)

    # --- reservation price ---------------------------------------------------
    if "reservation" in q:
        if agent_id and agent_id in cfgs:
            ev_reservation(agent_id)
            c = cfgs[agent_id]
            return finish(f"{agent_id}'s configured reservation price is ₹{c.reservation_price} "
                          f"(role {c.role}, budget ₹{c.budget}, inventory {c.inventory}). "
                          f"This is private config from default_agent_configs, hidden from other agents in play.", 0.95)
        parts = [f"{a} ₹{c.reservation_price}" for a, c in cfgs.items() if c.reservation_price]
        return finish("Configured reservation prices: " + ", ".join(parts) + ".", 0.9)

    # --- reward --------------------------------------------------------------
    if "reward" in q:
        pool = rows or all_rows
        if want_step is not None:
            pool = [t for t in pool if t.timestep == want_step] or pool
        t = pool[-1]
        ev_action(t); ev_reward(t)
        return finish(f"At step {t.timestep}, {t.agent_id} received reward {float(t.reward):.3f} "
                      f"for action {act_name(t)}. Recorded components: {comps_str(t)}. "
                      f"Reward is a shaped RL signal, not rupee surplus.", 0.9)

    # --- action-specific why: reject/offer/counteroffer/accept/wait/withdraw -
    wanted = None
    for name in _ACTION_NAMES:
        if name.lower() in q:
            wanted = name
            break
    if wanted or "why" in q:
        atype = _ACTION_NAMES.index(wanted) if wanted else None
        pool = [t for t in rows if atype is None or int(t.action[0]) == atype]
        if want_step is not None:
            stepped = [t for t in pool if t.timestep == want_step]
            if stepped:
                pool = stepped
        if not pool:
            who = agent_id or "that agent"
            what = wanted or "that action"
            if rows:
                ev_action(rows[-1]); ev_reward(rows[-1])
            if agent_id:
                ev_reservation(agent_id)
            return finish(f"No recorded {what} by {who} in {episode_id} yet "
                          f"({len(rows)} step(s) recorded for the selected scope). "
                          f"Ask again after more steps, or pick a different action/agent.", 0.55)
        t = pool[0] if want_step is None else pool[-1]
        ev_action(t); ev_reward(t); ev_reservation(t.agent_id)
        res = cfgs[t.agent_id].reservation_price if t.agent_id in cfgs else None
        extra = ""
        if wanted == "ACCEPT":
            done_deals = [d for d in deals if d.get("timestep") is not None and d["timestep"] <= t.timestep]
            if done_deals:
                d = done_deals[-1]
                extra = f" It closed a deal: {d['buyer']} / {d['seller']} at ₹{d['price']} x{d['quantity']}."
        answer = (f"At step {t.timestep}, {t.agent_id} chose {act_name(t)} "
                  f"(raw action {list(t.action)}) and received reward {float(t.reward):.3f} "
                  f"({comps_str(t)}). Its configured reservation price is ₹{res}. "
                  f"The trajectory records the choice and its reward components; it does not record a "
                  f"separate stated motive, so any deeper 'why' would be speculation.{extra}")
        return finish(answer, 0.9 if len(evidence) >= 3 else 0.7)

    # --- step-specific / summary fallback ------------------------------------
    if want_step is not None:
        pool = [t for t in all_rows if t.timestep == want_step]
        if pool:
            t = pool[0]
            ev_action(t); ev_reward(t)
            return finish(f"Step {want_step}: {t.agent_id} chose {act_name(t)} with reward "
                          f"{float(t.reward):.3f} ({comps_str(t)}).", 0.85)
        return finish(f"Step {want_step} is not recorded yet; {len(all_rows)} step(s) exist in {episode_id}.", 0.6)

    counts: dict[str, int] = {}
    for t in (rows or all_rows):
        counts[act_name(t)] = counts.get(act_name(t), 0) + 1
    detail = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
    who = agent_id or "all agents"
    return finish(f"In {episode_id}, {who} recorded {len(rows or all_rows)} step(s): {detail}. "
                  f"{len(deals)} deal(s) closed. Ask about a specific action (reject, offer, accept), "
                  f"a step number, reward, reservation price, or the best deal for a grounded answer.", 0.75)


@app.post("/api/v1/experiments")
def create_experiment():
    return {"experiment_id": "exp_smoke", "status": "use Week 6 harness (nexus.experiments)"}


@app.get("/api/v1/experiments/compare")
def compare():
    return {"algorithms": ["mappo", "ippo", "rule_based", "random"], "available": False,
            "reason": "run src/nexus/experiments harness for measured numbers"}


@app.websocket("/api/v1/simulation/{episode_id}/stream")
async def stream(websocket: WebSocket, episode_id: str):
    await websocket.accept()
    env = _sessions.get(episode_id)
    if env is None:
        await websocket.send_json({"error": "episode not found"})
        await websocket.close()
        return
    try:
        while True:
            data = await websocket.receive_text()
            if data == "state":
                await websocket.send_json(_state_payload(env))
            elif data == "ping":
                await websocket.send_json({"pong": True})
            else:
                await websocket.send_json({"echo": data})
    except WebSocketDisconnect:
        return
