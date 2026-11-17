"""Reward attribution: env already decomposes; this reads, never recomputes."""
from __future__ import annotations


def reward_attribution(transition) -> dict:
    comps = dict(transition.reward_components or {})
    total = float(transition.reward)
    dominant = max(comps, key=lambda k: abs(comps[k])) if comps else None
    return {"total_reward": total, "components": comps, "dominant_factor": dominant,
            "episode_id": transition.episode_id, "timestep": transition.timestep,
            "agent_id": transition.agent_id}


def episode_attribution(transitions) -> dict:
    out: dict[str, dict] = {}
    for t in transitions:
        agg = out.setdefault(t.agent_id, {"total_reward": 0.0, "components": {}})
        agg["total_reward"] += float(t.reward)
        for k, v in (t.reward_components or {}).items():
            agg["components"][k] = agg["components"].get(k, 0.0) + float(v)
    for agg in out.values():
        agg["dominant_factor"] = max(agg["components"], key=lambda k: abs(agg["components"][k])) if agg["components"] else None
    return out
