"""Counterfactual rollout via env snapshot/restore (Week 1 get_state/set_state)."""
from __future__ import annotations

import numpy as np


def run_counterfactual(env, snapshot: dict, agent: str, alternate_action, policy_fn, horizon: int = 10) -> dict:
    """Restore snapshot, force alternate_action for `agent`, roll out with policy_fn."""
    env.set_state(snapshot)
    total = 0.0
    steps = 0
    forced = False
    while env.agents and steps < horizon:
        cur = env.agent_selection
        if env.terminations[cur] or env.truncations[cur]:
            env.step(None)
            continue
        if cur == agent and not forced:
            env.step(np.asarray(alternate_action, dtype=np.int64))
            forced = True
        else:
            env.step(np.asarray(policy_fn(cur, env), dtype=np.int64))
        total += float(env.rewards.get(cur, 0.0))
        steps += 1
    return {"counterfactual_reward": float(total), "steps": steps, "forced": forced}
