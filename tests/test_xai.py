import json

import numpy as np
import pytest

from nexus.env import NexusMarketplaceEnv
from nexus.env.baselines import baseline_policy
from nexus.env.types import Transition
from nexus.xai import (TrajectoryRecorder, compare_policy, detect_exploits,
                       reward_attribution, run_counterfactual, state_action_analysis,
                       surrogate_feature_importance)


def _episode(seed=3):
    env = NexusMarketplaceEnv({"max_steps": 40})
    env.reset(seed=seed)
    pol = {a: baseline_policy(a, seed=1) for a in env.possible_agents}
    rng = np.random.default_rng(7)
    while env.agents:
        a = env.agent_selection
        if env.terminations[a] or env.truncations[a]:
            env.step(None); continue
        env.step(pol[a].act(env.observe(a), env.action_masks(a), rng))
    return env


def test_recorder_and_attribution():
    env = _episode()
    rec = TrajectoryRecorder(); rec.record_many(env.get_trajectory())
    ep = env.current_episode_id
    rows = rec.load(ep)
    assert rows and all(t.reward_components is not None for t in rows)
    att = reward_attribution(rows[0])
    assert "total_reward" in att and "components" in att
    assert rec.query(episode_id=ep, agent_id="buyer_a") is not None


def test_state_action_and_surrogate():
    env = _episode()
    traj = env.get_trajectory()
    out = state_action_analysis(traj, "buyer_a")
    assert "top_features" in out
    sur = surrogate_feature_importance(traj, "buyer_a")
    assert "policy" not in sur["method"] and sur["note"]


def test_policy_compare():
    a = {"type": [0.7, 0.1, 0.05, 0.05, 0.05, 0.05]}
    b = {"type": [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]}
    out = compare_policy(a, b)
    assert out["type"]["kl"] > 0 and out["mean_kl"] > 0


def test_counterfactual_runs():
    env = NexusMarketplaceEnv({"max_steps": 40})
    env.reset(seed=3)
    env.step(np.array([4, 0, 0]))
    snap = env.get_state()
    pol = {a: baseline_policy(a, seed=1) for a in env.possible_agents}
    rng = np.random.default_rng(7)
    out = run_counterfactual(env, snap, env.agent_selection, np.array([4, 0, 0]),
                             lambda cur, e: pol[cur].act(e.observe(cur), e.action_masks(cur), rng), horizon=5)
    assert np.isfinite(out["counterfactual_reward"]) and out["forced"]


def test_exploit_lowballing_synthetic():
    rows = []
    for i in range(8):
        rows.append(Transition(episode_id="ep_x", timestep=i, agent_id="buyer_a",
                               observation=np.zeros(50), global_state=np.zeros(254),
                               action=(0, 0, 5), reward=0.0, reward_components={},
                               slots={"price": 55.0}))
    flags = detect_exploits(rows)
    assert any(f["type"] == "repeated_lowballing" for f in flags)


def test_export_json(tmp_path):
    env = _episode()
    rec = TrajectoryRecorder(); rec.record_many(env.get_trajectory())
    p = rec.export_json(env.current_episode_id, str(tmp_path / "traj.json"))
    data = json.load(open(p))
    assert isinstance(data, list) and data
    p2 = rec.export_csv(env.current_episode_id, str(tmp_path / "traj.csv"))
    assert open(p2).readline().startswith("episode_id")
