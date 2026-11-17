"""Evaluation harness: mean reward / transaction rate per algorithm."""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from nexus.env.baselines import baseline_policy


def run_episode(env, policy_fn, seed: int = 0) -> dict:
    env.reset(seed=seed)
    rewards = defaultdict(float)
    steps = 0
    while env.agents and steps < 5000:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        action = policy_fn(agent, env)
        env.step(action)
        rewards[agent] += float(env.rewards.get(agent, 0.0))
        steps += 1
    traj = env.get_trajectory()
    deals = list(getattr(env, "_deals", []))
    return {"rewards": dict(rewards), "mean_reward": float(np.mean(list(rewards.values()))) if rewards else 0.0,
            "steps": steps, "accepts": len(deals), "deals": deals}


def evaluate(env_factory, algorithms: list[str], num_episodes: int = 10, trainers: dict | None = None,
             seed: int = 0) -> dict:
    trainers = trainers or {}
    results = {}
    for algo in algorithms:
        env = env_factory()
        if algo == "random":
            rng = np.random.default_rng(seed)
            fn = lambda a, e: e.action_space(a).sample()
        elif algo == "rule_based":
            pol = {a: baseline_policy(a, seed=seed) for a in env.possible_agents}
            rng = np.random.default_rng(seed)
            fn = lambda a, e: pol[a].act(e.observe(a), e.action_masks(a), rng)
        elif algo in ("mappo", "ippo"):
            tr = trainers[algo]
            fn = lambda a, e: tr.act_greedy(a, e.observe(a), e.action_masks(a))
        else:
            raise ValueError(f"unknown algorithm {algo}")
        eps = [run_episode(env_factory(), fn, seed=seed + i) for i in range(num_episodes)]
        results[algo] = {
            "mean_reward": float(np.mean([e["mean_reward"] for e in eps])),
            "mean_steps": float(np.mean([e["steps"] for e in eps])),
            "transaction_rate": float(np.mean([1.0 if e["accepts"] > 0 else 0.0 for e in eps])),
            "episodes": num_episodes,
        }
    return results


def save_results(results: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path
