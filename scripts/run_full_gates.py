"""Full RL gates: IPPO-500 improvement + Experiment 1 (5000 eps) + Exp4 shift.

Writes results/full_gates.json and merges measured numbers into
results/summary.json. Long-running (tens of minutes on CPU). Progress is
appended to results/full_gates.log by the caller redirect.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from nexus.env import NexusMarketplaceEnv
from nexus.env.types import AGENTS, default_agent_configs
from nexus.rl import IPPOTrainer, MAPPOTrainer
from nexus.rl.evaluate import run_episode
from nexus.experiments.metrics import buyer_savings, seller_profit

OUT = "results/full_gates.json"
CKPT_DIR = "results/checkpoints"
STAGES = [100, 400, 500, 4000]  # cumulative checkpoints at 100/500/1000/5000


def factory():
    return NexusMarketplaceEnv({"max_steps": 40})


def staged_train(trainer, label: str) -> list[dict]:
    os.makedirs(CKPT_DIR, exist_ok=True)
    hist: list[dict] = []
    for n in STAGES:
        t0 = time.time()
        hist += trainer.train(num_episodes=n)
        total = trainer.episode_count
        path = os.path.join(CKPT_DIR, f"{label}_ckpt_{total}.pt")
        trainer.save_checkpoint(path)
        print(f"[{label}] episodes={total} stage_seconds={time.time()-t0:.0f}", flush=True)
    return hist


def mean(xs) -> float:
    return float(np.mean(xs)) if xs else 0.0


def eval_algo(name: str, trainers: dict, episodes: int = 100) -> dict:
    cfgs = default_agent_configs()
    if name == "random":
        fn = lambda a, e: e.action_space(a).sample()
    elif name == "rule_based":
        from nexus.env.baselines import baseline_policy
        pol = {a: baseline_policy(a, seed=0) for a in AGENTS}
        rng = np.random.default_rng(0)
        fn = lambda a, e: pol[a].act(e.observe(a), e.action_masks(a), rng)
    else:
        tr = trainers[name]
        fn = lambda a, e: tr.act_greedy(a, e.observe(a), e.action_masks(a))
    eps = [run_episode(factory(), fn, seed=1000 + i) for i in range(episodes)]
    deals = [d for e in eps for d in e["deals"]]
    sav, prof = [], []
    for d in deals:
        sav.append((cfgs[d["buyer"]].reservation_price - d["price"]) / cfgs[d["buyer"]].reservation_price)
        prof.append((d["price"] - cfgs[d["seller"]].unit_cost) / cfgs[d["seller"]].unit_cost)
    return {
        "mean_reward": mean([e["mean_reward"] for e in eps]),
        "cumulative_reward_per_episode": mean([sum(e["rewards"].values()) for e in eps]),
        "transaction_rate": mean([1.0 if e["deals"] else 0.0 for e in eps]),
        "negotiation_duration_steps": mean([e["steps"] for e in eps]),
        "buyer_savings": mean(sav), "seller_profit": mean(prof),
        "deals_total": len(deals), "episodes": episodes,
    }


def policy_shift(label: str) -> dict:
    """Mean KL of buyer_a type-head between 100- and 5000-episode checkpoints."""
    from nexus.rl.networks import Actor
    obs = torch.zeros(1, 50)
    probs = {}
    for ep in (100, 5000):
        ck = torch.load(os.path.join(CKPT_DIR, f"{label}_ckpt_{ep}.pt"), map_location="cpu", weights_only=False)
        actor = Actor(50)
        actor.load_state_dict(ck["actors"]["buyer_a"])
        actor.eval()
        with torch.no_grad():
            ds = actor.dists(obs, None)
        probs[ep] = [d.probs.squeeze(0).numpy() for d in ds]
    from nexus.experiments.metrics import kl_divergence
    kls = [kl_divergence(probs[100][i], probs[5000][i]) for i in range(3)]
    return {"checkpoints": [100, 5000], "kl_by_head_type_price_qty": [float(k) for k in kls],
            "mean_kl": float(np.mean(kls))}


def main() -> int:
    t0 = time.time()
    print("=== IPPO 5000 (contains the 500-episode gate) ===", flush=True)
    ippo = IPPOTrainer(factory(), seed=0)
    ippo_hist = staged_train(ippo, "ippo")
    rewards = [h["mean_reward"] for h in ippo_hist]
    ippo500 = {"first50_mean_reward": mean(rewards[:50]), "episodes_451_500_mean_reward": mean(rewards[450:500]),
               "improved": bool(mean(rewards[450:500]) > mean(rewards[:50]))}
    print(f"IPPO500 {ippo500}", flush=True)

    print("=== MAPPO 5000 (Experiment 1) ===", flush=True)
    mappo = MAPPOTrainer(factory(), seed=0)
    staged_train(mappo, "mappo")

    print("=== Eval 100 episodes per algorithm ===", flush=True)
    trainers = {"mappo": mappo, "ippo": ippo}
    exp1 = {a: eval_algo(a, trainers, 100) for a in ("mappo", "ippo", "rule_based", "random")}
    # MAPPO wins on a metric if best (duration: lower is better)
    wins = []
    for m in ("mean_reward", "cumulative_reward_per_episode", "transaction_rate", "buyer_savings", "seller_profit"):
        if max(exp1, key=lambda a: exp1[a][m]) == "mappo":
            wins.append(m)
    if min(exp1, key=lambda a: exp1[a]["negotiation_duration_steps"]) == "mappo":
        wins.append("negotiation_duration_steps")
    result = {
        "mode": "full", "train_episodes_per_learned_algo": 5000, "eval_episodes": 100,
        "seconds_total": time.time() - t0,
        "ippo_500_gate": ippo500,
        "exp1": {"results": exp1, "mappo_wins": wins, "mappo_wins_count": len(wins),
                 "success_criterion": "MAPPO best on >=3 of the 5 PRD metrics (reward, savings, profit, transaction rate, duration)"},
        "exp4_policy_shift_mappo": policy_shift("mappo"),
        "exp4_policy_shift_ippo": policy_shift("ippo"),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    # Merge into summary.json without touching exp2/exp3 smoke entries.
    summary = json.load(open("results/summary.json")) if os.path.exists("results/summary.json") else {}
    summary.update({"mode": "full", "train_episodes_per_algo": 5000, "eval_episodes": 100,
                    "ippo_500_gate": ippo500, "exp1": result["exp1"],
                    "exp4_policy_shift": result["exp4_policy_shift_mappo"]})
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(result, indent=2)[:3000], flush=True)
    print(f"DONE seconds={time.time()-t0:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
