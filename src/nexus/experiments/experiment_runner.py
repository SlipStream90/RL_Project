"""Week 6 harness. Default run is a SMOKE (short training); full 5000-episode
runs require --full and are never inferred from smoke numbers."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from nexus.env import NexusMarketplaceEnv
from nexus.env.types import Transition
from nexus.rl import IPPOTrainer, MAPPOTrainer
from nexus.rl.evaluate import evaluate
from nexus.xai import compare_policy, detect_exploits


def env_factory(max_steps=40):
    return lambda: NexusMarketplaceEnv({"max_steps": max_steps})


def run_smoke(train_episodes: int = 20, eval_episodes: int = 10, out_dir: str = "results") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    factory = env_factory()
    mappo = MAPPOTrainer(factory(), seed=0)
    mappo.train(num_episodes=train_episodes)
    ippo = IPPOTrainer(factory(), seed=0)
    ippo.train(num_episodes=train_episodes)
    exp1 = evaluate(factory, ["mappo", "ippo", "rule_based", "random"],
                    num_episodes=eval_episodes, trainers={"mappo": mappo, "ippo": ippo})
    metrics = ["mean_reward", "mean_steps", "transaction_rate"]
    wins = 0
    for m in metrics:
        best = max(exp1, key=lambda a: exp1[a][m] if m != "mean_steps" else -exp1[a][m])
        if best == "mappo":
            wins += 1
    # Exp2: divergence between two recorded action-type distributions (synthetic check + smoke policies)
    p = np.array([0.6, 0.1, 0.1, 0.05, 0.1, 0.05]); q = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.1])
    div = compare_policy({"type": p.tolist()}, {"type": q.tolist()})
    # Exp3: seeded lowballing fixture through the real detector
    rows = [Transition(episode_id="exp3", timestep=i, agent_id="buyer_a",
                       observation=np.zeros(50), global_state=np.zeros(254),
                       action=(0, 0, 5), reward=0.0, reward_components={}, slots={"price": 55.0})
            for i in range(8)]
    exploits = detect_exploits(rows)
    # Exp4: stub checkpoints labelled 100/5000 + 3 unseen configs (eval only, no training claim)
    gen = {}
    for name, over in {"unseen_a": {"agents": {"buyer_a": {"reservation_price": 95.0}}},
                       "unseen_b": {"agents": {"seller_a": {"reservation_price": 100.0}}},
                       "unseen_c": {"agents": {"buyer_b": {"reservation_price": 90.0}}}}.items():
        f = lambda over=over: NexusMarketplaceEnv({"max_steps": 40, **over})
        gen[name] = evaluate(f, ["rule_based"], num_episodes=3)["rule_based"]
    summary = {
        "mode": "smoke" if train_episodes < 5000 else "full",
        "train_episodes_per_algo": train_episodes, "eval_episodes": eval_episodes,
        "exp1": {"results": exp1, "mappo_metric_wins_smoke_only": wins,
                 "note": "Smoke only: do NOT claim MAPPO outperforms baselines. Full gate is 5000 episodes."},
        "exp2": {"action_type_kl_fixture": div["type"]["kl"], "diverged": bool(div["type"]["kl"] > 0.1)},
        "exp3": {"exploits_found": len(exploits), "exploits": exploits, "verified": "fixture-seeded, manual audit still required for learned behaviour"},
        "exp4": {"policy_shift_fixture_kl": div["mean_kl"], "generalization_unseen_configs": gen,
                 "note": "Generalization uses rule_based on unseen configs; trained-policy gap needs full checkpoints."},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="5000-episode run (hours); default is smoke")
    ap.add_argument("--train-episodes", type=int, default=20)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    n = 5000 if a.full else a.train_episodes
    print(json.dumps(run_smoke(train_episodes=n, eval_episodes=a.eval_episodes, out_dir=a.out), indent=2)[:2000])
