import os

import numpy as np
import pytest
import torch

from nexus.env import NexusMarketplaceEnv
from nexus.env.types import D_GLOBAL, D_S
from nexus.rl import IPPOTrainer, MAPPOTrainer
from nexus.rl.evaluate import evaluate
from nexus.rl.networks import Actor, Critic


def test_network_shapes():
    a = Actor(D_S)
    lt, lp, lq = a(torch.zeros(2, D_S))
    assert lt.shape == (2, 6) and lp.shape == (2, 16) and lq.shape == (2, 10)
    assert Critic(D_GLOBAL)(torch.zeros(2, D_GLOBAL)).shape == (2,)
    assert Critic(D_S)(torch.zeros(2, D_S)).shape == (2,)


def test_masked_actions_never_sampled():
    torch.manual_seed(0)
    a = Actor(D_S)
    masks = {"type": np.array([0, 0, 0, 0, 1, 0], dtype=bool),
             "price": np.ones(16, dtype=bool), "quantity": np.ones(10, dtype=bool)}
    for _ in range(50):
        act, _, _ = a.act(np.zeros(D_S, dtype=np.float32), masks, torch.device("cpu"))
        assert int(act[0]) == 4  # WAIT only


@pytest.mark.parametrize("cls,central", [(MAPPOTrainer, True), (IPPOTrainer, False)])
def test_train_smoke_and_checkpoint(cls, central, tmp_path):
    env = NexusMarketplaceEnv({"max_steps": 30})
    tr = cls(env, seed=0)
    assert tr.centralized is central
    if central:
        assert next(tr.critic.parameters()).shape  # global critic exists
    hist = tr.train(num_episodes=4, checkpoint_dir=str(tmp_path), checkpoint_interval=2, log_dir=str(tmp_path / "runs"))
    assert len(hist) == 4
    assert all(np.isfinite(h["policy_loss"]) and np.isfinite(h["value_loss"]) for h in hist)
    ck = tmp_path / "ckpt_4.pt"
    assert ck.exists()
    assert (tmp_path / "runs" / "train_log.jsonl").exists()
    tr2 = cls(NexusMarketplaceEnv({"max_steps": 30}), seed=1)
    tr2.load_checkpoint(str(ck))
    assert tr2.episode_count == 4


def test_ctde_inference_local_only():
    env = NexusMarketplaceEnv({"max_steps": 20})
    tr = MAPPOTrainer(env, seed=0)
    orig = env.get_global_state
    env.get_global_state = lambda: (_ for _ in ()).throw(RuntimeError("global leaked"))
    try:
        act = tr.act_greedy("buyer_a", env.observe("buyer_a"), env.action_masks("buyer_a"))
        assert len(act) == 3
    finally:
        env.get_global_state = orig


def test_evaluate_random_and_rule():
    res = evaluate(lambda: NexusMarketplaceEnv({"max_steps": 30}), ["random", "rule_based"], num_episodes=3)
    assert set(res) == {"random", "rule_based"}
    for v in res.values():
        assert 0.0 <= v["transaction_rate"] <= 1.0
        assert v["episodes"] == 3
