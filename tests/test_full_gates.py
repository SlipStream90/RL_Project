import json
import os

import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("results/full_gates.json"),
                                reason="full gates not run yet")


def test_ippo_500_improved():
    g = json.load(open("results/full_gates.json"))
    assert g["ippo_500_gate"]["improved"] is True
    assert g["ippo_500_gate"]["episodes_451_500_mean_reward"] > g["ippo_500_gate"]["first50_mean_reward"]


def test_exp1_mappo_wins_at_least_3():
    g = json.load(open("results/full_gates.json"))
    assert g["train_episodes_per_learned_algo"] == 5000
    assert g["exp1"]["mappo_wins_count"] >= 3
    assert set(g["exp1"]["results"]) == {"mappo", "ippo", "rule_based", "random"}
    assert all(v["episodes"] == 100 for v in g["exp1"]["results"].values())


def test_checkpoints_exist():
    for label in ("mappo", "ippo"):
        for ep in (100, 500, 1000, 5000):
            assert os.path.exists(f"results/checkpoints/{label}_ckpt_{ep}.pt")


def test_bert_gate():
    m = json.load(open("results/bert_metrics_full.json"))
    assert m["accuracy"] >= 0.85, m
    assert m["held_out_templates"]
