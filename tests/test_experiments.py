import json
import os

from nexus.experiments.metrics import buyer_savings, seller_profit


def test_metrics_formulas():
    deals = [{"price": 80.0}, {"price": 90.0}]
    assert abs(buyer_savings(deals, 100.0) - 0.15) < 1e-9
    assert seller_profit(deals, 50.0) > 0


def test_summary_schema():
    path = "results/summary.json"
    assert os.path.exists(path), "run the experiment harness first"
    s = json.load(open(path))
    assert s["mode"] in ("smoke", "full")
    assert set(s["exp1"]["results"]) == {"mappo", "ippo", "rule_based", "random"}
    assert s["exp3"]["exploits_found"] >= 1
    assert len(s["exp4"]["generalization_unseen_configs"]) >= 3
