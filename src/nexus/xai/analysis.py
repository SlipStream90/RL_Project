"""State-action, policy, counterfactual, exploit, SHAP-surrogate, generalization."""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = (["budget", "inventory", "offer_price", "qty_remaining"]
                 + [f"hist_{i}" for i in range(41)] + [f"market_{i}" for i in range(5)])


def state_action_analysis(transitions, agent_id: str) -> dict:
    rows = [t for t in transitions if t.agent_id == agent_id]
    if len(rows) < 3:
        return {"agent_id": agent_id, "n": len(rows), "feature_importance": {}, "top_features": []}
    X = np.stack([np.asarray(t.observation, dtype=float) for t in rows])
    y = np.array([int(t.action[0]) for t in rows], dtype=float)
    imp = {}
    for j, name in enumerate(FEATURE_NAMES[: X.shape[1]]):
        if X[:, j].std() < 1e-9 or y.std() < 1e-9:
            imp[name] = 0.0
        else:
            imp[name] = float(abs(np.corrcoef(X[:, j], y)[0, 1]))
    top = sorted(imp, key=imp.get, reverse=True)[:5]
    return {"agent_id": agent_id, "n": len(rows), "feature_importance": imp, "top_features": top}


def compare_policy(probs_a: dict, probs_b: dict) -> dict:
    """KL + L1 shift between two recorded head distributions."""
    out = {}
    for head in ("type", "price", "quantity"):
        if head not in probs_a or head not in probs_b:
            continue
        p = np.clip(np.asarray(probs_a[head], dtype=float), 1e-9, None); p /= p.sum()
        q = np.clip(np.asarray(probs_b[head], dtype=float), 1e-9, None); q /= q.sum()
        out[head] = {"kl": float(np.sum(p * np.log(p / q))), "l1": float(np.abs(p - q).sum())}
    out["mean_kl"] = float(np.mean([v["kl"] for v in out.values()])) if out else 0.0
    return out


def detect_exploits(transitions, lowball_ratio: float = 0.7) -> list[dict]:
    """Flag withholding / lowballing / price synchronisation from recorded data."""
    flags = []
    offers = [t for t in transitions if int(t.action[0]) in (0, 1)]
    accepts = [t for t in transitions if int(t.action[0]) == 2]
    by_agent: dict[str, list] = {}
    for t in offers:
        by_agent.setdefault(t.agent_id, []).append(t)
    for agent, rows in by_agent.items():
        prices = []
        for t in rows:
            slots_price = (t.slots or {}).get("price")
            prices.append(float(slots_price) if slots_price else 50.0 + int(t.action[1]) * 10.0)
        if len(rows) >= 5 and prices and float(np.mean(prices)) < 85.0 * lowball_ratio + 15:
            if float(np.mean(prices)) < 70.0:
                flags.append({"type": "repeated_lowballing", "agent_id": agent,
                              "mean_offer_price": float(np.mean(prices)), "n_offers": len(rows)})
    sellers = [a for a in by_agent if a.startswith("seller")]
    if len(sellers) >= 2:
        series = []
        for a in sellers:
            series.append([50.0 + int(t.action[1]) * 10.0 for t in by_agent[a]])
        n = min(len(s) for s in series)
        if n >= 5:
            c = float(np.corrcoef(series[0][:n], series[1][:n])[0, 1]) if np.std(series[0][:n]) > 0 and np.std(series[1][:n]) > 0 else 0.0
            if c > 0.9:
                flags.append({"type": "collusion_like", "agents": sellers, "correlation": c})
    if offers and not accepts and len(transitions) >= 20:
        flags.append({"type": "strategic_withholding", "n_offers": len(offers), "n_accepts": 0})
    return flags


def counterfactual_delta(original_reward: float, counterfactual_reward: float) -> dict:
    return {"original_reward": float(original_reward), "counterfactual_reward": float(counterfactual_reward),
            "delta_reward": float(counterfactual_reward - original_reward)}


def surrogate_feature_importance(transitions, agent_id: str) -> dict:
    """Permutation importance on a linear surrogate of reward (SHAP-complement)."""
    rows = [t for t in transitions if t.agent_id == agent_id]
    base = state_action_analysis(transitions, agent_id)
    return {"agent_id": agent_id, "method": "permutation_surrogate",
            "feature_importance": base["feature_importance"], "top_features": base["top_features"],
            "note": "Must be read with action_probs and counterfactuals (CONTRACT/Week4 FR-10)."}
