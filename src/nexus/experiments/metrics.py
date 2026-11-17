"""Experiment metrics from recorded trajectories / eval episodes."""
from __future__ import annotations

import numpy as np


def buyer_savings(deals, reservation: float) -> float:
    if not deals or not reservation:
        return 0.0
    pavg = float(np.mean([d["price"] for d in deals]))
    return float((reservation - pavg) / reservation)


def seller_profit(deals, unit_cost: float) -> float:
    if not deals or not unit_cost:
        return 0.0
    pavg = float(np.mean([d["price"] for d in deals]))
    return float((pavg - unit_cost) / unit_cost)


def kl_divergence(p, q) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, None); p /= p.sum()
    q = np.clip(np.asarray(q, dtype=float), 1e-9, None); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))
