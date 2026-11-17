"""Shared types — single source per CONTRACT.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

AGENTS: list[str] = ["buyer_a", "buyer_b", "seller_a", "seller_b", "market_manager"]
BUYERS = {"buyer_a", "buyer_b"}
SELLERS = {"seller_a", "seller_b"}

D_S = 50
D_GLOBAL = 254
N_PRICE_BANDS = 16
Q_MAX = 10
P_MIN = 50.0
P_STEP = 10.0
HISTORY_K = 5


class ActionType(IntEnum):
    OFFER = 0
    COUNTEROFFER = 1
    ACCEPT = 2
    REJECT = 3
    WAIT = 4
    WITHDRAW = 5


INTENT_NAMES = [a.name for a in ActionType]


def band_to_price(band: int) -> float:
    return P_MIN + (int(band) - 1) * P_STEP


def price_to_band(price: float) -> int:
    b = int(round((float(price) - P_MIN) / P_STEP)) + 1
    return max(1, min(N_PRICE_BANDS, b))


@dataclass
class AgentConfig:
    agent_id: str
    role: str  # buyer | seller | manager
    budget: float
    inventory: int
    reservation_price: float
    required_qty: int
    risk_tolerance: float
    unit_cost: float = 40.0
    reward_weights: dict[str, float] = field(default_factory=dict)
    private_info: dict[str, Any] = field(default_factory=dict)


def default_agent_configs() -> dict[str, AgentConfig]:
    return {
        "buyer_a": AgentConfig("buyer_a", "buyer", 10000.0, 0, 85.0, 10, 0.3,
                               reward_weights={"price": 0.5, "speed": 0.2, "completion": 0.2, "penalty": 0.1}),
        "buyer_b": AgentConfig("buyer_b", "buyer", 10000.0, 0, 110.0, 10, 0.6,
                               reward_weights={"price": 0.2, "speed": 0.5, "completion": 0.2, "penalty": 0.1}),
        "seller_a": AgentConfig("seller_a", "seller", 0.0, 100, 90.0, 10, 0.3, unit_cost=60.0),
        "seller_b": AgentConfig("seller_b", "seller", 0.0, 100, 80.0, 10, 0.7, unit_cost=55.0),
        "market_manager": AgentConfig("market_manager", "manager", 0.0, 0, 0.0, 0, 0.5),
    }


@dataclass
class Transition:
    episode_id: str
    timestep: int
    agent_id: str
    observation: Any
    global_state: Any
    action: Any  # (type:int, band:int, qty:int)
    action_log_prob: float = 0.0
    action_probs: dict[str, Any] = field(default_factory=dict)
    available_actions: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)
    next_observation: Any = None
    done: bool = False
    utterance: str = ""
    intent: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    checkpoint_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        def conv(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [conv(v) for v in x]
            return x
        return {k: conv(v) for k, v in self.__dict__.items()}
