"""Rule-based baseline agents (Week 1).

Action contract (see nexus_env.py / types.py):
  act(obs, masks, rng) -> np.ndarray [type, band_idx, qty_idx]
  type: 0 OFFER, 1 COUNTEROFFER, 2 ACCEPT, 3 REJECT, 4 WAIT, 5 WITHDRAW
  band_idx: 0..15 (price = 50 + band_idx * 10), qty_idx: 0..9 (qty = idx + 1)
  masks: dict {"type": bool[6], "price": bool[16], "quantity": bool[10]}
"""
from __future__ import annotations

from typing import Any

import numpy as np

from nexus.env.types import (
    N_PRICE_BANDS, Q_MAX, ActionType, AgentConfig, band_to_price,
    default_agent_configs, price_to_band,
)

__all__ = ["FixedPriceSeller", "RandomBuyer", "TitForTatAgent", "RandomAgent", "baseline_policy"]

_N_TYPES = 6


def _coerce_rng(rng: Any, fallback: np.random.Generator) -> Any:
    if rng is None:
        return fallback
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


def _rand_index(rng: Any, n: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(n))
    if hasattr(rng, "randint"):
        return int(rng.randint(n))
    return int(rng.choice(n))


def _normalize_masks(masks: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.ones(_N_TYPES, dtype=bool)
    p = np.ones(N_PRICE_BANDS, dtype=bool)
    q = np.ones(Q_MAX, dtype=bool)
    if masks is None:
        return t, p, q
    if isinstance(masks, dict):
        def get(*keys: str) -> Any:
            for k in keys:
                if k in masks:
                    return masks[k]
            return None
        raw_t, raw_p, raw_q = (get("type", "types", "action_type"),
                               get("price", "prices", "band", "price_band"),
                               get("quantity", "quantities", "qty", "quantity_idx"))
    elif isinstance(masks, (tuple, list)) and len(masks) == 3:
        raw_t, raw_p, raw_q = masks
    else:
        raw_t, raw_p, raw_q = None, None, None
    for raw, target in ((raw_t, t), (raw_p, p), (raw_q, q)):
        if raw is not None:
            arr = np.asarray(raw, dtype=bool).reshape(-1)
            n = min(len(arr), len(target))
            target[:n] = arr[:n]
    if not t.any():
        t[int(ActionType.WAIT)] = True
    if not p.any():
        p[0] = True
    if not q.any():
        q[0] = True
    return t, p, q


def _allowed(mask: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(mask)]


def _closest(allowed: list[int], target: int) -> int:
    return min(allowed, key=lambda i: (abs(i - target), i))


class _BaseAgent:
    """Shared flexible constructor: (agent_id|config|reservation, seed=..., role=..., margin=...)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        agent_id = kwargs.pop("agent_id", None)
        reservation = kwargs.pop("reservation_price", kwargs.pop("reservation", None))
        config = kwargs.pop("config", None)
        role = kwargs.pop("role", None)
        seed = kwargs.pop("seed", None)
        self.margin: float = float(kwargs.pop("margin", 0.0))
        for a in args:
            if isinstance(a, AgentConfig):
                config = a
            elif isinstance(a, str):
                agent_id = a
            elif isinstance(a, (int, float, np.floating)) and reservation is None:
                reservation = float(a)
        if isinstance(config, AgentConfig):
            agent_id = agent_id or config.agent_id
            role = role or config.role
            if reservation is None:
                reservation = config.reservation_price
        self.agent_id: str | None = agent_id
        defaults = default_agent_configs()
        cfg = defaults.get(agent_id) if agent_id else None
        if cfg is not None:
            role = role or cfg.role
            if reservation is None:
                reservation = cfg.reservation_price
        self.role: str | None = role
        if reservation is None:
            reservation = 90.0 if role == "seller" else 85.0
        self.reservation_price: float = float(reservation)
        self._rng = np.random.default_rng(seed)

    # -- helpers ---------------------------------------------------------
    def _action(self, atype: int, band_idx: int, qty_idx: int) -> np.ndarray:
        return np.array([int(atype), int(np.clip(band_idx, 0, N_PRICE_BANDS - 1)),
                         int(np.clip(qty_idx, 0, Q_MAX - 1))], dtype=np.int64)

    def _fallback_type(self, type_mask: np.ndarray, prefs: tuple[int, ...]) -> int:
        for p in prefs:
            if type_mask[p]:
                return int(p)
        return _allowed(type_mask)[0]

    def _qty_idx(self, obs: np.ndarray, qty_mask: np.ndarray) -> int:
        allowed = _allowed(qty_mask)
        want = Q_MAX  # qty_idx + 1
        obs = np.asarray(obs, dtype=float).reshape(-1) if obs is not None else np.array([])
        if self.role == "buyer" and len(obs) > 3:
            want = int(round(float(obs[3]) * Q_MAX)) or Q_MAX
        elif self.role == "seller" and len(obs) > 1:
            want = min(Q_MAX, max(1, int(round(float(obs[1]) * 100.0))))
        return _closest(allowed, int(np.clip(want, 1, Q_MAX)) - 1)

    def _target_band(self) -> int:
        """Band idx at (seller) / just below (buyer) the reservation price."""
        band = price_to_band(self.reservation_price + (self.margin if self.role == "seller" else 0.0)) - 1
        if self.role == "buyer":
            # strictly below reservation: step down if band lands on/above it
            while band > 0 and band_to_price(band + 1) >= self.reservation_price:
                band -= 1
        return int(np.clip(band, 0, N_PRICE_BANDS - 1))

    def __call__(self, obs: Any, masks: Any = None, rng: Any = None) -> np.ndarray:
        return self.act(obs, masks, rng)


class FixedPriceSeller(_BaseAgent):
    """State-responsive seller (name kept for compatibility).

    Opens above its floor and concedes toward the reservation price as the
    episode clock runs down and as it makes more offers. Lot size rotates so
    consecutive offers are not identical. Accepts a buyer offer at/above the
    current ask; counters below it instead of repeating a fixed price.
    """

    _LOT_SIZES = (10, 7, 5, 8, 6)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("role", kwargs.get("role", "seller"))
        super().__init__(*args, **kwargs)
        self.role = "seller"
        self._offer_round = 0

    def _current_ask(self, obs_arr: np.ndarray) -> float:
        time_left = float(obs_arr[48]) if len(obs_arr) > 48 else 1.0
        opening_premium = 30.0 if self.agent_id == "seller_a" else 20.0
        concession = self._offer_round * 5.0
        premium = max(0.0, opening_premium * float(np.clip(time_left, 0.0, 1.0)) - concession)
        premium = float(int(premium // 10.0) * 10)
        return self.reservation_price + self.margin + premium

    def act(self, obs: Any, masks: Any = None, rng: Any = None) -> np.ndarray:
        _coerce_rng(rng, self._rng)
        t_mask, p_mask, q_mask = _normalize_masks(masks)
        obs_arr = np.asarray(obs, dtype=float).reshape(-1) if obs is not None else np.zeros(0)
        active_price = float(obs_arr[2]) * 200.0 if len(obs_arr) > 2 else 0.0
        ask = self._current_ask(obs_arr)
        band = _closest(_allowed(p_mask), price_to_band(ask) - 1)
        lot = self._LOT_SIZES[self._offer_round % len(self._LOT_SIZES)]
        inventory = int(round(float(obs_arr[1]) * 100.0)) if len(obs_arr) > 1 else lot
        qty = _closest(_allowed(q_mask), max(1, min(lot, inventory or lot)) - 1)
        if t_mask[ActionType.ACCEPT] and active_price >= self.reservation_price - 1e-4:
            return self._action(ActionType.ACCEPT, band, qty)
        for atype in (ActionType.OFFER, ActionType.COUNTEROFFER):
            if t_mask[atype]:
                self._offer_round += 1
                return self._action(atype, band, qty)
        return self._action(self._fallback_type(t_mask, (int(ActionType.WAIT),)), band, qty)


class RandomBuyer(_BaseAgent):
    """Concession buyer (name kept for compatibility).

    Bids start low and rise one band per own offer toward the reservation
    price, with a seeded one-band jitter. Accepts a seller price at/below its
    current bid ceiling (price-focused Buyer A holds out longer than the
    reservation alone would), rejects prices above reservation, and counters
    between the two instead of choosing actions uniformly at random.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.role = "buyer"
        self._bid_round = 0

    def act(self, obs: Any, masks: Any = None, rng: Any = None) -> np.ndarray:
        rng = _coerce_rng(rng, self._rng)
        t_mask, p_mask, q_mask = _normalize_masks(masks)
        obs_arr = np.asarray(obs, dtype=float).reshape(-1) if obs is not None else np.zeros(0)
        active_price = float(obs_arr[2]) * 200.0 if len(obs_arr) > 2 else 0.0
        time_left = float(obs_arr[48]) if len(obs_arr) > 48 else 1.0
        jitter = 10.0 if _rand_index(rng, 4) == 0 else 0.0
        ceiling = min(self.reservation_price,
                      60.0 + self._bid_round * 10.0 + jitter)
        if self.agent_id == "buyer_a" and time_left > 0.5:
            ceiling = min(ceiling, 70.0)
        # Quantize down to a band price so the ceiling never exceeds reservation.
        ceiling = 50.0 + int((ceiling - 50.0) // 10.0) * 10.0
        band = _closest(_allowed(p_mask), int((ceiling - 50.0) // 10.0))
        qty = self._qty_idx(obs_arr, q_mask)

        if active_price > 0.0:
            if t_mask[ActionType.ACCEPT] and active_price <= ceiling + 1e-4:
                return self._action(ActionType.ACCEPT, band, qty)
            if active_price > self.reservation_price and t_mask[ActionType.REJECT]:
                return self._action(ActionType.REJECT, band, qty)
            if t_mask[ActionType.COUNTEROFFER]:
                self._bid_round += 1
                return self._action(ActionType.COUNTEROFFER, band, qty)
        if t_mask[ActionType.OFFER]:
            self._bid_round += 1
            return self._action(ActionType.OFFER, band, qty)
        return self._action(self._fallback_type(t_mask, (int(ActionType.WAIT),)), band, qty)


class TitForTatAgent(_BaseAgent):
    """Mirrors the counterparty's last action from the obs history window."""

    def act(self, obs: Any, masks: Any = None, rng: Any = None) -> np.ndarray:
        _coerce_rng(rng, self._rng)  # deterministic: no randomness used
        t_mask, p_mask, q_mask = _normalize_masks(masks)
        obs_arr = np.asarray(obs, dtype=float).reshape(-1) if obs is not None else np.zeros(0)
        qty = self._qty_idx(obs_arr, q_mask)
        allowed_p = _allowed(p_mask)
        target = self._target_band()
        last_type, last_band = self._last_history(obs_arr)
        if last_type is None:
            for atype in (ActionType.OFFER, ActionType.COUNTEROFFER):
                if t_mask[atype]:
                    return self._action(atype, _closest(allowed_p, target), qty)
            return self._action(self._fallback_type(t_mask, (int(ActionType.WAIT),)),
                                _closest(allowed_p, target), qty)
        # concede one band toward own reservation from the last quoted price
        if self.role == "seller":
            band = _closest(allowed_p, max(target, last_band - 1))
        elif self.role == "buyer":
            band = _closest(allowed_p, min(target, last_band + 1))
        else:
            band = _closest(allowed_p, last_band - 1 if last_band > 7 else last_band + 1)
        active_price = float(obs_arr[2]) * 200.0 if len(obs_arr) > 2 else 0.0
        favourable = (self.role == "seller" and active_price >= self.reservation_price) or \
                     (self.role == "buyer" and 0.0 < active_price <= self.reservation_price)
        if last_type in (int(ActionType.OFFER), int(ActionType.COUNTEROFFER)):
            if favourable and t_mask[ActionType.ACCEPT]:
                return self._action(ActionType.ACCEPT, band, qty)
            if t_mask[ActionType.COUNTEROFFER]:
                return self._action(ActionType.COUNTEROFFER, band, qty)
            if t_mask[ActionType.OFFER]:
                return self._action(ActionType.OFFER, band, qty)
        elif last_type == int(ActionType.ACCEPT) and t_mask[ActionType.ACCEPT]:
            return self._action(ActionType.ACCEPT, band, qty)
        elif last_type == int(ActionType.REJECT) and t_mask[ActionType.REJECT]:
            return self._action(ActionType.REJECT, band, qty)
        return self._action(self._fallback_type(t_mask, (int(ActionType.WAIT), int(ActionType.OFFER))), band, qty)

    @staticmethod
    def _last_history(obs_arr: np.ndarray) -> tuple[int | None, int]:
        # obs[4:45] = 5 slots: price/200, qty/10, one-hot type (see _make_obs)
        last_type, last_band = None, 7
        for i in range(5):
            base = 4 + 10 + i * 6
            if len(obs_arr) <= base + 5:
                break
            one_hot = obs_arr[base:base + 6]
            if float(np.max(one_hot)) > 0.5 or (len(obs_arr) > 4 + i and float(obs_arr[4 + i]) > 0.0):
                last_type = int(np.argmax(one_hot)) if float(np.max(one_hot)) > 0.5 else last_type
                price = float(obs_arr[4 + i]) * 200.0
                if price > 0:
                    last_band = price_to_band(price) - 1
        return last_type, int(np.clip(last_band, 0, N_PRICE_BANDS - 1))


class RandomAgent(_BaseAgent):
    """Uniformly random choice among mask-allowed type/band/qty."""

    def act(self, obs: Any, masks: Any = None, rng: Any = None) -> np.ndarray:
        rng = _coerce_rng(rng, self._rng)
        t_mask, p_mask, q_mask = _normalize_masks(masks)
        allowed_t, allowed_p, allowed_q = _allowed(t_mask), _allowed(p_mask), _allowed(q_mask)
        return self._action(allowed_t[_rand_index(rng, len(allowed_t))],
                            allowed_p[_rand_index(rng, len(allowed_p))],
                            allowed_q[_rand_index(rng, len(allowed_q))])


class MarketManagerAgent(_BaseAgent):
    """Waits while the market is active; withdraws only after a long stall.

    The stall signal is observation slot 44 (time since last offer, /100),
    so the manager no longer withdraws on its first turn by chance.
    """

    def act(self, obs: Any, masks: Any = None, rng: Any = None) -> np.ndarray:
        _coerce_rng(rng, self._rng)
        t_mask, _p, _q = _normalize_masks(masks)
        obs_arr = np.asarray(obs, dtype=float).reshape(-1) if obs is not None else np.zeros(0)
        stall = float(obs_arr[44]) * 100.0 if len(obs_arr) > 44 else 0.0
        if stall >= 30.0 and t_mask[ActionType.WITHDRAW]:
            return self._action(ActionType.WITHDRAW, 0, 0)
        return self._action(ActionType.WAIT, 0, 0)


def baseline_policy(agent_id: str, seed: int | None = None) -> _BaseAgent:
    """Return a seeded baseline agent for `agent_id` (also callable as a policy)."""
    cfg = default_agent_configs().get(agent_id)
    role = cfg.role if cfg else None
    if role == "seller":
        agent: _BaseAgent = FixedPriceSeller(agent_id, seed=seed)
    elif agent_id == "buyer_a":
        agent = RandomBuyer(agent_id, seed=seed)
    elif role == "buyer":
        agent = TitForTatAgent(agent_id, seed=seed)
    elif role == "manager":
        agent = MarketManagerAgent(agent_id, seed=seed)
    else:
        agent = RandomAgent(agent_id, seed=seed) if cfg else RandomAgent(seed=seed)
    return agent
