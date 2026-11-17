"""NexusMarketplaceEnv — PettingZoo AEC marketplace (CONTRACT v1)."""
from __future__ import annotations

import copy
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from nexus.env.types import (
    AGENTS, BUYERS, SELLERS, D_GLOBAL, D_S, HISTORY_K, N_PRICE_BANDS,
    Q_MAX, ActionType, AgentConfig, Transition, band_to_price,
    default_agent_configs,
)


def env(config: dict | None = None, render_mode: str | None = None):
    e = NexusMarketplaceEnv(config=config, render_mode=render_mode)
    e = wrappers.OrderEnforcingWrapper(e)
    return e


class NexusMarketplaceEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "nexus_marketplace_v0"}

    def __init__(self, config: dict | None = None, render_mode: str | None = None):
        super().__init__()
        cfg = dict(config or {})
        self.render_mode = render_mode
        self.max_steps: int = int(cfg.get("max_steps", 100))
        self._configs: dict[str, AgentConfig] = default_agent_configs()
        for aid, over in (cfg.get("agents") or {}).items():
            base = self._configs[aid]
            for k, v in over.items():
                setattr(base, k, v)
        self.possible_agents: list[str] = list(AGENTS)
        self._agent_order = list(AGENTS)
        self.observation_spaces: dict[str, spaces.Box] = {
            a: spaces.Box(0.0, 1.0, shape=(D_S,), dtype=np.float32) for a in AGENTS
        }
        self.action_spaces: dict[str, spaces.MultiDiscrete] = {
            a: spaces.MultiDiscrete(np.array([6, N_PRICE_BANDS, Q_MAX], dtype=np.int64)) for a in AGENTS
        }
        self.agents: list[str] = []
        self.agent_selection: str = AGENTS[0]
        self.rewards: dict[str, float] = {a: 0.0 for a in AGENTS}
        self.terminations: dict[str, bool] = {a: False for a in AGENTS}
        self.truncations: dict[str, bool] = {a: False for a in AGENTS}
        self.infos: dict[str, dict] = {a: {} for a in AGENTS}
        self._cumulative_rewards: dict[str, float] = {a: 0.0 for a in AGENTS}
        self._np_random: np.random.Generator = np.random.default_rng(0)
        # dynamic state (also reset in reset())
        self._step_count = 0
        self._episode_id = "ep_0000"
        self._episode_counter = 0
        self._budgets: dict[str, float] = {}
        self._inventories: dict[str, int] = {}
        self._remaining: dict[str, int] = {}
        self._active_offer: tuple[str, float, int] | None = None
        self._history: list[tuple[float, int, int]] = []
        self._tx_prices: list[float] = []
        self._deals: list[dict[str, Any]] = []
        self._withdrawn: set[str] = set()
        self._dealt: set[str] = set()
        self._trajectory: list[Transition] = []
        self._agent_idx = 0

    # --- spaces helpers ---
    def observation_space(self, agent: str):  # type: ignore[override]
        return self.observation_spaces[agent]

    def action_space(self, agent: str):  # type: ignore[override]
        return self.action_spaces[agent]

    @property
    def current_episode_id(self) -> str:
        return self._episode_id

    def seed(self, seed: int | None = None) -> None:
        self._np_random = np.random.default_rng(seed)

    # --- core API ---
    def reset(self, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self._episode_counter += 1
        self._episode_id = f"ep_{self._episode_counter:04d}"
        self.agents = list(self._agent_order)
        self._step_count = 0
        self._agent_idx = 0
        self.agent_selection = self.agents[0]
        self.rewards = {a: 0.0 for a in AGENTS}
        self.terminations = {a: False for a in AGENTS}
        self.truncations = {a: False for a in AGENTS}
        self.infos = {a: {} for a in AGENTS}
        self._cumulative_rewards = {a: 0.0 for a in AGENTS}
        self._budgets = {a: float(self._configs[a].budget) for a in AGENTS}
        self._inventories = {a: int(self._configs[a].inventory) for a in AGENTS}
        self._remaining = {a: int(self._configs[a].required_qty) for a in AGENTS}
        self._active_offer = None
        self._history = []
        self._tx_prices = []
        self._deals = []
        self._withdrawn = set()
        self._dealt = set()
        self._trajectory = []
        return None

    def observe(self, agent: str) -> np.ndarray:  # type: ignore[override]
        return self._make_obs(agent)

    def state(self) -> np.ndarray:  # type: ignore[override]
        return self.get_global_state()

    def last(self, observe: bool = True):  # type: ignore[override]
        agent = self.agent_selection
        obs = self.observe(agent) if observe else None
        return obs, self._cumulative_rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]

    def _make_obs(self, agent: str) -> np.ndarray:
        cfg = self._configs[agent]
        b = np.clip(self._budgets.get(agent, 0.0) / 10000.0, 0, 1)
        inv = np.clip(self._inventories.get(agent, 0) / 100.0, 0, 1)
        p = np.clip((self._active_offer[1] / 200.0) if self._active_offer else 0.0, 0, 1)
        q = np.clip(self._remaining.get(agent, 0) / max(1, Q_MAX), 0, 1)
        hist = np.zeros(41, dtype=np.float32)
        recent = self._history[-HISTORY_K:]
        for i, (hp, hq, hi) in enumerate(recent):
            hist[i] = np.clip(hp / 200.0, 0, 1)
            hist[HISTORY_K + i] = np.clip(hq / Q_MAX, 0, 1)
            if 0 <= hi < 6:
                hist[2 * HISTORY_K + i * 6 + hi] = 1.0
        hist[40] = np.clip(self._steps_since_offer() / 100.0, 0, 1)
        avgp = float(np.mean(self._tx_prices)) if self._tx_prices else 0.0
        live_b = sum(1 for a in BUYERS if a in self.agents and a not in self._withdrawn)
        live_s = sum(1 for a in SELLERS if a in self.agents and a not in self._withdrawn)
        total_inv = sum(self._inventories.get(a, 0) for a in SELLERS)
        total_dem = sum(self._remaining.get(a, 0) for a in BUYERS) or 1
        market = np.array([
            np.clip(avgp / 200.0, 0, 1), live_b / 2.0, live_s / 2.0,
            np.clip(1.0 - self._step_count / max(1, self.max_steps), 0, 1),
            np.clip(total_inv / max(1, total_dem * 10), 0, 1),
        ], dtype=np.float32)
        obs = np.concatenate([np.array([b, inv, p, q], dtype=np.float32), hist, market]).astype(np.float32)
        assert obs.shape == (D_S,)
        # manager sees aggregates in budget/inventory slots
        if cfg.role == "manager":
            obs[0] = np.clip(total_inv / 200.0, 0, 1)
            obs[1] = np.clip(total_dem / 20.0, 0, 1)
        return obs

    def _steps_since_offer(self) -> int:
        if not self._history:
            return self._step_count
        return 0 if self._active_offer else min(self._step_count, 100)

    def get_global_state(self) -> np.ndarray:
        parts = [self._make_obs(a) for a in AGENTS]
        avgp = float(np.mean(self._tx_prices)) if self._tx_prices else 0.0
        agg = np.array([
            np.clip(avgp / 200.0, 0, 1),
            np.clip(sum(self._inventories.get(a, 0) for a in SELLERS) / 200.0, 0, 1),
            np.clip(sum(self._remaining.get(a, 0) for a in BUYERS) / 20.0, 0, 1),
            np.clip(self._step_count / max(1, self.max_steps), 0, 1),
        ], dtype=np.float32)
        g = np.concatenate(parts + [agg]).astype(np.float32)
        assert g.shape == (D_GLOBAL,), g.shape
        return g

    # --- masking ---
    def action_masks(self, agent: str) -> dict[str, np.ndarray]:
        type_mask = np.zeros(6, dtype=bool)
        type_mask[ActionType.WAIT] = True
        type_mask[ActionType.WITHDRAW] = True
        role = self._configs[agent].role
        has_offer = self._active_offer is not None
        is_offerer = has_offer and self._active_offer[0] == agent
        if role == "manager":
            return {"type": type_mask, "price": np.ones(N_PRICE_BANDS, dtype=bool),
                    "quantity": np.ones(Q_MAX, dtype=bool)}
        if not has_offer:
            type_mask[ActionType.OFFER] = True
        elif not is_offerer:
            type_mask[ActionType.COUNTEROFFER] = True
            type_mask[ActionType.REJECT] = True
            if self._can_accept(agent):
                type_mask[ActionType.ACCEPT] = True
        price_mask = np.ones(N_PRICE_BANDS, dtype=bool)
        qty_mask = np.ones(Q_MAX, dtype=bool)
        for band in range(1, N_PRICE_BANDS + 1):
            price = band_to_price(band)
            for qty in range(1, Q_MAX + 1):
                if not self._offer_valid(agent, price, qty):
                    pass  # per-pair validity checked in step; heads stay enabled unless no pair valid
        # disable price/qty heads entirely if no valid pair exists for offer-like actions
        if not any(self._offer_valid(agent, band_to_price(b), q)
                   for b in range(1, N_PRICE_BANDS + 1) for q in range(1, Q_MAX + 1)):
            price_mask[:] = False
            qty_mask[:] = False
            price_mask[0] = True
            qty_mask[0] = True
        return {"type": type_mask, "price": price_mask, "quantity": qty_mask}

    # Back-compat singular name used in PRDs
    def action_mask(self, agent: str) -> dict[str, np.ndarray]:
        return self.action_masks(agent)

    def invalid_action_mask(self, agent: str) -> dict[str, np.ndarray]:
        m = self.action_masks(agent)
        return {k: ~v for k, v in m.items()}

    def _offer_valid(self, agent: str, price: float, qty: int) -> bool:
        cfg = self._configs[agent]
        if cfg.role == "buyer":
            return price * qty <= self._budgets.get(agent, 0.0) + 1e-9 and qty <= max(1, self._remaining.get(agent, qty))
        if cfg.role == "seller":
            return qty <= self._inventories.get(agent, 0)
        return False

    def _can_accept(self, agent: str) -> bool:
        if not self._active_offer:
            return False
        offerer, price, qty = self._active_offer
        cfg = self._configs[agent]
        offerer_role = self._configs[offerer].role
        # A deal can only settle between opposite sides of the market.
        if {cfg.role, offerer_role} != {"buyer", "seller"}:
            return False
        if cfg.role == "buyer":
            return price * qty <= self._budgets.get(agent, 0.0) + 1e-9
        if cfg.role == "seller":
            return qty <= self._inventories.get(agent, 0)
        return False

    def _is_valid(self, agent: str, atype: int, band: int, qty: int) -> bool:
        masks = self.action_masks(agent)
        if not masks["type"][atype]:
            return False
        if atype in (int(ActionType.OFFER), int(ActionType.COUNTEROFFER)):
            if not (masks["price"][band - 1] and masks["quantity"][qty - 1]):
                return False
            return self._offer_valid(agent, band_to_price(band), qty)
        if atype == int(ActionType.ACCEPT):
            return self._can_accept(agent)
        return True

    # --- rewards ---
    def _terminal_components(self, agent: str) -> dict[str, float]:
        cfg = self._configs[agent]
        deals = [d for d in self._deals if agent in (d["buyer"], d["seller"])]
        completed = 1.0 if deals else 0.0
        w = cfg.reward_weights or {}
        if cfg.role == "buyer":
            if deals:
                pavg = float(np.mean([d["price"] for d in deals]))
                pe = 1.0 - pavg / cfg.reservation_price if cfg.reservation_price else 0.0
                tdeal = deals[0]["timestep"]
                speed = 1.0 - tdeal / max(1, self.max_steps)
            else:
                pe, speed = 0.0, 0.0
            return {"price_efficiency": float(w.get("price", 0.5) * pe),
                    "speed_bonus": float(w.get("speed", 0.2) * speed),
                    "completion_bonus": float(w.get("completion", 0.2) * completed),
                    "constraint_penalty": 0.0}
        if cfg.role == "seller" and agent == "seller_a":
            if deals:
                pavg = float(np.mean([d["price"] for d in deals]))
                denom = max(1e-9, cfg.reservation_price - cfg.unit_cost)
                margin = (pavg - cfg.unit_cost) / denom
            else:
                margin = 0.0
            unsold = self._inventories.get(agent, 0) / max(1, self._configs[agent].inventory or 1)
            return {"margin": float(0.6 * margin), "completion_bonus": float(0.3 * completed),
                    "volume": float(-0.1 * unsold)}
        if cfg.role == "seller":
            sold = sum(d["quantity"] for d in deals if d["seller"] == agent)
            fill = sold / max(1, cfg.required_qty or 10)
            return {"volume": float(0.6 * fill), "completion_bonus": float(0.3 * completed), "margin": 0.0}
        # manager
        if self._tx_prices and np.mean(self._tx_prices) > 0:
            cv = float(np.std(self._tx_prices) / np.mean(self._tx_prices))
            stab = 1.0 - min(cv, 1.0)
        else:
            stab = 0.0
        attempted = max(1, len(self._history))
        rate = len(self._deals) / attempted
        return {"stability": float(0.5 * stab), "completion_bonus": float(0.5 * min(rate, 1.0))}

    def _all_done(self) -> bool:
        if self._step_count >= self.max_steps:
            return True
        live = [a for a in self.agents]
        if not live:
            return True
        return all(a in self._withdrawn or a in self._dealt for a in AGENTS if self._configs[a].role != "manager") and (
            "market_manager" in self._withdrawn or self._step_count >= self.max_steps or len(self._dealt) >= 4)

    # --- step ---
    def step(self, action) -> None:  # type: ignore[override]
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        if action is None:
            raise ValueError("live agent requires an action, got None")
        # cumulative reward restarts once the agent has read it via last() and acts
        self._cumulative_rewards[agent] = 0.0
        obs_before = self._make_obs(agent)
        global_before = self.get_global_state()
        masks_before = self.action_masks(agent)
        arr = np.asarray(action, dtype=np.int64).reshape(-1)
        atype, band_idx, qty_idx = int(arr[0]), int(arr[1]), int(arr[2])
        band, qty = band_idx + 1, qty_idx + 1
        price = band_to_price(band)
        self._clear_rewards()
        comps: dict[str, float] = {}
        info: dict[str, Any] = {}
        valid = (0 <= atype < 6) and self._is_valid(agent, atype, band, qty)
        if not valid:
            self.rewards[agent] = -0.1
            comps = {"invalid_penalty": -0.1}
            info["invalid"] = True
        else:
            info["invalid"] = False
            if atype in (int(ActionType.OFFER), int(ActionType.COUNTEROFFER)):
                self._active_offer = (agent, price, qty)
                self.rewards[agent] = 0.01
                comps = {"activity_bonus": 0.01}
                self._history.append((price, qty, atype))
            elif atype == int(ActionType.ACCEPT):
                assert self._active_offer is not None
                offerer, p, q = self._active_offer
                buyer = agent if self._configs[agent].role == "buyer" else offerer
                seller = agent if self._configs[agent].role == "seller" else offerer
                # only settle buyer<->seller pairs; otherwise clear offer
                if self._configs[buyer].role == "buyer" and self._configs[seller].role == "seller":
                    self._budgets[buyer] -= p * q
                    self._inventories[seller] -= q
                    self._inventories[buyer] = self._inventories.get(buyer, 0) + q
                    self._remaining[buyer] = max(0, self._remaining.get(buyer, 0) - q)
                    self._tx_prices.append(p)
                    self._deals.append({"buyer": buyer, "seller": seller, "price": p,
                                        "quantity": q, "timestep": self._step_count})
                    self._dealt.add(buyer); self._dealt.add(seller)
                self._history.append((p, q, atype))
                self._active_offer = None
                comps = {"completion_bonus": 0.0}
            elif atype == int(ActionType.REJECT):
                if self._active_offer:
                    self._history.append((self._active_offer[1], self._active_offer[2], atype))
                self._active_offer = None
                comps = {"activity_bonus": 0.0}
            elif atype == int(ActionType.WAIT):
                self.rewards[agent] = -0.01
                comps = {"stall_penalty": -0.01}
            elif atype == int(ActionType.WITHDRAW):
                self._withdrawn.add(agent)
                self.rewards[agent] = -0.1
                comps = {"withdraw_penalty": -0.1}
        self._step_count += 1
        # termination checks (only agents still tracked; removed agents must not be re-added)
        for a in list(self.agents):
            if a in self._withdrawn:
                self.terminations[a] = True
        if self._step_count >= self.max_steps:
            for a in list(self.agents):
                if not self.terminations[a]:
                    self.truncations[a] = True
        if self._all_done():
            for a in list(self.agents):
                if a not in self._withdrawn and self._step_count < self.max_steps:
                    self.terminations[a] = True
            # add terminal rewards once
            for a in list(self.agents):
                tc = self._terminal_components(a)
                self.rewards[a] += float(sum(tc.values()))
                self.infos[a] = {**self.infos.get(a, {}), "terminal_components": tc}
        done = self.terminations[agent] or self.truncations[agent]
        self.infos[agent] = {**info, "reward_components": comps}
        self._accumulate_rewards()
        obs_after = self._make_obs(agent)
        self._trajectory.append(Transition(
            episode_id=self._episode_id, timestep=self._step_count - 1, agent_id=agent,
            observation=obs_before, global_state=global_before,
            action=(atype, band, qty), reward=float(self.rewards[agent]),
            reward_components=dict(comps), next_observation=obs_after, done=done,
            available_actions={k: v.tolist() for k, v in masks_before.items()},
        ))
        self._advance()
        self.agent_selection = self._deads_step_first()

    def _advance(self) -> None:
        if not self.agents:
            return
        # next live agent round-robin after current selection
        try:
            idx = self._agent_order.index(self.agent_selection)
        except ValueError:
            idx = -1
        for k in range(1, len(self._agent_order) + 1):
            cand = self._agent_order[(idx + k) % len(self._agent_order)]
            if cand in self.agents:
                self.agent_selection = cand
                return
        self.agent_selection = self.agents[0]

    def get_trajectory(self, episode_id: str | None = None) -> list[Transition]:
        if episode_id is None or episode_id == self._episode_id:
            return list(self._trajectory)
        return [t for t in self._trajectory if t.episode_id == episode_id]

    # --- snapshot ---
    def get_state(self) -> dict[str, Any]:
        return copy.deepcopy({
            "step_count": self._step_count, "episode_id": self._episode_id,
            "episode_counter": self._episode_counter, "budgets": self._budgets,
            "inventories": self._inventories, "remaining": self._remaining,
            "active_offer": self._active_offer, "history": self._history,
            "tx_prices": self._tx_prices, "deals": self._deals,
            "withdrawn": self._withdrawn, "dealt": self._dealt,
            "agents": self.agents, "agent_selection": self.agent_selection,
            "terminations": self.terminations, "truncations": self.truncations,
            "cumulative": self._cumulative_rewards,
            "rng_state": self._np_random.bit_generator.state,
        })

    def set_state(self, state: dict[str, Any]) -> None:
        s = copy.deepcopy(state)
        self._step_count = s["step_count"]; self._episode_id = s["episode_id"]
        self._episode_counter = s["episode_counter"]; self._budgets = s["budgets"]
        self._inventories = s["inventories"]; self._remaining = s["remaining"]
        self._active_offer = s["active_offer"]; self._history = s["history"]
        self._tx_prices = s["tx_prices"]; self._deals = s["deals"]
        self._withdrawn = s["withdrawn"]; self._dealt = s["dealt"]
        self.agents = s["agents"]; self.agent_selection = s["agent_selection"]
        self.terminations = s["terminations"]; self.truncations = s["truncations"]
        self._cumulative_rewards = s["cumulative"]
        self._np_random.bit_generator.state = s["rng_state"]

    def render(self):  # type: ignore[override]
        if self.render_mode == "human":
            print(f"step={self._step_count} offer={self._active_offer} deals={self._deals}")
        return None

    def close(self):  # type: ignore[override]
        return None
