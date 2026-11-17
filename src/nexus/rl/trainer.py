"""IPPO + MAPPO (CTDE) trainers. IPPO: per-agent local critic. MAPPO: shared global critic."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from nexus.env.types import AGENTS, D_GLOBAL, D_S
from nexus.rl.networks import Actor, Critic

DEFAULT_CFG = dict(lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, lam=0.95, clip=0.2,
                   epochs=4, entropy=0.01, value_coef=0.5, max_grad_norm=0.5)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Trainer:
    """Unified PPO trainer; centralized=True gives MAPPO, False gives IPPO."""

    def __init__(self, env, cfg: dict | None = None, centralized: bool = True, device=None, seed: int = 0):
        self.env = env
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.centralized = centralized
        self.device = device or get_device()
        torch.manual_seed(seed)
        self.actors = {a: Actor(D_S).to(self.device) for a in AGENTS}
        if centralized:
            self.critic = Critic(D_GLOBAL).to(self.device)
            self.critics = None
        else:
            self.critic = None
            self.critics = {a: Critic(D_S).to(self.device) for a in AGENTS}
        params_a = [p for ac in self.actors.values() for p in ac.parameters()]
        self.opt_actor = torch.optim.Adam(params_a, lr=self.cfg["lr_actor"])
        crit = self.critic if centralized else None
        params_c = list(crit.parameters()) if crit is not None else [p for c in self.critics.values() for p in c.parameters()]
        self.opt_critic = torch.optim.Adam(params_c, lr=self.cfg["lr_critic"])
        self.episode_count = 0
        self.history: list[dict] = []

    # --- collection ---
    def collect_episode(self, seed: int | None = None) -> dict:
        env = self.env
        env.reset(seed=seed)
        buf: dict[str, list] = defaultdict(list)
        ep_reward: dict[str, float] = defaultdict(float)
        steps = 0
        while env.agents and steps < 5000:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            obs = env.observe(agent)
            gstate = env.get_global_state()
            masks = env.action_masks(agent)
            action, logp, _ = self.actors[agent].act(obs, masks, self.device)
            with torch.no_grad():
                cin = gstate if self.centralized else obs
                crit = self.critic if self.centralized else self.critics[agent]
                value = float(crit(torch.as_tensor(cin, dtype=torch.float32, device=self.device).unsqueeze(0)).item())
            env.step(action)
            reward = float(env.rewards.get(agent, 0.0))
            done = bool(env.terminations.get(agent, False) or env.truncations.get(agent, False)) if agent in env.terminations else True
            buf[agent].append(dict(obs=obs, global_state=gstate, action=action, log_prob=logp,
                                   reward=reward, done=done, value=value, masks=masks))
            ep_reward[agent] += reward
            steps += 1
        return {"buffer": dict(buf), "episode_reward": dict(ep_reward), "steps": steps}

    # --- GAE + update ---
    def _gae(self, traj: list) -> tuple[np.ndarray, np.ndarray]:
        gamma, lam = self.cfg["gamma"], self.cfg["lam"]
        vals = [t["value"] for t in traj] + [0.0]
        adv = np.zeros(len(traj), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(traj))):
            nonterm = 0.0 if traj[t]["done"] else 1.0
            delta = traj[t]["reward"] + gamma * vals[t + 1] * nonterm - vals[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        returns = adv + np.array([t["value"] for t in traj], dtype=np.float32)
        return adv, returns

    def update(self, buffer: dict) -> dict:
        device = self.device
        tot = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, n=0)
        for agent, traj in buffer.items():
            if len(traj) < 2:
                continue
            adv, returns = self._gae(traj)
            adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
            obs_t = torch.as_tensor(np.stack([t["obs"] for t in traj]), dtype=torch.float32, device=device)
            glob_t = torch.as_tensor(np.stack([t["global_state"] for t in traj]), dtype=torch.float32, device=device)
            act_t = torch.as_tensor(np.stack([t["action"] for t in traj]), dtype=torch.int64, device=device)
            oldlp_t = torch.as_tensor([t["log_prob"] for t in traj], dtype=torch.float32, device=device)
            masks_t = {k: torch.as_tensor(np.stack([t["masks"][k] for t in traj]), dtype=torch.bool, device=device)
                       for k in ("type", "price", "quantity")}
            crit = self.critic if self.centralized else self.critics[agent]
            cin_t = glob_t if self.centralized else obs_t
            for _ in range(int(self.cfg["epochs"])):
                ds = self.actors[agent].dists(obs_t, masks_t)
                logp = sum(d.log_prob(act_t[:, i]) for i, d in enumerate(ds))
                ent = sum(d.entropy() for d in ds).mean()
                ratio = torch.exp(logp - oldlp_t)
                surr = torch.min(ratio * adv_t, torch.clamp(ratio, 1 - self.cfg["clip"], 1 + self.cfg["clip"]) * adv_t)
                ploss = -surr.mean()
                vpred = crit(cin_t)
                vloss = ((vpred - ret_t) ** 2).mean()
                loss_a = ploss - self.cfg["entropy"] * ent
                self.opt_actor.zero_grad(); loss_a.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actors[agent].parameters(), self.cfg["max_grad_norm"])
                self.opt_actor.step()
                loss_c = self.cfg["value_coef"] * vloss
                self.opt_critic.zero_grad(); loss_c.backward()
                nn.utils.clip_grad_norm_(crit.parameters(), self.cfg["max_grad_norm"])
                self.opt_critic.step()
                tot["policy_loss"] += float(ploss.item()); tot["value_loss"] += float(vloss.item())
                tot["entropy"] += float(ent.item()); tot["n"] += 1
        n = max(1, tot["n"])
        return {k: (v / n if k != "n" else v) for k, v in tot.items()}

    def train(self, num_episodes: int, checkpoint_dir: str | None = None, checkpoint_interval: int = 100,
              log_dir: str | None = None) -> list[dict]:
        writer = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir)
            except Exception:
                writer = None
        log_path = os.path.join(log_dir, "train_log.jsonl") if log_dir else None
        for ep in range(num_episodes):
            data = self.collect_episode(seed=ep)
            losses = self.update(data["buffer"])
            self.episode_count += 1
            mean_r = float(np.mean(list(data["episode_reward"].values()))) if data["episode_reward"] else 0.0
            rec = {"episode": self.episode_count, "mean_reward": mean_r, **losses}
            self.history.append(rec)
            if writer is not None:
                writer.add_scalar("reward/mean", mean_r, self.episode_count)
                writer.add_scalar("loss/policy", losses["policy_loss"], self.episode_count)
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if checkpoint_dir and self.episode_count % checkpoint_interval == 0:
                self.save_checkpoint(os.path.join(checkpoint_dir, f"ckpt_{self.episode_count}.pt"))
        if writer is not None:
            writer.close()
        return self.history

    # --- checkpoints ---
    def save_checkpoint(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actors": {a: m.state_dict() for a, m in self.actors.items()},
            "critic": self.critic.state_dict() if self.critic is not None else None,
            "critics": {a: m.state_dict() for a, m in self.critics.items()} if self.critics else None,
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "episode_count": self.episode_count, "centralized": self.centralized,
            "config_hash": config_hash(self.cfg), "cfg": self.cfg,
        }, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        for a, m in self.actors.items():
            m.load_state_dict(ck["actors"][a])
        if self.critic is not None and ck.get("critic") is not None:
            self.critic.load_state_dict(ck["critic"])
        if self.critics and ck.get("critics"):
            for a, m in self.critics.items():
                m.load_state_dict(ck["critics"][a])
        self.episode_count = int(ck.get("episode_count", 0))

    @torch.no_grad()
    def act_greedy(self, agent: str, obs, masks) -> np.ndarray:
        action, _, _ = self.actors[agent].act(obs, masks, self.device, greedy=True)
        return action


class MAPPOTrainer(Trainer):
    def __init__(self, env, cfg=None, device=None, seed=0):
        super().__init__(env, cfg, centralized=True, device=device, seed=seed)


class IPPOTrainer(Trainer):
    def __init__(self, env, cfg=None, device=None, seed=0):
        super().__init__(env, cfg, centralized=False, device=device, seed=seed)
