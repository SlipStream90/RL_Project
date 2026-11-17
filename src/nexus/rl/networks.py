"""Actor/critic networks (CONTRACT: actor 256x256 3 heads; critic 512/256/128)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from nexus.env.types import D_GLOBAL, D_S, N_PRICE_BANDS, Q_MAX

HEADS = (6, N_PRICE_BANDS, Q_MAX)


def _mask_logits(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return logits
    return logits.masked_fill(~mask.bool(), -1e8)


class Actor(nn.Module):
    def __init__(self, obs_dim: int = D_S):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.head_type = nn.Linear(256, HEADS[0])
        self.head_price = nn.Linear(256, HEADS[1])
        self.head_qty = nn.Linear(256, HEADS[2])

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.head_type(h), self.head_price(h), self.head_qty(h)

    def dists(self, obs: torch.Tensor, masks: dict | None = None):
        lt, lp, lq = self.forward(obs)
        if masks is not None:
            lt = _mask_logits(lt, masks.get("type"))
            lp = _mask_logits(lp, masks.get("price"))
            lq = _mask_logits(lq, masks.get("quantity"))
        return Categorical(logits=lt), Categorical(logits=lp), Categorical(logits=lq)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, masks_np: dict | None, device, greedy: bool = False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        masks = None
        if masks_np is not None:
            masks = {k: torch.as_tensor(v, dtype=torch.bool, device=device).unsqueeze(0)
                     for k, v in masks_np.items()}
        ds = self.dists(obs, masks)
        if greedy:
            acts = [d.probs.argmax(dim=-1) for d in ds]
        else:
            acts = [d.sample() for d in ds]
        logp = sum(d.log_prob(a) for d, a in zip(ds, acts))
        action = np.array([int(a.item()) for a in acts], dtype=np.int64)
        probs = {k: d.probs.squeeze(0).detach().cpu().numpy().tolist()
                 for k, d in zip(("type", "price", "quantity"), ds)}
        return action, float(logp.item()), probs


class Critic(nn.Module):
    def __init__(self, state_dim: int = D_GLOBAL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
