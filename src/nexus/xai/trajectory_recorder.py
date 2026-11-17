"""Trajectory store: in-memory + JSON/CSV export. Single Transition schema."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

from nexus.env.types import Transition


class TrajectoryRecorder:
    def __init__(self):
        self._by_episode: dict[str, list[Transition]] = defaultdict(list)

    def record(self, t: Transition) -> None:
        self._by_episode[t.episode_id].append(t)

    def record_many(self, ts) -> None:
        for t in ts:
            self.record(t)

    def load(self, episode_id: str) -> list[Transition]:
        return list(self._by_episode.get(episode_id, []))

    def episodes(self) -> list[str]:
        return sorted(self._by_episode)

    def query(self, episode_id=None, agent_id=None, action_type=None) -> list[Transition]:
        out = []
        pools = [self._by_episode[episode_id]] if episode_id else list(self._by_episode.values())
        for pool in pools:
            for t in pool:
                if agent_id and t.agent_id != agent_id:
                    continue
                if action_type is not None and int(t.action[0]) != int(action_type):
                    continue
                out.append(t)
        return out

    def get_step(self, episode_id: str, step_id: int) -> Transition:
        for t in self._by_episode.get(episode_id, []):
            if t.timestep == step_id:
                return t
        raise KeyError(f"step {step_id} not in {episode_id}")

    def export_json(self, episode_id: str, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump([t.to_dict() for t in self.load(episode_id)], f, indent=2)
        return path

    def export_csv(self, episode_id: str, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows = self.load(episode_id)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode_id", "timestep", "agent_id", "action", "reward", "done"])
            for t in rows:
                w.writerow([t.episode_id, t.timestep, t.agent_id, list(t.action), t.reward, t.done])
        return path
