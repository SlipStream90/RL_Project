"""Per-episode dialogue history (last-k turns)."""
from __future__ import annotations

from collections import defaultdict


class DialogueState:
    def __init__(self, k: int = 20):
        self.k = k
        self._hist: dict[str, list[dict]] = defaultdict(list)

    def add_turn(self, episode_id: str, speaker: str, utterance: str, intent: str, slots: dict, action=None) -> dict:
        rec = {"episode_id": episode_id, "speaker": speaker, "utterance": utterance,
               "intent": intent, "slots": dict(slots or {}), "action": action,
               "turn": len(self._hist[episode_id])}
        self._hist[episode_id].append(rec)
        return rec

    def history(self, episode_id: str) -> list[dict]:
        return list(self._hist.get(episode_id, []))

    def last_k(self, episode_id: str, k: int | None = None) -> list[dict]:
        h = self._hist.get(episode_id, [])
        return h[-(k or self.k):]

    def clear(self, episode_id: str) -> None:
        self._hist.pop(episode_id, None)
