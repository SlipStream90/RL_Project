"""Utterance -> structured action, with offer validation. NLP never decides."""
from __future__ import annotations

import numpy as np

from nexus.env.types import ActionType, price_to_band
from nexus.nlp.bert_negotiation import KeywordIntentClassifier, extract_slots

_TYPE_BY_INTENT = {"OFFER": 0, "COUNTEROFFER": 1, "ACCEPT": 2, "REJECT": 3, "WAIT": 4, "WITHDRAW": 5}


class DialogueParser:
    def __init__(self, classifier=None):
        self.classifier = classifier or KeywordIntentClassifier()

    def parse(self, utterance: str, agent_state: dict | None = None) -> dict:
        pred = self.classifier.predict(utterance)
        intent = pred["intent"]
        slots = extract_slots(utterance)
        atype = _TYPE_BY_INTENT[intent]
        price = slots.get("price")
        qty = slots.get("quantity") or 1
        band = price_to_band(price) if price is not None else 1
        action = np.array([atype, band - 1, int(qty) - 1], dtype=np.int64)
        result = {"intent": intent, "confidence": pred["confidence"], "slots": slots,
                  "action": action, "valid": True, "error": None}
        if agent_state is not None and atype in (0, 1):
            err = validate_offer(agent_state, price, qty)
            if err:
                result.update(valid=False, error=err)
        return result


def validate_offer(agent_state: dict, price, quantity) -> str | None:
    if price is None or quantity is None:
        return None
    role = agent_state.get("role")
    if role == "buyer":
        if price * quantity > float(agent_state.get("budget", np.inf)) + 1e-9:
            return "exceeds_budget"
        if price > float(agent_state.get("reservation_price", np.inf)):
            return "above_reservation"
    if role == "seller":
        if int(quantity) > int(agent_state.get("inventory", 0)):
            return "insufficient_inventory"
    return None


def parse_utterance(utterance: str, agent_state: dict | None = None) -> dict:
    return DialogueParser().parse(utterance, agent_state)


def utterance_to_rl_action(utterance: str, agent_state: dict | None = None):
    r = parse_utterance(utterance, agent_state)
    if not r["valid"]:
        raise ValueError(r["error"])
    return r["action"]
