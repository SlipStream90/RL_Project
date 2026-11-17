"""Synthetic negotiation corpus (split by template, never randomly).

Dialogue TEMPLATES (3-4/class) plus EXTRA paraphrase frames (8/class) so a
held-out template is a genuine generalization test, not the only phrasing.
"""
from __future__ import annotations

import numpy as np

from nexus.nlp.dialogue_generator import TEMPLATES

INTENT_BY_TYPE = {0: "OFFER", 1: "COUNTEROFFER", 2: "ACCEPT", 3: "REJECT", 4: "WAIT", 5: "WITHDRAW"}
PRICES = [50, 60, 70, 80, 85, 90, 100, 110, 120, 150, 180, 200]

EXTRA_TEMPLATES: dict[int, list[str]] = {
    0: ["We can provide {quantity} units priced at ₹{price} apiece.",
        "For {quantity} units, our asking price is ₹{price} each.",
        "Selling {quantity} units — ₹{price} per unit if that works.",
        "Here is my proposal: {quantity} units at ₹{price}.",
        "We have {quantity} units available for ₹{price} a unit.",
        "Pricing for {quantity} units comes to ₹{price} each.",
        "I would sell you {quantity} units at ₹{price} per unit.",
        "Quote: {quantity} units, ₹{price} each."],
    1: ["That price is high for us — how about {quantity} units at ₹{price}?",
        "We can meet at ₹{price} for {quantity} units instead.",
        "Come down to ₹{price} each for {quantity} units and we are close.",
        "Our budget allows ₹{price} per unit for {quantity} units.",
        "Split the difference: {quantity} units at ₹{price} apiece.",
        "I can go up to ₹{price} each if you supply {quantity} units.",
        "Lower it to ₹{price} for {quantity} units and I will consider signing.",
        "Revised terms from us: {quantity} units, ₹{price} each."],
    2: ["That works for us — signing off on {quantity} units at ₹{price} each.",
        "Agreed, we will take {quantity} units at ₹{price} per unit.",
        "Yes, those terms are acceptable: {quantity} units at ₹{price}.",
        "We have a deal at ₹{price} each for {quantity} units.",
        "Acceptable. Proceed with {quantity} units priced at ₹{price}.",
        "Fine by us — {quantity} units at ₹{price} apiece, agreed.",
        "Consider it settled: {quantity} units at ₹{price} per unit.",
        "Happy to agree to {quantity} units at ₹{price} each."],
    3: ["That price does not work for us on {quantity} units at ₹{price}.",
        "We must pass on {quantity} units priced at ₹{price} each.",
        "Not acceptable — ₹{price} per unit for {quantity} units is outside our range.",
        "We decline those terms ({quantity} units at ₹{price}).",
        "Sorry, we cannot proceed at ₹{price} each for {quantity} units.",
        "Those terms are refused: {quantity} units at ₹{price} apiece.",
        "We are turning down {quantity} units at ₹{price} per unit.",
        "No — ₹{price} each for {quantity} units will not be approved here."],
    4: ["Give us some time to evaluate {quantity} units at ₹{price} each.",
        "We need to check internally before answering on {quantity} units at ₹{price}.",
        "Pausing here — we will respond after reviewing ₹{price} per unit.",
        "Hold on while we assess {quantity} units at ₹{price} apiece.",
        "We are still deliberating over {quantity} units at ₹{price}.",
        "No answer yet; we are reviewing the ₹{price} terms for {quantity} units.",
        "Bear with us while we think through {quantity} units at ₹{price} each.",
        "Decision pending on {quantity} units priced at ₹{price}."],
    5: ["We are ending these talks and pulling our {quantity} units at ₹{price}.",
        "Exiting this deal — no further discussion on {quantity} units at ₹{price}.",
        "We walk away from {quantity} units priced at ₹{price} each.",
        "Cancelling our participation; {quantity} units at ₹{price} are off the table.",
        "We quit these negotiations over {quantity} units at ₹{price}.",
        "Terminating talks — {quantity} units at ₹{price} will not proceed with us.",
        "We leave the table and retract {quantity} units at ₹{price} apiece.",
        "Our side exits; no deal for {quantity} units at ₹{price}."],
}

SHORT_TEMPLATES: dict[int, list[str]] = {
    0: ["Quote — {quantity} units at ₹{price}.", "{quantity} units available, ₹{price} each.",
        "Price: ₹{price}, quantity {quantity}.", "{quantity} units, ₹{price} per unit.",
        "Supplying {quantity} at ₹{price} each.", "{quantity} units for ₹{price} each, from our side."],
    1: ["Revised price: {quantity} units at ₹{price}.", "Our terms: {quantity} units, ₹{price} each.",
        "Counter-terms — {quantity} at ₹{price}.", "We propose instead {quantity} units at ₹{price}.",
        "Modified terms: ₹{price} each, {quantity} units.", "New terms from us: {quantity} units at ₹{price}."],
    2: ["Agreed — {quantity} units at ₹{price}.", "Settled: {quantity} units, ₹{price} each.",
        "Confirmed — {quantity} at ₹{price}.", "Yes. {quantity} units at ₹{price} each.",
        "Approved: {quantity} units, ₹{price} per unit.", "Signed off: {quantity} units at ₹{price}."],
    3: ["Refused — {quantity} units at ₹{price}.", "Declined: {quantity} units, ₹{price} each.",
        "Not approved: {quantity} at ₹{price}.", "Passing on {quantity} units at ₹{price}.",
        "Turned down: {quantity} units, ₹{price} each.", "Unacceptable: {quantity} units at ₹{price}."],
    4: ["Pending: {quantity} units at ₹{price}.", "Still reviewing {quantity} units at ₹{price}.",
        "No decision yet on {quantity} at ₹{price}.", "Evaluating {quantity} units, ₹{price} each.",
        "Under review: {quantity} units at ₹{price}.", "Deliberating — {quantity} units at ₹{price}."],
    5: ["Exiting — {quantity} units at ₹{price} withdrawn.", "Retracted: {quantity} units, ₹{price} each.",
        "Out. No deal on {quantity} units at ₹{price}.", "Leaving: {quantity} units at ₹{price} cancelled.",
        "Withdrawn: {quantity} units at ₹{price}.", "We exit — {quantity} units at ₹{price} off the table."],
}

ALL_TEMPLATES: dict[int, list[tuple[str, str]]] = {
    atype: [(f"{INTENT_BY_TYPE[atype]}_{i}", t) for i, t in enumerate(TEMPLATES[atype])]
           + [(f"{INTENT_BY_TYPE[atype]}_x{i}", t) for i, t in enumerate(EXTRA_TEMPLATES[atype])]
           + [(f"{INTENT_BY_TYPE[atype]}_s{i}", t) for i, t in enumerate(SHORT_TEMPLATES[atype])]
    for atype in INTENT_BY_TYPE
}


def generate_corpus(n_per_class: int = 200, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for atype, intent in INTENT_BY_TYPE.items():
        frames = ALL_TEMPLATES[atype]
        for i in range(n_per_class):
            price = int(rng.choice(PRICES)); qty = int(rng.integers(1, 11))
            tid, tmpl = frames[int(rng.integers(0, len(frames)))]
            text = tmpl.format(price=price, quantity=qty)
            out.append({"text": text, "intent": intent, "slots": {"price": float(price), "quantity": qty},
                        "template_id": tid, "action_type": atype})
    return out


def split_by_template(corpus: list[dict], holdout_per_class: int = 1):
    held = set()
    by_intent: dict[str, set] = {}
    for r in corpus:
        by_intent.setdefault(r["intent"], set()).add(r["template_id"])
    for intent, tids in by_intent.items():
        for tid in sorted(tids)[-holdout_per_class:]:
            held.add(tid)
    train = [r for r in corpus if r["template_id"] not in held]
    test = [r for r in corpus if r["template_id"] in held]
    return train, test
