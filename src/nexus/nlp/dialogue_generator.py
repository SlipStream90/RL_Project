"""Template utterance generation: structured action -> text. RL decides."""
from __future__ import annotations

import numpy as np

from nexus.env.types import band_to_price

# Every template carries its class keyword so parsers (TF-IDF/keyword) can learn it.
TEMPLATES: dict[int, list[str]] = {
    0: ["I can offer {quantity} units at ₹{price} each.",
        "I'm offering {quantity} units for ₹{price} each.",
        "My offer price is ₹{price} per unit for {quantity} units.",
        "Offer: can you take {quantity} units at ₹{price}?"],
    1: ["Counteroffer: I can supply {quantity} units, but my price is ₹{price} each.",
        "Counteroffer: {quantity} units at ₹{price} each — can we meet there?",
        "My counteroffer is {quantity} units at ₹{price} instead."],
    2: ["I accept and agree to ₹{price} per unit for {quantity} units.",
        "Accept — deal, I accept ₹{price} each for {quantity} units.",
        "I accept. Confirmed: {quantity} units at ₹{price} per unit."],
    3: ["I reject this — I cannot accept ₹{price} per unit for {quantity} units.",
        "I reject that price of ₹{price} for {quantity} units.",
        "Reject: no deal at ₹{price} each, sorry."],
    4: ["Please wait — let me consider this offer.",
        "Wait while I review the terms, please.",
        "Wait — I need to think about {quantity} units at ₹{price} each."],
    5: ["I withdraw — I'm withdrawing from this negotiation.",
        "Withdraw: I withdraw and end this negotiation now.",
        "I'm withdrawing my offer of {quantity} units at ₹{price}."],
}


def generate_utterance(action, rng: np.random.Generator | None = None) -> str:
    atype, band, qty = int(action[0]), int(action[1]), int(action[2])
    # actions use 0-based band/qty indices; band index i -> price band i+1.
    price = band_to_price(min(band, 15) + 1)
    qty_val = min(qty, 9) + 1
    opts = TEMPLATES[atype]
    idx = int(rng.integers(0, len(opts))) if rng is not None else 0
    return opts[idx].format(price=int(price) if float(price).is_integer() else price, quantity=qty_val)


# CONTRACT alias
rl_action_to_utterance = generate_utterance
