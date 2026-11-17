import time

import numpy as np
import pytest

from nexus.nlp import (DialogueState, TfidfIntentClassifier, extract_slots,
                       generate_corpus, generate_utterance, parse_utterance,
                       split_by_template, utterance_to_rl_action, validate_offer)
from nexus.nlp.dialogue_generator import TEMPLATES


def test_templates_cover_six_types():
    assert set(TEMPLATES) == {0, 1, 2, 3, 4, 5}
    assert all(len(v) >= 3 for v in TEMPLATES.values())


def test_corpus_and_template_split():
    c = generate_corpus(n_per_class=200, seed=0)
    assert len(c) >= 1000
    train, test = split_by_template(c)
    assert train and test
    assert not ({r["template_id"] for r in train} & {r["template_id"] for r in test})


def test_tfidf_accuracy_template_holdout():
    train, test = split_by_template(generate_corpus(n_per_class=200, seed=1))
    clf = TfidfIntentClassifier().fit([r["text"] for r in train], [r["intent"] for r in train])
    acc = clf.score([r["text"] for r in test], [r["intent"] for r in test])
    assert acc >= 0.85, acc


def test_slots():
    s = extract_slots("I can offer 10 units at ₹90 each.")
    assert s["price"] == 90.0 and s["quantity"] == 10


def test_roundtrip():
    for atype in range(6):
        action = np.array([atype, 4, 9])  # band 5 -> ₹90, qty 10
        text = generate_utterance(action)
        back = utterance_to_rl_action(text)
        assert int(back[0]) == atype, (text, back)
        if atype in (0, 1, 2, 3):  # WAIT/WITHDRAW utterances carry no price/qty slots
            assert int(back[1]) == 4 and int(back[2]) == 9, (text, back)


def test_validation():
    assert validate_offer({"role": "buyer", "budget": 100, "reservation_price": 200}, 90, 10) == "exceeds_budget"
    assert validate_offer({"role": "seller", "inventory": 5}, 90, 10) == "insufficient_inventory"
    assert validate_offer({"role": "buyer", "budget": 10000, "reservation_price": 200}, 90, 10) is None


def test_dialogue_history_and_latency():
    ds = DialogueState()
    for i in range(5):
        ds.add_turn("ep_1", "buyer_a", f"turn {i}", "OFFER", {"price": 90})
    assert len(ds.history("ep_1")) == 5
    p = parse_utterance("I can offer 10 units at ₹90 each.")
    assert p["intent"] in ("OFFER", "COUNTEROFFER")
    t0 = time.perf_counter()
    for _ in range(20):
        parse_utterance("I can offer 10 units at ₹90 each.")
    assert (time.perf_counter() - t0) / 20 < 0.05
