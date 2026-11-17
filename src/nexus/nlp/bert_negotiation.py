"""Intent classification. Default: TF-IDF + LogisticRegression fallback (zero-train,
<1ms). Optional DistilBERT fine-tune path is used only if transformers+torch are
available and train_bert() is called explicitly. RL decides; this only parses."""
from __future__ import annotations

import re

import numpy as np

INTENTS = ["OFFER", "COUNTEROFFER", "ACCEPT", "REJECT", "WAIT", "WITHDRAW"]

_KEYWORDS = {
    "ACCEPT": ["agree", "accept", "deal", "confirm", "okay", "ok ", "yes"],
    "REJECT": ["cannot accept", "reject", "decline", "no deal", "refuse", "too high", "too low"],
    "WITHDRAW": ["withdraw", "exit", "leave", "end this negotiation", "pull out"],
    "WAIT": ["consider", "think", "wait", "hold on", "let me check", "review"],
    "COUNTEROFFER": ["counter", "but my price", "my price is", "instead", "meet me", "compromise"],
    "OFFER": ["offer", "supply", "provide", "sell", "purchase", "buy", "price is", "units at"],
}


def extract_slots(text: str) -> dict:
    t = text or ""
    price = None
    m = re.search(r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:/-|per unit|each|/unit)?", t)
    # prefer number near currency/unit words
    mp = re.search(r"(?:₹|rs\.?|inr)\s*(\d+(?:\.\d+)?)", t, re.I) or re.search(r"(\d+(?:\.\d+)?)\s*(?:per unit|each|/unit|rupees)", t, re.I)
    if mp:
        price = float(mp.group(1))
    elif m:
        price = float(m.group(1))
    qty = None
    mq = re.search(r"(\d+)\s*(?:units?|pcs|pieces|items)", t, re.I)
    if mq:
        qty = int(mq.group(1))
    res = None
    mr = re.search(r"\b(widgets?|gadgets?|components?|units?|goods|products?)\b", t, re.I)
    if mr:
        res = mr.group(1).lower()
    dl = re.search(r"(?:by|before|deadline[:\s]*)\s*(\d{1,2}[:/]\d{2}|\d+\s*(?:days?|hours?|weeks?))", t, re.I)
    return {"price": price, "quantity": qty, "resource_type": res,
            "deadline": dl.group(1) if dl else None}


class KeywordIntentClassifier:
    """Deterministic fallback; negation/withdrawal take priority over ACCEPT."""

    _RULES = [
        ("WITHDRAW", ["withdraw", "pull out", "exit this", "exits", "walk away", "off the table",
                      "retract", "we exit", "leaving:", "ending these talks", "quit these"]),
        ("REJECT", ["cannot accept", "reject", "decline", "no deal", "refuse"]),
        ("COUNTEROFFER", ["counteroffer", "counter offer", "but my price", "meet there", "instead",
                          "revised", "modified terms", "new terms", "our terms:"]),
        ("ACCEPT", ["agree", "accept", "deal —", "deal -", "confirmed", "settled:", "signed off"]),
        ("WAIT", ["consider", "wait while", "review the terms", "think about", "hold on",
                  "pending", "still reviewing", "under review", "evaluating", "deliberating",
                  "no decision yet", "decision pending"]),
        ("OFFER", ["offer", "offering", "supply", "purchase", "quote"]),
    ]

    def predict(self, text: str) -> dict:
        low = f" {(text or '').lower()} "
        for intent, kws in self._RULES:
            if any(k in low for k in kws):
                return {"intent": intent, "confidence": 0.9, "scores": {intent: 1}}
        scores = {}
        for intent, kws in _KEYWORDS.items():
            s = sum(1 for k in kws if k in low)
            if s:
                scores[intent] = s
        if not scores:
            return {"intent": "WAIT", "confidence": 0.3, "scores": {}}
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        return {"intent": best, "confidence": float(0.5 + 0.5 * scores[best] / max(1, total)), "scores": scores}


class TfidfIntentClassifier:
    def __init__(self):
        self._model = None
        self._vec = None

    def fit(self, texts: list[str], labels: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self._vec = TfidfVectorizer(ngram_range=(1, 2))
        X = self._vec.fit_transform(texts)
        self._model = LogisticRegression(max_iter=1000).fit(X, labels)
        return self

    def predict(self, text: str) -> dict:
        # Hybrid (PRD Week3 risk table): deterministic rules win when confident;
        # TF-IDF handles the remainder. DistilBERT fine-tune can replace TF-IDF.
        kw = KeywordIntentClassifier().predict(text)
        if kw["confidence"] >= 0.9:
            return kw
        if self._model is None:
            return kw
        X = self._vec.transform([text])
        proba = self._model.predict_proba(X)[0]
        i = int(np.argmax(proba))
        return {"intent": str(self._model.classes_[i]), "confidence": float(proba[i]),
                "scores": {str(c): float(p) for c, p in zip(self._model.classes_, proba)}}

    def score(self, texts: list[str], labels: list[str]) -> float:
        preds = [self.predict(t)["intent"] for t in texts]
        return float(np.mean([p == y for p, y in zip(preds, labels)])) if labels else 0.0


def get_classifier(kind: str = "keyword"):
    if kind == "tfidf":
        return TfidfIntentClassifier()
    return KeywordIntentClassifier()
