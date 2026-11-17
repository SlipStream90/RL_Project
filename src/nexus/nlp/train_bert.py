"""DistilBERT intent fine-tune on the synthetic corpus (template-held-out eval).

Measured gate: accuracy on templates never seen in training (split_by_template).
Run: .venv\\Scripts\\python.exe -m nexus.nlp.train_bert --epochs 2 --per-class 1000
Outputs results/bert_metrics.json and model dir results/distilbert-intent/.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nexus.nlp.bert_negotiation import INTENTS
from nexus.nlp.corpus_generator import generate_corpus, split_by_template

LABEL2ID = {name: i for i, name in enumerate(INTENTS)}


class CorpusDS(Dataset):
    def __init__(self, rows, tok, max_len=48):
        self.rows = rows; self.tok = tok; self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": torch.tensor(LABEL2ID[r["intent"]])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--freeze-bottom", action="store_true", help="freeze embeddings+3 layers (faster, weaker)")
    ap.add_argument("--out", default="results/distilbert-intent")
    ap.add_argument("--metrics", default="results/bert_metrics.json")
    a = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    corpus = generate_corpus(n_per_class=a.per_class, seed=7)
    train_rows, test_rows = split_by_template(corpus)
    print(f"train={len(train_rows)} test={len(test_rows)} held={sorted({r['template_id'] for r in test_rows})}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(INTENTS), id2label={i: n for i, n in enumerate(INTENTS)}, label2id=LABEL2ID)
    if a.freeze_bottom:
        for p in model.distilbert.embeddings.parameters():
            p.requires_grad = False
        for layer in model.distilbert.transformer.layer[:3]:
            for p in layer.parameters():
                p.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    dl = DataLoader(CorpusDS(train_rows, tok), batch_size=a.batch, shuffle=True)
    t0 = time.time()
    model.train()
    for ep in range(a.epochs):
        tot = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            opt.zero_grad(); out.loss.backward(); opt.step()
            tot += float(out.loss.item())
        print(f"epoch {ep} loss {tot / max(1, len(dl)):.4f} elapsed {time.time() - t0:.0f}s", flush=True)
    # Eval on held-out templates
    model.eval()
    dl_te = DataLoader(CorpusDS(test_rows, tok), batch_size=64)
    correct = 0; lat = []
    with torch.no_grad():
        for batch in dl_te:
            labels = batch.pop("labels")
            s = time.perf_counter()
            out = model(**{k: v.to(device) for k, v in batch.items()})
            lat.append(time.perf_counter() - s)
            correct += int((out.logits.argmax(-1).cpu() == labels).sum())
    acc = correct / max(1, len(test_rows))
    metrics = {"model": name, "train_n": len(train_rows), "test_n": len(test_rows),
               "held_out_templates": sorted({r["template_id"] for r in test_rows}),
               "accuracy": acc, "epochs": a.epochs, "device": str(device),
               "train_seconds": time.time() - t0, "batch_latency_s_median": float(np.median(lat)) if lat else None}
    os.makedirs(os.path.dirname(a.metrics) or ".", exist_ok=True)
    with open(a.metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if acc >= 0.85 else 2


if __name__ == "__main__":
    raise SystemExit(main())
