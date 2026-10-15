"""Held-out MLM perplexity for the three size-matched DAPT corpora.

Same protocol as evaluate_mlm.py: mask 15% of tokens in the 344 test-split
contexts at sequence length 256 under a fixed seed, and report perplexity. Run
after the training queue is empty so it does not compete for cores.
"""
import json
import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, set_seed)

BASE = Path(__file__).resolve().parents[2]   # repo root
RESULTS_DIR = BASE / "results" / "experiments" / "size_matched_dapt"
TEST_FILE = BASE / "data" / "splits" / "test.jsonl"
RUNS = BASE / "artifacts" / "models" / "size_matched"
MAX_SEQ_LENGTH, MLM_PROBABILITY, SEED = 256, 0.15, 42

MODELS = [
    ("InLegalBERT baseline", "law-ai/InLegalBERT"),
    ("auth901  (901 authentic)", RUNS / "auth901_mlm" / "final"),
    ("synth901 (901 synthetic)", RUNS / "synth901_mlm" / "final"),
    ("mixed901 (450a + 451s)", RUNS / "mixed901_mlm" / "final"),
]


def main():
    contexts = []
    seen = set()
    for line in TEST_FILE.open(encoding="utf-8"):
        line = line.strip()
        if line:
            c = json.loads(line)["context"]
            if c not in seen:
                seen.add(c)
                contexts.append(c)
    print(f"test-split contexts: {len(contexts)}\n")

    tok = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=MLM_PROBABILITY)

    rows = []
    for label, path in MODELS:
        set_seed(SEED)
        torch.manual_seed(SEED)
        model = AutoModelForMaskedLM.from_pretrained(
            str(path), local_files_only=not isinstance(path, str))
        model.eval()
        total, n = 0.0, 0
        for c in contexts:
            enc = tok(c, truncation=True, max_length=MAX_SEQ_LENGTH,
                      return_tensors="pt")
            batch = collator([{k: v[0] for k, v in enc.items()}])
            with torch.no_grad():
                out = model(**batch)
            if not math.isnan(float(out.loss)):
                total += float(out.loss)
                n += 1
        loss = total / n
        rows.append((label, loss, math.exp(loss)))
        print(f"  {label:28s} loss {loss:.4f}   ppl {math.exp(loss):7.2f}")

    base_ppl = rows[0][2]
    print("\n" + "=" * 62)
    print(f"{'model':28s}{'loss':>9s}{'PPL':>9s}{'reduction':>12s}")
    print("-" * 62)
    for label, loss, ppl in rows:
        red = "--" if ppl == base_ppl else f"{100*(1-ppl/base_ppl):.1f}%"
        print(f"{label:28s}{loss:9.4f}{ppl:9.2f}{red:>12s}")
    print("=" * 62)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "size_matched_mlm_perplexity.json").write_text(
        json.dumps([{"model": l, "loss": round(x, 4), "ppl": round(p, 2)}
                    for l, x, p in rows], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
