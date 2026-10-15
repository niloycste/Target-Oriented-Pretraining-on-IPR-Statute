"""STEP 3 (dapt.md sec. 5, 8-10): build the token-matched A/S/M DAPT corpora.

The archived corpora were matched on RECORD COUNT, which left authentic at
128,929 non-padding tokens against synthetic at 146,001 -- a 13.2% exposure
advantage confounded with composition. This script matches on TOKEN EXPOSURE
instead, which sec. 8 designates the primary criterion.

Conditions built here:
  A  authentic  -- every unique TRAINING-split statutory context, no exceptions
  S  synthetic  -- target-aligned passages sampled to A's token budget
  M  mixed      -- 50/50 authentic:synthetic by token exposure, at that budget

N (no DAPT) has no MLM stage by definition and is not built here (sec. 7).

Emits AUTHENTIC_TRAIN_ONLY_MANIFEST.json plus {A,S,M}_manifest.json and the
corresponding .jsonl corpora under experiments/dapt/final_cpu/.

Run: python scripts/09_dapt_final/build_final_matched_corpora.py
"""

import hashlib
import io
import json
import random
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
OUT_DIR = ROOT / "experiments" / "dapt" / "final_cpu"
SYNTH_CLEAN = OUT_DIR / "synthetic_clean_12b.jsonl"

# Fixed and independent of the training seeds (sec. 10).
CORPUS_SELECTION_SEED = 20260829
TOKENIZER = "law-ai/InLegalBERT"
MAX_LEN = 256
MIX_TARGET = 0.50  # authentic share of M, by tokens (sec. 9)

WS = re.compile(r"\s+")
norm = lambda t: WS.sub(" ", (t or "").strip().lower())
sha = lambda t: hashlib.sha256(norm(t).encode("utf8")).hexdigest()


def read_jsonl(path):
    with io.open(path, encoding="utf8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    if not SYNTH_CLEAN.exists():
        print(f"missing {SYNTH_CLEAN.relative_to(ROOT)} -- run audit_synthetic_provenance.py first")
        return 1

    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def ntok(text):
        """Non-padding ids after special tokens and truncation (audit definition)."""
        return len(tok(text, truncation=True, max_length=MAX_LEN)["input_ids"])

    # ---- A: every unique TRAIN context ------------------------------------
    train = read_jsonl(SPLITS / "train.jsonl")
    held = set()
    for s in ("val", "test"):
        held |= {norm(r["context"]) for r in read_jsonl(SPLITS / f"{s}.jsonl")}

    seen, authentic = {}, []
    for r in train:
        key = norm(r["context"])
        if key in seen:
            continue
        seen[key] = True
        assert key not in held, "train/held-out context collision"
        authentic.append({
            "context_id": sha(r["context"])[:16],
            "context": r["context"],
            "act": r.get("act"),
            "section_no": r.get("section_no"),
            "words": len(r["context"].split()),
            "tokens": ntok(r["context"]),
            "sha256": sha(r["context"]),
            "source": "train",
        })

    budget = sum(a["tokens"] for a in authentic)

    # ---- S: sample synthetic to A's token budget --------------------------
    pool = read_jsonl(SYNTH_CLEAN)
    rng = random.Random(CORPUS_SELECTION_SEED)
    rng.shuffle(pool)
    for p in pool:
        p["tokens"] = ntok(p["context"])

    def fill(records, target):
        """Greedily accumulate to target, then take whichever endpoint is closer.

        The trim must inspect out[-1]: after `break`, the loop variable holds the
        first record NOT appended, so using it would subtract a token count that
        was never added and leave `total` disagreeing with the emitted corpus.
        """
        out, total = [], 0
        for r in records:
            if total >= target:
                break
            out.append(r)
            total += r["tokens"]
        if len(out) > 1 and abs(total - out[-1]["tokens"] - target) < abs(total - target):
            total -= out.pop()["tokens"]
        return out, total

    synth, s_tokens = fill(pool, budget)

    # ---- M: 50/50 authentic:synthetic by tokens, same total budget --------
    a_half = int(round(budget * MIX_TARGET))
    auth_shuf = list(authentic)
    random.Random(CORPUS_SELECTION_SEED + 1).shuffle(auth_shuf)
    m_auth, m_a_tok = fill(auth_shuf, a_half)
    remaining = [p for p in pool if p not in synth] or pool
    m_syn, m_s_tok = fill(remaining, budget - m_a_tok)

    corpora = {
        "A": {"records": authentic, "tokens": budget},
        "S": {"records": synth, "tokens": s_tokens},
        "M": {"records": m_auth + m_syn, "tokens": m_a_tok + m_s_tok},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": "Token-matched train-only DAPT corpora (dapt.md sec. 5, 8-10)",
        "corpus_selection_seed": CORPUS_SELECTION_SEED,
        "tokenizer": TOKENIZER,
        "max_sequence_length": MAX_LEN,
        "token_definition": "non-padding token ids after special tokens and truncation",
        "primary_matching_criterion": "total non-padding MLM tokens",
        "synthetic_source": "gemma3:12b only (dapt.md sec. 6 single-generator control)",
        "token_budget": budget,
        "corpora": {},
    }

    for name, d in corpora.items():
        recs, total = d["records"], d["tokens"]
        texts = [r["context"] for r in recs]
        n_auth = sum(1 for r in recs if r.get("source") == "train")
        summary["corpora"][name] = {
            "unique_records": len(set(sha(t) for t in texts)),
            "records": len(recs),
            "repeated_records": len(recs) - len(set(sha(t) for t in texts)),
            "raw_words": sum(len(t.split()) for t in texts),
            "non_padding_tokens": total,
            "mean_tokens_per_record": round(total / max(1, len(recs)), 2),
            "delta_from_budget": total - budget,
            "delta_pct": round(100.0 * (total - budget) / budget, 3),
            "authentic_records": n_auth,
            "synthetic_records": len(recs) - n_auth,
            "authentic_token_share": round(
                sum(r["tokens"] for r in recs if r.get("source") == "train") / max(1, total), 4),
        }
        with io.open(OUT_DIR / f"{name}_manifest.json", "w", encoding="utf8") as fh:
            json.dump({"condition": name,
                       "corpus_selection_seed": CORPUS_SELECTION_SEED,
                       "stats": summary["corpora"][name],
                       "records": [{k: v for k, v in r.items() if k != "context"} for r in recs]},
                      fh, indent=2, ensure_ascii=False)
        with io.open(OUT_DIR / f"corpus_{name}.jsonl", "w", encoding="utf8") as fh:
            for r in recs:
                fh.write(json.dumps({"context": r["context"]}, ensure_ascii=False) + "\n")

    with io.open(OUT_DIR / "AUTHENTIC_TRAIN_ONLY_MANIFEST.json", "w", encoding="utf8") as fh:
        json.dump({"description": "Unique training-split statutory contexts (dapt.md sec. 5)",
                   "count": len(authentic),
                   "total_tokens": budget,
                   "records": [{k: v for k, v in r.items() if k != "context"} for r in authentic]},
                  fh, indent=2, ensure_ascii=False)

    with io.open(OUT_DIR / "MATCHED_CORPORA_SUMMARY.json", "w", encoding="utf8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"  token budget (A) : {budget:,}\n")
    hdr = f"  {'':3s} {'recs':>6s} {'uniq':>6s} {'tokens':>9s} {'d_tok':>7s} {'d%':>7s} {'auth_tok%':>10s}"
    print(hdr)
    for name in ("A", "S", "M"):
        c = summary["corpora"][name]
        print(f"  {name:3s} {c['records']:6d} {c['unique_records']:6d} "
              f"{c['non_padding_tokens']:9,d} {c['delta_from_budget']:+7d} "
              f"{c['delta_pct']:+7.3f} {100*c['authentic_token_share']:9.1f}%")
    print(f"\n  wrote manifests + corpora to {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
