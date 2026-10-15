"""Audit how evaluation contexts entered the archived DAPT corpora.

Reproduces every figure in the provenance paragraph of the paper:

  * the authentic block holds 3,710 records, one per training QA pair
  * those collapse to 901 distinct passages after normalisation
  * against the final split, 898 are training contexts and three are test
  * each of the three contributes exactly two corpus records and two test
    questions, the signature of whole sections reassigned by a section-level
    re-split rather than of piecewise inclusion of individual items
  * per-corpus exposure, including the size-matched corpora

Run:  python scripts/06_evaluation/audit_dapt_corpus_provenance.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
MERGE = ROOT / "data" / "dapt"

CORPORA = [
    ("H' authentic-only", "pretraining_contexts_original_only.jsonl"),
    ("C' mixed", "pretraining_contexts_v2.jsonl"),
    ("I' synthetic-only", "pretraining_contexts_synth_only.jsonl"),
    ("auth901", "pretraining_contexts_auth901.jsonl"),
    ("mixed901", "pretraining_contexts_mixed901.jsonl"),
    ("synth901", "pretraining_contexts_synth901.jsonl"),
]


def load(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def main():
    splits = {n: load(SPLITS / f"{n}.jsonl") for n in ("train", "val", "test")}
    ctx = {n: {norm(r["context"]) for r in rs} for n, rs in splits.items()}

    print("=== splits ===")
    for n, rs in splits.items():
        print(f"  {n:6s} QA pairs {len(rs):5d}   distinct normalised contexts {len(ctx[n]):5d}")
    print(f"  overlap train&test {len(ctx['train'] & ctx['test'])}   "
          f"train&val {len(ctx['train'] & ctx['val'])}   "
          f"val&test {len(ctx['val'] & ctx['test'])}")

    print()
    print("=== corpus exposure ===")
    print(f"  {'corpus':20s}{'records':>9s}{'distinct':>10s}{'train':>8s}{'test':>7s}{'test Qs':>9s}")
    for label, fname in CORPORA:
        p = MERGE / fname
        if not p.exists():
            print(f"  {label:20s}  (missing {fname})")
            continue
        recs = load(p)
        passages = [r["context"] for r in recs]
        uniq = {norm(c) for c in passages}
        hits = uniq & ctx["test"]
        qs = sum(1 for r in splits["test"] if norm(r["context"]) in hits)
        print(f"  {label:20s}{len(recs):9d}{len(uniq):10d}"
              f"{len(uniq & ctx['train']):8d}{len(hits):7d}{qs:9d}")

    # Provenance signature on the authentic block. The 4,022-record source
    # pool contains 3,710 authentic records plus 312 pilot synthetic records;
    # H' is the filtered 3,710-record original-only derivative.
    print()
    print("=== H' derivation from the archived source pool ===")
    source_pool = load(MERGE / "pretraining_contexts.jsonl")
    orig = [r for r in source_pool if r.get("source") == "original"]
    h_records = load(MERGE / "pretraining_contexts_original_only.jsonl")
    print(f"  source-pool records {len(source_pool)}")
    print(f"  source=original records {len(orig)}   H' records {len(h_records)}"
          f"   exact match: {orig == h_records}")
    print(f"  training QA pairs {len(splits['train'])}"
          f"   authentic-record count match: {len(orig) == len(splits['train'])}")

    counts = Counter(norm(r["context"]) for r in orig)
    stray = set(counts) & ctx["test"]
    print()
    print("=== the test contexts, and their record/question signature ===")
    for s in sorted(stray):
        qs = [r for r in splits["test"] if norm(r["context"]) == s]
        act = qs[0].get("act", "?")
        sec = qs[0].get("section_no", "?")
        print(f"  {act} s.{sec}: {counts[s]} corpus records, {len(qs)} test questions")
    print()
    print("  A section-level re-split moves every item of a section together, so equal")
    print("  record and question counts per context are the expected signature.")


if __name__ == "__main__":
    main()
