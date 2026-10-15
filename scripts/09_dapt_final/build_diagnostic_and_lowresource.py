"""STEPS 5-6 (dapt.md sec. 21, 24): MLM diagnostic set and low-resource manifests.

STEP 5 -- frozen MLM diagnostic (sec. 24). The archived perplexity diagnostic was
measured on TEST contexts. This replaces it with unique VALIDATION statutory
contexts verified absent from every DAPT corpus, so language-fit can be tracked
without touching the test set.

STEP 6 -- low-resource manifests (sec. 21). Subsets are drawn at the statutory
SECTION level, stratified by Act, not by sampling QA rows independently: rows
sharing a section are not independent, and row-level sampling would leak a
section's content across the retained/discarded boundary. One fixed sampling
seed; the same subset is reused for every condition and seed.

Run: python scripts/09_dapt_final/build_diagnostic_and_lowresource.py
"""

import hashlib
import io
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
OUT_DIR = ROOT / "experiments" / "dapt" / "final_cpu"

SAMPLING_SEED = 20260829  # fixed, independent of training seeds
FRACTIONS = (0.10, 0.25, 0.50)

WS = re.compile(r"\s+")
norm = lambda t: WS.sub(" ", (t or "").strip().lower())
sha = lambda t: hashlib.sha256(norm(t).encode("utf8")).hexdigest()


def read_jsonl(path):
    with io.open(path, encoding="utf8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- STEP 5: frozen validation MLM diagnostic --------------------------
    val = read_jsonl(SPLITS / "val.jsonl")
    dapt_ctx = set()
    for name in ("corpus_A", "corpus_S", "corpus_M"):
        p = OUT_DIR / f"{name}.jsonl"
        if p.exists():
            dapt_ctx |= {norm(r["context"]) for r in read_jsonl(p)}
    style = OUT_DIR / "corpus_S_STYLE.jsonl"
    if style.exists():
        dapt_ctx |= {norm(r["context"]) for r in read_jsonl(style)}

    seen, diag, excluded = set(), [], 0
    for r in val:
        k = norm(r["context"])
        if k in seen:
            continue
        seen.add(k)
        if k in dapt_ctx:
            excluded += 1
            continue
        diag.append({"context_id": sha(r["context"])[:16], "context": r["context"],
                     "act": r.get("act"), "section_no": r.get("section_no"),
                     "words": len(r["context"].split()), "sha256": sha(r["context"])})

    with io.open(OUT_DIR / "MLM_DIAGNOSTIC_MANIFEST.json", "w", encoding="utf8") as fh:
        json.dump({"description": "Frozen validation-split MLM diagnostic (dapt.md sec. 24)",
                   "purpose": "secondary language-fit measure; never used for model selection",
                   "source_split": "val",
                   "excluded_because_in_a_dapt_corpus": excluded,
                   "dapt_corpora_checked": sorted(
                       p.stem for p in OUT_DIR.glob("corpus_*.jsonl")),
                   "unique_validation_contexts": len(seen),
                   "diagnostic_contexts": len(diag),
                   "records": [{k: v for k, v in d.items() if k != "context"} for d in diag]},
                  fh, indent=2, ensure_ascii=False)
    with io.open(OUT_DIR / "mlm_diagnostic.jsonl", "w", encoding="utf8") as fh:
        for d in diag:
            fh.write(json.dumps({"context": d["context"]}, ensure_ascii=False) + "\n")

    print(f"  MLM diagnostic: {len(diag)} of {len(seen)} unique validation contexts "
          f"({excluded} excluded as present in a DAPT corpus)")

    # ---- STEP 6: section-level, Act-stratified low-resource subsets --------
    train = read_jsonl(SPLITS / "train.jsonl")
    train_squad = read_jsonl(SPLITS / "train_squad.jsonl")
    train_squad_by_id = {str(row["id"]): row for row in train_squad}
    by_section = defaultdict(list)
    for r in train:
        # Canonical section key; falls back to the context hash when unlabelled.
        key = (r.get("act"), r.get("section_no") or sha(r["context"])[:16])
        by_section[key].append(r)

    by_act = defaultdict(list)
    for key in by_section:
        by_act[key[0]].append(key)
    for act in by_act:
        by_act[act].sort()

    print(f"  train: {len(train)} QA pairs over {len(by_section)} sections, "
          f"{len(by_act)} Acts")

    for frac in FRACTIONS:
        rng = random.Random(SAMPLING_SEED)
        chosen = []
        for act in sorted(by_act):
            secs = list(by_act[act])
            rng.shuffle(secs)
            k = max(1, int(round(len(secs) * frac)))
            chosen.extend(secs[:k])
        rows = [r for key in chosen for r in by_section[key]]
        ids = sorted(str(r["id"]) for r in rows)
        pct = int(round(frac * 100))
        per_act = {a: sum(1 for key in chosen if key[0] == a) for a in sorted(by_act)}

        with io.open(OUT_DIR / f"LOW_RESOURCE_{pct}_MANIFEST.json", "w", encoding="utf8") as fh:
            json.dump({"description": f"{pct}% low-resource IndicIPR-QA subset (dapt.md sec. 21)",
                       "sampling_seed": SAMPLING_SEED,
                       "sampling_unit": "statutory section, stratified by Act",
                       "fraction_requested": frac,
                       "sections_selected": len(chosen),
                       "sections_total": len(by_section),
                       "qa_pairs_selected": len(rows),
                       "qa_pairs_total": len(train),
                       "realised_qa_fraction": round(len(rows) / len(train), 4),
                       "sections_per_act": per_act,
                       "question_ids": ids},
                      fh, indent=2, ensure_ascii=False)
        selected_ids = set(ids)
        selected_squad = [
            row for row in train_squad if str(row["id"]) in selected_ids
        ]
        if len(selected_squad) != len(ids):
            missing = sorted(set(ids) - set(train_squad_by_id))
            raise RuntimeError(
                f"{pct}% low-resource SQuAD materialisation missing IDs: {missing[:10]}"
            )
        with io.open(OUT_DIR / f"train_squad_{pct}.jsonl", "w", encoding="utf8") as fh:
            for row in selected_squad:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {pct:3d}%: {len(chosen):4d} sections -> {len(rows):5d} QA pairs "
              f"(realised {100*len(rows)/len(train):.1f}%)")

    print(f"\n  wrote manifests to {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
