"""STEPS 4b/7 (dapt.md sec. 18, 25, 26): trim S_STYLE to budget and characterise corpora.

Two jobs, deliberately in one script because the second validates the first.

1. Build corpus_S_STYLE.jsonl by trimming the generated style-only pool to S's
   exact non-padding token budget, using the same greedy fill as the A/S/M
   builder so the matching rule is identical across every condition.

2. Characterise A, S, M and S_STYLE (sec. 25) and run the near-duplicate audit
   (sec. 26). This exists to pre-empt the obvious referee objection to
   Experiment B: that S_STYLE underperformed because it is simply worse text.
   If S and S_STYLE match on length, type-token ratio, legal-marker rate and
   duplicate rate, that objection does not hold.

Analysis only -- no record is deleted on the basis of these numbers (sec. 26).

Run: python scripts/09_dapt_final/characterise_corpora.py
"""

import hashlib
import io
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "dapt" / "final_cpu"
GENERATED = OUT_DIR / "style_only_generated.jsonl"

TOKENIZER = "law-ai/InLegalBERT"
MAX_LEN = 256
CORPUS_SELECTION_SEED = 20260829
JACCARD_THRESHOLD = 0.80   # predefined, before any downstream result (sec. 26)
SHINGLE_N = 5

WS = re.compile(r"\s+")
norm = lambda t: WS.sub(" ", (t or "").strip().lower())
sha = lambda t: hashlib.sha256(norm(t).encode("utf8")).hexdigest()

LEGAL_MARKERS = ["shall", "may", "provided that", "subject to", "thereof",
                 "therein", "notwithstanding", "herein", "prescribed", "under this act"]


def read_jsonl(path):
    with io.open(path, encoding="utf8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def shingles(text, n=SHINGLE_N):
    t = norm(text).split()
    return {" ".join(t[i:i + n]) for i in range(max(0, len(t) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def describe(name, texts, tok):
    toks = [len(tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]) for t in texts]
    words = [len(t.split()) for t in texts]
    all_w = [w for t in texts for w in re.findall(r"\w+", norm(t))]
    bi = [" ".join(all_w[i:i + 2]) for i in range(len(all_w) - 1)]
    hashes = [sha(t) for t in texts]
    return {
        "records": len(texts),
        "total_tokens": sum(toks),
        "mean_tokens": round(st.mean(toks), 2) if toks else 0,
        "mean_words": round(st.mean(words), 2) if words else 0,
        "median_words": st.median(words) if words else 0,
        "p10_p90_words": [round(st.quantiles(words, n=10)[0]),
                          round(st.quantiles(words, n=10)[-1])] if len(words) > 10 else None,
        "type_token_ratio": round(len(set(all_w)) / max(1, len(all_w)), 4),
        "distinct_1": round(len(set(all_w)) / max(1, len(all_w)), 4),
        "distinct_2": round(len(set(bi)) / max(1, len(bi)), 4),
        "exact_duplicate_rate": round(1 - len(set(hashes)) / max(1, len(hashes)), 4),
        "legal_marker_rate_per_100w": {
            m: round(100.0 * sum(norm(t).count(m) for t in texts) / max(1, sum(words)), 3)
            for m in LEGAL_MARKERS},
    }


def near_dupe(texts, sample=400, seed=CORPUS_SELECTION_SEED):
    """Sampled pairwise n-gram Jaccard; exhaustive comparison is unnecessary here."""
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    if len(idx) > sample:
        idx = rng.sample(idx, sample)
    sh = {i: shingles(texts[i]) for i in idx}
    hits = 0
    pairs = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            pairs += 1
            if jaccard(sh[idx[a]], sh[idx[b]]) >= JACCARD_THRESHOLD:
                hits += 1
    return {"pairs_compared": pairs, "near_duplicate_pairs": hits,
            "rate": round(hits / max(1, pairs), 5), "sampled_records": len(idx)}


def cross_near(a_texts, b_texts, sample=300, seed=CORPUS_SELECTION_SEED):
    rng = random.Random(seed)
    A = a_texts if len(a_texts) <= sample else rng.sample(a_texts, sample)
    B = b_texts if len(b_texts) <= sample else rng.sample(b_texts, sample)
    sa = [shingles(t) for t in A]
    sb = [shingles(t) for t in B]
    best, hits = 0.0, 0
    for x in sa:
        for y in sb:
            j = jaccard(x, y)
            best = max(best, j)
            if j >= JACCARD_THRESHOLD:
                hits += 1
    return {"pairs_compared": len(sa) * len(sb), "max_jaccard": round(best, 4),
            "pairs_above_threshold": hits}


def metadata_coverage(name):
    path = OUT_DIR / f"{name}_manifest.json"
    if not path.exists():
        return None
    manifest = json.load(io.open(path, encoding="utf8"))
    records = manifest.get("records_detail", manifest.get("records", []))
    fields = {
        "generator_model": "model",
        "act": "non_target_act" if name == "S_STYLE" else "primary_act",
        "concept": "primary_entity",
        "style_family": "style_variant",
    }
    result = {}
    for label, field in fields.items():
        values = [str(row[field]) for row in records if row.get(field)]
        if values:
            result[label] = {
                "unique": len(set(values)),
                "distribution": dict(sorted(Counter(values).items())),
            }
    return result


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ntok = lambda t: len(tok(t, truncation=True, max_length=MAX_LEN)["input_ids"])

    summary = json.load(io.open(OUT_DIR / "MATCHED_CORPORA_SUMMARY.json", encoding="utf8"))
    budget = summary["token_budget"]

    # ---- trim S_STYLE to the same token budget ----------------------------
    pool = read_jsonl(GENERATED) if GENERATED.exists() else []
    pool_tokens = sum(ntok(r["synthetic_context"]) for r in pool)
    # A partially generated pool must not be emitted as a finished corpus: it would
    # silently under-fill the token budget and break the matching the experiment
    # depends on.
    if pool and pool_tokens >= budget:
        rng = random.Random(CORPUS_SELECTION_SEED)
        rng.shuffle(pool)
        out, total = [], 0
        for r in pool:
            if total >= budget:
                break
            r["tokens"] = ntok(r["synthetic_context"])
            out.append(r)
            total += r["tokens"]
        if len(out) > 1 and abs(total - out[-1]["tokens"] - budget) < abs(total - budget):
            total -= out.pop()["tokens"]
        with io.open(OUT_DIR / "corpus_S_STYLE.jsonl", "w", encoding="utf8") as fh:
            for r in out:
                fh.write(json.dumps({"context": r["synthetic_context"]}, ensure_ascii=False) + "\n")
        with io.open(OUT_DIR / "S_STYLE_manifest.json", "w", encoding="utf8") as fh:
            json.dump({"condition": "S_STYLE",
                       "corpus_selection_seed": CORPUS_SELECTION_SEED,
                       "generated_pool": len(pool), "records": len(out),
                       "non_padding_tokens": total,
                       "delta_from_budget": total - budget,
                       "delta_pct": round(100.0 * (total - budget) / budget, 3),
                       "records_detail": [{k: v for k, v in r.items()
                                           if k != "synthetic_context"} for r in out]},
                      fh, indent=2, ensure_ascii=False)
        status = (f"S_STYLE: {len(out)} records, {total:,} tokens "
                  f"({total - budget:+d}, {100.0*(total-budget)/budget:+.3f}%)")
    else:
        # Remove any corpus left by an earlier partial run so downstream steps and
        # the dry-run report see PENDING rather than a short corpus.
        for stale in ("corpus_S_STYLE.jsonl", "S_STYLE_manifest.json"):
            (OUT_DIR / stale).unlink(missing_ok=True)
        status = (f"S_STYLE: PENDING -- pool holds {pool_tokens:,} of "
                  f"{budget:,} required tokens ({len(pool)} passages)")
    print(f"  {status}\n")

    # ---- characterise every corpus ----------------------------------------
    corpora = {}
    for name in ("A", "S", "M", "S_STYLE"):
        p = OUT_DIR / f"corpus_{name}.jsonl"
        if p.exists():
            corpora[name] = [r["context"] for r in read_jsonl(p)]

    report = {"description": "Corpus characterisation and near-duplicate audit "
                             "(dapt.md sec. 18, 25, 26)",
              "jaccard_threshold": JACCARD_THRESHOLD, "shingle_n": SHINGLE_N,
              "note": "analysis only; no record removed on the basis of these numbers",
              "corpora": {}, "metadata_coverage": {}, "near_duplicate": {},
              "lexical_similarity_to_authentic": {}}

    for name, texts in corpora.items():
        report["corpora"][name] = describe(name, texts, tok)
        report["near_duplicate"][name] = near_dupe(texts)
        coverage = metadata_coverage(name)
        if coverage:
            report["metadata_coverage"][name] = coverage

    if "A" in corpora:
        for name in ("S", "S_STYLE"):
            if name in corpora:
                report["lexical_similarity_to_authentic"][name] = cross_near(
                    corpora["A"], corpora[name])

    with io.open(OUT_DIR / "CORPUS_CHARACTERISATION.json", "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"  {'':8s} {'recs':>5s} {'tokens':>9s} {'mean_w':>7s} {'TTR':>7s} "
          f"{'dist2':>7s} {'dup':>6s} {'neardup':>8s}")
    for name, c in report["corpora"].items():
        nd = report["near_duplicate"][name]
        print(f"  {name:8s} {c['records']:5d} {c['total_tokens']:9,d} "
              f"{c['mean_words']:7.1f} {c['type_token_ratio']:7.4f} "
              f"{c['distinct_2']:7.4f} {c['exact_duplicate_rate']:6.3f} "
              f"{nd['rate']:8.5f}")
    for name, s in report["lexical_similarity_to_authentic"].items():
        print(f"  A vs {name:8s}: max Jaccard {s['max_jaccard']:.4f}, "
              f"{s['pairs_above_threshold']} pairs >= {JACCARD_THRESHOLD}")
    print(f"\n  wrote CORPUS_CHARACTERISATION.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
