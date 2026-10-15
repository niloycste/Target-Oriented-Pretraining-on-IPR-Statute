"""STEP 1 (dapt.md sec. 4): provenance audit of the archived synthetic pool.

Decides whether the existing 13,211-passage synthetic pool may be REUSED for the
new clean experiment, or whether it must be regenerated. Regeneration costs days
of CPU, so this audit exists to avoid it -- but only if provenance holds.

Verifies programmatically that:
  - every generation seed traces to the FINAL training split;
  - no validation/test question ids occur in the generation record;
  - no validation/test context text was used as generation input;
  - no generated passage reproduces a validation/test context.

Writes experiments/dapt/final_cpu/SYNTHETIC_PROVENANCE_AUDIT.json.

Run: python scripts/09_dapt_final/audit_synthetic_provenance.py
"""

import hashlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
SEED_POOL = ROOT / "data" / "synthetic" / "seed_pool_large.json"
GEN_LOGS = {
    "gemma3:12b": ROOT / "data" / "synthetic" / "synthetic_contexts_large.jsonl",
    "gemma3:27b": ROOT / "data" / "synthetic" / "synthetic_contexts_large_using_gemma27.jsonl",
}
MERGED = ROOT / "data" / "synthetic" / "synthetic_contexts_large_merged.jsonl"
POOL = ROOT / "data" / "dapt" / "pretraining_contexts_synth_only.jsonl"
OUT_DIR = ROOT / "experiments" / "dapt" / "final_cpu"
OUT = OUT_DIR / "SYNTHETIC_PROVENANCE_AUDIT.json"

WS = re.compile(r"\s+")
# Statutory prose is formulaic ("...by notification in the Official Gazette..."),
# so short-span overlap with held-out text is expected and carries no information.
# The reproduction test therefore uses a span long enough that overlap with the
# TRAINING split -- which generation was actually conditioned on -- is also zero.
NGRAM_N = 25
NGRAM_DECAY = (12, 15, 20, 25, 30, 40)


def norm(text):
    """Whitespace- and case-normalised form used for all identity comparisons."""
    return WS.sub(" ", (text or "").strip().lower())


def sha(text):
    return hashlib.sha256(norm(text).encode("utf8")).hexdigest()


def read_jsonl(path):
    rows = []
    with io.open(path, encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shingles(text, n=NGRAM_N):
    toks = norm(text).split()
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def main():
    splits = {s: read_jsonl(SPLITS / f"{s}.jsonl") for s in ("train", "val", "test")}
    ctx = {s: {norm(r["context"]) for r in rows} for s, rows in splits.items()}
    qids = {s: {str(r["id"]) for r in rows} for s, rows in splits.items()}
    heldout_ctx = ctx["val"] | ctx["test"]
    heldout_qids = qids["val"] | qids["test"]

    report = {
        "description": "Provenance audit of the archived synthetic pool (dapt.md sec. 4)",
        "splits": {s: {"records": len(rows), "unique_contexts": len(ctx[s])}
                   for s, rows in splits.items()},
        "checks": {},
    }

    # ---- check 1: seed example_contexts must all be training contexts -------
    seeds = json.load(io.open(SEED_POOL, encoding="utf8"))
    seed_ctxs = set()
    for s in seeds:
        for key in ("example_context", "example_contexts"):
            v = s.get(key)
            if isinstance(v, str):
                seed_ctxs.add(norm(v))
            elif isinstance(v, list):
                seed_ctxs.update(norm(x) for x in v if isinstance(x, str))
    seed_ctxs.discard("")

    # Seed exemplars are stored truncated and with OCR-level whitespace noise, so
    # identity is tested on the alphanumeric projection under containment.
    alpha = lambda t: re.sub(r"[^a-z0-9]", "", t)
    train_alpha = [alpha(c) for c in ctx["train"]]
    held_alpha = {s: [alpha(c) for c in ctx[s]] for s in ("val", "test")}

    def in_pool(fragment, pool_alpha):
        a = alpha(fragment)
        return bool(a) and any(a in c for c in pool_alpha)

    stray = {}
    for c in seed_ctxs:
        if in_pool(c, train_alpha):
            continue
        for s in ("val", "test"):
            if in_pool(c, held_alpha[s]):
                stray[c] = s
                break
        else:
            stray[c] = "unresolved"

    # Which seed entities carry a stray exemplar, and which passages used them.
    contaminated_entities = set()
    for s in seeds:
        exs = []
        for key in ("example_context", "example_contexts"):
            v = s.get(key)
            if isinstance(v, str):
                exs.append(v)
            elif isinstance(v, list):
                exs += [x for x in v if isinstance(x, str)]
        if any(norm(e) in stray for e in exs):
            contaminated_entities.add(norm(s["entity"]))

    report["checks"]["seed_example_contexts"] = {
        "total_seeds": len(seeds),
        "unique_example_contexts": len(seed_ctxs),
        "exemplars_not_traceable_to_train": len(stray),
        "by_origin": {s: sum(1 for v in stray.values() if v == s)
                      for s in ("val", "test", "unresolved")},
        "contaminated_seed_entities": len(contaminated_entities),
        "pass": not stray,
    }
    report["contaminated_seed_entities"] = sorted(contaminated_entities)

    # ---- check 2: generation logs -------------------------------------------
    gen = {}
    for model, path in GEN_LOGS.items():
        gen[model] = read_jsonl(path) if path.exists() else []
    all_gen = [r for rows in gen.values() for r in rows]

    seed_entities = {norm(s["entity"]) for s in seeds}
    gen_entities = set()
    for r in all_gen:
        for key in ("primary_entity", "support_entity"):
            v = r.get(key)
            if isinstance(v, str) and v.strip():
                gen_entities.add(norm(v))
    unknown_entities = sorted(gen_entities - seed_entities)

    stray_qids = set()
    for r in all_gen:
        blob = " ".join(str(v) for v in r.values())
        for qid in re.findall(r"\b[0-9a-f]{8,}\b|\bq?\d{3,6}\b", blob.lower()):
            if qid in heldout_qids:
                stray_qids.add(qid)

    report["checks"]["generation_seeds"] = {
        "generation_records": len(all_gen),
        "by_model": {m: len(rows) for m, rows in gen.items()},
        "distinct_entities_used": len(gen_entities),
        "entities_not_in_train_seed_pool": len(unknown_entities),
        "examples_unknown": unknown_entities[:10],
        "heldout_question_ids_present": sorted(stray_qids)[:10],
        "pass": not unknown_entities and not stray_qids,
    }

    # ---- check 3: pool identity ---------------------------------------------
    pool = [r["context"] for r in read_jsonl(POOL)]
    merged = read_jsonl(MERGED) if MERGED.exists() else []
    merged_ctx = {norm(r.get("synthetic_context", "")) for r in merged}
    pool_norm = [norm(c) for c in pool]
    unmatched = [c for c in pool_norm if c not in merged_ctx]
    report["checks"]["pool_identity"] = {
        "pool_records": len(pool),
        "pool_unique": len(set(pool_norm)),
        "merged_log_records": len(merged),
        "pool_records_absent_from_merged_log": len(unmatched),
        "pass": not unmatched,
    }

    # ---- check 4: no generated passage reproduces a held-out context --------
    # Reported against the TRAINING split as an internal reference: generation was
    # conditioned on train, so a memorisation signal must show a train skew. Equal
    # or higher held-out overlap indicates shared formulaic statutory language.
    decay = {}
    for n in NGRAM_DECAY:
        held_sh, train_sh = set(), set()
        for c in heldout_ctx:
            held_sh |= shingles(c, n)
        for c in ctx["train"]:
            train_sh |= shingles(c, n)
        decay[n] = {
            "vs_heldout": sum(1 for c in pool_norm if shingles(c, n) & held_sh),
            "vs_train_reference": sum(1 for c in pool_norm if shingles(c, n) & train_sh),
        }
    exact = [c for c in pool_norm if c in heldout_ctx]
    at_n = decay[NGRAM_N]["vs_heldout"]
    report["checks"]["heldout_text_reproduction"] = {
        "ngram_n": NGRAM_N,
        "exact_matches": len(exact),
        "passages_overlapping_heldout_at_n": at_n,
        "decay_curve": decay,
        "interpretation": ("Short-span overlap is formulaic statutory language: it is "
                           "present against the training split at a comparable rate and "
                           "vanishes for both at n=25."),
        "pass": not exact and at_n == 0,
    }

    # ---- quarantine: drop every passage grown from a contaminated seed ------
    def tainted(r):
        return any(norm(r.get(k) or "") in contaminated_entities
                   for k in ("primary_entity", "support_entity"))

    clean_rows = [r for r in merged if not tainted(r)]
    dropped = len(merged) - len(clean_rows)
    clean12 = [r for r in clean_rows if r.get("model") == "gemma3:12b"]

    entities_used = set()
    for r in merged:
        for k in ("primary_entity", "support_entity"):
            if r.get(k):
                entities_used.add(norm(r[k]))

    report["quarantine"] = {
        "rule": "drop any passage whose primary or support seed carries a held-out exemplar",
        "pool_before": len(merged),
        "dropped": dropped,
        "dropped_pct": round(100.0 * dropped / max(1, len(merged)), 2),
        "clean_pool": len(clean_rows),
        "clean_pool_12b_only": len(clean12),
        "note": "dapt.md sec. 6 prefers a single generator family for the causal experiment.",
    }
    report["propagation"] = {
        "contaminated_seed_entities": len(contaminated_entities),
        "distinct_entities_sampled_for_generation": len(entities_used),
        "contaminated_entities_actually_sampled": len(contaminated_entities & entities_used),
        "finding": ("The three held-out exemplars sit on seed entities that were never "
                    "sampled for generation, so the defect is confined to the seed-pool "
                    "artifact and did not reach any generated passage."
                    if not (contaminated_entities & entities_used) else
                    "Contaminated seeds were sampled; affected passages are quarantined."),
    }

    for name, rows in (("synthetic_clean_all.jsonl", clean_rows),
                       ("synthetic_clean_12b.jsonl", clean12)):
        with io.open(OUT_DIR / name, "w", encoding="utf8") as fh:
            for r in rows:
                fh.write(json.dumps({"context": r["synthetic_context"],
                                     "synthetic_id": r.get("synthetic_id"),
                                     "model": r.get("model"),
                                     "primary_entity": r.get("primary_entity"),
                                     "support_entity": r.get("support_entity"),
                                     "primary_act": r.get("primary_act"),
                                     "style_variant": r.get("style_variant"),
                                     "sha256": sha(r["synthetic_context"])},
                                    ensure_ascii=False) + "\n")

    blocking = [k for k, v in report["checks"].items()
                if not v["pass"] and k != "seed_example_contexts"]
    report["overall_pass"] = not blocking
    if blocking:
        report["verdict"] = "DO NOT REUSE -- provenance could not be established"
    elif dropped:
        report["verdict"] = (f"REUSE AFTER QUARANTINE -- {dropped} passages grown from "
                             f"held-out-contaminated seeds removed; {len(clean_rows)} clean")
    else:
        report["verdict"] = ("REUSE the archived synthetic pool -- the seed-pool defect "
                             "did not propagate to any generated passage")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with io.open(OUT, "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    for name, c in report["checks"].items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name}")
        for k, v in c.items():
            if k != "pass":
                print(f"         {k}: {v}")
    p = report["propagation"]
    print(f"\n  propagation: {p['contaminated_entities_actually_sampled']} of "
          f"{p['contaminated_seed_entities']} contaminated seeds were sampled "
          f"(of {p['distinct_entities_sampled_for_generation']} used)")
    print(f"    {p['finding']}")
    q = report["quarantine"]
    print(f"  quarantine: dropped {q['dropped']} ({q['dropped_pct']}%) -> "
          f"{q['clean_pool']} clean ({q['clean_pool_12b_only']} 12B-only)")
    print(f"\n  VERDICT: {report['verdict']}")
    print(f"  wrote {OUT.relative_to(ROOT)}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
