"""Resume the size-matched multi-seed ablation after the 2026-08-12 reboot.

A Windows Update reboot killed size_matched_multiseed.py at 08:29, taking down
mixed901_s99_squad at step 225/1276 and the evaluation loop right after it.

size_matched_multiseed.py cannot simply be re-run: it calls drop_stage() on each
upstream directory once the downstream model exists, so auth901_s7_mlm,
auth901_s7_squad and their siblings are gone for every completed run. Its
run_stage() resume check looks for those directories, so a plain re-run would
retrain MLM + SQuAD for all five finished runs (~69 h) before skipping the IPR
stages it already has.

This driver does only what is actually missing:
  1. mixed901_s99  ->  SQuAD (from the surviving mixed901_s99_mlm/final), then IPR
  2. evaluation of every corpus/seed whose IPR final/ exists and is not yet scored
  3. the seed-42 merge and the spread-vs-SD report

Same semantics as the original: same seed patching, same checkpoint pruning,
same results file. Resumable in turn - rerunning after another interruption
skips whatever has since completed.

Run:  python resume_size_matched_multiseed.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import size_matched_dapt_experiment as base
import size_matched_multiseed as ms

ROOT = ms.ROOT
RESULTS_DIR = ms.RESULTS_DIR
WORK = ms.WORK
OUT_JSON = ms.OUT_JSON

CORPORA = ms.CORPORA
NEW_SEEDS = ms.NEW_SEEDS
BASE_SEED = ms.BASE_SEED

# the one run the reboot left unfinished
PENDING = [("mixed901", 99)]


def finish_pending():
    for corpus, seed in PENDING:
        tag = f"{corpus}_s{seed}"
        mlm_d = WORK / f"{tag}_mlm"
        sq_d = WORK / f"{tag}_squad"
        ipr_d = WORK / f"{tag}_ipr"

        if (ipr_d / "final").exists():
            print(f"[skip] {tag} already complete", flush=True)
            continue

        print(f"\n### {tag}   free disk {ms.free_gb():.1f} GB", flush=True)
        if ms.free_gb() < 8:
            print(f"[STOP] only {ms.free_gb():.1f} GB free; refusing to start {tag}",
                  flush=True)
            return

        if not (mlm_d / "final").exists():
            print(f"[abort] {tag}: {mlm_d / 'final'} is missing, cannot resume "
                  f"from the SQuAD stage", flush=True)
            continue

        if not base.run_stage(f"{tag}_squad",
                              ms.make("squad", seed, mlm_final=mlm_d / "final",
                                      outdir=sq_d),
                              sq_d / "final"):
            continue
        ms.prune_checkpoints(sq_d)
        ms.drop_stage(mlm_d)           # consumed by the SQuAD stage

        if base.run_stage(f"{tag}_ipr",
                          ms.make("ipr", seed, squad_final=sq_d / "final",
                                  outdir=ipr_d),
                          ipr_d / "final"):
            ms.prune_checkpoints(ipr_d)
            ms.drop_stage(sq_d)        # consumed by the IPR stage


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    finish_pending()

    # ------------------------------------------------------------- evaluate
    # Unlike the original, targets are discovered from disk rather than from
    # the stages this process happened to run, so runs finished before the
    # reboot still get scored.
    from multiseed_experiment import evaluate
    res = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}

    for corpus in CORPORA:
        for seed in NEW_SEEDS:
            tag = f"{corpus}_s{seed}"
            if tag in res:
                continue
            final = WORK / f"{tag}_ipr" / "final"
            if not final.exists():
                print(f"[missing] {tag}: {final}", flush=True)
                continue
            print(f"[eval] {tag}", flush=True)
            res[tag] = evaluate(final)
            print(f"       {res[tag]}", flush=True)
            OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # pull the seed-42 numbers from the single-seed run
    seed42 = json.loads((RESULTS_DIR / "size_matched_results.json").read_text())
    for c in CORPORA:
        if c in seed42:
            res[f"{c}_s{BASE_SEED}"] = seed42[c]
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- report
    print("\n" + "=" * 74)
    print(f"{'corpus':10s}{'seed':>6s}{'EM':>9s}{'F1':>9s}")
    print("-" * 74)
    summary = {}
    for c in CORPORA:
        f1s, ems = [], []
        for s in [BASE_SEED] + NEW_SEEDS:
            v = res.get(f"{c}_s{s}")
            if v:
                print(f"{c:10s}{s:6d}{v['em']:9.2f}{v['f1']:9.2f}")
                f1s.append(v["f1"])
                ems.append(v["em"])
        if len(f1s) >= 2:
            m, sd = st.mean(f1s), (st.stdev(f1s) if len(f1s) > 2 else st.pstdev(f1s))
            summary[c] = {"f1_mean": round(m, 2), "f1_sd": round(sd, 2),
                          "em_mean": round(st.mean(ems), 2), "n": len(f1s)}
            print(f"{c:10s}{'mean':>6s}{st.mean(ems):9.2f}{m:9.2f}   (F1 SD {sd:.2f}, n={len(f1s)})")
        print("-" * 74)

    if len(summary) == 3:
        best = max(summary, key=lambda k: summary[k]["f1_mean"])
        worst = min(summary, key=lambda k: summary[k]["f1_mean"])
        spread = summary[best]["f1_mean"] - summary[worst]["f1_mean"]
        pooled = math.sqrt(st.mean([summary[c]["f1_sd"] ** 2 for c in summary]))
        print(f"\ncomposition spread (mean F1): {spread:.2f}   pooled within-corpus SD: {pooled:.2f}")
        print(f"highest {best} ({summary[best]['f1_mean']:.2f}), "
              f"lowest {worst} ({summary[worst]['f1_mean']:.2f})")
        summary["_spread_vs_sd"] = {"spread": round(spread, 2), "pooled_sd": round(pooled, 2)}

    res["_summary"] = summary
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT_JSON}")


if __name__ == "__main__":
    main()
