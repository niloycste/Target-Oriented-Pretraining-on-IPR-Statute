"""Multi-seed repetition of the size-matched DAPT ablation.

The single-seed ablation in size_matched_dapt_experiment.py bounded four effects
below the seed-to-seed standard deviation but could not estimate any of them.
This script repeats the FULL pipeline (MLM -> SQuAD v1 -> IPR) for each of the
three size-matched corpora under two additional seeds, so each corpus has three
complete runs and the composition effect can be reported as a mean with a spread.

Seed 42 is already done by size_matched_dapt_experiment.py and is reused.

Every stage keeps the lowest-validation-loss checkpoint, matching the seed-42
runs, and the seed is applied to MLM, SQuAD and IPR alike so that DAPT-stage
variance is captured rather than only fine-tuning variance.

Resumable: any stage whose final/ exists is skipped.
Run:  python size_matched_multiseed.py
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[2]   # repo root: data lives here
RESULTS_DIR = ROOT / "results" / "experiments" / "size_matched_dapt"
WORK = ROOT / "artifacts" / "models" / "size_matched"
MERGE = ROOT / "data" / "dapt"
OUT_JSON = RESULTS_DIR / "size_matched_multiseed_results.json"

CORPORA = ["auth901", "synth901", "mixed901"]
NEW_SEEDS = [7, 99]          # 42 already complete
BASE_SEED = 42

import size_matched_dapt_experiment as base


def set_seed(text, seed):
    return re.sub(r"^SEED\s*=\s*\d+.*$", f"SEED = {seed}", text, count=1, flags=re.M)


def make(kind, seed, **kw):
    src = {"mlm": base.make_mlm, "squad": base.make_squad, "ipr": base.make_ipr}[kind](**kw)
    out = set_seed(src, seed)
    if out == src and "SEED" in src:
        raise RuntimeError(f"seed patch failed for {kind}")
    return out


def free_gb():
    import shutil
    return shutil.disk_usage(str(ROOT)).free / 1024 ** 3


def prune_checkpoints(stage_dir):
    """Drop per-epoch checkpoints once final/ exists.

    load_best_model_at_end=True means final/ already holds the selected
    weights, so the checkpoint-* directories are pure overhead after a stage
    completes. This is what filled the disk on the first attempt.
    """
    import shutil
    d = Path(stage_dir)
    if not (d / "final").exists():
        return 0.0
    freed = 0.0
    for ck in d.glob("checkpoint-*"):
        if ck.is_dir():
            sz = sum(f.stat().st_size for f in ck.rglob("*") if f.is_file())
            shutil.rmtree(ck, ignore_errors=True)
            freed += sz / 1024 ** 3
    if freed:
        print(f"       pruned checkpoints from {d.name}: {freed:.2f} GB freed", flush=True)
    return freed


def drop_stage(stage_dir):
    """Remove an upstream stage once the downstream model has been produced."""
    import shutil
    d = Path(stage_dir)
    if not d.exists():
        return 0.0
    sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024 ** 3
    shutil.rmtree(d, ignore_errors=True)
    print(f"       removed intermediate {d.name}: {sz:.2f} GB freed", flush=True)
    return sz


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    targets = {}

    for corpus in CORPORA:
        for seed in NEW_SEEDS:
            tag = f"{corpus}_s{seed}"
            mlm_d = WORK / f"{tag}_mlm"
            sq_d = WORK / f"{tag}_squad"
            ipr_d = WORK / f"{tag}_ipr"

            print(f"\n### {tag}   free disk {free_gb():.1f} GB", flush=True)
            if free_gb() < 8:
                print(f"[STOP] only {free_gb():.1f} GB free; refusing to start {tag}",
                      flush=True)
                break

            if not base.run_stage(f"{tag}_mlm",
                                  make("mlm", seed, variant=corpus, outdir=mlm_d),
                                  mlm_d / "final"):
                continue
            prune_checkpoints(mlm_d)

            if not base.run_stage(f"{tag}_squad",
                                  make("squad", seed, mlm_final=mlm_d / "final", outdir=sq_d),
                                  sq_d / "final"):
                continue
            prune_checkpoints(sq_d)
            drop_stage(mlm_d)          # consumed by the SQuAD stage

            if base.run_stage(f"{tag}_ipr",
                              make("ipr", seed, squad_final=sq_d / "final", outdir=ipr_d),
                              ipr_d / "final"):
                targets[tag] = ipr_d / "final"
                prune_checkpoints(ipr_d)
                drop_stage(sq_d)       # consumed by the IPR stage

    # ---------------------------------------------------------- evaluate
    from multiseed_experiment import evaluate
    res = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for name, path in targets.items():
        print(f"[eval] {name}", flush=True)
        res[name] = evaluate(path)
        print(f"       {res[name]}", flush=True)
        OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # pull the seed-42 numbers from the single-seed run
    seed42 = json.loads((RESULTS_DIR / "size_matched_results.json").read_text())
    for c in CORPORA:
        if c in seed42:
            res[f"{c}_s{BASE_SEED}"] = seed42[c]
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # ---------------------------------------------------------- report
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
