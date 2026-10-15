"""Statistical and telemetry report for Experiment D."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "dapt" / "equal_update"
WORK = ROOT / "artifacts" / "models" / "equal_update"
SEEDS = [7, 42, 99, 123, 2026]
CONDITIONS = ["N", "A", "S", "M"]

stats_base = importlib.import_module("scripts.09_dapt_final.analyse_final_experiments")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf8")
    os.replace(temporary, path)


def load_runs(split: str, allow_missing: bool = False) -> dict:
    path = EXP / ("results.json" if split == "test" else "validation_results.json")
    if not path.exists() and not allow_missing:
        raise FileNotFoundError(path)
    d_runs = (
        json.loads(path.read_text(encoding="utf8")).get("runs", {})
        if path.exists() else {}
    )
    # N may be evaluated directly into Experiment D or reused from frozen A.
    frozen_name = "results.json" if split == "test" else "validation_results.json"
    frozen = json.loads((ROOT / "experiments/dapt/final_cpu" / frozen_name).read_text(encoding="utf8"))["runs"]
    for seed in SEEDS:
        d_key = f"D_N_s{seed}_100"
        if d_key not in d_runs:
            source = dict(frozen[f"A_N_s{seed}_100"])
            source.update({"experiment": "D", "condition": "N"})
            d_runs[d_key] = source
    return d_runs


def telemetry_report() -> dict:
    rows = {}
    complete = True
    for condition in "ASM":
        for seed in SEEDS:
            key = f"D_{condition}_s{seed}"
            path = WORK / f"{key}_mlm" / "training_telemetry.json"
            if not path.exists():
                rows[key] = {"status": "missing"}
                complete = False
                continue
            data = json.loads(path.read_text(encoding="utf8"))
            okay = (
                data.get("completed_optimizer_updates") == 100
                and data.get("configured", {}).get("warmup_steps") == 10
                and len(data.get("tokens_by_step", {})) == 100
            )
            complete &= okay
            rows[key] = {
                "status": "complete" if okay else "invalid",
                "optimizer_updates": data.get("completed_optimizer_updates"),
                "warmup_steps": data.get("configured", {}).get("warmup_steps"),
                "configured_peak_learning_rate": data.get("configured", {}).get("peak_learning_rate"),
                "observed_peak_learning_rate": data.get("observed_peak_learning_rate"),
                "observed_final_learning_rate": data.get("observed_final_learning_rate"),
                "total_non_padding_tokens_processed": data.get("total_non_padding_tokens_processed"),
                "telemetry_file": path.relative_to(ROOT).as_posix(),
            }
    return {"all_complete": complete, "runs": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    runs = load_runs(args.split, allow_missing=args.allow_partial)
    table = {condition: {} for condition in CONDITIONS}
    for row in runs.values():
        if row.get("experiment") == "D" and row.get("condition") in table:
            table[row["condition"]][int(row["seed"])] = row
    present = sum(seed in table[condition] for condition in CONDITIONS for seed in SEEDS)
    if present != 20 and not args.allow_partial:
        raise RuntimeError(f"incomplete Experiment D result matrix: {present}/20")
    report = {
        "schema_version": 1,
        "split": args.split,
        "design": "post-hoc equal-update sensitivity; seed is the replication unit",
        "cells_present": present,
        "training_telemetry": telemetry_report(),
    }
    for metric in ("f1", "em"):
        metric_report = {
            "descriptives": {},
            "blocked_anova": stats_base.blocked_anova(table, metric),
            "dapt_minus_n": {},
            "per_seed": {},
        }
        for condition in CONDITIONS:
            values = [table[condition][seed][metric] for seed in SEEDS if seed in table[condition]]
            metric_report["descriptives"][condition] = stats_base.mean_sd_ci(values)
            metric_report["per_seed"][condition] = {
                str(seed): table[condition][seed][metric]
                for seed in SEEDS if seed in table[condition]
            }
        for condition in "ASM":
            common = [seed for seed in SEEDS if seed in table[condition] and seed in table["N"]]
            metric_report["dapt_minus_n"][f"{condition}-N"] = stats_base.paired(
                [table[condition][seed][metric] for seed in common],
                [table["N"][seed][metric] for seed in common],
                f"{condition}-N", common,
            )
        stats_base.holm(metric_report["dapt_minus_n"])
        report[metric] = metric_report
    output = EXP / f"EXPERIMENT_D_{args.split.upper()}_STATISTICS.json"
    atomic_json(output, report)
    print(f"Wrote: {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
