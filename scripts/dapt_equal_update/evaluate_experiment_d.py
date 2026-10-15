"""Canonical validation/test evaluator for Experiment D.

Test evaluation remains locked until ``freeze_test_analysis.py --confirm`` has
recorded all completed checkpoints and frozen inputs.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments" / "dapt" / "equal_update"
WORK = ROOT / "artifacts" / "models" / "equal_update"
FROZEN_WORK = ROOT / "artifacts" / "models" / "final_cpu"
SEEDS = [7, 42, 99, 123, 2026]
CONDITIONS = ["N", "A", "S", "M"]

evaluator = importlib.import_module("scripts.09_dapt_final.evaluate_final_experiments")
evaluator.FINAL = EXP
evaluator.WORK = WORK
evaluator.LOCK = EXP / "EXPERIMENT_D_TEST_LOCK.json"


def verify_experiment_d_lock() -> dict:
    lock = evaluator.LOCK
    if not lock.exists():
        raise RuntimeError(
            "Experiment D test evaluation is locked; run "
            "freeze_test_analysis.py --confirm after reviewing validation results"
        )
    payload = json.loads(lock.read_text(encoding="utf8"))
    if not payload.get("locked"):
        raise RuntimeError("EXPERIMENT_D_TEST_LOCK.json is not marked locked")
    mismatches = []
    for name, expected in payload.get("sha256", {}).items():
        path = ROOT / name
        actual = evaluator.sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, found {actual}")
    if mismatches:
        raise RuntimeError("Experiment D frozen input mismatch:\n" + "\n".join(mismatches))
    return payload


def target_table() -> dict[str, dict]:
    targets = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            key = f"D_{condition}_s{seed}_100"
            checkpoint = (
                FROZEN_WORK / f"A_N_s{seed}_ipr" / "final"
                if condition == "N"
                else WORK / f"D_{condition}_s{seed}_ipr" / "final"
            )
            targets[key] = {
                "experiment": "D", "condition": condition, "seed": seed,
                "fraction": 100, "checkpoint": checkpoint,
            }
    return targets


evaluator.target_table = target_table
evaluator.verify_analysis_lock = verify_experiment_d_lock


if __name__ == "__main__":
    raise SystemExit(evaluator.main())
