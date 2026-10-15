"""Freeze completed Experiment D artifacts before canonical test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "dapt" / "equal_update"
LOCK = EXP / "EXPERIMENT_D_TEST_LOCK.json"
SEEDS = [7, 42, 99, 123, 2026]

evaluation = importlib.import_module("scripts.dapt_equal_update.evaluate_experiment_d")
canonical = evaluation.evaluator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                        help="required acknowledgment that this freezes the pre-test analysis state")
    args = parser.parse_args()
    if not args.confirm:
        print("Refused: inspect validation results, then rerun with --confirm.")
        return 2
    targets = evaluation.target_table()
    missing = [key for key, row in targets.items() if not row["checkpoint"].exists()]
    if missing:
        raise RuntimeError(f"cannot freeze; missing checkpoints: {missing}")
    validation = EXP / "validation_results.json"
    if not validation.exists():
        raise RuntimeError("cannot freeze before validation_results.json exists")
    validation_runs = json.loads(validation.read_text(encoding="utf8")).get("runs", {})
    expected_adapted = {f"D_{condition}_s{seed}_100" for condition in "ASM" for seed in SEEDS}
    absent_validation = sorted(expected_adapted - set(validation_runs))
    if absent_validation:
        raise RuntimeError(f"cannot freeze; missing validation runs: {absent_validation}")

    frozen_files = [
        EXP / "EXPERIMENT_D_PROTOCOL.json", EXP / "DRY_RUN_REPORT.json", validation,
        ROOT / "data/splits/train_squad.jsonl", ROOT / "data/splits/val_squad.jsonl",
        ROOT / "data/splits/val.jsonl",
        ROOT / "data/splits/test.jsonl",
        *(ROOT / "experiments/dapt/final_cpu" / f"corpus_{condition}.jsonl" for condition in "ASM"),
        *sorted((EXP / "generated_scripts").glob("*.py")),
        *sorted((ROOT / "scripts/dapt_equal_update").glob("*.py")),
    ]
    reviewed = {}
    for key, row in targets.items():
        checkpoint = row["checkpoint"]
        reviewed[key] = {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": canonical.checkpoint_sha256(checkpoint),
        }
    file_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in frozen_files}
    protocol = json.loads((EXP / "EXPERIMENT_D_PROTOCOL.json").read_text(encoding="utf8"))
    for raw_path in protocol.get("external_input_sha256", {}):
        path = Path(raw_path)
        file_hashes[str(path)] = sha256(path)
    payload = {
        "schema_version": 1,
        "locked": True,
        "status": "post-hoc sensitivity; frozen before Experiment D canonical test evaluation",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sha256": file_hashes,
        "reviewed_checkpoints": reviewed,
        "analysis": {
            "primary": "seed-blocked ANOVA for F1 across N/A/S/M",
            "secondary": "paired DAPT-minus-N intervals and Holm-adjusted tests",
            "replication_unit": "complete training seed",
        },
    }
    atomic_json(LOCK, payload)
    print(f"Frozen: {LOCK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
