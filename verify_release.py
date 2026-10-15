from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


required = [
    "papers/target_pretraining/target-pretraining.tex",
    "data/splits/test.jsonl",
    "experiments/dapt/final_cpu/FINAL_STATISTICS.json",
    "experiments/dapt/final_cpu/results.json",
    "experiments/dapt/equal_update/EXPERIMENT_D_TEST_STATISTICS.json",
    "experiments/dapt/equal_update/results.json",
]
missing = [path for path in required if not (ROOT / path).is_file()]
if missing:
    raise SystemExit(f"Missing required release files: {missing}")

with (ROOT / "data/splits/test.jsonl").open(encoding="utf-8") as handle:
    test_count = sum(1 for line in handle if line.strip())
assert test_count == 344, test_count

with (ROOT / "experiments/dapt/equal_update/EXPERIMENT_D_TEST_STATISTICS.json").open(
    encoding="utf-8"
) as handle:
    statistics = json.load(handle)

means = statistics["f1"]["descriptives"]
assert round(means["N"]["mean"], 2) == 85.86
assert round(means["A"]["mean"], 2) == 85.85
assert round(means["S"]["mean"], 2) == 85.91
assert round(means["M"]["mean"], 2) == 85.58
assert round(statistics["f1"]["blocked_anova"]["p_condition"], 3) == 0.706

print("DAPT release verified: 344 test questions and final/equal-update evidence present.")
