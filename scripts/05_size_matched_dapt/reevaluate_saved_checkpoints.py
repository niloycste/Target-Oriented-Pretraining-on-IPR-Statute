"""Re-evaluate archived ablation checkpoints with the canonical QA decoder.

This script performs inference only. It updates the three tracked ablation
reports without retraining, pruning, or deleting any checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import multiseed_experiment as evaluator  # noqa: E402


RESULTS_DIR = ROOT / "results" / "experiments" / "size_matched_dapt"
MODELS_DIR = ROOT / "artifacts" / "models"


def load_main_seed42() -> dict[str, dict]:
    report = json.loads(
        (ROOT / "results" / "evaluation" / "qa_eval_report.json").read_text(encoding="utf-8")
    )
    by_code = {row["code"]: row for row in report["results_summary"]}
    return {
        code: {
            "n": report["test_records"],
            "em": by_code[code]["em"],
            "f1": by_code[code]["f1"],
            "inference_protocol": report["inference_protocol"],
        }
        for code in ("A", "C'")
    }


def require_checkpoint(path: Path) -> Path:
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return path


def evaluate(path: Path) -> dict:
    print(f"[eval] {path.relative_to(ROOT)}", flush=True)
    result = evaluator.evaluate(require_checkpoint(path))
    print(f"       EM {result['em']:.2f} | F1 {result['f1']:.2f}", flush=True)
    return result


def write_multiseed_report(seed42: dict[str, dict]) -> None:
    paths = {
        "A": {
            "7": MODELS_DIR / "multiseed" / "A_seed7" / "final",
            "99": MODELS_DIR / "multiseed" / "A_seed99" / "final",
        },
        "C'": {
            "7": MODELS_DIR / "multiseed" / "Cprime_seed7" / "final",
            "99": MODELS_DIR / "multiseed" / "Cprime_seed99" / "final",
        },
    }
    descriptions = {
        "A": "InLegalBERT -> IPR direct",
        "C'": "InLegalBERT + DAPT(Orig+Synth) -> SQuAD v1 -> IPR",
    }
    report = {}
    for code, seed_paths in paths.items():
        seeds = {"42": seed42[code]}
        for seed, path in seed_paths.items():
            seeds[seed] = evaluate(path)
        em_values = [seeds[seed]["em"] for seed in ("42", "7", "99")]
        f1_values = [seeds[seed]["f1"] for seed in ("42", "7", "99")]
        report[code] = {
            "description": descriptions[code],
            "seeds": seeds,
            "em_mean": round(stats.mean(em_values), 2),
            "em_sd": round(stats.stdev(em_values), 2),
            "em_min": min(em_values),
            "em_max": max(em_values),
            "f1_mean": round(stats.mean(f1_values), 2),
            "f1_sd": round(stats.stdev(f1_values), 2),
            "f1_min": min(f1_values),
            "f1_max": max(f1_values),
        }
    path = RESULTS_DIR / "multiseed_results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {path.relative_to(ROOT)}")


def write_size_matched_reports() -> None:
    work = MODELS_DIR / "size_matched"
    all_runs = {}
    for corpus in ("auth901", "synth901", "mixed901"):
        for seed in (42, 7, 99):
            suffix = f"{corpus}_ipr" if seed == 42 else f"{corpus}_s{seed}_ipr"
            all_runs[f"{corpus}_s{seed}"] = evaluate(work / suffix / "final")

    summary = {}
    for corpus in ("auth901", "synth901", "mixed901"):
        rows = [all_runs[f"{corpus}_s{seed}"] for seed in (42, 7, 99)]
        f1_values = [row["f1"] for row in rows]
        em_values = [row["em"] for row in rows]
        summary[corpus] = {
            "f1_mean": round(stats.mean(f1_values), 2),
            "f1_sd": round(stats.stdev(f1_values), 2),
            "em_mean": round(stats.mean(em_values), 2),
            "n": 3,
        }
    best = max(summary, key=lambda corpus: summary[corpus]["f1_mean"])
    worst = min(summary, key=lambda corpus: summary[corpus]["f1_mean"])
    summary["_spread_vs_sd"] = {
        "spread": round(summary[best]["f1_mean"] - summary[worst]["f1_mean"], 2),
        "pooled_sd": round(
            math.sqrt(
                sum(
                    summary[corpus]["f1_sd"] ** 2
                    for corpus in ("auth901", "synth901", "mixed901")
                )
                / 3
            ),
            2,
        ),
    }
    multiseed_report = all_runs | {"_summary": summary}
    multiseed_path = RESULTS_DIR / "size_matched_multiseed_results.json"
    multiseed_path.write_text(
        json.dumps(multiseed_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[write] {multiseed_path.relative_to(ROOT)}")

    single_seed_report = {
        "H' (loss rule)": evaluate(work / "Hprime_lossrule_ipr" / "final"),
        **{corpus: all_runs[f"{corpus}_s42"] for corpus in ("auth901", "synth901", "mixed901")},
    }
    single_seed_path = RESULTS_DIR / "size_matched_results.json"
    single_seed_path.write_text(
        json.dumps(single_seed_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[write] {single_seed_path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluator.EVAL_BATCH_SIZE = args.batch_size
    seed42 = load_main_seed42()
    write_multiseed_report(seed42)
    write_size_matched_reports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
