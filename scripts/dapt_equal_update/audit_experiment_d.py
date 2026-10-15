"""Read-only audit of Experiment D evidence and manuscript reporting.

This script performs no training or inference. It independently re-scores the
saved validation and test predictions, checks the frozen inputs and checkpoint
fingerprints, recomputes the main statistics, verifies training telemetry, and
checks the Experiment D manuscript tables.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics as st
import string

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "dapt" / "equal_update"
PAPER = ROOT / "papers" / "target_pretraining"
OUT = PAPER / "submission_audit_equal_update_20261002.json"
SEEDS = (7, 42, 99, 123, 2026)
CONDITIONS = "NASM"
ERRORS: list[str] = []


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def check(condition: bool, message: str) -> None:
    if not condition:
        ERRORS.append(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_digest(folder: Path) -> str:
    files: set[Path] = set()
    for pattern in (
        "config.json", "*.safetensors", "pytorch_model*.bin", "*.index.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.txt",
        "special_tokens_map.json",
    ):
        files.update(folder.glob(pattern))
    h = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.name):
        h.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def normalise(text: str) -> str:
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def score(prediction: str, answer: str) -> tuple[float, float]:
    pred = normalise(prediction)
    gold = normalise(answer)
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    common = sum((collections.Counter(pred_tokens) & collections.Counter(gold_tokens)).values())
    em = float(pred == gold)
    f1 = 2 * common / (len(pred_tokens) + len(gold_tokens)) if pred_tokens and gold_tokens else 0.0
    return em, f1


def table_body(tex: str, label: str) -> str:
    marker = "\\label{" + label + "}"
    check(marker in tex, f"Missing manuscript table: {label}")
    return tex.split(marker, 1)[1].split("\\end{table}", 1)[0] if marker in tex else ""


def close(actual: float, expected: float, tolerance: float, message: str) -> None:
    check(abs(actual - expected) <= tolerance, f"{message}: {actual} != {expected}")


def main() -> int:
    expected = {f"D_{condition}_s{seed}_100" for condition in CONDITIONS for seed in SEEDS}
    report: dict = {
        "scope": (
            "Read-only rescoring, fingerprint, telemetry, statistical, and "
            "manuscript audit of Experiment D; no training or inference."
        ),
        "errors": ERRORS,
    }
    all_runs: dict[str, dict] = {}

    for split, filename, count in (
        ("test", "results.json", 344),
        ("val", "validation_results.json", 408),
    ):
        runs = read(EXP / filename)["runs"]
        if split == "val":
            # Experiment D reuses the five frozen Experiment A no-DAPT readers.
            frozen = read(ROOT / "experiments" / "dapt" / "final_cpu" / filename)["runs"]
            for seed in SEEDS:
                source = dict(frozen[f"A_N_s{seed}_100"])
                source.update({"experiment": "D", "condition": "N"})
                runs[f"D_N_s{seed}_100"] = source
        check(set(runs) == expected, f"{split}: expected 20 condition--seed cells")
        gold = {str(row["id"]): row for row in jsonl(ROOT / f"data/splits/{split}.jsonl")}
        summaries: dict[str, dict] = {}
        by_condition: dict[str, list[dict]] = collections.defaultdict(list)
        for key, run in sorted(runs.items()):
            saved = read(ROOT / run["predictions_file"])
            predictions = saved["predictions"]
            source_key = key if run["condition"] != "N" else key.replace("D_N_", "A_N_")
            check(run["run_key"] in {key, source_key} and saved["run_key"] in {key, source_key},
                  f"{split}:{key}: run-key mismatch")
            check(run["split"] == split and saved["split"] == split, f"{split}:{key}: split mismatch")
            check(len(predictions) == count, f"{split}:{key}: prediction count mismatch")
            check({str(row["id"]) for row in predictions} == set(gold), f"{split}:{key}: ID mismatch")
            scores: list[tuple[float, float]] = []
            for row in predictions:
                item = gold[str(row["id"])]
                check(row["question"] == item["question"], f"{split}:{key}:{row['id']}: question mismatch")
                check(row["gold"] == item["answer"], f"{split}:{key}:{row['id']}: answer mismatch")
                check(not row["prediction"] or row["prediction"] in item["context"],
                      f"{split}:{key}:{row['id']}: prediction is not a literal context span")
                em, f1 = score(row["prediction"], item["answer"])
                close(em, row["em"], 1e-12, f"{split}:{key}:{row['id']}: EM")
                close(f1, row["f1"], 6e-9, f"{split}:{key}:{row['id']}: F1")
                scores.append((em, f1))
            fresh_em = 100 * st.fmean(value[0] for value in scores)
            fresh_f1 = 100 * st.fmean(value[1] for value in scores)
            for metric, fresh in (("em", fresh_em), ("f1", fresh_f1)):
                close(fresh, run[metric], 2e-6, f"{split}:{key}: aggregate {metric}")
                close(fresh, saved[metric], 2e-6, f"{split}:{key}: saved aggregate {metric}")
            check(run["checkpoint_sha256"] == saved["checkpoint_sha256"],
                  f"{split}:{key}: prediction fingerprint mismatch")
            by_condition[run["condition"]].append(run)
        for condition in CONDITIONS:
            rows = by_condition[condition]
            summaries[condition] = {
                "seeds": len(rows),
                "mean_em": st.fmean(row["em"] for row in rows),
                "mean_f1": st.fmean(row["f1"] for row in rows),
                "sd_f1": st.stdev(row["f1"] for row in rows),
            }
        report[split] = {
            "condition_seed_cells": len(runs),
            "prediction_rows": count * len(runs),
            "summaries": summaries,
        }
        all_runs[split] = runs
        print(f"Rescored {count * len(runs):,} {split} predictions", flush=True)

    lock = read(EXP / "EXPERIMENT_D_TEST_LOCK.json")
    check(lock.get("locked") is True, "Experiment D test lock is not active")
    lock_mismatches = []
    for name, expected_hash in lock.get("sha256", {}).items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        actual = digest(path) if path.exists() else "MISSING"
        if actual != expected_hash:
            lock_mismatches.append(name)
    check(not lock_mismatches, f"Frozen-input hash mismatches: {lock_mismatches}")
    report["lock"] = {
        "created_at": lock.get("created_at"),
        "files_checked": len(lock.get("sha256", {})),
        "mismatches": lock_mismatches,
    }

    reviewed = lock.get("reviewed_checkpoints", {})
    check(set(reviewed) == expected, "Reviewed-checkpoint inventory mismatch")
    checkpoint_checks = {}
    for key in sorted(expected):
        entry = reviewed[key]
        result = all_runs["test"][key]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = ROOT / path
        actual = checkpoint_digest(path)
        okay = (
            actual == entry["sha256"]
            and actual == result["checkpoint_sha256"]
            and actual == all_runs["val"][key]["checkpoint_sha256"]
        )
        checkpoint_checks[key] = okay
        check(okay, f"Checkpoint fingerprint mismatch: {key}")
    report["checkpoint_fingerprints"] = checkpoint_checks
    print("Checked all 20 Experiment D checkpoint fingerprints", flush=True)

    test_stats = read(EXP / "EXPERIMENT_D_TEST_STATISTICS.json")
    val_stats = read(EXP / "EXPERIMENT_D_VAL_STATISTICS.json")
    telemetry = test_stats["training_telemetry"]
    check(telemetry.get("all_complete") is True, "Training telemetry is incomplete")
    token_summary = {}
    for condition in "ASM":
        totals = []
        for seed in SEEDS:
            key = f"D_{condition}_s{seed}"
            row = telemetry["runs"][key]
            check(row["optimizer_updates"] == 100, f"{key}: optimizer updates")
            check(row["warmup_steps"] == 10, f"{key}: warmup updates")
            close(row["observed_peak_learning_rate"], 2e-5, 1e-15, f"{key}: peak learning rate")
            close(row["observed_final_learning_rate"], 0.0, 1e-15, f"{key}: final learning rate")
            raw = read(ROOT / row["telemetry_file"])
            check(raw["completed_optimizer_updates"] == 100, f"{key}: raw telemetry updates")
            check(len(raw["tokens_by_step"]) == 100, f"{key}: raw telemetry step count")
            check(raw["total_non_padding_tokens_processed"] == row["total_non_padding_tokens_processed"],
                  f"{key}: token total mismatch")
            totals.append(row["total_non_padding_tokens_processed"])
        token_summary[condition] = {
            "mean": round(st.fmean(totals)),
            "minimum": min(totals),
            "maximum": max(totals),
        }
    report["training_telemetry"] = token_summary

    independently_recomputed = {}
    for split, saved_stats in (("test", test_stats), ("val", val_stats)):
        runs = all_runs[split]
        independently_recomputed[split] = {}
        for metric in ("f1", "em"):
            matrix = np.array([
                [runs[f"D_{condition}_s{seed}_100"][metric] for condition in CONDITIONS]
                for seed in SEEDS
            ])
            grand = matrix.mean()
            condition_means = matrix.mean(axis=0)
            seed_means = matrix.mean(axis=1)
            residual = matrix - condition_means - seed_means[:, None] + grand
            ss_condition = len(SEEDS) * np.sum((condition_means - grand) ** 2)
            ss_error = np.sum(residual ** 2)
            f_value = (ss_condition / 3) / (ss_error / 12)
            p_value = stats.f.sf(f_value, 3, 12)
            eta = ss_condition / (ss_condition + ss_error)
            stored = saved_stats[metric]
            close(f_value, stored["blocked_anova"]["F_condition"], 6e-7, f"{split}:{metric}: F")
            close(p_value, stored["blocked_anova"]["p_condition"], 6e-9, f"{split}:{metric}: p")
            close(eta, stored["blocked_anova"]["partial_eta_sq"], 6e-7, f"{split}:{metric}: eta")
            paired_rows = {}
            raw_ps = {}
            for condition in "ASM":
                differences = np.array([
                    runs[f"D_{condition}_s{seed}_100"][metric]
                    - runs[f"D_N_s{seed}_100"][metric]
                    for seed in SEEDS
                ])
                mean = float(differences.mean())
                interval = stats.t.interval(
                    0.95, len(differences) - 1,
                    loc=mean, scale=stats.sem(differences),
                )
                p = float(stats.ttest_1samp(differences, 0).pvalue)
                key = f"{condition}-N"
                saved = stored["dapt_minus_n"][key]
                close(mean, saved["mean_difference"], 6e-7, f"{split}:{metric}:{key}: mean")
                close(interval[0], saved["ci95"][0], 6e-7, f"{split}:{metric}:{key}: lower CI")
                close(interval[1], saved["ci95"][1], 6e-7, f"{split}:{metric}:{key}: upper CI")
                close(p, saved["p_uncorrected"], 6e-9, f"{split}:{metric}:{key}: p")
                paired_rows[key] = {"mean": mean, "ci95": [float(interval[0]), float(interval[1])], "p": p}
                raw_ps[key] = p
            ordered = sorted(raw_ps, key=raw_ps.get)
            running = 0.0
            holm = {}
            for index, key in enumerate(ordered):
                running = max(running, min(1.0, (len(ordered) - index) * raw_ps[key]))
                holm[key] = running
            for key, value in holm.items():
                close(value, stored["dapt_minus_n"][key]["p_holm"], 2e-8,
                      f"{split}:{metric}:{key}: Holm p")
                paired_rows[key]["holm_p"] = value
            independently_recomputed[split][metric] = {
                "F": float(f_value), "p": float(p_value), "partial_eta_sq": float(eta),
                "paired": paired_rows,
            }
    report["independently_recomputed_statistics"] = independently_recomputed

    tex = (PAPER / "target-pretraining.tex").read_text(encoding="utf8")
    names = {"No DAPT": "N", "Authentic": "A", "Synthetic": "S", "Mixed": "M"}
    test_table = table_body(tex, "tab:equal_update_test")
    test_rows = re.findall(
        r"^(No DAPT|Authentic|Synthetic|Mixed)\s*&\s*([\d.]+)\s*&\s*\$([\d.]+)\\pm([\d.]+)\$\s*&\s*(.*?)\\\\$",
        test_table, re.M,
    )
    check(len(test_rows) == 4, "Equal-update test table should contain four rows")
    for name, em, f1, sd, contrast in test_rows:
        code = names[name]
        summary = report["test"]["summaries"][code]
        check(em == f"{summary['mean_em']:.2f}", f"Test table {name}: EM")
        check(f1 == f"{summary['mean_f1']:.2f}", f"Test table {name}: F1")
        check(sd == f"{summary['sd_f1']:.2f}", f"Test table {name}: SD")
        if code == "N":
            check(contrast.strip() == "---", "Test table No DAPT contrast")
        else:
            row = test_stats["f1"]["dapt_minus_n"][f"{code}-N"]
            expected_text = (
                f"${row['mean_difference']:+.2f}\\;"
                f"[{row['ci95'][0]:.2f},{row['ci95'][1]:.2f}]$"
            )
            check(contrast.strip() == expected_text, f"Test table {name}: contrast interval")

    val_table = table_body(tex, "tab:equal_update_validation")
    val_rows = re.findall(
        r"^(No DAPT|Authentic|Synthetic|Mixed)\s*&\s*(\d+)\s*&\s*([\d.]+)\s*&\s*\$([\d.]+)\\pm([\d.]+)\$\s*\\\\$",
        val_table, re.M,
    )
    check(len(val_rows) == 4, "Equal-update validation table should contain four rows")
    for name, seeds, em, f1, sd in val_rows:
        summary = report["val"]["summaries"][names[name]]
        check(int(seeds) == 5, f"Validation table {name}: seed count")
        check(em == f"{summary['mean_em']:.2f}", f"Validation table {name}: EM")
        check(f1 == f"{summary['mean_f1']:.2f}", f"Validation table {name}: F1")
        check(sd == f"{summary['sd_f1']:.2f}", f"Validation table {name}: SD")

    budget_table = table_body(tex, "tab:equal_update_budget")
    budget_rows = re.findall(
        r"^([ASM])\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*([\d,]+)\s*\(([\d,]+)--([\d,]+)\)\s*\\\\$",
        budget_table, re.M,
    )
    check(len(budget_rows) == 3, "Equal-update budget table should contain three rows")
    for condition, updates, warmup, mean, minimum, maximum in budget_rows:
        summary = token_summary[condition]
        check(int(updates) == 100 and int(warmup) == 10, f"Budget table {condition}: schedule")
        check(int(mean.replace(",", "")) == summary["mean"], f"Budget table {condition}: mean tokens")
        check(int(minimum.replace(",", "")) == summary["minimum"], f"Budget table {condition}: minimum tokens")
        check(int(maximum.replace(",", "")) == summary["maximum"], f"Budget table {condition}: maximum tokens")

    required_claims = (
        "All 62 readers have saved",
        "\\(F(3,12)=0.474\\), \\(p=0.706\\)",
        "Mean test F1 is 85.86 without DAPT, 85.85 with authentic text, 85.91 with",
        "post hoc schedule sensitivity analysis",
    )
    for claim in required_claims:
        check(claim in tex, f"Missing or changed Experiment D claim: {claim}")
    abstract = tex.split("\\abstract{", 1)[1].split("}\n", 1)[0]
    abstract_plain = re.sub(r"\\[A-Za-z]+|[{}$]", " ", abstract)
    abstract_words = re.findall(r"\b[\w]+(?:[-–—][\w]+)*\b", abstract_plain)
    check(150 <= len(abstract_words) <= 250, f"Abstract word count is {len(abstract_words)}")
    report["manuscript"] = {
        "test_rows_checked": len(test_rows),
        "validation_rows_checked": len(val_rows),
        "budget_rows_checked": len(budget_rows),
        "abstract_words": len(abstract_words),
    }

    OUT.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"Wrote {OUT.relative_to(ROOT)}", flush=True)
    if ERRORS:
        print(f"Audit failed with {len(ERRORS)} error(s):", flush=True)
        for error in ERRORS:
            print(f"- {error}", flush=True)
        return 1
    print("Experiment D audit passed with no errors", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
