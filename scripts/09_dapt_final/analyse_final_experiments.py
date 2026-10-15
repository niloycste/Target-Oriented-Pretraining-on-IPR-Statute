"""Complete statistical and difficulty analysis for final DAPT A/B/C results."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "experiments" / "dapt" / "final_cpu"
RESULTS = FINAL / "results.json"
OUT = FINAL / "FINAL_STATISTICS.json"
DIFFICULTY_OUT = FINAL / "QA_DIFFICULTY_DIAGNOSTICS.json"
MLM_RESULTS = FINAL / "MLM_DIAGNOSTIC_RESULTS.json"
SEEDS_A = [7, 42, 99, 123, 2026]
SEEDS_BC = [7, 42, 99]
CONDITIONS = ["N", "A", "S", "M"]
CONTRASTS_A = [
    ("A", "N"), ("S", "N"), ("M", "N"),
    ("A", "S"), ("A", "M"), ("S", "M"),
]


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf8")
    os.replace(temp, path)


def mean_sd_ci(values: list[float]) -> dict:
    count = len(values)
    if not count:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    mean = st.mean(values)
    sd = st.stdev(values) if count > 1 else 0.0
    if count > 1:
        half = float(stats.t.ppf(0.975, count - 1)) * sd / math.sqrt(count)
        interval = [round(mean - half, 6), round(mean + half, 6)]
    else:
        interval = None
    return {
        "n": count, "mean": round(mean, 6), "sd": round(sd, 6), "ci95": interval,
    }


def paired(values_a: list[float], values_b: list[float], label: str,
           seeds: list[int]) -> dict:
    differences = [a - b for a, b in zip(values_a, values_b)]
    summary = mean_sd_ci(differences)
    if len(differences) < 2:
        return {"contrast": label, "seeds": seeds, "error": "insufficient paired seeds"}
    mean = st.mean(differences)
    sd = st.stdev(differences)
    standard_error = sd / math.sqrt(len(differences))
    if standard_error:
        t_value: float | str = mean / standard_error
        p_value = float(2 * stats.t.sf(abs(t_value), len(differences) - 1))
    elif mean == 0:
        t_value = 0.0
        p_value = 1.0
    else:
        t_value = "inf" if mean > 0 else "-inf"
        p_value = 0.0
    return {
        "contrast": label,
        "seeds": seeds,
        "per_seed_differences": [round(value, 6) for value in differences],
        "mean_difference": summary["mean"],
        "sd": summary["sd"],
        "ci95": summary["ci95"],
        "t": round(t_value, 6) if isinstance(t_value, float) else t_value,
        "p_uncorrected": round(p_value, 8),
    }


def holm(rows: dict[str, dict]) -> None:
    ordered = sorted(
        ((label, row["p_uncorrected"]) for label, row in rows.items()
         if "p_uncorrected" in row),
        key=lambda item: item[1],
    )
    previous = 0.0
    total = len(ordered)
    for index, (label, p_value) in enumerate(ordered):
        adjusted = min(1.0, max(previous, (total - index) * p_value))
        rows[label]["p_holm"] = round(adjusted, 8)
        previous = adjusted


def blocked_anova(table: dict[str, dict[int, dict]], metric: str) -> dict:
    if any(seed not in table.get(condition, {}) for condition in CONDITIONS for seed in SEEDS_A):
        present = sum(seed in table.get(condition, {}) for condition in CONDITIONS for seed in SEEDS_A)
        return {"error": "incomplete design", "cells_present": present, "cells_required": 20}
    cells = {
        (condition, seed): table[condition][seed][metric]
        for condition in CONDITIONS for seed in SEEDS_A
    }
    grand = st.mean(cells.values())
    condition_ss = len(SEEDS_A) * sum(
        (st.mean(cells[(condition, seed)] for seed in SEEDS_A) - grand) ** 2
        for condition in CONDITIONS
    )
    seed_ss = len(CONDITIONS) * sum(
        (st.mean(cells[(condition, seed)] for condition in CONDITIONS) - grand) ** 2
        for seed in SEEDS_A
    )
    total_ss = sum((value - grand) ** 2 for value in cells.values())
    error_ss = max(0.0, total_ss - condition_ss - seed_ss)
    condition_df = len(CONDITIONS) - 1
    error_df = condition_df * (len(SEEDS_A) - 1)
    condition_ms = condition_ss / condition_df
    error_ms = error_ss / error_df
    if error_ms:
        f_value: float | str = condition_ms / error_ms
        p_value = float(stats.f.sf(f_value, condition_df, error_df))
    elif condition_ms == 0:
        f_value = 0.0
        p_value = 1.0
    else:
        f_value = "inf"
        p_value = 0.0
    eta_denominator = condition_ss + error_ss
    return {
        "F_condition": round(f_value, 6) if isinstance(f_value, float) else f_value,
        "df": [condition_df, error_df],
        "p_condition": round(p_value, 8),
        "partial_eta_sq": round(condition_ss / eta_denominator, 6) if eta_denominator else None,
        "ss_condition": round(condition_ss, 6),
        "ss_seed": round(seed_ss, 6),
        "ss_error": round(error_ss, 6),
    }


def table_from_runs(runs: dict, experiment: str, fraction: int,
                    conditions: list[str]) -> dict[str, dict[int, dict]]:
    table = {condition: {} for condition in conditions}
    for row in runs.values():
        if row["experiment"] != experiment or int(row["fraction"]) != fraction:
            continue
        if row["condition"] in table:
            table[row["condition"]][int(row["seed"])] = row
    return table


def experiment_a(table: dict, metric: str) -> dict:
    block = {
        "descriptives": {
            condition: mean_sd_ci([table[condition][seed][metric] for seed in SEEDS_A])
            for condition in CONDITIONS if all(seed in table[condition] for seed in SEEDS_A)
        },
        "blocked_anova": blocked_anova(table, metric),
        "contrasts": {},
    }
    for first, second in CONTRASTS_A:
        common = [seed for seed in SEEDS_A if seed in table[first] and seed in table[second]]
        block["contrasts"][f"{first}-{second}"] = paired(
            [table[first][seed][metric] for seed in common],
            [table[second][seed][metric] for seed in common],
            f"{first}-{second}", common,
        )
    holm(block["contrasts"])
    return block


def experiment_b(a_table: dict, b_table: dict, metric: str) -> dict:
    common = [seed for seed in SEEDS_BC if seed in a_table["S"] and seed in b_table["S_STYLE"]]
    result = paired(
        [a_table["S"][seed][metric] for seed in common],
        [b_table["S_STYLE"][seed][metric] for seed in common],
        "S_TARGET-S_STYLE", common,
    )
    if result.get("mean_difference") is not None:
        result["effect_direction"] = (
            "target_aligned_higher" if result["mean_difference"] > 0
            else "style_only_higher" if result["mean_difference"] < 0
            else "no_observed_difference"
        )
    result["interpretation_note"] = "n=3 paired seeds; treat significance cautiously"
    return result


def experiment_c(a_table: dict, c_tables: dict[int, dict], metric: str) -> dict:
    tables = {25: c_tables[25], 50: c_tables[50], 100: a_table}
    report = {"fractions": {}, "interaction_pattern": {}}
    for fraction, table in tables.items():
        seeds = SEEDS_BC
        descriptives = {}
        contrasts = {}
        for condition in CONDITIONS:
            values = [table[condition][seed][metric] for seed in seeds if seed in table[condition]]
            descriptives[condition] = mean_sd_ci(values)
        for condition in ("A", "S", "M"):
            common = [seed for seed in seeds if seed in table[condition] and seed in table["N"]]
            contrasts[f"{condition}-N"] = paired(
                [table[condition][seed][metric] for seed in common],
                [table["N"][seed][metric] for seed in common],
                f"{condition}-N@{fraction}", common,
            )
        holm(contrasts)
        report["fractions"][str(fraction)] = {
            "descriptives": descriptives, "dapt_minus_n": contrasts,
        }

    baseline = tables[100]
    for fraction in (25, 50):
        table = tables[fraction]
        report["interaction_pattern"][str(fraction)] = {}
        for condition in ("A", "S", "M"):
            common = [
                seed for seed in SEEDS_BC
                if all(seed in source[name] for source in (table, baseline) for name in (condition, "N"))
            ]
            low_advantage = [
                table[condition][seed][metric] - table["N"][seed][metric] for seed in common
            ]
            full_advantage = [
                baseline[condition][seed][metric] - baseline["N"][seed][metric] for seed in common
            ]
            report["interaction_pattern"][str(fraction)][condition] = paired(
                low_advantage, full_advantage,
                f"({condition}-N)@{fraction}-({condition}-N)@100", common,
            )
    return report


def difficulty_analysis(runs: dict) -> dict:
    per_run = {}
    categories = [
        "short_1_10", "medium_11_30", "long_31_plus",
        "semicolon_yes", "semicolon_no", "enumerated_yes", "enumerated_no",
        "proviso_yes", "proviso_no",
    ]
    for key, row in runs.items():
        path = ROOT / row["predictions_file"]
        if not path.exists():
            continue
        predictions = json.loads(path.read_text(encoding="utf8"))["predictions"]
        groups = defaultdict(list)
        for prediction in predictions:
            groups[prediction["answer_length_bin"]].append(prediction)
            groups["semicolon_yes" if prediction["semicolon"] else "semicolon_no"].append(prediction)
            groups["enumerated_yes" if prediction["enumerated_clause"] else "enumerated_no"].append(prediction)
            groups["proviso_yes" if prediction["proviso_or_exception"] else "proviso_no"].append(prediction)
        per_run[key] = {}
        for category in categories:
            values = groups.get(category, [])
            if values:
                per_run[key][category] = {
                    "n": len(values),
                    "em": round(100 * st.mean(item["em"] for item in values), 6),
                    "f1": round(100 * st.mean(item["f1"] for item in values), 6),
                }

    grouped = defaultdict(lambda: defaultdict(lambda: {"em": [], "f1": [], "n": []}))
    for key, categories_for_run in per_run.items():
        row = runs[key]
        group_key = f"{row['experiment']}:{row['condition']}:{row['fraction']}"
        for category, metrics in categories_for_run.items():
            for metric in ("em", "f1", "n"):
                grouped[group_key][category][metric].append(metrics[metric])
    aggregate = {}
    for group_key, grouped_categories in sorted(grouped.items()):
        aggregate[group_key] = {}
        for category, values in sorted(grouped_categories.items()):
            aggregate[group_key][category] = {
                "runs": len(values["f1"]),
                "items_per_run": values["n"][0],
                "em": mean_sd_ci(values["em"]),
                "f1": mean_sd_ci(values["f1"]),
            }
    return {
        "description": "Exploratory deterministic answer-span difficulty diagnostics",
        "predefined_bins": {
            "answer_length": {"short": "1-10 words", "medium": "11-30 words", "long": "31+ words"},
            "semicolon": "literal semicolon in gold answer",
            "enumerated_clause": "parenthesized alphanumeric/Roman marker in gold answer",
            "proviso_or_exception": "predefined proviso/exception phrase in gold answer",
        },
        "note": "exploratory; no categories were selected using performance",
        "aggregate": aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not RESULTS.exists():
        print(f"  no final test results at {RESULTS.relative_to(ROOT)}")
        return 1
    result_file = json.loads(RESULTS.read_text(encoding="utf8"))
    runs = result_file.get("runs", {})
    expected = 20 + 3 + 24
    if len(runs) != expected and not args.allow_partial:
        print(f"  incomplete final result set: {len(runs)}/{expected} runs")
        return 1

    a_table = table_from_runs(runs, "A", 100, CONDITIONS)
    b_table = table_from_runs(runs, "B", 100, ["S_STYLE"])
    c_tables = {
        fraction: table_from_runs(runs, "C", fraction, CONDITIONS)
        for fraction in (25, 50)
    }
    report = {
        "schema_version": 1,
        "description": "Final DAPT statistics; complete training seed is the replication unit",
        "test_runs_present": len(runs),
        "reporting_rule": "p>0.05 is non-detection, never equivalence",
    }
    for metric in ("f1", "em"):
        report[f"experiment_A_{metric}"] = experiment_a(a_table, metric)
        report[f"experiment_B_{metric}"] = experiment_b(a_table, b_table, metric)
        report[f"experiment_C_{metric}"] = experiment_c(a_table, c_tables, metric)
    if MLM_RESULTS.exists():
        report["mlm_diagnostic_file"] = MLM_RESULTS.relative_to(ROOT).as_posix()

    difficulty = difficulty_analysis(runs)
    atomic_json(DIFFICULTY_OUT, difficulty)
    report["qa_difficulty_file"] = DIFFICULTY_OUT.relative_to(ROOT).as_posix()
    atomic_json(OUT, report)
    print(f"  wrote {OUT.relative_to(ROOT)}")
    print(f"  wrote {DIFFICULTY_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
