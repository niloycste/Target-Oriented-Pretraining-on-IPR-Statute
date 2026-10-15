"""Freeze final DAPT inputs and code before training and test evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "experiments" / "dapt" / "final_cpu"
LOCK_JSON = FINAL / "FINAL_ANALYSIS_LOCK.json"
LOCK_MD = FINAL / "FINAL_ANALYSIS_LOCK.md"

LOCKED_FILES = [
    "requirements.txt",
    "papers/target_pretraining/dapt.md",
    "experiments/dapt/EXPERIMENT_PROTOCOL.md",
    "data/splits/train.jsonl",
    "data/splits/val.jsonl",
    "data/splits/test.jsonl",
    "data/splits/train_squad.jsonl",
    "data/splits/val_squad.jsonl",
    "experiments/dapt/final_cpu/corpus_A.jsonl",
    "experiments/dapt/final_cpu/corpus_S.jsonl",
    "experiments/dapt/final_cpu/corpus_M.jsonl",
    "experiments/dapt/final_cpu/corpus_S_STYLE.jsonl",
    "experiments/dapt/final_cpu/A_manifest.json",
    "experiments/dapt/final_cpu/S_manifest.json",
    "experiments/dapt/final_cpu/M_manifest.json",
    "experiments/dapt/final_cpu/MATCHED_CORPORA_SUMMARY.json",
    "experiments/dapt/final_cpu/S_STYLE_manifest.json",
    "experiments/dapt/final_cpu/SYNTHETIC_PROVENANCE_AUDIT.json",
    "experiments/dapt/final_cpu/CORPUS_CHARACTERISATION.json",
    "experiments/dapt/final_cpu/DRY_RUN_REPORT.md",
    "experiments/dapt/final_cpu/MLM_DIAGNOSTIC_MANIFEST.json",
    "experiments/dapt/final_cpu/mlm_diagnostic.jsonl",
    "experiments/dapt/final_cpu/LOW_RESOURCE_25_MANIFEST.json",
    "experiments/dapt/final_cpu/LOW_RESOURCE_50_MANIFEST.json",
    "experiments/dapt/final_cpu/train_squad_25.jsonl",
    "experiments/dapt/final_cpu/train_squad_50.jsonl",
    "experiments/dapt/final_cpu/cpu_benchmark.json",
    "scripts/02_synthetic_dapt/generate_synthetic_large.py",
    "scripts/02_synthetic_dapt/pretrain_mlm_original_only.py",
    "scripts/03_finetuning/finetune_squad_v1_original_only.py",
    "scripts/03_finetuning/finetune_ipr_original_only.py",
    "scripts/09_dapt_final/audit_synthetic_provenance.py",
    "scripts/09_dapt_final/benchmark_cpu_config.py",
    "scripts/09_dapt_final/build_final_matched_corpora.py",
    "scripts/09_dapt_final/generate_style_only_corpus.py",
    "scripts/09_dapt_final/characterise_corpora.py",
    "scripts/09_dapt_final/build_diagnostic_and_lowresource.py",
    "scripts/09_dapt_final/write_dry_run_report.py",
    "scripts/09_dapt_final/run_full_study.py",
    "scripts/09_dapt_final/prepare_final_analysis_lock.py",
    "scripts/09_dapt_final/run_final_experiments.py",
    "scripts/09_dapt_final/evaluate_final_experiments.py",
    "scripts/09_dapt_final/evaluate_final_mlm.py",
    "scripts/09_dapt_final/analyse_final_experiments.py",
    "scripts/qa_evaluation.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
        capture_output=True, text=True,
    )
    return result.stdout.strip() or "unknown"


def validate_inputs() -> dict:
    missing = [relative for relative in LOCKED_FILES if not (ROOT / relative).exists()]
    if missing:
        raise RuntimeError("cannot lock; required files are missing:\n  " + "\n  ".join(missing))

    matched = json.loads((FINAL / "MATCHED_CORPORA_SUMMARY.json").read_text(encoding="utf8"))
    style = json.loads((FINAL / "S_STYLE_manifest.json").read_text(encoding="utf8"))
    diagnostic = json.loads(
        (FINAL / "MLM_DIAGNOSTIC_MANIFEST.json").read_text(encoding="utf8")
    )
    token_rows = {
        name: {
            "records": values["records"],
            "tokens": values["non_padding_tokens"],
            "delta_pct": values["delta_pct"],
        }
        for name, values in matched["corpora"].items()
    }
    token_rows["S_STYLE"] = {
        "records": style["records"],
        "tokens": style["non_padding_tokens"],
        "delta_pct": style["delta_pct"],
    }
    bad_match = {
        name: row["delta_pct"] for name, row in token_rows.items()
        if abs(float(row["delta_pct"])) > 0.05
    }
    if bad_match:
        raise RuntimeError(f"token matching exceeds 0.05%: {bad_match}")
    if diagnostic.get("source_split") != "val":
        raise RuntimeError("MLM diagnostic is not validation-only")
    return {
        "token_budget": matched["token_budget"],
        "corpora": token_rows,
        "mlm_diagnostic_contexts": diagnostic["diagnostic_contexts"],
        "mlm_diagnostic_source": diagnostic["source_split"],
    }


def main() -> int:
    checks = validate_inputs()
    hashes = {relative: sha256(ROOT / relative) for relative in LOCKED_FILES}
    lock = {
        "locked": True,
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "purpose": "Freeze code, corpora, splits and diagnostics before final DAPT training/test evaluation",
        "design": {
            "experiment_A": {"conditions": ["N", "A", "S", "M"], "seeds": [7, 42, 99, 123, 2026]},
            "experiment_B": {"conditions": ["S", "S_STYLE"], "seeds": [7, 42, 99]},
            "experiment_C": {
                "conditions": ["N", "A", "S", "M"],
                "seeds": [7, 42, 99], "fractions": [25, 50, 100],
            },
            "test_records": 344,
            "test_use": "canonical evaluation only after this lock",
        },
        "checks": checks,
        "sha256": hashes,
    }
    FINAL.mkdir(parents=True, exist_ok=True)
    temp = LOCK_JSON.with_suffix(".json.tmp")
    temp.write_text(json.dumps(lock, indent=2), encoding="utf8")
    os.replace(temp, LOCK_JSON)

    lines = [
        "# Final DAPT analysis lock",
        "",
        f"Locked: **{lock['locked_at']}** at Git revision `{lock['git_revision']}`.",
        "",
        "This lock freezes all corpus, split, training, evaluation and analysis inputs before final test evaluation.",
        "Any hash mismatch causes the canonical test evaluator to refuse execution.",
        "",
        "## Frozen design",
        "",
        "- Experiment A: N/A/S/M × seeds 7, 42, 99, 123, 2026.",
        "- Experiment B: S versus S_STYLE × seeds 7, 42, 99.",
        "- Experiment C: N/A/S/M × seeds 7, 42, 99 × 25%, 50%, 100% supervision.",
        "- Primary metrics: token-level EM and F1 under the canonical sliding-window decoder.",
        "- MLM loss/perplexity: validation-only secondary diagnostic.",
        "",
        "## Corpus lock",
        "",
        "| Condition | Records | Tokens | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name, row in checks["corpora"].items():
        lines.append(
            f"| {name} | {row['records']} | {row['tokens']:,} | {row['delta_pct']:+.3f}% |"
        )
    lines += [
        "",
        f"Validation MLM diagnostic: {checks['mlm_diagnostic_contexts']} contexts.",
        "",
        f"Machine-readable hashes: `{LOCK_JSON.relative_to(ROOT).as_posix()}`.",
    ]
    LOCK_MD.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"  wrote {LOCK_JSON.relative_to(ROOT)}")
    print(f"  wrote {LOCK_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
