"""Resumable driver for final DAPT Experiments A, B and C.

Experiment A  N/A/S/M x seeds 7,42,99,123,2026        = 20 full pipelines
Experiment B  S_STYLE x seeds 7,42,99                 =  3 full pipelines
Experiment C  N/A/S/M x seeds 7,42,99 x {25%,50%}     = 24 IPR-only runs

The driver is deliberately serial. Each stage writes recovery checkpoints;
re-running resumes the latest checkpoint and skips completed ``final/``
directories. Every completed IPR checkpoint is evaluated on validation and
written to ``validation_results.json``. Test evaluation remains locked behind
``FINAL_ANALYSIS_LOCK.json``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "experiments" / "dapt" / "final_cpu"
WORK = ROOT / "artifacts" / "models" / "final_cpu"
LEDGER = FINAL / "RUN_LEDGER.csv"
CPU_BENCHMARK = FINAL / "cpu_benchmark.json"
EVALUATOR = ROOT / "scripts" / "09_dapt_final" / "evaluate_final_experiments.py"
PYTHON = sys.executable

BASE_MODEL = "law-ai/InLegalBERT"
SEEDS_A = [7, 42, 99, 123, 2026]
SEEDS_B = [7, 42, 99]
SEEDS_C = [7, 42, 99]
FRACTIONS_C = [25, 50]
CONDITIONS_A = ["N", "A", "S", "M"]
CORPUS_FILE = {
    "A": "corpus_A.jsonl",
    "S": "corpus_S.jsonl",
    "M": "corpus_M.jsonl",
    "S_STYLE": "corpus_S_STYLE.jsonl",
}

LEDGER_COLS = [
    "experiment", "condition", "seed", "stage", "status", "start_time",
    "end_time", "checkpoint", "runtime_hours", "notes",
]

sys.path.insert(0, str(ROOT / "scripts" / "05_size_matched_dapt"))
import size_matched_dapt_experiment as base  # noqa: E402


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_ledger() -> dict:
    if not LEDGER.exists():
        return {}
    with io.open(LEDGER, encoding="utf8", newline="") as handle:
        return {
            (row["experiment"], row["condition"], row["seed"], row["stage"]): row
            for row in csv.DictReader(handle)
        }


def write_ledger(rows: dict) -> None:
    """Atomically rewrite the ledger so interruption cannot truncate it."""
    FINAL.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".csv.tmp")
    with io.open(tmp, "w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
    os.replace(tmp, LEDGER)


def reconcile_ledger(rows: dict) -> None:
    """Remove impossible completion claims left by historical dry runs."""
    changed = False
    for row in rows.values():
        if row.get("status") not in {
            "MLM_DONE", "SQUAD_DONE", "IPR_DONE", "EVALUATED", "SKIPPED"
        }:
            continue
        checkpoint = row.get("checkpoint", "")
        if checkpoint and Path(checkpoint).exists():
            continue
        row.update({
            "status": "NOT_STARTED",
            "start_time": "",
            "end_time": "",
            "checkpoint": "",
            "runtime_hours": "",
            "notes": "reconciled: historical completion row had no artifact",
        })
        changed = True
    if changed:
        write_ledger(rows)


def mark(rows, exp, cond, seed, stage, status, **fields):
    """Write one truthful stage state, clearing stale dry-run/failure fields."""
    key = (exp, cond, str(seed), stage)
    row = {column: "" for column in LEDGER_COLS}
    row.update(rows.get(key, {}))
    row.update({
        "experiment": exp,
        "condition": cond,
        "seed": str(seed),
        "stage": stage,
        "status": status,
    })
    if status == "RUNNING":
        row.update({
            "start_time": fields.pop("start_time", now()),
            "end_time": "",
            "checkpoint": fields.pop("checkpoint", ""),
            "runtime_hours": "",
            "notes": "",
        })
    elif status in {"FAILED", "RESUMABLE"}:
        row["end_time"] = fields.pop("end_time", now())
        row["notes"] = ""
    elif status in {"MLM_DONE", "SQUAD_DONE", "IPR_DONE", "EVALUATED", "SKIPPED"}:
        row["end_time"] = fields.pop("end_time", now())
        row["notes"] = ""
    row.update({name: str(value) for name, value in fields.items()})
    rows[key] = row
    write_ledger(rows)
    return row


def set_seed(text: str, seed: int) -> str:
    patched, count = re.subn(
        r"^SEED\s*=\s*\d+.*$", f"SEED = {seed}", text, count=1, flags=re.M
    )
    if count != 1:
        raise RuntimeError("seed patch failed")
    return patched


def set_const(text: str, name: str, path: Path | str) -> str:
    pattern = rf"^{name}\s*=\s*.*$"
    patched, count = re.subn(
        pattern,
        lambda _match: f'{name} = Path(r"{path}")',
        text,
        count=1,
        flags=re.M,
    )
    if count != 1:
        raise RuntimeError(f"constant {name} patch failed")
    return patched


def set_int_const(text: str, name: str, value: int) -> str:
    pattern = rf"^{name}\s*=\s*\d+.*$"
    patched, count = re.subn(
        pattern, f"{name} = {value}", text, count=1, flags=re.M
    )
    if count != 1:
        raise RuntimeError(f"integer constant {name} patch failed")
    return patched


def cpu_config(stage: str = "squad") -> dict:
    """Return the measured CPU configuration or the frozen baseline."""
    if CPU_BENCHMARK.exists():
        report = json.loads(CPU_BENCHMARK.read_text(encoding="utf8"))
        stage_specific = report.get("recommended_by_stage", {}).get(stage, {})
        if "micro_batch" in stage_specific:
            result = dict(stage_specific)
            if os.name == "nt":
                result["dataloader_workers"] = 0
            return result
        recommended = report.get("recommended", {})
        if "micro_batch" in recommended:
            result = dict(recommended)
            if os.name == "nt":
                result["dataloader_workers"] = 0
            return result
    return {
        "micro_batch": 8,
        "grad_accum": 4,
        "threads": min(24, os.cpu_count() or 1),
        "dataloader_workers": 0,
        "source": "frozen baseline; benchmark not yet available",
    }


def apply_cpu_config(source: str, stage: str) -> str:
    config = cpu_config(stage)
    source = set_int_const(source, "BATCH_SIZE", int(config["micro_batch"]))
    grad_name = "GRAD_ACCUM_STEPS" if stage == "mlm" else "GRAD_ACCUM"
    source = set_int_const(source, grad_name, int(config["grad_accum"]))
    return set_int_const(
        source, "DATALOADER_WORKERS", int(config.get("dataloader_workers", 0))
    )


def is_valid_recovery_checkpoint(path: Path) -> bool:
    """Return whether a checkpoint has the files required by Trainer resume.

    A process can be interrupted while serialising a checkpoint.  Such a
    directory may contain model weights but lack ``trainer_state.json`` or an
    optimizer/scheduler state; passing it to ``Trainer.train`` then causes
    repeated failed retries.  Completed stages use ``final/`` and do not rely
    on these recovery directories.
    """
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    required = ("config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt")
    return path.is_dir() and bool(weights) and all((path / name).exists() for name in required)


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    for path in output_dir.glob("checkpoint-*") if output_dir.exists() else []:
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if is_valid_recovery_checkpoint(path):
            candidates.append((step, path))
        else:
            print(f"      ignoring incomplete recovery checkpoint: {path}", flush=True)
    return max(candidates, default=(None, None))[1]


def cleanup_completed_checkpoints(output_dir: Path, final_dir: Path) -> None:
    """Reclaim recovery-only checkpoint storage after a verified completion.

    The final model is retained for the next stage and for evaluation.  The
    ledger records that final directory, so recovery checkpoints are redundant
    once it exists.  Cleanup is intentionally best-effort: a filesystem error
    must never invalidate a completed experimental result.
    """
    if not is_complete_model(final_dir):
        return
    for checkpoint in output_dir.glob("checkpoint-*"):
        if not checkpoint.is_dir():
            continue
        try:
            shutil.rmtree(checkpoint)
            print(f"      removed completed-stage recovery checkpoint: {checkpoint}", flush=True)
        except OSError as exc:
            print(f"      warning: could not remove {checkpoint}: {exc}", flush=True)


def is_complete_model(path: Path) -> bool:
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    tokenizer = (path / "tokenizer.json").exists() or (path / "vocab.txt").exists()
    return path.is_dir() and (path / "config.json").exists() and bool(weights) and tokenizer


def stage_environment(stage: str = "squad") -> dict:
    env = os.environ.copy()
    config = cpu_config(stage)
    threads = str(config.get("threads", min(24, os.cpu_count() or 1)))
    env.update({
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    return env


def run_stage(
    tag: str, stage: str, source: str, dry_run: bool, max_retries: int,
) -> tuple[float, int]:
    """Write and execute one patched stage, retrying from its latest checkpoint."""
    WORK.mkdir(parents=True, exist_ok=True)
    script = WORK / f"{tag}.py"
    script.write_text(source, encoding="utf-8")
    if dry_run:
        print(f"      [dry-run] would execute {script.relative_to(ROOT)}")
        return 0.0, 0

    started = time.time()
    for attempt in range(1, max_retries + 2):
        print(f"      attempt {attempt}/{max_retries + 1}", flush=True)
        process = subprocess.run(
            [PYTHON, "-X", "utf8", str(script)],
            cwd=str(ROOT),
            env=stage_environment(stage),
        )
        if process.returncode == 0:
            return (time.time() - started) / 3600.0, attempt
        if attempt <= max_retries:
            print(
                f"      stage exited {process.returncode}; retrying from the latest "
                "checkpoint in 10 seconds",
                flush=True,
            )
            time.sleep(10)
            continue
        raise RuntimeError(
            f"stage {tag} failed with code {process.returncode} after {attempt} attempts"
        )
    raise AssertionError("unreachable")


def materialise_low_resource_split(percent: int) -> Path:
    dest = FINAL / f"train_squad_{percent}.jsonl"
    manifest = json.loads(
        (FINAL / f"LOW_RESOURCE_{percent}_MANIFEST.json").read_text(encoding="utf8")
    )
    keep = set(manifest["question_ids"])
    source = ROOT / "data" / "splits" / "train_squad.jsonl"
    rows = [
        line + "\n" for line in source.read_text(encoding="utf8").splitlines()
        if line.strip() and str(json.loads(line)["id"]) in keep
    ]
    if len(rows) != len(keep):
        raise RuntimeError(
            f"low-resource {percent}%: matched {len(rows)} of {len(keep)} ids"
        )
    expected = "".join(rows)
    if not dest.exists() or dest.read_text(encoding="utf8") != expected:
        dest.write_text(expected, encoding="utf8")
    return dest


def materialise_base_encoder() -> Path:
    dest = WORK / "base_encoder"
    if is_complete_model(dest):
        existing = json.loads((dest / "config.json").read_text(encoding="utf8"))
        if existing.get("tie_word_embeddings") is False:
            return dest
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dest.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True).save_pretrained(dest)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL, local_files_only=True)
    model.config.tie_word_embeddings = False
    model.save_pretrained(dest)
    return dest


def make_mlm_source(corpus: Path, output: Path, seed: int) -> str:
    source = base.make_mlm(variant="original_only", outdir=str(output))
    source = set_const(source, "BASE_DIR", ROOT)
    source = set_const(source, "INPUT_FILE", corpus)
    source = set_seed(source, seed)
    return apply_cpu_config(source, "mlm")


def make_squad_source(model: Path, output: Path, seed: int) -> str:
    source = base.make_squad(mlm_final=str(model), outdir=str(output))
    source = set_const(source, "BASE_DIR", ROOT)
    source = set_seed(source, seed)
    return apply_cpu_config(source, "squad")


def make_ipr_source(model: Path, output: Path, seed: int, train_file: Path | None = None) -> str:
    source = base.make_ipr(squad_final=str(model), outdir=str(output))
    source = set_const(source, "BASE_DIR", ROOT)
    if train_file is not None:
        source = set_const(source, "TRAIN_FILE", train_file)
    source = set_seed(source, seed)
    return apply_cpu_config(source, "ipr")


def run_key(experiment: str, condition: str, seed: int, fraction: int = 100) -> str:
    return f"{experiment}_{condition}_s{seed}_{fraction}"


def validation_evaluation(rows, experiment, condition, seed, fraction, dry_run, max_retries):
    key = run_key(experiment, condition, seed, fraction)
    result_file = FINAL / "evaluations" / "val" / f"{key}.json"
    ledger_condition = condition if fraction == 100 else f"{condition}@{fraction}"
    command = [
        PYTHON, str(EVALUATOR), "--split", "val", "--run-key", key,
        "--device", "cpu",
    ]
    if dry_run:
        print("      [dry-run] would execute " + " ".join(command))
        return
    mark(rows, experiment, ledger_condition, seed, "val_eval", "RUNNING")
    started = time.time()
    for attempt in range(1, max_retries + 2):
        process = subprocess.run(command, cwd=str(ROOT), env=stage_environment("ipr"))
        if process.returncode == 0:
            mark(
                rows, experiment, ledger_condition, seed, "val_eval", "EVALUATED",
                checkpoint=str(result_file),
                runtime_hours=round((time.time() - started) / 3600.0, 3),
                notes=f"canonical sliding-window validation; attempts={attempt}",
            )
            return
        if attempt <= max_retries:
            time.sleep(5)
            continue
        mark(
            rows, experiment, ledger_condition, seed, "val_eval", "FAILED",
            notes=f"canonical validation failed after {attempt} attempts",
        )
        raise RuntimeError(f"validation evaluation failed for {key}")


def execute_stage(rows, exp, cond, seed, stage, output, final_dir, source_builder,
                  dry_run, max_retries):
    if is_complete_model(final_dir):
        print(f"    {stage:5s} already complete -> skip")
        if not dry_run:
            cleanup_completed_checkpoints(output, final_dir)
            mark(
                rows, exp, cond, seed, stage, f"{stage.upper()}_DONE",
                checkpoint=str(final_dir), notes="pre-existing validated final directory",
            )
        return

    recovery = latest_checkpoint(output)
    print(f"    {stage:5s} {'resuming' if recovery else 'running'}")
    if not dry_run:
        mark(
            rows, exp, cond, seed, stage, "RUNNING",
            checkpoint=str(recovery or ""),
            notes=f"resume checkpoint: {recovery}" if recovery else "",
        )
    try:
        runtime, attempts = run_stage(
            f"{exp}_{cond.replace('@', '')}_s{seed}_{stage}",
            stage, source_builder(), dry_run, max_retries,
        )
    except Exception as exc:
        if not dry_run:
            recovery = latest_checkpoint(output)
            mark(
                rows, exp, cond, seed, stage,
                "RESUMABLE" if recovery else "FAILED",
                checkpoint=str(recovery or ""), notes=str(exc)[:300],
            )
        raise
    if not dry_run:
        if not final_dir.exists():
            raise RuntimeError(f"stage exited successfully but final directory is missing: {final_dir}")
        mark(
            rows, exp, cond, seed, stage, f"{stage.upper()}_DONE",
            checkpoint=str(final_dir), runtime_hours=round(runtime, 3),
            notes=f"attempts={attempts}",
        )
        cleanup_completed_checkpoints(output, final_dir)


def full_pipeline(rows, experiment, condition, seed, dry_run, max_retries):
    tag = f"{experiment}_{condition}_s{seed}"
    if condition == "N":
        mlm_final = WORK / "base_encoder"
        if not dry_run:
            mlm_final = materialise_base_encoder()
            mark(
                rows, experiment, condition, seed, "mlm", "SKIPPED",
                checkpoint=str(mlm_final), notes="no DAPT stage by definition",
            )
    else:
        corpus = FINAL / CORPUS_FILE[condition]
        if not corpus.exists():
            raise FileNotFoundError(f"missing required corpus: {corpus}")
        mlm_output = WORK / f"{tag}_mlm"
        mlm_final = mlm_output / "final"
        execute_stage(
            rows, experiment, condition, seed, "mlm", mlm_output, mlm_final,
            lambda: make_mlm_source(corpus, mlm_output, seed), dry_run, max_retries,
        )

    squad_output = WORK / f"{tag}_squad"
    squad_final = squad_output / "final"
    execute_stage(
        rows, experiment, condition, seed, "squad", squad_output, squad_final,
        lambda: make_squad_source(mlm_final, squad_output, seed), dry_run, max_retries,
    )

    ipr_output = WORK / f"{tag}_ipr"
    ipr_final = ipr_output / "final"
    execute_stage(
        rows, experiment, condition, seed, "ipr", ipr_output, ipr_final,
        lambda: make_ipr_source(squad_final, ipr_output, seed), dry_run, max_retries,
    )
    validation_evaluation(rows, experiment, condition, seed, 100, dry_run, max_retries)


def low_resource_pipeline(rows, condition, fraction, seed, dry_run, max_retries):
    squad_final = WORK / f"A_{condition}_s{seed}_squad" / "final"
    ledger_condition = f"{condition}@{fraction}"
    if not squad_final.exists() and not dry_run:
        mark(
            rows, "C", ledger_condition, seed, "ipr", "FAILED",
            notes=f"missing Experiment A post-SQuAD checkpoint: {squad_final}",
        )
        raise FileNotFoundError(f"Experiment C requires {squad_final}")
    train_file = FINAL / f"train_squad_{fraction}.jsonl"
    if not dry_run:
        train_file = materialise_low_resource_split(fraction)
    output = WORK / f"C_{condition}{fraction}_s{seed}_ipr"
    final_dir = output / "final"
    execute_stage(
        rows, "C", ledger_condition, seed, "ipr", output, final_dir,
        lambda: make_ipr_source(squad_final, output, seed, train_file),
        dry_run, max_retries,
    )
    validation_evaluation(rows, "C", condition, seed, fraction, dry_run, max_retries)


def experiment_plan(experiment: str) -> list[tuple]:
    plans = {
        "A": [("A", condition, seed, 100) for condition in CONDITIONS_A for seed in SEEDS_A],
        "B": [("B", "S_STYLE", seed, 100) for seed in SEEDS_B],
        "C": [
            ("C", condition, seed, fraction)
            for condition in CONDITIONS_A for seed in SEEDS_C for fraction in FRACTIONS_C
        ],
    }
    return plans["A"] + plans["B"] + plans["C"] if experiment == "all" else plans[experiment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["A", "B", "C", "all"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="automatic retries per failed stage; recovery checkpoints are reused",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    rows = read_ledger()
    if not args.dry_run:
        reconcile_ledger(rows)
    plan = experiment_plan(args.experiment)
    print(f"  Experiment {args.experiment}: {len(plan)} pipeline entries")
    qa_config = cpu_config("squad")
    mlm_config = cpu_config("mlm")
    print(
        f"  CPU QA config: micro={qa_config['micro_batch']} accum={qa_config['grad_accum']} "
        f"threads={qa_config.get('threads')} workers={qa_config.get('dataloader_workers', 0)}"
    )
    print(
        f"  CPU MLM config: micro={mlm_config['micro_batch']} accum={mlm_config['grad_accum']} "
        f"threads={mlm_config.get('threads')} workers={mlm_config.get('dataloader_workers', 0)}"
    )
    if args.dry_run:
        print("  DRY RUN -- scripts are generated; no model, result, or ledger is changed")

    for experiment, condition, seed, fraction in plan:
        label = condition if fraction == 100 else f"{condition}@{fraction}"
        print(f"\n  == {experiment} / {label} / seed {seed} ==", flush=True)
        try:
            if experiment == "C":
                low_resource_pipeline(rows, condition, fraction, seed, args.dry_run, args.max_retries)
            else:
                full_pipeline(rows, experiment, condition, seed, args.dry_run, args.max_retries)
        except Exception as exc:
            print(f"     FAILED: {exc}", flush=True)
            if not args.dry_run:
                return 1

    print(f"\n  ledger: {LEDGER.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
