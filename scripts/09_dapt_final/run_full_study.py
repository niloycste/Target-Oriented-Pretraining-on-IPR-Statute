"""Run the final DAPT study as one resumable, strictly serial queue.

The queue performs all preflight gates, freezes the analysis, runs Experiments
A/B/C, evaluates the untouched test split, computes the validation-only MLM
diagnostic, and writes the final statistics. A failed stage stops the queue;
rerunning this command resumes generation/training from their durable state.

Run:
    python -u scripts/09_dapt_final/run_full_study.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FINAL = ROOT / "experiments" / "dapt" / "final_cpu"
STATE_PATH = FINAL / "FULL_STUDY_STATE.json"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf8")
    os.replace(temp, path)


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "description": "Resumable state for the serial final DAPT study",
            "created_at": now(),
            "updated_at": now(),
            "stages": {},
        }
    return json.loads(STATE_PATH.read_text(encoding="utf8"))


def py(script: str, *arguments: str) -> list[str]:
    return [sys.executable, "-u", str(HERE / script), *arguments]


def render(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def outputs_exist(paths: tuple[Path, ...]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def run_stage(
    state: dict,
    name: str,
    command: list[str],
    outputs: tuple[Path, ...] = (),
    accept_existing: bool = False,
    force: bool = False,
) -> None:
    previous = state["stages"].get(name, {})
    complete_in_state = previous.get("status") == "DONE"
    can_skip = outputs_exist(outputs) and (complete_in_state or accept_existing)
    if can_skip and not force:
        print(f"\n=== {name}: already complete ===", flush=True)
        return

    print(f"\n=== {name} ===", flush=True)
    print(f"$ {render(command)}", flush=True)
    state["stages"][name] = {
        "status": "RUNNING",
        "started_at": now(),
        "ended_at": None,
        "command": command,
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
    }
    state["updated_at"] = now()
    atomic_json(STATE_PATH, state)

    result = subprocess.run(command, cwd=str(ROOT), env=stage_environment())
    row = state["stages"][name]
    row["ended_at"] = now()
    row["return_code"] = result.returncode
    if result.returncode != 0:
        row["status"] = "FAILED"
        state["updated_at"] = now()
        atomic_json(STATE_PATH, state)
        raise RuntimeError(f"stage {name!r} failed with exit code {result.returncode}")
    missing = [str(path.relative_to(ROOT)) for path in outputs if not path.exists()]
    if missing:
        row["status"] = "FAILED"
        row["missing_outputs"] = missing
        state["updated_at"] = now()
        atomic_json(STATE_PATH, state)
        raise RuntimeError(f"stage {name!r} did not create: {missing}")
    row["status"] = "DONE"
    state["updated_at"] = now()
    atomic_json(STATE_PATH, state)


def stage_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def stages(force_benchmark: bool = False) -> list[dict]:
    return [
        {
            "name": "01_provenance_audit",
            "command": py("audit_synthetic_provenance.py"),
            "outputs": (FINAL / "SYNTHETIC_PROVENANCE_AUDIT.json",),
        },
        {
            "name": "02_matched_corpora",
            "command": py("build_final_matched_corpora.py"),
            "outputs": (
                FINAL / "corpus_A.jsonl",
                FINAL / "corpus_S.jsonl",
                FINAL / "corpus_M.jsonl",
                FINAL / "MATCHED_CORPORA_SUMMARY.json",
            ),
        },
        {
            "name": "03_cpu_benchmark",
            "command": py("benchmark_cpu_config.py"),
            "outputs": (FINAL / "cpu_benchmark.json", FINAL / "CPU_OPTIMIZATION_REPORT.md"),
            "accept_existing": True,
            "force": force_benchmark,
        },
        {
            "name": "04_style_only_generation",
            "command": py("generate_style_only_corpus.py"),
            "outputs": (FINAL / "style_only_generated.jsonl",),
        },
        {
            "name": "05_corpus_characterisation",
            "command": py("characterise_corpora.py"),
            "outputs": (
                FINAL / "corpus_S_STYLE.jsonl",
                FINAL / "S_STYLE_manifest.json",
                FINAL / "CORPUS_CHARACTERISATION.json",
            ),
            "accept_existing": True,
        },
        {
            "name": "06_diagnostics_and_low_resource",
            "command": py("build_diagnostic_and_lowresource.py"),
            "outputs": (
                FINAL / "MLM_DIAGNOSTIC_MANIFEST.json",
                FINAL / "mlm_diagnostic.jsonl",
                FINAL / "LOW_RESOURCE_25_MANIFEST.json",
                FINAL / "LOW_RESOURCE_50_MANIFEST.json",
                FINAL / "train_squad_25.jsonl",
                FINAL / "train_squad_50.jsonl",
            ),
        },
        {
            "name": "07_dry_run_report",
            "command": py("write_dry_run_report.py"),
            "outputs": (FINAL / "DRY_RUN_REPORT.md",),
        },
        {
            "name": "08_analysis_lock",
            "command": py("prepare_final_analysis_lock.py"),
            "outputs": (FINAL / "FINAL_ANALYSIS_LOCK.json", FINAL / "FINAL_ANALYSIS_LOCK.md"),
        },
        {
            "name": "09_experiments_A_B_C",
            "command": py("run_final_experiments.py", "--experiment", "all"),
            "outputs": (FINAL / "RUN_LEDGER.csv", FINAL / "validation_results.json"),
        },
        {
            "name": "10_locked_test_evaluation",
            "command": py("evaluate_final_experiments.py", "--split", "test"),
            "outputs": (FINAL / "results.json",),
        },
        {
            "name": "11_validation_mlm_diagnostic",
            "command": py("evaluate_final_mlm.py"),
            "outputs": (FINAL / "MLM_DIAGNOSTIC_RESULTS.json",),
        },
        {
            "name": "12_final_statistics",
            "command": py("analyse_final_experiments.py"),
            "outputs": (
                FINAL / "FINAL_STATISTICS.json",
                FINAL / "QA_DIFFICULTY_DIAGNOSTICS.json",
            ),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the complete serial queue without changing files",
    )
    parser.add_argument(
        "--force-benchmark", action="store_true",
        help="repeat the CPU benchmark even if a valid report already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = stages(force_benchmark=args.force_benchmark)
    print("Final DAPT study: strictly serial, stop-on-failure, resumable queue")
    print(f"Workspace: {ROOT}")
    if args.dry_run:
        for index, stage in enumerate(queue, 1):
            print(f"{index:02d}. {stage['name']}: {render(stage['command'])}")
        print("\nDry run only: no command was executed and no state was changed.")
        return 0

    state = load_state()
    try:
        for stage in queue:
            run_stage(
                state,
                stage["name"],
                stage["command"],
                stage.get("outputs", ()),
                stage.get("accept_existing", False),
                stage.get("force", False),
            )
    except Exception as exc:
        print(f"\nQUEUE STOPPED: {exc}", file=sys.stderr, flush=True)
        print("Fix the reported issue and rerun the same command; completed stages are retained.")
        return 1

    state["status"] = "DONE"
    state["completed_at"] = now()
    state["updated_at"] = now()
    atomic_json(STATE_PATH, state)
    print("\nFinal DAPT queue complete: models, predictions, metrics and statistics are saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
