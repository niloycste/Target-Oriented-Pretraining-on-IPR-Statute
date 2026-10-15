"""Prepare or execute Experiment D: equal-update full-data DAPT sensitivity.

Default invocation is safe and performs a dry run only::

    python scripts/dapt_equal_update/run_experiment_d.py

Costly training starts only with ``--execute``.  Experiment D trains the
authentic (A), synthetic (S), and mixed (M) DAPT routes for exactly 100 MLM
optimizer updates with 10 warmup updates.  The five no-DAPT (N) routes are
reused from Experiment A because they contain no MLM stage and all downstream
settings are unchanged.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "dapt" / "equal_update"
WORK = ROOT / "artifacts" / "models" / "equal_update"
FROZEN = ROOT / "experiments" / "dapt" / "final_cpu"
FROZEN_WORK = ROOT / "artifacts" / "models" / "final_cpu"
GENERATED = EXP / "generated_scripts"
PROTOCOL = EXP / "EXPERIMENT_D_PROTOCOL.json"
DRY_REPORT = EXP / "DRY_RUN_REPORT.json"
LEDGER = EXP / "RUN_LEDGER.csv"
SEEDS = [7, 42, 99, 123, 2026]
CONDITIONS = ["A", "S", "M"]
CORPORA = {condition: FROZEN / f"corpus_{condition}.jsonl" for condition in CONDITIONS}
MAX_STEPS = 100
WARMUP_STEPS = 10

base = importlib.import_module("scripts.09_dapt_final.run_final_experiments")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf8")
    os.replace(temporary, path)


def patch_once(text: str, pattern: str, replacement: str, label: str) -> str:
    patched, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return patched


def make_mlm_source(corpus: Path, output: Path, seed: int) -> str:
    source = base.make_mlm_source(corpus, output, seed)
    source = base.set_int_const(source, "WARMUP_STEPS", WARMUP_STEPS)
    source = base.set_int_const(source, "LOGGING_STEPS", 10)
    source = patch_once(
        source,
        r"^(NUM_EPOCHS\s*=\s*\d+.*)$",
        rf"\1\nMAX_STEPS       = {MAX_STEPS}",
        "insert MAX_STEPS",
    )
    source = patch_once(
        source,
        r"^(\s*num_train_epochs=NUM_EPOCHS,)$",
        r"\1\n    max_steps=MAX_STEPS,",
        "TrainingArguments max_steps",
    )
    source = patch_once(
        source,
        r"^(BASE_DIR\s*=.*)$",
        r"\1\n\nimport sys\nsys.path.insert(0, str(BASE_DIR))\n"
        r"from scripts.dapt_equal_update.telemetry import (\n"
        r"    TokenCountingTrainer, TrainingTelemetry,\n)",
        "telemetry import",
    )
    source = patch_once(
        source,
        r"^trainer = Trainer\($",
        "telemetry = TrainingTelemetry(\n"
        "    OUTPUT_DIR / \"training_telemetry.json\",\n"
        "    configured={\n"
        "        \"optimizer_updates\": MAX_STEPS,\n"
        "        \"warmup_steps\": WARMUP_STEPS,\n"
        "        \"peak_learning_rate\": LEARNING_RATE,\n"
        "        \"scheduler\": \"linear\",\n"
        "        \"effective_batch_size\": BATCH_SIZE * GRAD_ACCUM_STEPS,\n"
        "        \"max_seq_length\": MAX_SEQ_LENGTH,\n"
        "        \"seed\": SEED,\n"
        "        \"corpus_sha256\": hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest(),\n"
        "    },\n"
        ")\n"
        "trainer = TokenCountingTrainer(\n"
        "    telemetry=telemetry,",
        "TokenCountingTrainer",
    )
    source = patch_once(
        source,
        r"^\s*callbacks=\[RecoveryCheckpointCallback\(\)\],$",
        "    callbacks=[RecoveryCheckpointCallback(), telemetry],",
        "telemetry callback",
    )
    source = source.replace(
        'print(f"  Epochs               : {NUM_EPOCHS}")',
        'print(f"  Optimizer updates    : {MAX_STEPS}")\n'
        'print(f"  Warmup updates       : {WARMUP_STEPS}")',
    )
    return source


def make_squad_source(model: Path, output: Path, seed: int) -> str:
    return base.make_squad_source(model, output, seed)


def make_ipr_source(model: Path, output: Path, seed: int) -> str:
    return base.make_ipr_source(model, output, seed)


def generated_sources(condition: str, seed: int) -> dict[str, tuple[str, Path, Path]]:
    tag = f"D_{condition}_s{seed}"
    mlm_output = WORK / f"{tag}_mlm"
    squad_output = WORK / f"{tag}_squad"
    ipr_output = WORK / f"{tag}_ipr"
    return {
        "mlm": (make_mlm_source(CORPORA[condition], mlm_output, seed), mlm_output, mlm_output / "final"),
        "squad": (make_squad_source(mlm_output / "final", squad_output, seed), squad_output, squad_output / "final"),
        "ipr": (make_ipr_source(squad_output / "final", ipr_output, seed), ipr_output, ipr_output / "final"),
    }


def materialise_scripts() -> dict[str, dict[str, Path]]:
    GENERATED.mkdir(parents=True, exist_ok=True)
    paths: dict[str, dict[str, Path]] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            tag = f"D_{condition}_s{seed}"
            paths[tag] = {}
            for stage, (source, _output, _final) in generated_sources(condition, seed).items():
                path = GENERATED / f"{tag}_{stage}.py"
                path.write_text(source, encoding="utf8")
                paths[tag][stage] = path
    return paths


def literal_constants(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def prepare_protocol() -> dict:
    required = [*CORPORA.values(), ROOT / "data/splits/train_squad.jsonl",
                ROOT / "data/splits/val_squad.jsonl", ROOT / "data/splits/val.jsonl",
                ROOT / "data/splits/test.jsonl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required inputs missing: {missing}")
    from datasets import config as datasets_config
    squad_cache = Path(datasets_config.HF_DATASETS_CACHE) / "rajpurkar___squad" / "plain_text"
    squad_train = sorted(
        squad_cache.glob("**/squad-train.arrow"),
        key=lambda path: path.stat().st_mtime, reverse=True,
    )
    if not squad_train or not squad_train[0].with_name("squad-validation.arrow").exists():
        raise FileNotFoundError("the frozen downstream SQuAD Arrow cache is unavailable")
    external = [squad_train[0], squad_train[0].with_name("squad-validation.arrow")]
    protocol = {
        "schema_version": 1,
        "status": "post-hoc sensitivity protocol; not preregistered",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "objective": "Test corpus-composition conclusions under an equal 100-update MLM schedule.",
        "conditions": ["N (reused from Experiment A)", "A", "S", "M"],
        "trained_conditions": CONDITIONS,
        "seeds": SEEDS,
        "mlm": {"optimizer_updates": MAX_STEPS, "warmup_steps": WARMUP_STEPS,
                "scheduler": "linear", "peak_learning_rate": 2e-5},
        "primary_analysis": "seed-blocked ANOVA over N/A/S/M full-data F1",
        "secondary_analyses": [
            "seed-paired A-N, S-N, and M-N differences with two-sided t intervals",
            "the same analyses for exact match",
            "Holm correction across the three DAPT-vs-N tests within each metric",
        ],
        "reporting_rule": "A non-significant result is non-detection, not evidence of equivalence.",
        "input_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in required},
        "external_input_sha256": {str(path): sha256(path) for path in external},
        "reused_no_dapt_runs": [f"A_N_s{seed}_100" for seed in SEEDS],
    }
    atomic_json(PROTOCOL, protocol)
    return protocol


def dry_run_audit(paths: dict[str, dict[str, Path]], protocol: dict) -> dict:
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected_mlm = {
        "BASE_MODEL": "law-ai/InLegalBERT", "MAX_SEQ_LENGTH": 256,
        "MLM_PROBABILITY": 0.15, "LEARNING_RATE": 2e-5,
        "WEIGHT_DECAY": 0.01, "MAX_STEPS": 100, "WARMUP_STEPS": 10,
    }
    expected_downstream_names = [
        "MAX_SEQ_LENGTH", "DOC_STRIDE", "TRAIN_SUBSET", "VAL_SUBSET",
        "BATCH_SIZE", "GRAD_ACCUM", "LEARNING_RATE", "WEIGHT_DECAY",
        "NUM_EPOCHS", "WARMUP_STEPS", "SUBSET_SEED", "DATALOADER_WORKERS",
    ]

    def normalise_downstream(text: str) -> str:
        return re.sub(
            r"^(MODEL_DIR|OUTPUT_DIR)\s*=.*$", r"\1 = <PIPELINE_PATH>",
            text, flags=re.M,
        )
    for condition in CONDITIONS:
        for seed in SEEDS:
            tag = f"D_{condition}_s{seed}"
            mlm_path = paths[tag]["mlm"]
            mlm = literal_constants(mlm_path)
            check(f"{tag}: MLM syntax", True, "AST parse passed")
            for name, expected in expected_mlm.items():
                check(f"{tag}: {name}", mlm.get(name) == expected,
                      f"observed={mlm.get(name)!r}; expected={expected!r}")
            check(f"{tag}: seed", mlm.get("SEED") == seed, f"observed={mlm.get('SEED')}")
            text = mlm_path.read_text(encoding="utf8")
            check(f"{tag}: max_steps wired", "max_steps=MAX_STEPS" in text,
                  "TrainingArguments uses MAX_STEPS")
            check(f"{tag}: telemetry wired",
                  "TokenCountingTrainer" in text and "TrainingTelemetry" in text,
                  "loss/LR and exact non-padding-token telemetry enabled")
            for stage in ("squad", "ipr"):
                current = literal_constants(paths[tag][stage])
                frozen_path = FROZEN_WORK / f"A_{condition}_s{seed}_{stage}.py"
                frozen = literal_constants(frozen_path)
                check(
                    f"{tag}: {stage} exact frozen logic",
                    normalise_downstream(paths[tag][stage].read_text(encoding="utf8"))
                    == normalise_downstream(frozen_path.read_text(encoding="utf8")),
                    "only MODEL_DIR and OUTPUT_DIR may differ",
                )
                for name in expected_downstream_names:
                    if name in frozen or name in current:
                        check(f"{tag}: {stage} {name} unchanged",
                              current.get(name) == frozen.get(name),
                              f"D={current.get(name)!r}; frozen={frozen.get(name)!r}")
                check(f"{tag}: {stage} seed", current.get("SEED") == seed,
                      f"observed={current.get('SEED')}")
    for condition, corpus in CORPORA.items():
        relative = corpus.relative_to(ROOT).as_posix()
        check(f"corpus {condition} hash frozen",
              sha256(corpus) == protocol["input_sha256"][relative], sha256(corpus))
    for raw_path, expected_hash in protocol["external_input_sha256"].items():
        path = Path(raw_path)
        check(f"external input frozen: {path.name}", path.exists() and sha256(path) == expected_hash,
              expected_hash)
    results_file = FROZEN / "results.json"
    existing = json.loads(results_file.read_text(encoding="utf8")).get("runs", {})
    for seed in SEEDS:
        key = f"A_N_s{seed}_100"
        check(f"reuse {key}", key in existing, "existing no-DAPT test result present")
    passed = all(row["passed"] for row in checks)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": passed,
        "summary": {"passed": sum(row["passed"] for row in checks), "total": len(checks)},
        "checks": checks,
        "generated_script_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for stages in paths.values() for path in stages.values()
        },
    }
    atomic_json(DRY_REPORT, report)
    if not passed:
        failures = [row for row in checks if not row["passed"]]
        raise RuntimeError(f"dry-run audit failed ({len(failures)} checks); see {DRY_REPORT}")
    return report


def configure_base_module() -> None:
    base.FINAL = EXP
    base.WORK = WORK
    base.LEDGER = LEDGER
    base.EVALUATOR = ROOT / "scripts" / "dapt_equal_update" / "evaluate_experiment_d.py"


def execute(paths: dict[str, dict[str, Path]], max_retries: int) -> None:
    configure_base_module()
    rows = base.read_ledger()
    base.reconcile_ledger(rows)
    for condition in CONDITIONS:
        for seed in SEEDS:
            tag = f"D_{condition}_s{seed}"
            print(f"\n== {tag} ==", flush=True)
            mlm_output = WORK / f"{tag}_mlm"
            mlm_final = mlm_output / "final"
            mlm_source = paths[tag]["mlm"].read_text(encoding="utf8")
            base.execute_stage(rows, "D", condition, seed, "mlm", mlm_output, mlm_final,
                               lambda source=mlm_source: source, False, max_retries)
            squad_output = WORK / f"{tag}_squad"
            squad_final = squad_output / "final"
            squad_source = paths[tag]["squad"].read_text(encoding="utf8")
            base.execute_stage(rows, "D", condition, seed, "squad", squad_output, squad_final,
                               lambda source=squad_source: source, False, max_retries)
            ipr_output = WORK / f"{tag}_ipr"
            ipr_final = ipr_output / "final"
            ipr_source = paths[tag]["ipr"].read_text(encoding="utf8")
            base.execute_stage(rows, "D", condition, seed, "ipr", ipr_output, ipr_final,
                               lambda source=ipr_source: source, False, max_retries)
            base.validation_evaluation(rows, "D", condition, seed, 100, False, max_retries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="start costly training; omitted means dry run only")
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    protocol = prepare_protocol()
    paths = materialise_scripts()
    report = dry_run_audit(paths, protocol)
    print(f"Experiment D dry run: PASS ({report['summary']['passed']}/{report['summary']['total']} checks)")
    print(f"Protocol: {PROTOCOL.relative_to(ROOT)}")
    print(f"Audit:    {DRY_REPORT.relative_to(ROOT)}")
    print(f"Scripts:  {GENERATED.relative_to(ROOT)}")
    if not args.execute:
        print("No training was started. Use --execute only after reviewing the protocol and audit.")
        return 0
    execute(paths, args.max_retries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
