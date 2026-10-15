"""Canonical validation/test evaluation for every final DAPT reader checkpoint.

Validation evaluation may run throughout training. Test evaluation is refused
unless ``FINAL_ANALYSIS_LOCK.json`` exists and every locked input still matches
its SHA-256 hash. Per-run predictions are written atomically and an aggregate
``validation_results.json`` or ``results.json`` is updated after every model.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "experiments" / "dapt" / "final_cpu"
WORK = ROOT / "artifacts" / "models" / "final_cpu"
LOCK = FINAL / "FINAL_ANALYSIS_LOCK.json"
SEEDS_A = [7, 42, 99, 123, 2026]
SEEDS_BC = [7, 42, 99]
CONDITIONS = ["N", "A", "S", "M"]
FRACTIONS = [25, 50]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qa_evaluation import (  # noqa: E402
    QADecodingConfig,
    compute_char_f1,
    compute_em_f1,
    predict_answers,
    resolve_device,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sha256(checkpoint: Path) -> str:
    """Hash model and tokenizer artifacts so cached metrics cannot go stale."""
    candidates = []
    for pattern in (
        "config.json", "*.safetensors", "pytorch_model*.bin", "*.index.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
    ):
        candidates.extend(checkpoint.glob(pattern))
    files = sorted({path for path in candidates if path.is_file()}, key=lambda path: path.name)
    if not files or not (checkpoint / "config.json").exists():
        raise RuntimeError(f"incomplete checkpoint: {checkpoint}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")
    os.replace(temp, path)


def verify_analysis_lock() -> dict:
    if not LOCK.exists():
        raise RuntimeError(
            "test evaluation is locked: run prepare_final_analysis_lock.py first"
        )
    lock = json.loads(LOCK.read_text(encoding="utf8"))
    if not lock.get("locked"):
        raise RuntimeError("FINAL_ANALYSIS_LOCK.json is not marked locked")
    mismatches = []
    for relative, expected in lock.get("sha256", {}).items():
        path = ROOT / relative
        actual = sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            mismatches.append(f"{relative}: expected {expected}, found {actual}")
    if mismatches:
        from importlib import import_module
        amendment = import_module("scripts.09_dapt_final.analysis_amendment")
        reviewed = amendment.verify_amendment(lock)
        lock = {**lock, "reviewed_checkpoints": reviewed["checkpoints"]}
    return lock


def target_table() -> dict[str, dict]:
    targets = {}
    for condition in CONDITIONS:
        for seed in SEEDS_A:
            key = f"A_{condition}_s{seed}_100"
            targets[key] = {
                "experiment": "A", "condition": condition, "seed": seed,
                "fraction": 100,
                "checkpoint": WORK / f"A_{condition}_s{seed}_ipr" / "final",
            }
    for seed in SEEDS_BC:
        key = f"B_S_STYLE_s{seed}_100"
        targets[key] = {
            "experiment": "B", "condition": "S_STYLE", "seed": seed,
            "fraction": 100,
            "checkpoint": WORK / f"B_S_STYLE_s{seed}_ipr" / "final",
        }
    for condition in CONDITIONS:
        for seed in SEEDS_BC:
            for fraction in FRACTIONS:
                key = f"C_{condition}_s{seed}_{fraction}"
                targets[key] = {
                    "experiment": "C", "condition": condition, "seed": seed,
                    "fraction": fraction,
                    "checkpoint": WORK / f"C_{condition}{fraction}_s{seed}_ipr" / "final",
                }
    return targets


def load_records(split: str) -> list[dict]:
    path = ROOT / "data" / "splits" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]


def deterministic_tags(record: dict) -> dict:
    answer = record["answer"]
    words = re.findall(r"\w+", answer)
    if len(words) <= 10:
        length_bin = "short_1_10"
    elif len(words) <= 30:
        length_bin = "medium_11_30"
    else:
        length_bin = "long_31_plus"
    return {
        "gold_answer_words": len(words),
        "answer_length_bin": length_bin,
        "semicolon": ";" in answer,
        "enumerated_clause": bool(re.search(r"(?:^|\s)\([a-z0-9ivx]+\)", answer, re.I)),
        "proviso_or_exception": bool(
            re.search(r"\b(?:provided that|proviso|subject to|except(?:ion)?)\b", answer, re.I)
        ),
    }


def evaluate_one(key: str, target: dict, records: list[dict], split: str,
                 device: torch.device, config: QADecodingConfig,
                 checkpoint_hash: str) -> dict:
    checkpoint = target["checkpoint"]
    print(f"\n  [{split}] {key}: {checkpoint.relative_to(ROOT)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    model.to(device)
    model.eval()
    decoded = predict_answers(
        tokenizer,
        model,
        device,
        [record["question"] for record in records],
        [record["context"] for record in records],
        config,
    )

    predictions = []
    per_act = collections.defaultdict(lambda: {"em": [], "f1": []})
    for record, (prediction, span_score) in zip(records, decoded):
        em, f1 = compute_em_f1(prediction, record["answer"])
        char_f1 = compute_char_f1(prediction, record["answer"])
        act = record.get("act", "unknown")
        per_act[act]["em"].append(em)
        per_act[act]["f1"].append(f1)
        predictions.append({
            "id": str(record["id"]),
            "act": act,
            "question": record["question"],
            "gold": record["answer"],
            "prediction": prediction,
            "span_score": round(float(span_score), 6),
            "em": em,
            "f1": round(f1, 8),
            "char_f1": round(char_f1, 8),
            **deterministic_tags(record),
        })
    count = len(predictions)
    result = {
        "schema_version": 2,
        "run_key": key,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "split": split,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        **{name: target[name] for name in ("experiment", "condition", "seed", "fraction")},
        "n": count,
        "em": round(100 * sum(row["em"] for row in predictions) / count, 6),
        "f1": round(100 * sum(row["f1"] for row in predictions) / count, 6),
        "char_f1_diagnostic": round(
            100 * sum(row["char_f1"] for row in predictions) / count, 6
        ),
        "per_act": {
            act: {
                "n": len(values["em"]),
                "em": round(100 * sum(values["em"]) / len(values["em"]), 6),
                "f1": round(100 * sum(values["f1"]) / len(values["f1"]), 6),
            }
            for act, values in sorted(per_act.items())
        },
        "inference_protocol": config.as_dict(),
        "predictions": predictions,
    }
    del model
    gc.collect()
    return result


def update_aggregate(split: str, per_run: dict[str, dict], config: QADecodingConfig) -> Path:
    aggregate_path = FINAL / ("results.json" if split == "test" else "validation_results.json")
    aggregate = {
        "schema_version": 2,
        "split": split,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inference_protocol": config.as_dict(),
        "runs": {},
    }
    if aggregate_path.exists():
        prior = json.loads(aggregate_path.read_text(encoding="utf8"))
        if prior.get("schema_version") == 2 and prior.get("split") == split:
            aggregate["runs"].update(prior.get("runs", {}))
    for key, result in per_run.items():
        prediction_file = FINAL / "evaluations" / split / f"{key}.json"
        aggregate["runs"][key] = {
            name: value for name, value in result.items() if name != "predictions"
        }
        aggregate["runs"][key]["predictions_file"] = prediction_file.relative_to(ROOT).as_posix()
    atomic_json(aggregate_path, aggregate)
    return aggregate_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--run-key", action="append", help="evaluate only this exact run key")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--list-targets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = target_table()
    if args.list_targets:
        print("\n".join(targets))
        return 0
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    test_lock = verify_analysis_lock() if args.split == "test" else None

    selected = args.run_key or list(targets)
    unknown = sorted(set(selected) - set(targets))
    if unknown:
        raise ValueError(f"unknown run keys: {unknown}")
    missing = [key for key in selected if not targets[key]["checkpoint"].exists()]
    if missing and not args.allow_partial:
        print("  missing checkpoints; evaluation refused:")
        for key in missing:
            print(f"    {key}: {targets[key]['checkpoint'].relative_to(ROOT)}")
        return 1
    selected = [key for key in selected if key not in missing]

    records = load_records(args.split)
    device = resolve_device(args.device)
    config = QADecodingConfig(
        max_seq_length=384, doc_stride=128, batch_size=args.batch_size,
        max_answer_tokens=None,
    )
    output_dir = FINAL / "evaluations" / args.split
    results = {}
    for key in selected:
        output = output_dir / f"{key}.json"
        current_hash = checkpoint_sha256(targets[key]["checkpoint"])
        if test_lock and "reviewed_checkpoints" in test_lock:
            frozen = test_lock["reviewed_checkpoints"].get(key)
            relative = targets[key]["checkpoint"].relative_to(ROOT).as_posix()
            if not frozen or frozen["path"] != relative or frozen["sha256"] != current_hash:
                raise RuntimeError(f"Checkpoint differs from pre-test amendment: {key}")
        result = None
        if output.exists() and not args.force:
            try:
                cached = json.loads(output.read_text(encoding="utf8"))
                valid_cache = (
                    cached.get("schema_version") == 2
                    and cached.get("run_key") == key
                    and cached.get("split") == args.split
                    and cached.get("checkpoint_sha256") == current_hash
                    and cached.get("inference_protocol") == config.as_dict()
                )
                if valid_cache:
                    result = cached
                    print(f"  [{args.split}] {key}: verified existing result reused")
            except (OSError, json.JSONDecodeError):
                result = None
        if result is None:
            result = evaluate_one(
                key, targets[key], records, args.split, device, config, current_hash
            )
            atomic_json(output, result)
        results[key] = result
        aggregate = update_aggregate(args.split, {key: result}, config)
        print(f"       EM={result['em']:.3f} F1={result['f1']:.3f}")
        print(f"       aggregate: {aggregate.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
