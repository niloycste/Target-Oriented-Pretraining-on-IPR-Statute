"""Evaluate IndicIPR-QA readers on train, validation, and test splits.

This diagnostic uses exactly the same sliding-window constrained decoder as
``evaluate_qa.py``. It is intentionally separate from the headline test report
because evaluating the training split is substantially more expensive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_qa import MODELS, load_jsonl, relative_path  # noqa: E402
from scripts.qa_evaluation import (  # noqa: E402
    QADecodingConfig,
    compute_char_f1,
    compute_em_f1,
    predict_answers,
    resolve_device,
)


SPLIT_FILES = {
    "train": BASE_DIR / "data" / "splits" / "train.jsonl",
    "val": BASE_DIR / "data" / "splits" / "val.jsonl",
    "test": BASE_DIR / "data" / "splits" / "test.jsonl",
}
REPORT_PATH = BASE_DIR / "results" / "evaluation" / "qa_split_eval_report.json"


def evaluate_records(tokenizer, model, device, records: list[dict], config: QADecodingConfig):
    decoded = predict_answers(
        tokenizer,
        model,
        device,
        [record["question"] for record in records],
        [record["context"] for record in records],
        config,
    )
    em_scores, f1_scores, char_f1_scores = [], [], []
    for record, (prediction, _) in zip(records, decoded):
        em, f1 = compute_em_f1(prediction, record["answer"])
        em_scores.append(em)
        f1_scores.append(f1)
        char_f1_scores.append(compute_char_f1(prediction, record["answer"]))
    count = len(records)
    return {
        "em": round(sum(em_scores) / count * 100, 4),
        "f1": round(sum(f1_scores) / count * 100, 4),
        "char_f1": round(sum(char_f1_scores) / count * 100, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", help="optional model codes")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.models or [model["code"] for model in MODELS])
    unknown = selected - {model["code"] for model in MODELS}
    if unknown:
        raise ValueError(f"unknown model codes: {sorted(unknown)}")
    config = QADecodingConfig(
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        batch_size=args.batch_size,
        max_answer_tokens=None,
    )
    device = resolve_device(args.device)
    split_data = {name: load_jsonl(path) for name, path in SPLIT_FILES.items()}
    print(
        f"device: {device} | split sizes: "
        + ", ".join(f"{name}={len(rows):,}" for name, rows in split_data.items())
    )

    all_results = []
    for model_info in MODELS:
        if model_info["code"] not in selected:
            continue
        path = model_info["path"]
        if not path.exists():
            print(f"[SKIP] {model_info['label']} — missing checkpoint")
            continue
        print(f"\n{'=' * 72}\n{model_info['label']}\n{'=' * 72}")
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        model = AutoModelForQuestionAnswering.from_pretrained(str(path), local_files_only=True)
        model.to(device)
        model.eval()

        split_results = {}
        for split_name, records in split_data.items():
            metrics = evaluate_records(tokenizer, model, device, records, config)
            split_results[split_name] = metrics
            print(
                f"{split_name:<5} EM {metrics['em']:6.2f} | "
                f"F1 {metrics['f1']:6.2f} | diagnostic Char-F1 {metrics['char_f1']:6.2f}"
            )
        train_f1 = split_results["train"]["f1"]
        val_f1 = split_results["val"]["f1"]
        test_f1 = split_results["test"]["f1"]
        all_results.append(
            {
                "code": model_info["code"],
                "label": model_info["label"],
                "model_path": relative_path(path),
                "splits": split_results,
                "gaps": {
                    "train_minus_val_f1": round(train_f1 - val_f1, 4),
                    "train_minus_test_f1": round(train_f1 - test_f1, 4),
                    "val_minus_test_f1": round(val_f1 - test_f1, 4),
                },
            }
        )
        del model

    report = {
        "inference_protocol": config.as_dict(),
        "split_files": {name: relative_path(path) for name, path in SPLIT_FILES.items()},
        "results": all_results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved canonical split report to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
