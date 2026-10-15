"""Canonical gold-context evaluation for all IndicIPR-QA reader models.

Long contexts are evaluated with overlapping windows. Every model uses the
same exhaustive constrained span decoder from :mod:`scripts.qa_evaluation`.
Predictions are literal slices of the supplied statutory context.
"""

from __future__ import annotations

import argparse
import collections
import datetime
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

from scripts.qa_evaluation import (  # noqa: E402
    QADecodingConfig,
    compute_char_f1,
    compute_em_f1,
    predict_answers,
    resolve_device,
)


TEST_FILE = BASE_DIR / "data" / "splits" / "test.jsonl"
REPORT_PATH = BASE_DIR / "results" / "evaluation" / "qa_eval_report.json"

MODELS = [
    {
        "code": "A",
        "label": "A : InLegalBERT → IPR direct",
        "path": BASE_DIR / "artifacts" / "models" / "direct" / "baseline_a" / "final",
    },
    {
        "code": "D",
        "label": "D : BERT-base → IPR direct",
        "path": BASE_DIR / "artifacts" / "models" / "direct" / "baseline_d" / "final",
    },
    {
        "code": "E",
        "label": "E : LegalBERT(US) → IPR direct",
        "path": BASE_DIR / "artifacts" / "models" / "direct" / "baseline_e" / "final",
    },
    {
        "code": "G",
        "label": "G : InCaseLawBERT → IPR direct",
        "path": BASE_DIR / "artifacts" / "models" / "direct" / "baseline_g" / "final",
    },
    {
        "code": "B'",
        "label": "B' : InLegalBERT → SQuAD v1 → IPR",
        "path": BASE_DIR / "artifacts" / "models" / "squad_v1" / "baseline_b_squadv1" / "final",
    },
    {
        "code": "D2'",
        "label": "D2': BERT-base → SQuAD v1 → IPR",
        "path": BASE_DIR / "artifacts" / "models" / "squad_v1" / "baseline_d2_squadv1" / "final",
    },
    {
        "code": "E2'",
        "label": "E2': LegalBERT(US) → SQuAD v1 → IPR",
        "path": BASE_DIR / "artifacts" / "models" / "squad_v1" / "baseline_e2_squadv1" / "final",
    },
    {
        "code": "F'",
        "label": "F' : InCaseLawBERT → SQuAD v1 → IPR",
        "path": BASE_DIR / "artifacts" / "models" / "squad_v1" / "baseline_f_squadv1" / "final",
    },
    {
        "code": "C'",
        "label": "C' : Proposed Model: pretrained → SQuAD v1 → IPR  ★",
        "path": BASE_DIR / "artifacts" / "models" / "squad_v1" / "squad_v1_modelc_v2" / "ipr_final" / "final",
    },
    {
        "code": "H'",
        "label": "H' : InLegalBERT → Original-only MLM → SQuAD v1 → IPR",
        "path": BASE_DIR / "artifacts" / "models" / "squad_v1" / "original_only_squadv1" / "ipr_final" / "final",
    },
    {
        "code": "I'",
        "label": "I' : InLegalBERT → Synth-only MLM → SQuAD v1 → IPR",
        "path": BASE_DIR / "artifacts" / "models" / "squad_v1" / "squad_v1_synth_only" / "ipr_final" / "final",
    },
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relative_path(path: Path) -> str:
    return path.relative_to(BASE_DIR).as_posix()


def audit_window_coverage(records: list[dict], tokenizer, config: QADecodingConfig) -> dict:
    """Count cases that a first-window-only evaluator would mishandle."""

    multi_window = 0
    gold_absent_from_first = 0
    for record in records:
        encoded = tokenizer(
            record["question"],
            record["context"],
            max_length=config.max_seq_length,
            truncation="only_second",
            stride=config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        if len(encoded["input_ids"]) <= 1:
            continue
        multi_window += 1
        sequence_ids = encoded.sequence_ids(0)
        context_offsets = [
            offset
            for offset, sequence_id in zip(encoded["offset_mapping"][0], sequence_ids)
            if sequence_id == 1 and offset[1] > offset[0]
        ]
        answer_start = int(record["answerspan"])
        answer_end = answer_start + len(record["answer"])
        if not context_offsets or not (
            context_offsets[0][0] <= answer_start
            and context_offsets[-1][1] >= answer_end
        ):
            gold_absent_from_first += 1
    return {
        "inputs_requiring_multiple_windows": multi_window,
        "gold_spans_absent_from_first_window": gold_absent_from_first,
        "denominator": len(records),
    }


def evaluate_model(model_info: dict, records: list[dict], config: QADecodingConfig, device):
    model_path = model_info["path"]
    if not model_path.exists():
        print(f"[SKIP] {model_info['label']} — missing checkpoint: {model_path}")
        return None

    print(f"\n{'=' * 72}\n{model_info['label']}\n{'=' * 72}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(str(model_path), local_files_only=True)
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

    act_scores = collections.defaultdict(lambda: {"em": [], "f1": [], "char_f1": []})
    predictions = []
    for record, (prediction, span_score) in zip(records, decoded):
        em, f1 = compute_em_f1(prediction, record["answer"])
        char_f1 = compute_char_f1(prediction, record["answer"])
        act = record.get("act", "unknown")
        act_scores[act]["em"].append(em)
        act_scores[act]["f1"].append(f1)
        act_scores[act]["char_f1"].append(char_f1)
        predictions.append(
            {
                "id": record["id"],
                "question": record["question"],
                "gold": record["answer"],
                "pred": prediction,
                "span_score": round(span_score, 6),
                "em": em,
                "f1": round(f1, 6),
                "char_f1": round(char_f1, 6),
                "act": act,
            }
        )

    count = len(predictions)
    average = lambda metric: sum(row[metric] for row in predictions) / count * 100
    per_act = {
        act: {
            metric: round(sum(values[metric]) / len(values[metric]) * 100, 4)
            for metric in ("em", "f1", "char_f1")
        }
        | {"count": len(values["em"])}
        for act, values in sorted(act_scores.items())
    }
    result = {
        "code": model_info["code"],
        "label": model_info["label"],
        "model_path": relative_path(model_path),
        "em": round(average("em"), 4),
        "f1": round(average("f1"), 4),
        "char_f1": round(average("char_f1"), 4),
        "per_act": per_act,
        "predictions": predictions,
    }
    print(
        f"EM {result['em']:.2f} | F1 {result['f1']:.2f} | "
        f"diagnostic Char-F1 {result['char_f1']:.2f}"
    )
    del model
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        help="optional model codes (for example: --models A C')",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    selected = set(args.models or [model["code"] for model in MODELS])
    unknown = selected - {model["code"] for model in MODELS}
    if unknown:
        raise ValueError(f"unknown model codes: {sorted(unknown)}")

    records = load_jsonl(TEST_FILE)
    device = resolve_device(args.device)
    config = QADecodingConfig(
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        batch_size=args.batch_size,
        max_answer_tokens=None,
    )
    print(f"Test records: {len(records):,} | device: {device} | protocol: {config.as_dict()}")

    results = []
    for model_info in MODELS:
        if model_info["code"] not in selected:
            continue
        result = evaluate_model(model_info, records, config, device)
        if result:
            results.append(result)

    if selected != {model["code"] for model in MODELS}:
        print("\nPartial model selection: report not overwritten.")
        return 0

    audit_model = next(model for model in MODELS if model["path"].exists())
    audit_tokenizer = AutoTokenizer.from_pretrained(
        str(audit_model["path"]), local_files_only=True, use_fast=True
    )
    window_audit = audit_window_coverage(records, audit_tokenizer, config)
    window_audit["tokenizer_model_code"] = audit_model["code"]

    report = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "test_file": relative_path(TEST_FILE),
        "test_records": len(records),
        "inference_protocol": config.as_dict(),
        "window_coverage_audit": window_audit,
        "results_summary": [
            {key: value for key, value in result.items() if key != "predictions"}
            for result in results
        ],
        "metrics": {
            "primary": ["em", "f1"],
            "diagnostic": ["char_f1"],
            "diagnostic_note": (
                "char_f1 is order-insensitive character-frequency overlap retained "
                "for backward-compatible auditing; it is not a manuscript metric"
            ),
        },
        "per_act_summary": {result["label"]: result["per_act"] for result in results},
        "predictions": {result["label"]: result["predictions"] for result in results},
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\nCanonical report saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
