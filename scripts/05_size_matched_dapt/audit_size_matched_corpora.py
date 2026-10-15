"""Audit record, token, and split exposure in the legacy 901-record DAPT corpora.

The paper calls these corpora record-matched, not token-matched.  This script
creates the canonical machine-readable artifact used for that statement.

Run from the repository root:
    python scripts/05_size_matched_dapt/audit_size_matched_corpora.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DAPT = ROOT / "data" / "dapt"
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "results" / "experiments" / "size_matched_dapt" / "size_matched_corpus_audit.json"
TOKENIZER = "law-ai/InLegalBERT"
MAX_LENGTH = 256

CORPORA = {
    "auth901": "pretraining_contexts_auth901.jsonl",
    "synth901": "pretraining_contexts_synth901.jsonl",
    "mixed901": "pretraining_contexts_mixed901.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    test = load_jsonl(SPLITS / "test.jsonl")
    test_contexts = {norm(row["context"]) for row in test}

    report: dict[str, object] = {
        "description": "Legacy record-matched DAPT corpus audit",
        "tokenizer": TOKENIZER,
        "max_sequence_length": MAX_LENGTH,
        "token_definition": "non-padding token ids after special tokens and truncation",
        "corpora": {},
    }

    for label, filename in CORPORA.items():
        rows = load_jsonl(DAPT / filename)
        texts = [row["context"] for row in rows]
        unique = {norm(text) for text in texts}
        overlap = unique & test_contexts
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )["input_ids"]
        report["corpora"][label] = {
            "records": len(rows),
            "normalized_unique_contexts": len(unique),
            "non_padding_tokens": sum(len(ids) for ids in encoded),
            "test_context_overlap": len(overlap),
            "affected_test_questions": sum(
                1 for row in test if norm(row["context"]) in overlap
            ),
        }

    auth_tokens = report["corpora"]["auth901"]["non_padding_tokens"]
    synth_tokens = report["corpora"]["synth901"]["non_padding_tokens"]
    mixed_tokens = report["corpora"]["mixed901"]["non_padding_tokens"]
    report["relative_to_authentic_percent"] = {
        "synth901": round(100 * (synth_tokens / auth_tokens - 1), 4),
        "mixed901": round(100 * (mixed_tokens / auth_tokens - 1), 4),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
