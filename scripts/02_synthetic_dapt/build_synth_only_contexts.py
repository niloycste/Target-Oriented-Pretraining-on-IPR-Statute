"""
Build pretraining_contexts_synth_only.jsonl
Extracts the 13,211 synthetic contexts from synthetic_contexts_large_merged.jsonl
and writes them in the standard {"context": "..."} format used by all MLM pretraining scripts.

Input  : data/synthetic/synthetic_contexts_large_merged.jsonl
         (13,211 records, field: synthetic_context)
Output : data/dapt/pretraining_contexts_synth_only.jsonl
         (13,211 records, field: context — same format as pretraining_contexts_v2.jsonl)

Run: python build_synth_only_contexts.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "synthetic" / "synthetic_contexts_large_merged.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "dapt" / "pretraining_contexts_synth_only.jsonl"


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    kept = 0
    skipped = 0

    with open(INPUT_FILE, encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            context = record.get("synthetic_context", "").strip()
            if context:
                fout.write(json.dumps({"context": context}, ensure_ascii=False) + "\n")
                kept += 1
            else:
                skipped += 1

    print(f"Done.")
    print(f"  Written (synthetic) : {kept:,}")
    print(f"  Skipped (empty)     : {skipped:,}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"\nNext step: run pretrain_mlm_synth_only.py")


if __name__ == "__main__":
    main()
