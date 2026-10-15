"""
Build pretraining_contexts_original_only.jsonl
Extracts only the 3,710 original IPR contexts from pretraining_contexts.jsonl
(filters out the 312 synthetic entries).

Input  : data/dapt/pretraining_contexts.jsonl     (3,710 original + 312 synthetic)
Output : data/dapt/pretraining_contexts_original_only.jsonl  (3,710 original only)

Run: python build_original_only_contexts.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "dapt" / "pretraining_contexts.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "dapt" / "pretraining_contexts_original_only.jsonl"


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
            if record.get("source", "") == "original":
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                skipped += 1

    print(f"Done.")
    print(f"  Kept (original)  : {kept:,}")
    print(f"  Skipped (synthetic): {skipped:,}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"\nNext step: run pretrain_mlm_original_only.py")


if __name__ == "__main__":
    main()
