"""
Merge two accepted synthetic context files (gemma3:12b + gemma3:27b),
remove duplicates based on synthetic_context text (normalized),
and save the merged result.

Input:
  data/synthetic/synthetic_contexts_large.jsonl (gemma3:12b)
  data/synthetic/synthetic_contexts_large_using_gemma27.jsonl (gemma3:27b)

Output:
  data/synthetic/synthetic_contexts_large_merged.jsonl
"""

import json
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DIR = BASE_DIR / "data" / "synthetic"

FILE_12B  = DIR / "synthetic_contexts_large.jsonl"
FILE_27B  = DIR / "synthetic_contexts_large_using_gemma27.jsonl"
OUT_FILE  = DIR / "synthetic_contexts_large_merged.jsonl"


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace for dedup comparison."""
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    return hashlib.md5(normalize(text).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [SKIP] Missing optional file: {path}")
        return []

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Load ───────────────────────────────────────────────────────────────────────
print(f"Loading {FILE_12B.name} ...")
records_12b = load_jsonl(FILE_12B)
print(f"  {len(records_12b):,} records")

print(f"Loading {FILE_27B.name} ...")
records_27b = load_jsonl(FILE_27B)
print(f"  {len(records_27b):,} records")

# ── Merge with dedup ───────────────────────────────────────────────────────────
seen_hashes  = set()
seen_ids     = set()
merged       = []
dup_text     = 0
dup_id       = 0

def add_records(records, source_label):
    global dup_text, dup_id
    for rec in records:
        rid  = rec.get("synthetic_id", "")
        ctx  = rec.get("synthetic_context", "")
        h    = text_hash(ctx)

        if h in seen_hashes:
            dup_text += 1
            continue
        if rid and rid in seen_ids:
            dup_id += 1
            # still add it — same ID but different content (keep content, update id)
            rec = dict(rec)
            rec["synthetic_id"] = rid + "__dedup_id"

        seen_hashes.add(h)
        if rid:
            seen_ids.add(rec["synthetic_id"])
        merged.append(rec)

# Add 12b first (larger, treated as primary), then 27b
add_records(records_12b, "gemma3:12b")
add_records(records_27b, "gemma3:27b")

# ── Save ───────────────────────────────────────────────────────────────────────
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for rec in merged:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ── Report ─────────────────────────────────────────────────────────────────────
print("\n=== Merge Report ===")
print(f"  gemma3:12b records   : {len(records_12b):,}")
print(f"  gemma3:27b records   : {len(records_27b):,}")
print(f"  Total input          : {len(records_12b) + len(records_27b):,}")
print(f"  Duplicate text (skip): {dup_text:,}")
print(f"  Duplicate ID (rename): {dup_id:,}")
print(f"  Final merged count   : {len(merged):,}")
print(f"\n  Output → {OUT_FILE}")
