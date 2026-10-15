"""
Merge new large-scale synthetic contexts into pretraining corpus (v2).

Input:
  - data/dapt/pretraining_contexts.jsonl (original: 3,710 + 312 = 4,022 contexts)
  - data/synthetic/synthetic_contexts_large_merged.jsonl (gemma3:12b + gemma3:27b merged, ~13,000 contexts)

Output:
  - data/dapt/pretraining_contexts_v2.jsonl (combined: ~17,000+ contexts)

After running this:
  → Update pretrain_mlm.py: change PRETRAINING_FILE to "pretraining_contexts_v2.jsonl"
  → Re-run pretrain_mlm.py  (Model C only pipeline)
  → Re-run finetune_squad_v1_modelc_v2.py
  → Re-run finetune_ipr.py
  (Baselines A/B/D/D2/E/E2/F/G do not use MLM pretraining — no re-run needed)

Run: python merge_and_prepare_pretraining_v2.py
"""

import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[2]

ORIGINAL_FILE = BASE_DIR / "data" / "dapt" / "pretraining_contexts.jsonl"
NEW_SYNTH_FILE = BASE_DIR / "data" / "synthetic" / "synthetic_contexts_large_merged.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "dapt" / "pretraining_contexts_v2.jsonl"


def load_contexts(path, context_key="context"):
    contexts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get(context_key, "").strip()
            if text:
                contexts.append(text)
    return contexts


def main():
    print("Loading original pretraining contexts ...")
    original_contexts = load_contexts(ORIGINAL_FILE, context_key="context")
    print(f"  Original : {len(original_contexts):,}")

    print("Loading new large-scale synthetic contexts ...")
    synth_contexts = load_contexts(NEW_SYNTH_FILE, context_key="synthetic_context")
    print(f"  New synthetic : {len(synth_contexts):,}")

    # Dedup by exact text (rare but possible if same entity+style generated same output)
    all_texts   = original_contexts + synth_contexts
    seen        = set()
    deduped     = []
    for text in all_texts:
        key = text.strip().lower()[:200]   # first 200 chars as fingerprint
        if key not in seen:
            seen.add(key)
            deduped.append(text)

    duplicates_removed = len(all_texts) - len(deduped)
    print(f"  Duplicates removed : {duplicates_removed:,}")
    print(f"  Total (after dedup): {len(deduped):,}")

    # Act breakdown from synthetic file (for report)
    act_counts = Counter()
    with open(NEW_SYNTH_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                act_counts[row.get("primary_act", "unknown")] += 1
            except Exception:
                pass

    print("\nSynthetic contexts per act:")
    for act, cnt in act_counts.most_common():
        print(f"  {act:<48} : {cnt:>5}")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for text in deduped:
            f.write(json.dumps({"context": text}, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(deduped):,} contexts to: {OUTPUT_FILE}")
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Open pretrain_mlm.py")
    print('     Change: PRETRAINING_FILE = ... / "pretraining_contexts.jsonl"')
    print('     To:     PRETRAINING_FILE = ... / "pretraining_contexts_v2.jsonl"')
    print("  2. Run: python pretrain_mlm.py")
    print("  3. Run: python finetune_squad_v1_modelc_v2.py")
    print("  4. Run: python finetune_ipr.py")
    print("  5. Run: python evaluate_qa.py  (Model C will now use the improved pretrained model)")


if __name__ == "__main__":
    main()
