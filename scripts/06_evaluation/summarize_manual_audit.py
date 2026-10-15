"""
Summarize completed manual_audit_sample.csv

Expected labels:
  manual_faithful   -> yes / no
  manual_extractive -> yes / no
  manual_grammar    -> good / ok / bad
"""

import csv
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
AUDIT_CSV = BASE_DIR / "results" / "evaluation" / "manual_audit_sample.csv"

VALID_FAITHFUL = {"yes", "no"}
VALID_EXTRACTIVE = {"yes", "no"}
VALID_GRAMMAR = {"good", "ok", "bad"}


def normalize(value):
    return (value or "").strip().lower()


def pct(count, total):
    return 0.0 if total == 0 else count * 100.0 / total


def main():
    if not AUDIT_CSV.exists():
        print(f"manual_audit_sample.csv not found: {AUDIT_CSV}")
        return

    with open(AUDIT_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total_rows = len(rows)
    completed_rows = []
    incomplete_ids = []
    invalid_rows = []

    faithful_counts = Counter()
    extractive_counts = Counter()
    grammar_counts = Counter()

    for row in rows:
        rid = row.get("id", "")
        faithful = normalize(row.get("manual_faithful", ""))
        extractive = normalize(row.get("manual_extractive", ""))
        grammar = normalize(row.get("manual_grammar", ""))

        if not faithful and not extractive and not grammar:
            incomplete_ids.append(rid)
            continue

        errors = []
        if faithful not in VALID_FAITHFUL:
            errors.append(f"manual_faithful={faithful!r}")
        if extractive not in VALID_EXTRACTIVE:
            errors.append(f"manual_extractive={extractive!r}")
        if grammar not in VALID_GRAMMAR:
            errors.append(f"manual_grammar={grammar!r}")

        if errors:
            invalid_rows.append({"id": rid, "errors": errors})
            continue

        completed_rows.append(row)
        faithful_counts[faithful] += 1
        extractive_counts[extractive] += 1
        grammar_counts[grammar] += 1

    completed = len(completed_rows)

    print("=" * 60)
    print("MANUAL AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total sampled rows      : {total_rows}")
    print(f"Completed valid rows    : {completed}")
    print(f"Blank rows remaining    : {len(incomplete_ids)}")
    print(f"Rows with invalid labels: {len(invalid_rows)}")

    if invalid_rows:
        print("\nInvalid label examples:")
        for item in invalid_rows[:10]:
            print(f"  {item['id']}: {', '.join(item['errors'])}")

    if completed == 0:
        print("\nNo completed valid audit rows yet.")
        print("Use labels:")
        print("  manual_faithful   -> yes / no")
        print("  manual_extractive -> yes / no")
        print("  manual_grammar    -> good / ok / bad")
        return

    print("\nFaithfulness")
    print(f"  yes: {faithful_counts['yes']} ({pct(faithful_counts['yes'], completed):.2f}%)")
    print(f"  no : {faithful_counts['no']} ({pct(faithful_counts['no'], completed):.2f}%)")

    print("\nExtractiveness")
    print(f"  yes: {extractive_counts['yes']} ({pct(extractive_counts['yes'], completed):.2f}%)")
    print(f"  no : {extractive_counts['no']} ({pct(extractive_counts['no'], completed):.2f}%)")

    print("\nGrammar")
    print(f"  good: {grammar_counts['good']} ({pct(grammar_counts['good'], completed):.2f}%)")
    print(f"  ok  : {grammar_counts['ok']} ({pct(grammar_counts['ok'], completed):.2f}%)")
    print(f"  bad : {grammar_counts['bad']} ({pct(grammar_counts['bad'], completed):.2f}%)")

    good_or_ok = grammar_counts["good"] + grammar_counts["ok"]
    print(f"  good+ok: {good_or_ok} ({pct(good_or_ok, completed):.2f}%)")

    print("\nPaper-ready summary")
    print(
        f"  Manual audit over {completed} sampled QA pairs found "
        f"{pct(faithful_counts['yes'], completed):.2f}% faithfulness, "
        f"{pct(extractive_counts['yes'], completed):.2f}% extractiveness, and "
        f"{pct(good_or_ok, completed):.2f}% good-or-acceptable question grammar."
    )


if __name__ == "__main__":
    main()
