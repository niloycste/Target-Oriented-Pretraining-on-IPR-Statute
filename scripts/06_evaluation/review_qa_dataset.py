"""
QA Dataset Quality Review - Two-layer evaluation
================================================
Layer 1: Programmatic checks on all pairs (instant, 13 checks)
Layer 2: LLM-as-Judge via Ollama (4 criteria, scored 1-3)

Input  : data/processed/indic_ipr_qa/merged_ip_qa.jsonl
Output : qa_review_report.json
         qa_review_flagged.jsonl
         qa_review_progress.jsonl

Run: python review_qa_dataset.py
"""

import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# Config
OLLAMA_MODEL = "gemma3:12b"
TEMPERATURE = 0.0
REQUEST_DELAY = 0.2
REQUEST_TIMEOUT = 180.0
SAMPLE_SIZE = None
SAMPLE_SEED = 42
STRATIFIED_SAMPLE = True

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "indic_ipr_qa" / "merged_ip_qa.jsonl"
RESULTS_DIR = BASE_DIR / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = RESULTS_DIR / "qa_review_report.json"
FLAGGED_FILE = RESULTS_DIR / "qa_review_flagged.jsonl"
PROGRESS_FILE = RESULTS_DIR / "qa_review_progress.jsonl"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=REQUEST_TIMEOUT,
)

CRITERIA = ("Q_SPECIFIC", "A_SUFFICIENT", "A_MINIMAL", "LEGAL_RELEVANT")

JUDGE_PROMPT = """You are a quality reviewer for an extractive QA dataset built from Indian IPR (Intellectual Property Rights) statutes.

In extractive QA, the answer must be a verbatim substring copied directly from the context - no paraphrasing.

You are given one QA pair. The relevant context excerpt around the answer is shown.

Context excerpt:
{context_excerpt}

Question: {question}
Answer:   {answer}

Score each criterion from 1 to 3 ONLY:

Q_SPECIFIC (question specificity):
  3 = Asks about a specific legal element: time period, authority, penalty, condition, right, duty, proviso, procedure
  2 = Somewhat specific but could target a more precise element
  1 = Generic ("What does this section state?") OR yes/no question OR asks about section number/title only

A_SUFFICIENT (answer answers the question):
  3 = The answer span fully and directly answers the question
  2 = Partially answers - missing a minor element
  1 = Does not answer the question at all, or answers a different question

A_MINIMAL (answer span length):
  3 = Shortest span that answers - not too short, not unnecessarily long
  2 = Slightly too long (includes one extra clause not needed)
  1 = Far too long (nearly the whole context) OR too short (incomplete legal phrase)

LEGAL_RELEVANT (legal substance):
  3 = Question asks about substantive legal content (rights, duties, procedures, conditions)
  2 = Marginally legal - could be more substantive
  1 = Trivial - asks only about section number, act name, date, or marginal heading

Output valid JSON only - no explanation, no markdown:
{{"Q_SPECIFIC": <1-3>, "A_SUFFICIENT": <1-3>, "A_MINIMAL": <1-3>, "LEGAL_RELEVANT": <1-3>, "comment": "<max 10 words on worst issue>"}}"""

GENERIC_Q_RE = re.compile(
    r"what does (section|this section|the section|clause|sub-?section|this act|the act)"
    r"\s*[\dA-Z]*\s*(state|provide|say|stipulate|mention|specify|contain)\?",
    re.IGNORECASE,
)

EDITORIAL_RE = re.compile(
    r"\b(Ins\.|Subs\.|Omitted|w\.e\.f\.|w\.e\.f|Inserted by|Substituted by)",
    re.IGNORECASE,
)


def record_fingerprint(record) -> str:
    payload = {
        "id": record.get("id", ""),
        "act": record.get("act", ""),
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
        "context": record.get("context", ""),
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def record_key(record) -> str:
    return record.get("id") or record_fingerprint(record)


def is_valid_llm_scores(scores) -> bool:
    return isinstance(scores, dict) and all(scores.get(k) in (1, 2, 3) for k in CRITERIA)


def get_context_excerpt(context: str, answer: str, window: int = 300) -> str:
    idx = context.find(answer)
    if idx < 0 or len(context) <= window * 2:
        return context
    start = max(0, idx - window)
    end = min(len(context), idx + len(answer) + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(context) else ""
    return prefix + context[start:end] + suffix


def select_sample(records):
    if SAMPLE_SIZE is None or SAMPLE_SIZE >= len(records):
        return list(records)

    rng = random.Random(SAMPLE_SEED)
    if not STRATIFIED_SAMPLE:
        return rng.sample(records, SAMPLE_SIZE)

    by_act = defaultdict(list)
    for record in records:
        by_act[record.get("act", "unknown")].append(record)

    if SAMPLE_SIZE < len(by_act):
        return rng.sample(records, SAMPLE_SIZE)

    selected = []
    selected_keys = set()

    for act in sorted(by_act):
        picked = rng.choice(by_act[act])
        selected.append(picked)
        selected_keys.add(record_key(picked))

    remaining_needed = SAMPLE_SIZE - len(selected)
    if remaining_needed > 0:
        remaining_pool = [r for r in records if record_key(r) not in selected_keys]
        selected.extend(rng.sample(remaining_pool, remaining_needed))

    rng.shuffle(selected)
    return selected


def load_scored_cache(records_by_id):
    scored_cache = {}
    skipped_invalid = 0

    if not PROGRESS_FILE.exists():
        return scored_cache, skipped_invalid

    with PROGRESS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue

            rid = entry.get("id", "")
            current_record = records_by_id.get(rid)
            if current_record is None:
                skipped_invalid += 1
                continue

            if entry.get("_record_hash") != record_fingerprint(current_record):
                skipped_invalid += 1
                continue

            scores = entry.get("llm_scores", {})
            if not is_valid_llm_scores(scores):
                skipped_invalid += 1
                continue

            scored_cache[rid] = {**current_record, "llm_scores": scores}

    return scored_cache, skipped_invalid


def llm_judge(context: str, question: str, answer: str):
    excerpt = get_context_excerpt(context, answer)
    prompt = JUDGE_PROMPT.format(
        context_excerpt=excerpt,
        question=question,
        answer=answer,
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=120,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("Judge returned no JSON object.")

            scores = json.loads(match.group())
            if is_valid_llm_scores(scores):
                return scores
            raise ValueError(f"Invalid judge scores: {scores}")
        except Exception:
            if attempt < 2:
                time.sleep(1)

    return None


def programmatic_checks(records):
    issues = defaultdict(list)
    q_types = Counter()

    for record in records:
        context = record.get("context", "")
        question = record.get("question", "")
        answer = record.get("answer", "")
        record_id = record.get("id", "?")

        question_lower = question.lower()
        if question_lower.startswith("what is"):
            q_types["what_is"] += 1
        elif question_lower.startswith("who"):
            q_types["who"] += 1
        elif question_lower.startswith("how"):
            q_types["how"] += 1
        elif question_lower.startswith("which"):
            q_types["which"] += 1
        elif question_lower.startswith("under what"):
            q_types["under_what"] += 1
        elif question_lower.startswith("what shall"):
            q_types["what_shall"] += 1
        elif question_lower.startswith("what are"):
            q_types["what_are"] += 1
        elif question_lower.startswith("what right"):
            q_types["what_right"] += 1
        elif question_lower.startswith("what must"):
            q_types["what_must"] += 1
        elif question_lower.startswith("what happen"):
            q_types["what_happens"] += 1
        elif question_lower.startswith("how many"):
            q_types["how_many"] += 1
        else:
            q_types["other"] += 1

        if context.find(answer) < 0:
            issues["answer_not_in_context"].append(record_id)
        if len(answer) < 5:
            issues["answer_too_short_chars"].append(record_id)
        if len(answer) > 400:
            issues["answer_too_long_chars"].append(record_id)
        if len(answer.split()) < 3:
            issues["answer_under_3_words"].append(record_id)
        if len(context) > 0 and len(answer) / len(context) > 0.80:
            issues["answer_covers_most_context"].append(record_id)
        if len(question.split()) < 5:
            issues["question_too_short"].append(record_id)
        if not question.strip().endswith("?"):
            issues["no_question_mark"].append(record_id)
        if GENERIC_Q_RE.search(question):
            issues["generic_question_pattern"].append(record_id)
        if len(answer) > 20 and answer.lower() in question.lower():
            issues["answer_inside_question"].append(record_id)
        if answer and answer[-1] in ("(", "-", "—"):
            issues["answer_ends_with_dangling_punctuation"].append(record_id)
        if EDITORIAL_RE.search(answer):
            issues["answer_contains_editorial_artifact"].append(record_id)
        if re.fullmatch(r"\d{1,4}(\s+\w+)?", answer.strip()):
            issues["answer_is_just_number_or_year"].append(record_id)

    seen_questions = defaultdict(set)
    for record in records:
        act = record.get("act", "unknown")
        normalized_question = re.sub(r"\W+", " ", record.get("question", "").lower()).strip()
        if normalized_question in seen_questions[act]:
            issues["duplicate_question_within_act"].append(record.get("id", "?"))
        else:
            seen_questions[act].add(normalized_question)

    return issues, q_types


def count_cross_act_duplicate_questions(records) -> int:
    seen = {}
    duplicates = 0
    for record in records:
        normalized_question = re.sub(r"\W+", " ", record.get("question", "").lower()).strip()
        if normalized_question in seen:
            duplicates += 1
        else:
            seen[normalized_question] = record.get("id", "?")
    return duplicates


def _save_report(
    records,
    sample,
    sample_act_counts,
    scored,
    issues,
    q_types,
    answer_lengths,
    question_lengths,
    context_lengths,
    act_counts,
    cross_act_duplicates,
    score_sums=None,
    score_dist=None,
    act_scores=None,
    flagged=None,
    n_valid=0,
    failed=0,
):
    distinct_issue_ids = sorted({rid for values in issues.values() for rid in values})

    report = {
        "total_pairs_in_file": len(records),
        "pairs_reviewed": len(sample),
        "input_file": str(INPUT_FILE.name),
        "sample_config": {
            "requested_sample_size": SAMPLE_SIZE,
            "actual_sample_size": len(sample),
            "seed": SAMPLE_SEED,
            "stratified": STRATIFIED_SAMPLE,
            "sample_act_distribution": dict(sample_act_counts),
        },
        "layer1_programmatic": {
            "total_issue_hits": sum(len(values) for values in issues.values()),
            "distinct_flagged_pairs": len(distinct_issue_ids),
            "distinct_flagged_rate": round(len(distinct_issue_ids) / max(len(records), 1) * 100, 1),
            "issue_counts": {key: len(values) for key, values in issues.items()},
            "flagged_ids": {key: values[:20] for key, values in issues.items()},
            "cross_act_duplicate_questions": cross_act_duplicates,
        },
        "question_type_distribution": dict(q_types),
        "length_stats": {
            "answer_words": {
                "min": min(answer_lengths),
                "max": max(answer_lengths),
                "avg": round(sum(answer_lengths) / len(answer_lengths), 1),
            },
            "question_words": {
                "min": min(question_lengths),
                "max": max(question_lengths),
                "avg": round(sum(question_lengths) / len(question_lengths), 1),
            },
            "context_words": {
                "min": min(context_lengths),
                "max": max(context_lengths),
                "avg": round(sum(context_lengths) / len(context_lengths), 1),
            },
        },
        "act_distribution": dict(act_counts),
    }

    if scored:
        report["layer2_llm_judge"] = {
            "model": OLLAMA_MODEL,
            "reviewed": len(sample),
            "valid_judgements": n_valid,
            "failed_judgements": failed,
            "score_averages": {
                key: round(score_sums[key] / max(n_valid, 1), 2) for key in CRITERIA
            } if score_sums else {},
            "score_distribution": {
                key: dict(score_dist[key]) for key in CRITERIA
            } if score_dist else {},
            "per_act_averages": {
                act: {
                    key: round(sum(act_scores[act][key]) / max(len(act_scores[act][key]), 1), 2)
                    for key in CRITERIA
                }
                for act in (act_scores or {})
            },
            "flagged_count": len(flagged) if flagged else 0,
            "flagged_rate": (
                f"{len(flagged) / max(n_valid, 1) * 100:.1f}%"
                if flagged is not None else "0%"
            ),
        }

    with REPORT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"  Report saved   : {REPORT_FILE.name}")


def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE} not found.")
        return

    with INPUT_FILE.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    sample = select_sample(records)
    sample_act_counts = Counter(record.get("act", "unknown") for record in sample)

    print(f"Total in file  : {len(records)} QA pairs")
    print(
        f"Reviewing      : {len(sample)} pairs  "
        f"({'ALL' if len(sample) == len(records) else f'SAMPLE of {len(sample)}'})"
    )
    print(f"Input          : {INPUT_FILE.name}\n")

    if len(sample) != len(records):
        sample_mode = "stratified" if STRATIFIED_SAMPLE else "random"
        print(f"Sample config  : seed={SAMPLE_SEED}  mode={sample_mode}")
        print(
            "Sample acts    : " + ", ".join(
                f"{act}={sample_act_counts[act]}" for act in sorted(sample_act_counts)
            ) + "\n"
        )

    print("=" * 60)
    print("LAYER 1 - Programmatic checks  (all records)")
    print("=" * 60)
    issues, q_types = programmatic_checks(records)

    total_issue_hits = sum(len(values) for values in issues.values())
    distinct_issue_ids = sorted({rid for values in issues.values() for rid in values})
    cross_act_duplicates = count_cross_act_duplicate_questions(records)

    print(f"\nIssue hits     : {total_issue_hits}")
    print(
        f"Flagged pairs  : {len(distinct_issue_ids)} / {len(records)} "
        f"({len(distinct_issue_ids) / max(len(records), 1) * 100:.1f}%)"
    )
    print(f"Cross-act dups : {cross_act_duplicates}  (informational)\n")

    if total_issue_hits == 0:
        print("  No programmatic issues found.")
    for key, values in sorted(issues.items(), key=lambda item: -len(item[1])):
        pct = len(values) / len(records) * 100
        print(f"  {key:<45} {len(values):>5}  ({pct:.1f}%)")

    print("\nQuestion type distribution:")
    for key, value in sorted(q_types.items(), key=lambda item: -item[1]):
        pct = value / len(records) * 100
        print(f"  {key:<28} {value:>5}  ({pct:.1f}%)")

    answer_lengths = [len(record["answer"].split()) for record in records]
    question_lengths = [len(record["question"].split()) for record in records]
    context_lengths = [len(record["context"].split()) for record in records]
    print("\nLength statistics (words):")
    print(
        f"  Answer   min={min(answer_lengths):>3}  "
        f"max={max(answer_lengths):>3}  avg={sum(answer_lengths)/len(answer_lengths):.1f}"
    )
    print(
        f"  Question min={min(question_lengths):>3}  "
        f"max={max(question_lengths):>3}  avg={sum(question_lengths)/len(question_lengths):.1f}"
    )
    print(
        f"  Context  min={min(context_lengths):>3}  "
        f"max={max(context_lengths):>3}  avg={sum(context_lengths)/len(context_lengths):.1f}"
    )

    act_counts = Counter(record.get("act", "unknown") for record in records)
    print("\nAct distribution:")
    for act, count in sorted(act_counts.items(), key=lambda item: -item[1]):
        print(f"  {act:<50} {count}")

    print("\n" + "=" * 60)
    print("LAYER 2 - LLM-as-Judge")
    print("=" * 60)

    try:
        client.models.list()
        print(f"Ollama connected - model: {OLLAMA_MODEL}\n")
    except Exception as exc:
        print(f"[SKIP] Cannot connect to Ollama: {exc}")
        print("Run 'ollama serve' and retry.\n")
        _save_report(
            records,
            sample,
            sample_act_counts,
            [],
            issues,
            q_types,
            answer_lengths,
            question_lengths,
            context_lengths,
            act_counts,
            cross_act_duplicates,
        )
        return

    records_by_id = {record.get("id", ""): record for record in records}
    scored_cache, skipped_cache_entries = load_scored_cache(records_by_id)
    if scored_cache:
        print(f"Resuming - {len(scored_cache)} valid cached judgements found.")
    if skipped_cache_entries:
        print(f"Skipped cache : {skipped_cache_entries} stale or invalid cached entries.")
    if scored_cache or skipped_cache_entries:
        print()

    scored = []
    flagged = []
    failed = 0

    with PROGRESS_FILE.open("a", encoding="utf-8") as progress_handle:
        for record in tqdm(sample, desc="LLM judging"):
            record_id = record.get("id", "")

            if record_id in scored_cache:
                entry = scored_cache[record_id]
            else:
                time.sleep(REQUEST_DELAY)
                scores = llm_judge(record["context"], record["question"], record["answer"])
                if is_valid_llm_scores(scores):
                    entry = {**record, "llm_scores": scores}
                    progress_entry = {
                        **entry,
                        "_record_hash": record_fingerprint(record),
                    }
                    progress_handle.write(json.dumps(progress_entry, ensure_ascii=False) + "\n")
                    progress_handle.flush()
                else:
                    entry = {**record, "llm_scores": {"comment": "judge_failed"}}

            scored.append(entry)
            scores = entry.get("llm_scores", {})
            if not is_valid_llm_scores(scores):
                failed += 1
            elif any(scores.get(key, 0) == 1 for key in CRITERIA):
                flagged.append(entry)

    valid = [entry for entry in scored if is_valid_llm_scores(entry.get("llm_scores", {}))]
    n_valid = len(valid)

    score_sums = {key: sum(entry["llm_scores"].get(key, 0) for entry in valid) for key in CRITERIA}
    score_dist = {key: Counter(entry["llm_scores"].get(key, 0) for entry in valid) for key in CRITERIA}
    act_scores = defaultdict(lambda: defaultdict(list))
    for entry in valid:
        act = entry.get("act", "unknown")
        for key in CRITERIA:
            act_scores[act][key].append(entry["llm_scores"].get(key, 0))

    print(f"\nLLM Judge results  ({n_valid} valid / {failed} failed / {len(scored)} total):")
    print("\n  Score averages (out of 3.0):")
    for key in CRITERIA:
        avg = score_sums[key] / max(n_valid, 1)
        bar = "#" * int(avg * 10)
        print(f"  {key:<20} {avg:.2f}  {bar}")

    print("\n  Score distribution (1=poor  2=ok  3=good):")
    print(f"  {'Criterion':<20} {'1 (poor)':>10} {'2 (ok)':>10} {'3 (good)':>10}")
    print(f"  {'-' * 54}")
    for key in CRITERIA:
        distribution = score_dist[key]
        p1 = distribution[1] / max(n_valid, 1) * 100
        p2 = distribution[2] / max(n_valid, 1) * 100
        p3 = distribution[3] / max(n_valid, 1) * 100
        print(
            f"  {key:<20} {distribution[1]:>5} ({p1:.0f}%)"
            f"  {distribution[2]:>5} ({p2:.0f}%)"
            f"  {distribution[3]:>5} ({p3:.0f}%)"
        )

    print("\n  Per-act average scores:")
    print(f"  {'Act':<45} {'Q_SP':>5} {'A_SU':>5} {'A_MI':>5} {'LG_R':>5}")
    print(f"  {'-' * 65}")
    for act in sorted(act_scores):
        averages = {
            key: sum(act_scores[act][key]) / max(len(act_scores[act][key]), 1)
            for key in CRITERIA
        }
        print(
            f"  {act:<45} {averages['Q_SPECIFIC']:>5.2f} {averages['A_SUFFICIENT']:>5.2f} "
            f"{averages['A_MINIMAL']:>5.2f} {averages['LEGAL_RELEVANT']:>5.2f}"
        )

    print(
        f"\n  Flagged (any score == 1): {len(flagged)} / {n_valid} "
        f"({len(flagged) / max(n_valid, 1) * 100:.1f}%)"
    )

    if flagged:
        print("\n  Sample of flagged pairs (up to 5):")
        for entry in flagged[:5]:
            scores = entry["llm_scores"]
            print(f"    ID: {entry['id']}")
            print(f"    Q : {entry['question'][:80]}")
            print(f"    A : {entry['answer'][:60]}")
            print(
                f"    Scores: Q_SPEC={scores.get('Q_SPECIFIC')} "
                f"A_SUFF={scores.get('A_SUFFICIENT')} "
                f"A_MIN={scores.get('A_MINIMAL')} "
                f"LG_REL={scores.get('LEGAL_RELEVANT')}"
            )
            print(f"    Note: {scores.get('comment', '')}\n")

    with FLAGGED_FILE.open("w", encoding="utf-8") as handle:
        for entry in flagged:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Flagged saved  : {FLAGGED_FILE.name}")

    if failed == 0 and n_valid == len(sample):
        PROGRESS_FILE.unlink(missing_ok=True)
        print("  Progress file  : deleted (run complete)")
    else:
        print("  Progress file  : kept (incomplete run or retryable failures)")

    _save_report(
        records,
        sample,
        sample_act_counts,
        scored,
        issues,
        q_types,
        answer_lengths,
        question_lengths,
        context_lengths,
        act_counts,
        cross_act_duplicates,
        score_sums,
        score_dist,
        act_scores,
        flagged,
        n_valid,
        failed,
    )


if __name__ == "__main__":
    main()
