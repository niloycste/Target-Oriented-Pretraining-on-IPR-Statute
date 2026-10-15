"""
QA Dataset Quality Review - Three-judge ensemble evaluation
===========================================================

Layer 1: Programmatic checks on all pairs
Layer 2: LLM-as-Judge via Ollama using multiple judge models

Input  : data/processed/indic_ipr_qa/merged_ip_qa.jsonl
Output : qa_review_report.json
         qa_review_flagged.jsonl
         qa_review_progress.jsonl
         qa_review_errors.log

Run: python review_qa_dataset_ensemble.py
"""

import hashlib
import json
import logging
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# =========================================================
# Config
# =========================================================

JUDGE_MODELS = [
    "gemma3:12b",
    "llama3.1:8b",
    "mistral:7b",
]

# Aggregation: "majority" or "average"
ENSEMBLE_METHOD = "majority"

TEMPERATURE = 0.0
REQUEST_DELAY = 0.2
REQUEST_TIMEOUT = 180.0

SAMPLE_SIZE = None
SAMPLE_SEED = 42
STRATIFIED_SAMPLE = True

PROMPT_VERSION = "v2_multi_judge_ensemble_fixed"

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "indic_ipr_qa" / "merged_ip_qa.jsonl"
RESULTS_DIR = BASE_DIR / "results" / "evaluation"
LOGS_DIR = BASE_DIR / "results" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = RESULTS_DIR / "llm_as_judge_review_report.json"
FLAGGED_FILE = RESULTS_DIR / "llm_as_judge_review_flagged.jsonl"
PROGRESS_FILE = RESULTS_DIR / "llm_as_judge_review_progress.jsonl"
LOG_FILE = LOGS_DIR / "llm_as_judge_review_errors.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

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
  3 = Asks about a specific legal element: time period, authority, penalty, condition, right, duty, proviso, procedure, exception, defined term
  2 = Somewhat specific but could target a more precise legal element
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
  3 = Question asks about substantive legal content (rights, duties, procedures, conditions, exceptions, defined legal concepts)
  2 = Marginally legal - could be more substantive
  1 = Trivial - asks only about section number, act name, date, or marginal heading

Output valid JSON only - no explanation, no markdown:
{{"Q_SPECIFIC": <1-3>, "A_SUFFICIENT": <1-3>, "A_MINIMAL": <1-3>, "LEGAL_RELEVANT": <1-3>, "comment": "<max 10 words on worst issue>"}}"""

GENERIC_Q_RE = re.compile(
    r"what does (section|this section|the section|clause|sub-?section|this act|the act)"
    r"\s*[\dA-Z()\-]*\s*(state|provide|say|stipulate|mention|specify|contain)\?",
    re.IGNORECASE,
)

EDITORIAL_RE = re.compile(
    r"\b(Ins\.|Subs\.|Omitted|w\.e\.f\.|w\.e\.f|Inserted by|Substituted by)",
    re.IGNORECASE,
)

LOW_SUBSTANCE_RE = re.compile(
    r"\b(short title|extent|commencement|title of the act|act may be called)\b",
    re.IGNORECASE,
)


# =========================================================
# Utility / record helpers
# =========================================================

def record_fingerprint(record) -> str:
    payload = {
        "id": record.get("id", ""),
        "act": record.get("act", ""),
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
        "context": record.get("context", ""),
        "answerspan": record.get("answerspan", None),
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def record_key(record) -> str:
    return record.get("id") or record_fingerprint(record)


def is_valid_single_judge_scores(scores) -> bool:
    return isinstance(scores, dict) and all(scores.get(k) in (1, 2, 3) for k in CRITERIA)


def is_valid_final_scores(scores) -> bool:
    if not isinstance(scores, dict):
        return False
    for k in CRITERIA:
        v = scores.get(k)
        if not isinstance(v, (int, float)):
            return False
        if not (1 <= v <= 3):
            return False
    return True


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


# =========================================================
# Cache loading
# =========================================================

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

            if entry.get("_prompt_version") != PROMPT_VERSION:
                skipped_invalid += 1
                continue

            if entry.get("_judge_models") != JUDGE_MODELS:
                skipped_invalid += 1
                continue

            if entry.get("_ensemble_method") != ENSEMBLE_METHOD:
                skipped_invalid += 1
                continue

            final_scores = entry.get("llm_scores", {})
            judge_runs = entry.get("judge_runs", {})
            agreement = entry.get("judge_agreement", {})

            if not is_valid_final_scores(final_scores):
                skipped_invalid += 1
                continue

            if not isinstance(judge_runs, dict):
                skipped_invalid += 1
                continue

            if not isinstance(agreement, dict):
                skipped_invalid += 1
                continue

            scored_cache[rid] = {
                **current_record,
                "llm_scores": final_scores,
                "judge_runs": judge_runs,
                "judge_agreement": agreement,
            }

    return scored_cache, skipped_invalid


# =========================================================
# LLM judging
# =========================================================

def llm_judge(model_name: str, context: str, question: str, answer: str):
    excerpt = get_context_excerpt(context, answer)
    prompt = JUDGE_PROMPT.format(
        context_excerpt=excerpt,
        question=question,
        answer=answer,
    )

    last_error = None

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=120,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
            match = re.search(r"\{.*\}", raw, re.DOTALL)

            if not match:
                raise ValueError(f"Judge returned no JSON object. Raw output: {raw[:300]}")

            scores = json.loads(match.group())
            if is_valid_single_judge_scores(scores):
                return scores

            raise ValueError(f"Invalid judge scores from {model_name}: {scores}")

        except Exception as exc:
            last_error = exc
            logging.warning(
                "Judge failed | model=%s | attempt=%s | qhash=%s | error=%s",
                model_name,
                attempt + 1,
                hashlib.sha1((question + answer).encode("utf-8")).hexdigest()[:12],
                repr(exc),
            )
            if attempt < 2:
                time.sleep(1)

    logging.error(
        "Judge permanently failed | model=%s | error=%s | question=%r | answer=%r",
        model_name,
        repr(last_error),
        question[:200],
        answer[:200],
    )
    return None


def aggregate_scores_majority(score_list):
    if not score_list:
        return None

    aggregated = {}
    tie_flags = []

    for key in CRITERIA:
        counts = Counter(scores[key] for scores in score_list)
        most_common = counts.most_common()
        top_freq = most_common[0][1]
        top_vals = sorted([val for val, freq in most_common if freq == top_freq])

        if len(top_vals) == 1:
            aggregated[key] = top_vals[0]
        else:
            # deterministic tie-break: choose the upper median among tied values
            aggregated[key] = top_vals[len(top_vals) // 2]
            tie_flags.append(key)

    comment = f"majority_of_{len(score_list)}_judges"
    if tie_flags:
        comment += f" | tie:{','.join(tie_flags)}"
    aggregated["comment"] = comment
    return aggregated


def aggregate_scores_average(score_list):
    if not score_list:
        return None

    aggregated = {}
    for key in CRITERIA:
        avg = sum(scores[key] for scores in score_list) / len(score_list)
        aggregated[key] = round(avg, 2)

    aggregated["comment"] = f"avg_of_{len(score_list)}_judges"
    return aggregated


def compute_agreement(judge_runs: dict):
    valid = {
        model: scores
        for model, scores in judge_runs.items()
        if is_valid_single_judge_scores(scores)
    }

    if not valid:
        return {"valid_judges": 0, "all_agree": False}

    agreement = {"valid_judges": len(valid)}
    all_criterion_agree = []

    for key in CRITERIA:
        vals = [scores[key] for scores in valid.values()]
        agreement[key] = {
            "all_equal": len(set(vals)) == 1,
            "values": vals,
        }
        all_criterion_agree.append(len(set(vals)) == 1)

    agreement["all_agree"] = all(all_criterion_agree)
    agreement["disagreement_count"] = sum(1 for ok in all_criterion_agree if not ok)
    return agreement


def judge_with_ensemble(context: str, question: str, answer: str):
    judge_runs = {}

    for model_name in JUDGE_MODELS:
        time.sleep(REQUEST_DELAY)
        scores = llm_judge(model_name, context, question, answer)
        judge_runs[model_name] = scores

    valid_scores = [
        scores for scores in judge_runs.values()
        if is_valid_single_judge_scores(scores)
    ]

    if not valid_scores:
        return None, judge_runs, {"valid_judges": 0, "all_agree": False}

    if ENSEMBLE_METHOD == "average":
        final_scores = aggregate_scores_average(valid_scores)
    else:
        final_scores = aggregate_scores_majority(valid_scores)

    agreement = compute_agreement(judge_runs)

    if final_scores is not None and agreement.get("all_agree") is False:
        final_scores["comment"] = final_scores.get("comment", "") + " | judges_disagree"

    return final_scores, judge_runs, agreement


# =========================================================
# Layer 1 programmatic checks
# =========================================================

def programmatic_checks(records):
    issues = defaultdict(list)
    q_types = Counter()

    for record in records:
        context = record.get("context", "")
        question = record.get("question", "")
        answer = record.get("answer", "")
        answerspan = record.get("answerspan", None)
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

        idx = context.find(answer)
        if idx < 0:
            issues["answer_not_in_context"].append(record_id)

        if isinstance(answerspan, int):
            if 0 <= answerspan < len(context):
                sliced = context[answerspan:answerspan + len(answer)]
                if sliced != answer:
                    issues["answerspan_mismatch"].append(record_id)
            else:
                issues["answerspan_out_of_bounds"].append(record_id)

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
        if LOW_SUBSTANCE_RE.search(question) or LOW_SUBSTANCE_RE.search(record.get("title", "")):
            issues["low_substance_pattern"].append(record_id)
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


# =========================================================
# Save report
# =========================================================

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
    agreement_summary=None,
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
            "judge_models": JUDGE_MODELS,
            "ensemble_method": ENSEMBLE_METHOD,
            "prompt_version": PROMPT_VERSION,
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
            "agreement_summary": agreement_summary or {},
        }

    with REPORT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"  Report saved   : {REPORT_FILE.name}")


# =========================================================
# Main
# =========================================================

def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE} not found.")
        logging.error("Input file not found: %s", INPUT_FILE)
        return

    with INPUT_FILE.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    if not records:
        print(f"[ERROR] No records found in {INPUT_FILE}.")
        logging.error("No records found in input file: %s", INPUT_FILE)
        return

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

    answer_lengths = [len(record.get("answer", "").split()) for record in records]
    question_lengths = [len(record.get("question", "").split()) for record in records]
    context_lengths = [len(record.get("context", "").split()) for record in records]

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
    print("LAYER 2 - LLM-as-Judge (Multi-judge ensemble)")
    print("=" * 60)

    try:
        available_models = {m.id for m in client.models.list().data}
        missing = [m for m in JUDGE_MODELS if m not in available_models]
        if missing:
            print(f"[SKIP] Missing models in Ollama: {missing}")
            logging.error("Missing models in Ollama: %s", missing)
            print("Please pull or load those models, then retry.\n")
            _save_report(
                records,
                records[:0],
                Counter(),
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

        print(f"Ollama connected - judges: {', '.join(JUDGE_MODELS)}")
        print(f"Ensemble method : {ENSEMBLE_METHOD}\n")
    except Exception as exc:
        print(f"[SKIP] Cannot connect to Ollama: {exc}")
        logging.error("Cannot connect to Ollama: %r", exc)
        print("Run 'ollama serve' and retry.\n")
        _save_report(
            records,
            records[:0],
            Counter(),
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
    agreement_counts = Counter()

    with PROGRESS_FILE.open("a", encoding="utf-8") as progress_handle:
        for record in tqdm(sample, desc="LLM judging"):
            record_id = record.get("id", "")

            if record_id in scored_cache:
                entry = scored_cache[record_id]
            else:
                final_scores, judge_runs, agreement = judge_with_ensemble(
                    record["context"],
                    record["question"],
                    record["answer"],
                )

                if is_valid_final_scores(final_scores):
                    entry = {
                        **record,
                        "llm_scores": final_scores,
                        "judge_runs": judge_runs,
                        "judge_agreement": agreement,
                    }
                    progress_entry = {
                        **entry,
                        "_record_hash": record_fingerprint(record),
                        "_prompt_version": PROMPT_VERSION,
                        "_judge_models": JUDGE_MODELS,
                        "_ensemble_method": ENSEMBLE_METHOD,
                    }
                    progress_handle.write(json.dumps(progress_entry, ensure_ascii=False) + "\n")
                    progress_handle.flush()
                else:
                    entry = {
                        **record,
                        "llm_scores": {"comment": "judge_failed"},
                        "judge_runs": judge_runs,
                        "judge_agreement": agreement,
                    }

            scored.append(entry)

            scores = entry.get("llm_scores", {})
            agreement = entry.get("judge_agreement", {})

            if not is_valid_final_scores(scores):
                failed += 1
                continue

            if agreement.get("all_agree") is True:
                agreement_counts["all_agree"] += 1
            else:
                agreement_counts["disagree"] += 1

            # Flag if any criterion is weak or borderline weak
            if any(scores.get(key, 0) <= 1.5 for key in CRITERIA):
                flagged.append(entry)

    valid = [entry for entry in scored if is_valid_final_scores(entry.get("llm_scores", {}))]
    n_valid = len(valid)

    score_sums = {
        key: sum(entry["llm_scores"].get(key, 0) for entry in valid)
        for key in CRITERIA
    }

    score_dist = {key: Counter() for key in CRITERIA}
    for entry in valid:
        for key in CRITERIA:
            score_dist[key][str(entry["llm_scores"].get(key, 0))] += 1

    act_scores = defaultdict(lambda: defaultdict(list))
    for entry in valid:
        act = entry.get("act", "unknown")
        for key in CRITERIA:
            act_scores[act][key].append(entry["llm_scores"].get(key, 0))

    agreement_summary = {
        "all_agree_count": agreement_counts["all_agree"],
        "disagree_count": agreement_counts["disagree"],
        "all_agree_rate": round(
            agreement_counts["all_agree"] / max(n_valid, 1) * 100, 1
        ),
    }

    print(f"\nLLM Judge results  ({n_valid} valid / {failed} failed / {len(scored)} total):")
    print("\n  Score averages:")
    for key in CRITERIA:
        avg = score_sums[key] / max(n_valid, 1)
        bar = "#" * int(avg * 10)
        print(f"  {key:<20} {avg:.2f}  {bar}")

    print("\n  Score distribution:")
    for key in CRITERIA:
        print(f"  {key:<20} {dict(score_dist[key])}")

    print("\n  Per-act average scores:")
    print(f"  {'Act':<45} {'Q_SP':>6} {'A_SU':>6} {'A_MI':>6} {'LG_R':>6}")
    print(f"  {'-' * 73}")
    for act in sorted(act_scores):
        averages = {
            key: sum(act_scores[act][key]) / max(len(act_scores[act][key]), 1)
            for key in CRITERIA
        }
        print(
            f"  {act:<45} {averages['Q_SPECIFIC']:>6.2f} {averages['A_SUFFICIENT']:>6.2f} "
            f"{averages['A_MINIMAL']:>6.2f} {averages['LEGAL_RELEVANT']:>6.2f}"
        )

    print(
        f"\n  Agreement      : {agreement_summary['all_agree_count']} all-agree / "
        f"{agreement_summary['disagree_count']} disagree "
        f"({agreement_summary['all_agree_rate']:.1f}% full agreement)"
    )

    print(
        f"  Flagged        : {len(flagged)} / {n_valid} "
        f"({len(flagged) / max(n_valid, 1) * 100:.1f}%)"
    )

    if flagged:
        print("\n  Sample flagged pairs (up to 5):")
        for entry in flagged[:5]:
            scores = entry["llm_scores"]
            agreement = entry.get("judge_agreement", {})
            print(f"    ID: {entry['id']}")
            print(f"    Q : {entry['question'][:90]}")
            print(f"    A : {entry['answer'][:70]}")
            print(
                f"    Scores: Q_SPEC={scores.get('Q_SPECIFIC')} "
                f"A_SUFF={scores.get('A_SUFFICIENT')} "
                f"A_MIN={scores.get('A_MINIMAL')} "
                f"LG_REL={scores.get('LEGAL_RELEVANT')}"
            )
            print(f"    All agree: {agreement.get('all_agree', False)}")
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
        agreement_summary,
    )


if __name__ == "__main__":
    main()
