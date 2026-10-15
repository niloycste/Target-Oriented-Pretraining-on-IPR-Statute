"""
LLM-as-Judge Evaluation of QA Model Predictions
=================================================
Reads qa_eval_report.json (produced by evaluate_qa.py) and scores each
model's predictions against the gold answer using three local LLM judges
via Ollama.  The judge is asked to rate how acceptable the predicted
answer is on a 4-point scale:

  4 = Exact / Essentially identical
  3 = Acceptable — correct meaning, minor wording difference
  2 = Partial — contains part of the answer but is incomplete or imprecise
  1 = Wrong — incorrect or off-topic

Aggregation: majority vote across 3 judges; ties broken by mean score.
Acceptability rate = fraction of predictions scoring >= 3 (Exact or Acceptable).

Models evaluated by default: all 11 models in qa_eval_report.json.
Use --model_filter substring to restrict (e.g. "C'" or "A").

Outputs:
  llm_judge_eval_report.json   — full per-item scores + aggregate stats
  llm_judge_eval_progress.jsonl — checkpoint (resume on restart)

Run:
    python scripts/06_evaluation/llm_judge_evaluation.py
    python scripts/06_evaluation/llm_judge_evaluation.py --model_filter "C'"
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
JUDGE_MODELS = [
    "gemma3:12b",
    "llama3.1:8b",
    "mistral:7b",
]

TEMPERATURE    = 0.0
REQUEST_DELAY  = 0.3          # seconds between Ollama calls
REQUEST_TIMEOUT = 120.0
ENSEMBLE_METHOD = "majority"  # "majority" or "average"
ACCEPT_THRESHOLD = 3          # score >= this → acceptable

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "evaluation"
LOGS_DIR = BASE_DIR / "results" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_IN = RESULTS_DIR / "qa_eval_report.json"
REPORT_OUT = RESULTS_DIR / "llm_judge_eval_report.json"
PROGRESS_FILE = RESULTS_DIR / "llm_judge_eval_progress.jsonl"
LOG_FILE = LOGS_DIR / "llm_judge_eval.log"

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

# ── Prompt ────────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are evaluating the quality of a predicted answer produced by a QA model
for a question about Indian Intellectual Property Rights (IPR) statutes.
The task is extractive QA: the correct answer is a verbatim substring from the legal text.

Question    : {question}
Gold answer : {gold}
Predicted   : {pred}

Rate the predicted answer on a 4-point scale:
  4 = Exact or essentially identical (only trivial case difference)
  3 = Acceptable — conveys the correct legal meaning, minor wording difference is OK
  2 = Partial — contains part of the correct answer but is incomplete or adds extra noise
  1 = Wrong — incorrect, off-topic, or empty

Output valid JSON only — no explanation, no markdown fences:
{{"score": <1|2|3|4>, "reason": "<max 12 words>"}}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def call_judge(model: str, question: str, gold: str, pred: str) -> dict | None:
    """Call a single judge and parse its JSON response."""
    prompt = JUDGE_PROMPT.format(
        question=question.strip(),
        gold=gold.strip(),
        pred=pred.strip() if pred.strip() else "(empty)",
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if model wrapped in ```json ... ```
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        score = parsed.get("score")
        if score not in (1, 2, 3, 4):
            return None
        return {"score": int(score), "reason": parsed.get("reason", "")}
    except Exception as exc:
        logging.warning("Judge %s failed: %s", model, exc)
        return None


def ensemble_score(judge_scores: list[int]) -> int:
    """Return final score from a list of per-judge scores."""
    if not judge_scores:
        return 1
    if ENSEMBLE_METHOD == "majority":
        cnt = Counter(judge_scores)
        max_count = max(cnt.values())
        candidates = [s for s, c in cnt.items() if c == max_count]
        # tie-break: pick candidate closest to mean
        mean = sum(judge_scores) / len(judge_scores)
        return min(candidates, key=lambda s: abs(s - mean))
    else:
        return round(sum(judge_scores) / len(judge_scores))


def load_progress() -> dict:
    """Key: "model_label|||pred_id", Value: final_score int."""
    cache = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        cache[entry["key"]] = entry
                    except Exception:
                        pass
    return cache


def save_progress_entry(fh, key: str, entry: dict):
    entry["key"] = key
    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    fh.flush()


def prediction_cache_key(
    model_label: str,
    pred_id: str,
    question: str,
    gold: str,
    prediction: str,
) -> str:
    """Key cached judgments to the exact prediction and judging protocol."""
    payload = {
        "schema": 2,
        "model_label": model_label,
        "pred_id": str(pred_id),
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "judge_models": JUDGE_MODELS,
        "temperature": TEMPERATURE,
        "ensemble_method": ENSEMBLE_METHOD,
        "accept_threshold": ACCEPT_THRESHOLD,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return f"{model_label}|||{pred_id}|||{digest}"


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_model_predictions(
    model_label: str,
    predictions: list[dict],
    progress_cache: dict,
    progress_fh,
) -> dict:
    """
    Score every prediction for one model.
    Returns a dict with per-item results and aggregate stats.
    """
    item_results = []

    for pred_item in tqdm(predictions, desc=model_label[:45], ncols=90, leave=False):
        pred_id   = pred_item.get("id", "?")
        question  = pred_item.get("question", "")
        gold      = pred_item.get("gold", "")
        pred_text = pred_item.get("pred", "")
        cache_key = prediction_cache_key(
            model_label, pred_id, question, gold, pred_text
        )

        # Resume from checkpoint
        if cache_key in progress_cache:
            item_results.append(progress_cache[cache_key])
            continue

        judge_scores = []
        judge_details = {}
        for judge_model in JUDGE_MODELS:
            result = call_judge(judge_model, question, gold, pred_text)
            if result is not None:
                judge_scores.append(result["score"])
                judge_details[judge_model] = result
            time.sleep(REQUEST_DELAY)

        final = ensemble_score(judge_scores) if judge_scores else 1
        acceptable = int(final >= ACCEPT_THRESHOLD)

        entry = {
            "id":         pred_id,
            "question":   question,
            "gold":       gold,
            "pred":       pred_text,
            "act":        pred_item.get("act", ""),
            "em":         pred_item.get("em", 0.0),
            "f1":         pred_item.get("f1", 0.0),
            "char_f1":    pred_item.get("char_f1", 0.0),
            "judge_scores": judge_details,
            "final_score":  final,
            "acceptable":   acceptable,
        }

        item_results.append(entry)
        save_progress_entry(progress_fh, cache_key, entry)

    # Aggregate
    n = len(item_results)
    if n == 0:
        return {"label": model_label, "n": 0}

    accept_rate = sum(r["acceptable"] for r in item_results) / n * 100
    score_dist  = Counter(r["final_score"] for r in item_results)
    avg_judge_score = sum(r["final_score"] for r in item_results) / n

    # Per-act acceptability
    per_act: dict[str, list] = {}
    for r in item_results:
        per_act.setdefault(r["act"], []).append(r["acceptable"])
    per_act_rate = {act: sum(v) / len(v) * 100 for act, v in per_act.items()}

    return {
        "label":            model_label,
        "n":                n,
        "acceptability_pct": round(accept_rate, 2),
        "avg_judge_score":   round(avg_judge_score, 3),
        "score_distribution": {str(k): v for k, v in sorted(score_dist.items())},
        "per_act_acceptability": {k: round(v, 2) for k, v in sorted(per_act_rate.items())},
        "items":            item_results,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation of QA predictions")
    parser.add_argument("--model_filter", default="", help="Substring filter on model label")
    args = parser.parse_args()

    if not REPORT_IN.exists():
        print(f"[ERROR] {REPORT_IN} not found. Run evaluate_qa.py first.", file=sys.stderr)
        sys.exit(1)

    with open(REPORT_IN, encoding="utf-8") as f:
        report = json.load(f)

    all_predictions: dict[str, list] = report.get("predictions", {})
    results_summary: list[dict]       = report.get("results_summary", [])

    if not all_predictions:
        print("[ERROR] No 'predictions' key in qa_eval_report.json.", file=sys.stderr)
        print("        Re-run evaluate_qa.py to regenerate with per-item predictions.", file=sys.stderr)
        sys.exit(1)

    # Build label → predictions map; filter if requested
    model_labels = list(all_predictions.keys())
    if args.model_filter:
        model_labels = [l for l in model_labels if args.model_filter in l]
    if not model_labels:
        print(f"[ERROR] No models matched filter '{args.model_filter}'", file=sys.stderr)
        sys.exit(1)

    print(f"LLM-as-Judge Evaluation")
    print(f"  Judges    : {', '.join(JUDGE_MODELS)}")
    print(f"  Threshold : score >= {ACCEPT_THRESHOLD} → Acceptable")
    print(f"  Models    : {len(model_labels)}")
    for l in model_labels:
        print(f"    {l.replace(chr(8594), '->')}")
    print()

    progress_cache = load_progress()
    print(f"  Resumed {len(progress_cache)} cached items from {PROGRESS_FILE.name}")

    all_results = []
    with open(PROGRESS_FILE, "a", encoding="utf-8") as progress_fh:
        for label in model_labels:
            preds = all_predictions[label]
            print(f"\n{'='*65}")
            print(f"  {label.replace(chr(8594), '->')}")
            print(f"  {len(preds)} predictions")
            print(f"{'='*65}")
            result = evaluate_model_predictions(label, preds, progress_cache, progress_fh)
            all_results.append({k: v for k, v in result.items() if k != "items"})
            print(
                f"  Acceptability : {result['acceptability_pct']:.2f}%"
                f"  AvgScore : {result['avg_judge_score']:.3f}"
                f"  ScoreDist : {result['score_distribution']}"
            )

    # Build final report (no per-item to keep file manageable — write separate detailed file)
    summary_sorted = sorted(all_results, key=lambda x: x.get("acceptability_pct", 0), reverse=True)

    # Print ranked table
    print(f"\n{'='*78}")
    print(f"  {'Rank':<5} {'Model':<46} {'Accept%':>8}  {'AvgScore':>9}")
    print(f"  {'-'*5} {'-'*46} {'-'*8}  {'-'*9}")
    for rank, res in enumerate(summary_sorted, 1):
        lbl = res["label"].replace(chr(8594), "->")[:46]
        print(f"  {rank:<5} {lbl:<46} {res['acceptability_pct']:>8.2f}  {res['avg_judge_score']:>9.3f}")

    # Enrich results with existing EM/F1 for easy comparison
    em_f1_map = {r["label"]: r for r in results_summary}
    for res in summary_sorted:
        lbl = res["label"]
        if lbl in em_f1_map:
            res["em"]       = em_f1_map[lbl]["em"]
            res["f1"]       = em_f1_map[lbl]["f1"]
            res["char_f1"]  = em_f1_map[lbl]["char_f1"]

    final_report = {
        "source_report":     str(REPORT_IN.relative_to(BASE_DIR)),
        "source_protocol":   report.get("inference_protocol", {}),
        "cache_schema":      2,
        "judge_models":     JUDGE_MODELS,
        "ensemble_method":  ENSEMBLE_METHOD,
        "accept_threshold": ACCEPT_THRESHOLD,
        "n_models":         len(all_results),
        "results_ranked":   summary_sorted,
    }

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report saved: {REPORT_OUT}")
    print(f"  Progress   : {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
