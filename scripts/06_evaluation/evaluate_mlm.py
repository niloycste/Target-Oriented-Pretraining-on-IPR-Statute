"""
MLM Evaluation – Perplexity on test-split contexts
4-way comparison:
  1. Original InLegalBERT (no pretraining)
  2. MLM pretrained on Original-only corpus         → pretrained_model_original_only/final  (H' base)
  3. MLM pretrained on Original + Synthetic corpus  → pretrained_model_v2/final             (C' base, 13,211 synth)
  4. MLM pretrained on Synthetic-only corpus        → pretrained_model_synth_only/final     (I' base)

Input  : data/splits/test.jsonl contexts (three contexts overlap H'/C' DAPT corpora)
Metric : Perplexity (lower is better)
"""

import json
import math
import warnings
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[2]
TEST_FILE = BASE_DIR / "data" / "splits" / "test.jsonl"
MODEL_DIR_ORIG_ONLY = BASE_DIR / "data" / "dapt" / "pretrained_model_original_only" / "final"
MODEL_DIR_V2 = BASE_DIR / "data" / "dapt" / "pretrained_model_v2" / "final"
MODEL_DIR_SYNTH_ONLY = BASE_DIR / "data" / "dapt" / "pretrained_model_synth_only" / "final"
RESULTS_DIR = BASE_DIR / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BASE_MODEL           = "law-ai/InLegalBERT"   # original model — compare against this

MAX_SEQ_LENGTH  = 256
MLM_PROBABILITY = 0.15
SEED            = 42

# ── Load test-split contexts ─────────────────────────────────────────────────
print("Loading test-split contexts …")
seen = set()
eval_texts = []
with open(TEST_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ctx = json.loads(line).get("context", "").strip()
        if ctx and ctx not in seen:
            seen.add(ctx)
            eval_texts.append(ctx)

print(f"  Eval contexts (test.jsonl, unique) : {len(eval_texts):,}")

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR_V2), local_files_only=True)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        return_special_tokens_mask=True,
    )

eval_dataset_raw = Dataset.from_dict({"text": eval_texts})
eval_dataset = eval_dataset_raw.map(tokenize, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
)

# ── Evaluation args (CPU, no training) ───────────────────────────────────────
eval_args = TrainingArguments(
    output_dir=str(RESULTS_DIR / "mlm_eval_tmp"),
    per_device_eval_batch_size=8,
    use_cpu=True,
    report_to="none",
    seed=SEED,
)

def evaluate_model(label, model_path):
    print(f"\n{'='*55}")
    print(f"  Evaluating: {label}")
    print(f"  Path      : {model_path}")
    print(f"{'='*55}")
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), ignore_mismatched_sizes=True, local_files_only=True)
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    results = trainer.evaluate()
    loss = results["eval_loss"]
    ppl  = math.exp(loss)
    print(f"\n  Eval Loss   : {loss:.4f}")
    print(f"  Perplexity  : {ppl:.2f}")
    return {"label": label, "loss": loss, "perplexity": ppl}

# ── Run evaluation on all models ─────────────────────────────────────────────
results = []
results.append(evaluate_model("InLegalBERT baseline (no MLM)",              BASE_MODEL))
if MODEL_DIR_ORIG_ONLY.exists():
    results.append(evaluate_model("H' base — MLM Orig-only",                MODEL_DIR_ORIG_ONLY))
else:
    print(f"  [SKIP] pretrained_model_original_only not found at {MODEL_DIR_ORIG_ONLY}")
if MODEL_DIR_V2.exists():
    results.append(evaluate_model("C' base — MLM Orig+Synth (13,211 synth)", MODEL_DIR_V2))
else:
    print(f"  [SKIP] pretrained_model_v2 not found at {MODEL_DIR_V2}")
if MODEL_DIR_SYNTH_ONLY.exists():
    results.append(evaluate_model("I' base — MLM Synth-only (13,211 synth)", MODEL_DIR_SYNTH_ONLY))
else:
    print(f"  [SKIP] pretrained_model_synth_only not found at {MODEL_DIR_SYNTH_ONLY}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  SUMMARY")
print("="*55)
print(f"  {'Model':<38} {'Loss':>7}  {'PPL':>8}")
print(f"  {'-'*38}  {'-'*7}  {'-'*8}")
for r in results:
    print(f"  {r['label']:<38} {r['loss']:>7.4f}  {r['perplexity']:>8.2f}")

baseline_ppl = results[0]["perplexity"]
best = min(results, key=lambda r: r["perplexity"])
improvement = baseline_ppl - best["perplexity"]
if improvement > 0:
    print(f"\n  Best model : {best['label']}")
    print(f"  Improvement over baseline: {improvement:.2f} PPL points.")
else:
    print(f"\n  No improvement over baseline — consider more pretraining.")
print()

# ── Save results to JSON ──────────────────────────────────────────────────────
import datetime
report = {
    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "eval_source": "test.jsonl contexts (three contexts overlap H'/C' DAPT corpora; see leakage audit)",
    "eval_contexts": len(eval_texts),
    "max_seq_length": MAX_SEQ_LENGTH,
    "mlm_probability": MLM_PROBABILITY,
    "note": "Eval data is test.jsonl contexts — not seen during MLM pretraining. Fair 3-way comparison.",
    "results": results,
    "best_model": best["label"],
    "improvement_over_baseline_ppl": round(improvement, 4),
}
report_path = RESULTS_DIR / "mlm_eval_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"  Report saved to: {report_path}")
