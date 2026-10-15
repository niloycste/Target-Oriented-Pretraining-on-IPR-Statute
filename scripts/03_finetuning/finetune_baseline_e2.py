"""
Model E2' — Baseline: LegalBERT(US) → SQuAD v1 → IPR  (no domain pretraining)
Step 1: Fine-tune LegalBERT(US) on SQuAD v1  → saved to baseline_e2_squadv1/squad/
Step 2: Fine-tune that model on IPR data     → saved to baseline_e2_squadv1/final/
"""

import json, string, collections, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    DefaultDataCollator, TrainingArguments, Trainer,
)
import torch

BASE_DIR   = Path(__file__).resolve().parents[2]
TRAIN_FILE = BASE_DIR / "data" / "splits" / "train_squad.jsonl"
VAL_FILE = BASE_DIR / "data" / "splits" / "val_squad.jsonl"
BASE_MODEL = "nlpaueb/legal-bert-base-uncased"  # US LegalBERT baseline
OUTPUT_DIR = BASE_DIR / "artifacts" / "models" / "squad_v1" / "baseline_e2_squadv1"

MAX_SEQ_LENGTH = 384
DOC_STRIDE     = 128
BATCH_SIZE     = 8
GRAD_ACCUM     = 4
SEED           = 42

# ── Shared tokenize helper ────────────────────────────────────────────────────
def make_preprocess_fn(tokenizer):
    def preprocess(examples):
        inputs = tokenizer(
            [q.strip() for q in examples["question"]],
            examples["context"],
            max_length=MAX_SEQ_LENGTH,
            truncation="only_second",
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map     = inputs.pop("overflow_to_sample_mapping")
        answers_list   = examples["answers"]
        start_positions, end_positions = [], []

        for i, offset in enumerate(offset_mapping):
            sample_idx   = sample_map[i]
            answers      = answers_list[sample_idx]
            sequence_ids = inputs.sequence_ids(i)
            ctx_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
            ctx_end   = len(sequence_ids) - 1 - next(j for j, s in enumerate(reversed(sequence_ids)) if s == 1)

            # SQuAD v1: all questions are answerable — no unanswerable handling needed
            ans_start_char = answers["answer_start"][0]
            ans_end_char   = ans_start_char + len(answers["text"][0])

            if offset[ctx_start][0] > ans_end_char or offset[ctx_end][1] < ans_start_char:
                start_positions.append(0); end_positions.append(0); continue

            tok_start = ctx_start
            while tok_start <= ctx_end and offset[tok_start][0] <= ans_start_char:
                tok_start += 1
            start_positions.append(tok_start - 1)

            tok_end = ctx_end
            while tok_end >= ctx_start and offset[tok_end][1] >= ans_end_char:
                tok_end -= 1
            end_positions.append(tok_end + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"]   = end_positions
        return inputs
    return preprocess

def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def compute_em_f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common      = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same    = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0
    p  = num_same / len(pred_tokens)
    r  = num_same / len(gold_tokens)
    f1 = 2 * p * r / (p + r)
    return float(normalize(pred) == normalize(gold)), f1

def make_qa_trainer_class(tokenizer_ref, val_records_ref):
    class QATrainer(Trainer):
        def evaluate(self, **kwargs):
            results = super().evaluate(**kwargs)
            self.model.eval()
            em_scores, f1_scores = [], []
            for record in val_records_ref:
                enc = tokenizer_ref(record["question"], record["context"],
                                    return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
                with torch.no_grad():
                    out = self.model(**enc)
                start = out.start_logits.argmax().item()
                end   = out.end_logits.argmax().item() + 1
                pred  = tokenizer_ref.convert_tokens_to_string(
                    tokenizer_ref.convert_ids_to_tokens(enc["input_ids"][0][start:end]))
                em, f1 = compute_em_f1(pred, record["answers"]["text"][0])
                em_scores.append(em); f1_scores.append(f1)
            avg_em = sum(em_scores) / len(em_scores) * 100
            avg_f1 = sum(f1_scores) / len(f1_scores) * 100
            print(f"\n  ── Val EM : {avg_em:.2f}%   F1 : {avg_f1:.2f}% ──\n")
            results["eval_em"] = avg_em
            results["eval_f1"] = avg_f1
            return results
    return QATrainer

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — InLegalBERT → SQuAD v1
# ════════════════════════════════════════════════════════════════════════════════
squad_dir = OUTPUT_DIR / "squad"

print("=" * 60)
print("  STEP 1 — Fine-tuning InLegalBERT on SQuAD v1")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model     = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL, ignore_mismatched_sizes=True)

squad = load_dataset("rajpurkar/squad")
train_squad = squad["train"].shuffle(seed=SEED).select(range(20_000))
val_squad   = squad["validation"].shuffle(seed=SEED).select(range(2_000))

preprocess = make_preprocess_fn(tokenizer)
train_sq_tok = train_squad.map(preprocess, batched=True, remove_columns=train_squad.column_names, desc="SQuAD Train")
val_sq_tok   = val_squad.map(preprocess,   batched=True, remove_columns=val_squad.column_names,   desc="SQuAD Val")

sq_args = TrainingArguments(
    output_dir=str(squad_dir),
    num_train_epochs=2,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=200,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    use_cpu=True, fp16=False, bf16=False,
    logging_steps=50,
    report_to="none",
    seed=SEED, data_seed=SEED,
)

Trainer(
    model=model, args=sq_args,
    train_dataset=train_sq_tok, eval_dataset=val_sq_tok,
    data_collator=DefaultDataCollator(),
    processing_class=tokenizer,
).train()

squad_final = squad_dir / "final"
model.save_pretrained(str(squad_final))
tokenizer.save_pretrained(str(squad_final))
print(f"\n  SQuAD v1 model saved to: {squad_final}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — SQuAD v1 model → IPR fine-tuning
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2 — Fine-tuning on IPR QA data")
print("=" * 60)

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

train_records = load_jsonl(TRAIN_FILE)
val_records   = load_jsonl(VAL_FILE)

def records_to_dataset(records):
    return Dataset.from_dict({
        "id":       [r["id"]       for r in records],
        "context":  [r["context"]  for r in records],
        "question": [r["question"] for r in records],
        "answers":  [r["answers"]  for r in records],
    })

tokenizer = AutoTokenizer.from_pretrained(str(squad_final), local_files_only=True)
model     = AutoModelForQuestionAnswering.from_pretrained(str(squad_final), local_files_only=True)

preprocess = make_preprocess_fn(tokenizer)
train_tok  = records_to_dataset(train_records).map(preprocess, batched=True,
             remove_columns=["id","context","question","answers"], desc="IPR Train")
val_tok    = records_to_dataset(val_records).map(preprocess, batched=True,
             remove_columns=["id","context","question","answers"], desc="IPR Val")

ipr_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "ipr"),
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    save_total_limit=2,
    use_cpu=True, fp16=False, bf16=False,
    logging_steps=20,
    report_to="none",
    seed=SEED, data_seed=SEED,
)

QATrainer = make_qa_trainer_class(tokenizer, val_records)
QATrainer(
    model=model, args=ipr_args,
    train_dataset=train_tok, eval_dataset=val_tok,
    data_collator=DefaultDataCollator(),
    processing_class=tokenizer,
).train()

final_dir = OUTPUT_DIR / "final"
model.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"\nModel E2 saved to: {final_dir}")
