"""
Step 3 (Model I') — Fine-tune on IPR QA Data

Ablation pipeline:
  I' : InLegalBERT  ->  MLM(Synth-only)  ->  SQuAD v1  ->  IPR   <-- this script is step 3

Base model  : artifacts/models/squad_v1/squad_v1_synth_only/squad_v1/final
Train data  : data/splits/train_squad.jsonl  (3,710 pairs)
Val data    : data/splits/val_squad.jsonl    (408 pairs)
Output      : artifacts/models/squad_v1/squad_v1_synth_only/ipr_final/final/
"""

import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
TRAIN_FILE = BASE_DIR / "data" / "splits" / "train_squad.jsonl"
VAL_FILE = BASE_DIR / "data" / "splits" / "val_squad.jsonl"
MODEL_DIR = BASE_DIR / "artifacts" / "models" / "squad_v1" / "squad_v1_synth_only" / "squad_v1" / "final"
OUTPUT_DIR = BASE_DIR / "artifacts" / "models" / "squad_v1" / "squad_v1_synth_only" / "ipr_final"

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 384
DOC_STRIDE     = 128
BATCH_SIZE     = 8
GRAD_ACCUM     = 4
NUM_EPOCHS     = 3
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 0.01
WARMUP_STEPS   = 100
SAVE_STEPS     = 200
LOGGING_STEPS  = 50
SEED           = 42

# ── Load data ─────────────────────────────────────────────────────────────────
def load_squad_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

print("Loading IPR QA data ...")
train_rows = load_squad_jsonl(TRAIN_FILE)
val_rows   = load_squad_jsonl(VAL_FILE)
print(f"  Train : {len(train_rows):,}")
print(f"  Val   : {len(val_rows):,}")

# ── Tokenizer & model ─────────────────────────────────────────────────────────
print(f"\nLoading model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForQuestionAnswering.from_pretrained(str(MODEL_DIR), ignore_mismatched_sizes=True)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

# ── Preprocessing ─────────────────────────────────────────────────────────────
def rows_to_hf(rows):
    return Dataset.from_dict({
        "id":       [r["id"] for r in rows],
        "context":  [r["context"] for r in rows],
        "question": [r["question"] for r in rows],
        "answers":  [r["answers"] for r in rows],
    })

def preprocess(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
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

    start_positions = []
    end_positions   = []

    for i, offset in enumerate(offset_mapping):
        sample_idx   = sample_map[i]
        answers      = answers_list[sample_idx]
        sequence_ids = inputs.sequence_ids(i)

        ctx_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
        ctx_end   = len(sequence_ids) - 1 - next(
            j for j, s in enumerate(reversed(sequence_ids)) if s == 1
        )

        if len(answers["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        ans_start_char = answers["answer_start"][0]
        ans_end_char   = ans_start_char + len(answers["text"][0])

        if offset[ctx_start][0] > ans_end_char or offset[ctx_end][1] < ans_start_char:
            start_positions.append(0)
            end_positions.append(0)
            continue

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

print("\nTokenising IPR data ...")
train_ds = rows_to_hf(train_rows)
val_ds   = rows_to_hf(val_rows)

train_tokenized = train_ds.map(
    preprocess, batched=True, remove_columns=train_ds.column_names, desc="Train"
)
val_tokenized = val_ds.map(
    preprocess, batched=True, remove_columns=val_ds.column_names, desc="Val"
)
print(f"  Train examples : {len(train_tokenized):,}")
print(f"  Val examples   : {len(val_tokenized):,}")

# ── Training arguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    use_cpu=True,
    fp16=False,
    bf16=False,
    logging_steps=LOGGING_STEPS,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DefaultDataCollator(),
    processing_class=tokenizer,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting IPR fine-tuning (Model I') ...")
print(f"  Base model      : {MODEL_DIR}")
print(f"  Effective batch : {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Epochs          : {NUM_EPOCHS}")
print(f"  Output          : {OUTPUT_DIR}\n")

trainer.train()

# ── Save final model ──────────────────────────────────────────────────────────
final_dir = OUTPUT_DIR / "final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"\nFinal model saved to: {final_dir}")
print("Next step: run evaluate_qa.py (Model I' will be compared against H' and C')")
