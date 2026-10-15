"""
Step 3 — Fine-tune on IPR QA Data (Original-only MLM ablation)
Base model : artifacts/models/squad_v1/original_only_squadv1/squad_v1/final
Train data : data/splits/train_squad.jsonl
Val data   : data/splits/val_squad.jsonl
Output     : artifacts/models/squad_v1/original_only_squadv1/ipr_final/final
Metrics    : Exact Match (EM) + F1 on val set after each epoch
"""

import hashlib
import json
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    TrainerCallback,
    TrainingArguments,
    Trainer,
    set_seed as set_global_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"E:\Dataset on IPR")
TRAIN_FILE = BASE_DIR / "data" / "splits" / "train_squad.jsonl"
VAL_FILE = BASE_DIR / "data" / "splits" / "val_squad.jsonl"
MODEL_DIR  = Path(r"E:\Dataset on IPR\artifacts\models\equal_update\D_A_s42_squad\final")
OUTPUT_DIR = Path(r"E:\Dataset on IPR\artifacts\models\equal_update\D_A_s42_ipr")
CACHE_DIR  = BASE_DIR / "experiments" / "dapt" / "final_cpu" / "cache"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.qa_evaluation import (
    QADecodingConfig,
    compute_em_f1 as canonical_compute_em_f1,
    predict_answers,
)

# ── Hyperparameters ──────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 384
DOC_STRIDE     = 128
BATCH_SIZE = 16
GRAD_ACCUM = 2
NUM_EPOCHS     = 3
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 0.01
WARMUP_STEPS   = 50
LOGGING_STEPS  = 20
SEED = 42
RECOVERY_SAVE_STEPS = 100
DATALOADER_WORKERS = 0


class RecoveryCheckpointCallback(TrainerCallback):
    """Save recovery checkpoints without changing epoch-level model selection."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % RECOVERY_SAVE_STEPS == 0:
            control.should_save = True
        return control


set_global_seed(SEED)


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


print("Loading IPR QA data …")
train_records = load_jsonl(TRAIN_FILE)
val_records   = load_jsonl(VAL_FILE)
print(f"  Train : {len(train_records):,}")
print(f"  Val   : {len(val_records):,}")


def records_to_dataset(records):
    return Dataset.from_dict({
        "id":       [r["id"]       for r in records],
        "context":  [r["context"]  for r in records],
        "question": [r["question"] for r in records],
        "answers":  [r["answers"]  for r in records],
    })


train_ds = records_to_dataset(train_records)
val_ds   = records_to_dataset(val_records)

print(f"\nLoading model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model     = AutoModelForQuestionAnswering.from_pretrained(str(MODEL_DIR), ignore_mismatched_sizes=True, local_files_only=True)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")


def preprocess_train(examples):
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

    start_positions, end_positions = [], []

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


def preprocess_val(examples):
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
    sample_map = inputs["overflow_to_sample_mapping"]
    inputs["example_id"] = [examples["id"][sample_map[i]] for i in range(len(sample_map))]

    offset_mapping = inputs["offset_mapping"]
    answers_list   = examples["answers"]
    start_positions, end_positions = [], []

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


print("\nTokenising …")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
train_key = hashlib.sha256(TRAIN_FILE.read_bytes()).hexdigest()[:12]
val_key = hashlib.sha256(VAL_FILE.read_bytes()).hexdigest()[:12]
train_tokenized = train_ds.map(
    preprocess_train, batched=True, remove_columns=train_ds.column_names, desc="Train",
    cache_file_name=str(
        CACHE_DIR / f"ipr_train_{train_key}_l{MAX_SEQ_LENGTH}_d{DOC_STRIDE}.arrow"
    ),
)
val_tokenized = val_ds.map(
    preprocess_val, batched=True, remove_columns=val_ds.column_names, desc="Val",
    cache_file_name=str(
        CACHE_DIR / f"ipr_val_{val_key}_l{MAX_SEQ_LENGTH}_d{DOC_STRIDE}.arrow"
    ),
)
print(f"  Train examples : {len(train_tokenized):,}")
print(f"  Val examples   : {len(val_tokenized):,}")


class QATrainer(Trainer):
    def __init__(self, val_records, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validation_rows = list(val_records)

    def evaluate(self, **kwargs):
        results = super().evaluate(**kwargs)
        decoded = predict_answers(
            tokenizer,
            self.model,
            self.args.device,
            [row["question"] for row in self._validation_rows],
            [row["context"] for row in self._validation_rows],
            QADecodingConfig(
                max_seq_length=MAX_SEQ_LENGTH,
                doc_stride=DOC_STRIDE,
                batch_size=BATCH_SIZE,
                max_answer_tokens=None,
            ),
        )
        em_scores, f1_scores = [], []
        for row, (prediction, _score) in zip(self._validation_rows, decoded):
            gold = row["answers"]["text"][0]
            em, f1 = canonical_compute_em_f1(prediction, gold)
            em_scores.append(em)
            f1_scores.append(f1)

        avg_em = sum(em_scores) / len(em_scores) * 100 if em_scores else 0
        avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0
        print(f"\n  ── Val EM : {avg_em:.2f}%   F1 : {avg_f1:.2f}% ──\n")
        results["eval_em"] = avg_em
        results["eval_f1"] = avg_f1
        return results


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
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    dataloader_num_workers=DATALOADER_WORKERS,
    use_cpu=True,
    fp16=False,
    bf16=False,
    logging_steps=LOGGING_STEPS,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
)

trainer = QATrainer(
    val_records=val_records,
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DefaultDataCollator(),
    processing_class=tokenizer,
    callbacks=[RecoveryCheckpointCallback()],
)

print("\nStarting IPR QA fine-tuning (original-only ablation) …")
print(f"  Base model      : {MODEL_DIR}")
print(f"  Train examples  : {len(train_tokenized):,}")
print(f"  Val examples    : {len(val_tokenized):,}")
print(f"  Effective batch : {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Epochs          : {NUM_EPOCHS}")
print(f"  Output          : {OUTPUT_DIR}\n")

last_checkpoint = get_last_checkpoint(str(OUTPUT_DIR)) if OUTPUT_DIR.exists() else None
if last_checkpoint:
    print(f"  Resuming IPR from: {last_checkpoint}")
trainer.train(resume_from_checkpoint=last_checkpoint)

final_dir = OUTPUT_DIR / "final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"\nFinal model saved to: {final_dir}")
print("Next step: add this model to evaluate_qa.py for ablation comparison.")
