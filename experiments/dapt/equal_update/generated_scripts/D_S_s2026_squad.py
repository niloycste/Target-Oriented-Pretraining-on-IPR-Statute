"""
Step 2 — Fine-tune on SQuAD v1 (Original-only MLM ablation)
Base model : data/dapt/pretrained_model_original_only/final
Dataset    : rajpurkar/squad  (SQuAD v1 — all questions answerable, matches IPR dataset)
Objective  : Teach the model QA span extraction format
Output     : artifacts/models/squad_v1/original_only_squadv1/squad_v1/final/

Note: Using SQuAD v1 (not v2) because our IPR dataset has no unanswerable questions.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datasets import Dataset, DatasetDict, config as datasets_config, load_dataset
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

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"E:\Dataset on IPR")
MODEL_DIR  = Path(r"E:\Dataset on IPR\artifacts\models\equal_update\D_S_s2026_mlm\final")
OUTPUT_DIR = Path(r"E:\Dataset on IPR\artifacts\models\equal_update\D_S_s2026_squad")
CACHE_DIR  = BASE_DIR / "experiments" / "dapt" / "final_cpu" / "cache"

# ── Hyperparameters ─────────────────────────────────────────────────────────
MAX_SEQ_LENGTH  = 384
DOC_STRIDE      = 128
TRAIN_SUBSET    = 20_000
VAL_SUBSET      = 2_000
BATCH_SIZE = 16
GRAD_ACCUM = 2
NUM_EPOCHS      = 2
LEARNING_RATE   = 3e-5
WEIGHT_DECAY    = 0.01
WARMUP_STEPS    = 200
LOGGING_STEPS   = 50
SEED = 2026
SUBSET_SEED     = 42
RECOVERY_SAVE_STEPS = 100
DATALOADER_WORKERS = 0


class RecoveryCheckpointCallback(TrainerCallback):
    """Save recovery checkpoints without changing epoch-level model selection."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % RECOVERY_SAVE_STEPS == 0:
            control.should_save = True
        return control


def load_squad_v1():
    """Load SQuAD, bypassing a datasets 4.7 cache-lock deadlock when possible."""
    cache_root = Path(datasets_config.HF_DATASETS_CACHE) / "rajpurkar___squad" / "plain_text"
    train_files = sorted(
        cache_root.glob("**/squad-train.arrow"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for train_file in train_files:
        val_file = train_file.with_name("squad-validation.arrow")
        if val_file.exists():
            print(f"  Using local Arrow cache: {train_file.parent}")
            return DatasetDict({
                "train": Dataset.from_file(str(train_file)),
                "validation": Dataset.from_file(str(val_file)),
            })
    return load_dataset("rajpurkar/squad")

# ── Load tokenizer & model ───────────────────────────────────────────────────
set_global_seed(SEED)
print(f"Loading tokenizer and model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(str(MODEL_DIR), ignore_mismatched_sizes=True, local_files_only=True)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

# ── Load SQuAD v1 (all questions answerable — matches our IPR dataset) ────────
print("\nLoading SQuAD v1 dataset …")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
squad = load_squad_v1()
train_data = squad["train"].shuffle(
    seed=SUBSET_SEED,
    indices_cache_file_name=str(CACHE_DIR / f"squad_train_shuffle_s{SUBSET_SEED}.arrow"),
).select(range(TRAIN_SUBSET))
val_data = squad["validation"].shuffle(
    seed=SUBSET_SEED,
    indices_cache_file_name=str(CACHE_DIR / f"squad_val_shuffle_s{SUBSET_SEED}.arrow"),
).select(range(VAL_SUBSET))
print(f"  Train subset : {len(train_data):,}")
print(f"  Val subset   : {len(val_data):,}")


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
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers_list = examples["answers"]

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answers = answers_list[sample_idx]
        sequence_ids = inputs.sequence_ids(i)

        ctx_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
        ctx_end = len(sequence_ids) - 1 - next(
            j for j, s in enumerate(reversed(sequence_ids)) if s == 1
        )

        # SQuAD v1: all questions are answerable — no unanswerable handling needed
        ans_start_char = answers["answer_start"][0]
        ans_end_char = ans_start_char + len(answers["text"][0])

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
    inputs["end_positions"] = end_positions
    return inputs


print("\nTokenising SQuAD v1 …")
train_tokenized = train_data.map(
    preprocess, batched=True, remove_columns=train_data.column_names, desc="Train",
    cache_file_name=str(
        CACHE_DIR / f"squad_train_n{TRAIN_SUBSET}_s{SUBSET_SEED}_l{MAX_SEQ_LENGTH}_d{DOC_STRIDE}.arrow"
    ),
)
val_tokenized = val_data.map(
    preprocess, batched=True, remove_columns=val_data.column_names, desc="Val",
    cache_file_name=str(
        CACHE_DIR / f"squad_val_n{VAL_SUBSET}_s{SUBSET_SEED}_l{MAX_SEQ_LENGTH}_d{DOC_STRIDE}.arrow"
    ),
)
print(f"  Train examples after tokenisation : {len(train_tokenized):,}")
print(f"  Val examples after tokenisation   : {len(val_tokenized):,}")

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DefaultDataCollator(),
    processing_class=tokenizer,
    callbacks=[RecoveryCheckpointCallback()],
)

print("\nStarting SQuAD v1 fine-tuning (original-only ablation) …")
print(f"  Base model       : {MODEL_DIR}")
print(f"  Train examples   : {len(train_tokenized):,}")
print(f"  Effective batch  : {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Epochs           : {NUM_EPOCHS}")
print(f"  Output           : {OUTPUT_DIR}\n")

last_checkpoint = get_last_checkpoint(str(OUTPUT_DIR)) if OUTPUT_DIR.exists() else None
if last_checkpoint:
    print(f"  Resuming SQuAD from: {last_checkpoint}")
trainer.train(resume_from_checkpoint=last_checkpoint)

final_dir = OUTPUT_DIR / "final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"\nFinal model saved to: {final_dir}")
print("Next step: run finetune_ipr_original_only.py")
