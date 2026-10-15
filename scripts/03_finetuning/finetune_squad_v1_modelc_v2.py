"""
Step 2 (Model C v2) - Fine-tune on SQuAD v1 (Intermediate QA Training)
Base model : data/dapt/pretrained_model_v2/final
Dataset    : rajpurkar/squad (SQuAD v1 - all questions answerable, matches IPR)
Objective  : Teach the model QA span extraction format
Output     : artifacts/models/squad_v1/squad_v1_modelc_v2/squad_v1/final
"""

import warnings
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data" / "dapt" / "pretrained_model_v2" / "final"
OUTPUT_DIR = BASE_DIR / "artifacts" / "models" / "squad_v1" / "squad_v1_modelc_v2" / "squad_v1"

MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
TRAIN_SUBSET = 20_000
VAL_SUBSET = 2_000
BATCH_SIZE = 8
GRAD_ACCUM = 4
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
LOGGING_STEPS = 50
SEED = 42

print(f"Loading tokenizer and model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(
    str(MODEL_DIR), ignore_mismatched_sizes=True, local_files_only=True
)
print(f"  Parameters: {sum(param.numel() for param in model.parameters()) / 1e6:.1f} M")

print("\nLoading SQuAD v1 dataset ...")
squad = load_dataset("rajpurkar/squad")
train_data = squad["train"].shuffle(seed=SEED).select(range(TRAIN_SUBSET))
val_data = squad["validation"].shuffle(seed=SEED).select(range(VAL_SUBSET))
print(f"  Train subset : {len(train_data):,}")
print(f"  Val subset   : {len(val_data):,}")


def preprocess(examples):
    questions = [question.strip() for question in examples["question"]]
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

        ctx_start = next(j for j, seq_id in enumerate(sequence_ids) if seq_id == 1)
        ctx_end = len(sequence_ids) - 1 - next(
            j for j, seq_id in enumerate(reversed(sequence_ids)) if seq_id == 1
        )

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


print("\nTokenising SQuAD v1 ...")
train_tokenized = train_data.map(
    preprocess,
    batched=True,
    remove_columns=train_data.column_names,
    desc="Train",
)
val_tokenized = val_data.map(
    preprocess,
    batched=True,
    remove_columns=val_data.column_names,
    desc="Val",
)
print(f"  Train examples : {len(train_tokenized):,}")
print(f"  Val examples   : {len(val_tokenized):,}")

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DefaultDataCollator(),
    processing_class=tokenizer,
)

print("\nStarting SQuAD v1 fine-tuning (Model C v2) ...")
print(f"  Base model       : {MODEL_DIR}")
print(f"  Effective batch  : {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Epochs           : {NUM_EPOCHS}")
print(f"  Output           : {OUTPUT_DIR}\n")

trainer.train()

final_dir = OUTPUT_DIR / "final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"\nFinal model saved to: {final_dir}")
print("Next step: run finetune_ipr_modelc_v2.py")
