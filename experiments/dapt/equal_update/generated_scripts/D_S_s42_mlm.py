"""
Continued MLM Pretraining on Original IPR Contexts Only

This script is for the ablation:
  original-only pretraining  vs  original+synthetic pretraining

It uses the same mixed corpus file as `pretrain_mlm.py`, but filters to:
  source == "original"

Base model : law-ai/InLegalBERT
Input      : data/dapt/pretraining_contexts_original_only.jsonl
Output     : data/dapt/pretrained_model_original_only/final/
"""

import hashlib
import json
import warnings
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainingArguments,
    Trainer,
    set_seed as set_global_seed,
)
from transformers.trainer_utils import get_last_checkpoint

warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"E:\Dataset on IPR")

import sys
sys.path.insert(0, str(BASE_DIR))
from scripts.dapt_equal_update.telemetry import (
    TokenCountingTrainer, TrainingTelemetry,
)
INPUT_FILE = Path(r"E:\Dataset on IPR\experiments\dapt\final_cpu\corpus_S.jsonl")
OUTPUT_DIR = Path(r"E:\Dataset on IPR\artifacts\models\equal_update\D_S_s42_mlm")
CACHE_DIR  = BASE_DIR / "experiments" / "dapt" / "final_cpu" / "cache"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BASE_MODEL       = "law-ai/InLegalBERT"
MAX_SEQ_LENGTH   = 256
MLM_PROBABILITY  = 0.15
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS       = 3
MAX_STEPS       = 100
LEARNING_RATE    = 2e-5
WEIGHT_DECAY     = 0.01
SAVE_STEPS       = 500
LOGGING_STEPS = 10
WARMUP_STEPS = 10
SEED = 42
RECOVERY_SAVE_STEPS = 25
DATALOADER_WORKERS = 0


class RecoveryCheckpointCallback(TrainerCallback):
    """Save resumable checkpoints between epoch-level scientific checkpoints."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % RECOVERY_SAVE_STEPS == 0:
            control.should_save = True
        return control


set_global_seed(SEED)


def load_original_contexts(path):
    texts = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            context = obj.get("context", "").strip()
            if context:
                texts.append(context)

    return texts


print("Loading pretraining corpus (original contexts only) ...")
texts = load_original_contexts(INPUT_FILE)
print(f"  Original train contexts kept: {len(texts):,}")

if not texts:
    raise ValueError(
        "No original contexts found. First run build_original_pretraining_contexts.py."
    )

# ── Tokenizer ──────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ── Tokenise ───────────────────────────────────────────────────────────────────
print("Tokenising ...")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
corpus_key = hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest()[:12]
token_cache = CACHE_DIR / (
    f"mlm_{corpus_key}_len{MAX_SEQ_LENGTH}.arrow"
)
raw_dataset = Dataset.from_dict({"text": texts})
tokenized = raw_dataset.map(
    lambda batch: tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        return_special_tokens_mask=True,
    ),
    batched=True,
    remove_columns=["text"],
    desc="Tokenising",
    cache_file_name=str(token_cache),
)
print(f"  {len(tokenized):,} examples after tokenisation")

# ── Data collator (dynamic masking) ───────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROBABILITY,
)

# ── Model ──────────────────────────────────────────────────────────────────────
print(f"\nLoading model: {BASE_MODEL}")
model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL, ignore_mismatched_sizes=True)
# The published InLegalBERT checkpoint stores distinct input-embedding and MLM
# decoder weights even though its legacy config says they are tied. Preserve the
# actual checkpoint architecture and make the saved DAPT config truthful.
model.config.tie_word_embeddings = False
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

# ── Training arguments ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="linear",
    fp16=False,
    bf16=False,
    use_cpu=True,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    dataloader_num_workers=DATALOADER_WORKERS,
    seed=SEED,
    data_seed=SEED,
    report_to="none",
)

# ── Trainer ────────────────────────────────────────────────────────────────────
telemetry = TrainingTelemetry(
    OUTPUT_DIR / "training_telemetry.json",
    configured={
        "optimizer_updates": MAX_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "peak_learning_rate": LEARNING_RATE,
        "scheduler": "linear",
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "seed": SEED,
        "corpus_sha256": hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest(),
    },
)
trainer = TokenCountingTrainer(
    telemetry=telemetry,
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[RecoveryCheckpointCallback(), telemetry],
)

# ── Train ──────────────────────────────────────────────────────────────────────
print("\nStarting continued pretraining (original-only) ...")
print(f"  Effective batch size : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Max seq length       : {MAX_SEQ_LENGTH}")
print(f"  Optimizer updates    : {MAX_STEPS}")
print(f"  Warmup updates       : {WARMUP_STEPS}")
print(f"  Output               : {OUTPUT_DIR}\n")

last_checkpoint = get_last_checkpoint(str(OUTPUT_DIR)) if OUTPUT_DIR.exists() else None
if last_checkpoint:
    print(f"  Resuming MLM from: {last_checkpoint}")
trainer.train(resume_from_checkpoint=last_checkpoint)

# ── Save final model ───────────────────────────────────────────────────────────
final_dir = OUTPUT_DIR / "final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"\nFinal model saved to: {final_dir}")
print("Next step: fine-tune this model on SQuAD v2, then on IPR QA.")
