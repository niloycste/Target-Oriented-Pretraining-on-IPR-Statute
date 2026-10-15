"""
Continued MLM Pretraining on Synthetic Contexts Only  (Model I')

Ablation:
  H' : Original-only  MLM  ->  SQuAD v1  ->  IPR
  C' : Original+Synth  MLM  ->  SQuAD v1  ->  IPR
  I' : Synth-only      MLM  ->  SQuAD v1  ->  IPR   <-- this script

Base model  : law-ai/InLegalBERT
Input       : data/dapt/pretraining_contexts_synth_only.jsonl
              (13,211 synthetic contexts, field: context — same format as pretraining_contexts_v2.jsonl)
Output      : data/dapt/pretrained_model_synth_only/
"""

import json
import warnings
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "dapt" / "pretraining_contexts_synth_only.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "dapt" / "pretrained_model_synth_only"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BASE_MODEL       = "law-ai/InLegalBERT"
MAX_SEQ_LENGTH   = 256
MLM_PROBABILITY  = 0.15
BATCH_SIZE       = 8
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS       = 3
LEARNING_RATE    = 2e-5
WEIGHT_DECAY     = 0.01
SAVE_STEPS       = 500
LOGGING_STEPS    = 50
WARMUP_STEPS     = 100
SEED             = 42


def load_contexts(path):
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


print("Loading synthetic-only pretraining corpus ...")
texts = load_contexts(INPUT_FILE)
print(f"  Synthetic contexts loaded: {len(texts):,}")

if not texts:
    raise ValueError(f"No contexts found. Check file: {INPUT_FILE}")

# ── Tokenizer ──────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ── Tokenise ───────────────────────────────────────────────────────────────────
print("Tokenising ...")
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
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

# ── Training arguments ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
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
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    seed=SEED,
    data_seed=SEED,
    report_to="none",
)

# ── Trainer ────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ── Train ──────────────────────────────────────────────────────────────────────
print("\nStarting continued pretraining (synthetic-only, Model I') ...")
print(f"  Corpus                : synthetic contexts only ({len(texts):,} passages)")
print(f"  Effective batch size  : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Max seq length        : {MAX_SEQ_LENGTH}")
print(f"  Epochs                : {NUM_EPOCHS}")
print(f"  Output                : {OUTPUT_DIR}\n")

trainer.train()

# ── Save final model ───────────────────────────────────────────────────────────
final_dir = OUTPUT_DIR / "final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"\nFinal model saved to: {final_dir}")
print("Next step: run finetune_squad_v1_synth_only.py")
