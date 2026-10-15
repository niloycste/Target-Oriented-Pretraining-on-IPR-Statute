"""
Model D — Baseline: BERT-base-uncased → IPR direct (no SQuAD, no pretraining)
Train data : data/splits/train_squad.jsonl  (3,710 pairs)
Val data   : data/splits/val_squad.jsonl    (408 pairs)
Output     : artifacts/models/direct/baseline_d/
"""

import json, string, collections, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    DefaultDataCollator, TrainingArguments, Trainer,
)
import torch

BASE_DIR   = Path(__file__).resolve().parents[2]
TRAIN_FILE = BASE_DIR / "data" / "splits" / "train_squad.jsonl"
VAL_FILE = BASE_DIR / "data" / "splits" / "val_squad.jsonl"
BASE_MODEL = "bert-base-uncased"         # general BERT baseline
OUTPUT_DIR = BASE_DIR / "artifacts" / "models" / "direct" / "baseline_d"

MAX_SEQ_LENGTH = 384
DOC_STRIDE     = 128
BATCH_SIZE     = 8
GRAD_ACCUM     = 4
NUM_EPOCHS     = 3
LEARNING_RATE  = 3e-5
WEIGHT_DECAY   = 0.01
WARMUP_STEPS   = 50
LOGGING_STEPS  = 20
SEED           = 42

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
print(f"  Train : {len(train_records):,}  |  Val : {len(val_records):,}")

def records_to_dataset(records):
    return Dataset.from_dict({
        "id":       [r["id"]       for r in records],
        "context":  [r["context"]  for r in records],
        "question": [r["question"] for r in records],
        "answers":  [r["answers"]  for r in records],
    })

train_ds = records_to_dataset(train_records)
val_ds   = records_to_dataset(val_records)

print(f"\nLoading model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model     = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL, ignore_mismatched_sizes=True)

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

        if len(answers["answer_start"]) == 0:
            start_positions.append(0); end_positions.append(0); continue

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

print("Tokenising …")
train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names, desc="Train")
val_tok   = val_ds.map(preprocess,   batched=True, remove_columns=val_ds.column_names,   desc="Val")

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

class QATrainer(Trainer):
    def __init__(self, val_records, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._val_records = val_records

    def evaluate(self, **kwargs):
        results = super().evaluate(**kwargs)
        self.model.eval()
        em_scores, f1_scores = [], []
        for record in self._val_records:
            enc = tokenizer(record["question"], record["context"],
                            return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
            with torch.no_grad():
                out = self.model(**enc)
            start = out.start_logits.argmax().item()
            end   = out.end_logits.argmax().item() + 1
            pred  = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(enc["input_ids"][0][start:end]))
            em, f1 = compute_em_f1(pred, record["answers"]["text"][0])
            em_scores.append(em); f1_scores.append(f1)
        avg_em = sum(em_scores) / len(em_scores) * 100
        avg_f1 = sum(f1_scores) / len(f1_scores) * 100
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
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    save_total_limit=2,
    use_cpu=True, fp16=False, bf16=False,
    logging_steps=LOGGING_STEPS,
    report_to="none",
    seed=SEED, data_seed=SEED,
)

trainer = QATrainer(
    val_records=val_records,
    model=model, args=training_args,
    train_dataset=train_tok, eval_dataset=val_tok,
    data_collator=DefaultDataCollator(),
    processing_class=tokenizer,
)

print(f"\nModel D — BERT-base-uncased → IPR direct")
print(f"  Effective batch : {BATCH_SIZE * GRAD_ACCUM}  |  Epochs : {NUM_EPOCHS}\n")
trainer.train()

final_dir = OUTPUT_DIR / "final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"\nModel D saved to: {final_dir}")
