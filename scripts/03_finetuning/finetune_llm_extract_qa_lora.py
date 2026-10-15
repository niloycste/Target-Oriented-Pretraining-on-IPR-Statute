"""
LoRA fine-tuning for extractive legal QA with decoder-only LLMs
===============================================================

This script is the closest LLM analogue to the repo's BERT QA pipeline:

  BERT path:
    base encoder -> SQuAD span QA -> IPR span QA

  LLM path:
    base instruct LLM -> SQuAD instruction tuning -> IPR instruction tuning

Instead of learning start/end span positions, the LLM is trained to
generate the exact answer text copied from the passage.

Supported families:
  - Llama 3.1 Instruct
  - Qwen 2.5 Instruct
  - Gemma Instruct / IT models
  - any Hugging Face causal LM with a working tokenizer

Examples
--------

1) Intermediate SQuAD tuning (similar to finetune_squad_v1_modelc_v2.py)

python finetune_llm_extract_qa_lora.py ^
  --dataset squad ^
  --base_model meta-llama/Llama-3.1-8B-Instruct ^
  --output_dir artifacts/models/llm/llama31/squad_lora ^
  --load_in_4bit

2) Final IPR tuning on top of the SQuAD adapter
   (similar to finetune_ipr_modelc_v2.py)

python finetune_llm_extract_qa_lora.py ^
  --dataset ipr ^
  --base_model meta-llama/Llama-3.1-8B-Instruct ^
  --resume_adapter artifacts/models/llm/llama31/squad_lora/final ^
  --output_dir artifacts/models/llm/llama31/ipr_lora ^
  --load_in_4bit

3) Direct IPR tuning without the SQuAD intermediate stage

python finetune_llm_extract_qa_lora.py ^
  --dataset ipr ^
  --base_model Qwen/Qwen2.5-7B-Instruct ^
  --output_dir artifacts/models/llm/qwen25/ipr_direct_lora ^
  --load_in_4bit

Notes
-----
- For 7B-8B models, GPU training is strongly recommended.
- On Windows, 4-bit QLoRA often works best inside WSL/Linux.
- This saves a LoRA adapter under output_dir/final, not a merged full model.
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

try:
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: peft\n"
        "Install with: pip install peft"
    ) from exc

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_FILE = BASE_DIR / "data" / "splits" / "train.jsonl"
DEFAULT_VAL_FILE = BASE_DIR / "data" / "splits" / "val.jsonl"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "artifacts" / "models" / "llm"

SYSTEM_PROMPT = (
    "You are an expert legal extractive QA assistant. "
    "Answer by copying the shortest exact answer span from the passage. "
    "Do not paraphrase. Do not explain. Return only the answer text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an instruction LLM for extractive QA using LoRA/QLoRA."
    )
    parser.add_argument(
        "--dataset",
        choices=("ipr", "squad"),
        required=True,
        help="Training source. Use 'squad' for the intermediate stage or 'ipr' for the final legal stage.",
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Base Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--resume_adapter",
        default=None,
        help="Optional existing LoRA adapter to continue training from.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write checkpoints. Defaults under artifacts/models/llm/.",
    )
    parser.add_argument(
        "--train_file",
        default=str(DEFAULT_TRAIN_FILE),
        help="IPR train JSONL. Used when --dataset ipr.",
    )
    parser.add_argument(
        "--val_file",
        default=str(DEFAULT_VAL_FILE),
        help="IPR validation JSONL. Used when --dataset ipr.",
    )
    parser.add_argument(
        "--squad_train_subset",
        type=int,
        default=20_000,
        help="Train subset size for SQuAD stage. Use 0 or negative for full train split.",
    )
    parser.add_argument(
        "--squad_val_subset",
        type=int,
        default=2_000,
        help="Validation subset size for SQuAD stage. Use 0 or negative for full validation split.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1536,
        help="Maximum prompt+answer token length.",
    )
    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=2800,
        help="If a context is long, crop around the gold answer before tokenization.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
        help="Per-device train/eval batch size. 1 is typical for 7B-8B QLoRA.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA tuning.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Training log frequency.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum checkpoint count to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names for LoRA.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Enable QLoRA 4-bit loading. Recommended for 7B-8B models on GPU.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loading if needed.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_model_name(name: str) -> str:
    return name.replace("\\", "_").replace("/", "_").replace(":", "_")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return DEFAULT_OUTPUT_ROOT / sanitize_model_name(args.base_model) / f"{args.dataset}_lora"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_ipr_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        answer_start = row.get("answerspan")
        if answer_start is None and isinstance(row.get("answers"), dict):
            starts = row["answers"].get("answer_start", [])
            answer_start = starts[0] if starts else None

        answer = row.get("answer")
        if answer is None and isinstance(row.get("answers"), dict):
            texts = row["answers"].get("text", [])
            answer = texts[0] if texts else ""

        records.append(
            {
                "id": row.get("id", ""),
                "act": row.get("act", "ipr"),
                "title": row.get("title", ""),
                "context": row["context"],
                "question": row["question"],
                "answer": answer or "",
                "answer_start": answer_start,
            }
        )
    return records


def load_ipr_dataset(train_file: Path, val_file: Path) -> Tuple[Dataset, Dataset]:
    train_rows = normalize_ipr_rows(load_jsonl(train_file))
    val_rows = normalize_ipr_rows(load_jsonl(val_file))
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


def load_squad_subset(split: str, subset_size: int, seed: int) -> Dataset:
    ds = load_dataset("rajpurkar/squad", split=split)
    if subset_size and subset_size > 0:
        subset_size = min(subset_size, len(ds))
        ds = ds.shuffle(seed=seed).select(range(subset_size))
    else:
        ds = ds.shuffle(seed=seed)

    rows = []
    for row in ds:
        answers = row.get("answers", {})
        texts = answers.get("text", [])
        starts = answers.get("answer_start", [])
        rows.append(
            {
                "id": row.get("id", ""),
                "act": "squad",
                "title": row.get("title", ""),
                "context": row["context"],
                "question": row["question"],
                "answer": texts[0] if texts else "",
                "answer_start": starts[0] if starts else None,
            }
        )
    return Dataset.from_list(rows)


def crop_context_around_answer(
    context: str,
    answer: str,
    answer_start: Optional[int],
    max_context_chars: int,
) -> str:
    if not context or len(context) <= max_context_chars:
        return context

    if answer_start is None:
        located = context.find(answer)
        answer_start = located if located >= 0 else None

    if answer_start is None:
        return context[:max_context_chars]

    answer_end = answer_start + len(answer)
    half_window = max_context_chars // 2
    start = max(0, answer_start - half_window)
    end = min(len(context), start + max_context_chars)
    start = max(0, end - max_context_chars)

    cropped = context[start:end]
    if answer not in cropped:
        fallback_start = max(0, answer_start - max_context_chars // 3)
        fallback_end = min(len(context), fallback_start + max_context_chars)
        cropped = context[fallback_start:fallback_end]
    return cropped


def build_user_prompt(record: Dict[str, Any], max_context_chars: int) -> str:
    context = crop_context_around_answer(
        context=record["context"],
        answer=record["answer"],
        answer_start=record.get("answer_start"),
        max_context_chars=max_context_chars,
    )
    act = record.get("act", "")
    title = record.get("title", "")
    return (
        f"Act: {act}\n"
        f"Title: {title}\n\n"
        f"Passage:\n{context}\n\n"
        f"Question:\n{record['question']}\n\n"
        "Return only the shortest exact answer span copied from the passage."
    )


def render_chat_pair(
    tokenizer: AutoTokenizer,
    record: Dict[str, Any],
    max_context_chars: int,
) -> Tuple[str, str]:
    user_prompt = build_user_prompt(record, max_context_chars=max_context_chars)
    answer = record["answer"].strip()

    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        prompt_text = (
            f"System: {SYSTEM_PROMPT}\n\n"
            f"User:\n{user_prompt}\n\n"
            "Assistant:"
        )
        full_text = f"{prompt_text} {answer}"

    eos = tokenizer.eos_token or ""
    if eos and not full_text.endswith(eos):
        full_text = f"{full_text}{eos}"

    return prompt_text, full_text


def preprocess_dataset(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_context_chars: int,
) -> Dataset:
    def preprocess_row(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text, full_text = render_chat_pair(
            tokenizer=tokenizer,
            record=row,
            max_context_chars=max_context_chars,
        )

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        model_inputs = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        labels = input_ids.copy()

        prompt_len = min(len(prompt_ids), len(labels))
        for idx in range(prompt_len):
            labels[idx] = -100

        for idx, mask in enumerate(attention_mask):
            if mask == 0:
                labels[idx] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return ds.map(
        preprocess_row,
        remove_columns=ds.column_names,
        desc="Tokenising",
    )


def build_quant_config(use_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if not use_4bit:
        return None

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(base_model: str, trust_remote_code: bool) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    base_model: str,
    resume_adapter: Optional[str],
    load_in_4bit: bool,
    lora_target_modules: List[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    trust_remote_code: bool,
) -> torch.nn.Module:
    quantization_config = build_quant_config(load_in_4bit)
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    if resume_adapter:
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, peft_config)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model


def print_trainable_params(model: torch.nn.Module) -> None:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count

    pct = 100 * trainable / total if total else 0.0
    print(f"  Total params      : {total / 1e6:.1f} M")
    print(f"  Trainable params  : {trainable / 1e6:.1f} M ({pct:.3f}%)")


def build_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    optim_name = "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch"

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        bf16=bf16,
        fp16=fp16,
        use_cpu=not torch.cuda.is_available(),
        optim=optim_name,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_targets = [part.strip() for part in args.lora_target_modules.split(",") if part.strip()]

    print("=" * 72)
    print("LLM Extractive QA Fine-tuning")
    print("=" * 72)
    print(f"  Dataset          : {args.dataset}")
    print(f"  Base model       : {args.base_model}")
    print(f"  Resume adapter   : {args.resume_adapter or '(none)'}")
    print(f"  Output dir       : {output_dir}")
    print(f"  4-bit QLoRA      : {args.load_in_4bit}")
    print(f"  CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print()

    print("Loading tokenizer ...")
    tokenizer = load_tokenizer(args.base_model, trust_remote_code=args.trust_remote_code)

    print("Loading dataset ...")
    if args.dataset == "ipr":
        train_ds, val_ds = load_ipr_dataset(
            train_file=Path(args.train_file),
            val_file=Path(args.val_file),
        )
    else:
        train_ds = load_squad_subset("train", args.squad_train_subset, args.seed)
        val_ds = load_squad_subset("validation", args.squad_val_subset, args.seed)

    print(f"  Train rows       : {len(train_ds):,}")
    print(f"  Val rows         : {len(val_ds):,}")

    print("\nTokenising datasets ...")
    train_tokenized = preprocess_dataset(
        train_ds,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_context_chars=args.max_context_chars,
    )
    val_tokenized = preprocess_dataset(
        val_ds,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_context_chars=args.max_context_chars,
    )
    print(f"  Train examples   : {len(train_tokenized):,}")
    print(f"  Val examples     : {len(val_tokenized):,}")

    print("\nLoading model ...")
    model = load_model(
        base_model=args.base_model,
        resume_adapter=args.resume_adapter,
        load_in_4bit=args.load_in_4bit,
        lora_target_modules=lora_targets,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        trust_remote_code=args.trust_remote_code,
    )
    print_trainable_params(model)

    training_args = build_training_args(args, output_dir=output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DefaultDataCollator(),
        processing_class=tokenizer,
    )

    print("\nStarting fine-tuning ...")
    print(f"  Effective batch  : {args.per_device_batch_size * args.gradient_accumulation_steps}")
    print(f"  Epochs           : {args.num_train_epochs}")
    print(f"  Max length       : {args.max_length}")
    print(f"  LoRA targets     : {', '.join(lora_targets)}")
    print()

    trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\nTraining complete.")
    print(f"  Adapter saved to : {final_dir}")

    if args.dataset == "squad":
        print("  Next step        : run the same script with --dataset ipr and --resume_adapter")
    else:
        print("  Next step        : evaluate the adapter with a generative QA evaluation script")


if __name__ == "__main__":
    main()
