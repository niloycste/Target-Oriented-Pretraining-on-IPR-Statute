"""Deterministic validation-only MLM diagnostic for final DAPT checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "experiments" / "dapt" / "final_cpu"
WORK = ROOT / "artifacts" / "models" / "final_cpu"
DIAGNOSTIC = FINAL / "mlm_diagnostic.jsonl"
OUT = FINAL / "MLM_DIAGNOSTIC_RESULTS.json"
MASK_SEED = 20260829
MAX_LENGTH = 256
MLM_PROBABILITY = 0.15
SEEDS_A = [7, 42, 99, 123, 2026]
SEEDS_B = [7, 42, 99]


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf8")
    os.replace(temp, path)


def targets() -> dict[str, Path]:
    result = {"N_base": WORK / "base_encoder"}
    for condition in ("A", "S", "M"):
        for seed in SEEDS_A:
            result[f"A_{condition}_s{seed}"] = WORK / f"A_{condition}_s{seed}_mlm" / "final"
    for seed in SEEDS_B:
        result[f"B_S_STYLE_s{seed}"] = WORK / f"B_S_STYLE_s{seed}_mlm" / "final"
    return result


def build_masked_inputs(tokenizer) -> dict[str, torch.Tensor]:
    texts = [
        json.loads(line)["context"]
        for line in DIAGNOSTIC.read_text(encoding="utf8").splitlines() if line
    ]
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention = encoded["attention_mask"]
    special = torch.tensor([
        tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True)
        for row in input_ids
    ], dtype=torch.bool)
    generator = torch.Generator().manual_seed(MASK_SEED)
    candidate = attention.bool() & ~special
    masked = (torch.rand(input_ids.shape, generator=generator) < MLM_PROBABILITY) & candidate
    if not masked.any():
        raise RuntimeError("deterministic masking selected no tokens")

    labels = input_ids.clone()
    labels[~masked] = -100
    corrupted = input_ids.clone()
    replace_draw = torch.rand(input_ids.shape, generator=generator)
    mask_token = masked & (replace_draw < 0.8)
    random_token = masked & (replace_draw >= 0.8) & (replace_draw < 0.9)
    corrupted[mask_token] = tokenizer.mask_token_id
    random_ids = torch.randint(
        low=0, high=len(tokenizer), size=input_ids.shape, generator=generator
    )
    corrupted[random_token] = random_ids[random_token]
    return {
        "input_ids": corrupted,
        "attention_mask": attention,
        "labels": labels,
        "contexts": len(texts),
        "masked_tokens": int(masked.sum().item()),
    }


def evaluate(path: Path, tensors: dict, batch_size: int) -> dict:
    model = AutoModelForMaskedLM.from_pretrained(str(path), local_files_only=True)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, tensors["contexts"], batch_size):
            end = min(start + batch_size, tensors["contexts"])
            ids = tensors["input_ids"][start:end]
            attention = tensors["attention_mask"][start:end]
            labels = tensors["labels"][start:end]
            logits = model(input_ids=ids, attention_mask=attention).logits
            selected = labels.ne(-100)
            loss = functional.cross_entropy(
                logits[selected], labels[selected], reduction="sum"
            )
            total_loss += float(loss.item())
            total_tokens += int(selected.sum().item())
    mean_loss = total_loss / total_tokens
    result = {
        "loss": round(mean_loss, 8),
        "perplexity": round(math.exp(mean_loss), 8),
        "masked_tokens": total_tokens,
    }
    del model
    gc.collect()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    table = targets()
    missing = {key: path for key, path in table.items() if not path.exists()}
    if missing and not args.allow_partial:
        print("  missing MLM checkpoints; diagnostic refused:")
        for key, path in missing.items():
            print(f"    {key}: {path.relative_to(ROOT)}")
        return 1

    tokenizer_path = WORK / "base_encoder"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    tensors = build_masked_inputs(tokenizer)
    report = {
        "schema_version": 1,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": DIAGNOSTIC.relative_to(ROOT).as_posix(),
        "source_split": "val",
        "contexts": tensors["contexts"],
        "mask_seed": MASK_SEED,
        "mlm_probability": MLM_PROBABILITY,
        "max_length": MAX_LENGTH,
        "masking_protocol": "one frozen 80/10/10 mask shared by every checkpoint",
        "results": {},
    }
    if OUT.exists() and not args.force:
        prior = json.loads(OUT.read_text(encoding="utf8"))
        if prior.get("schema_version") == 1 and prior.get("mask_seed") == MASK_SEED:
            report["results"].update(prior.get("results", {}))

    for key, path in table.items():
        if not path.exists():
            continue
        if key in report["results"] and not args.force:
            print(f"  {key}: existing diagnostic reused")
            continue
        print(f"  evaluating MLM {key}", flush=True)
        report["results"][key] = {
            "checkpoint": path.relative_to(ROOT).as_posix(),
            **evaluate(path, tensors, args.batch_size),
        }
        atomic_json(OUT, report)

    baseline = report["results"].get("N_base")
    if baseline:
        baseline_ppl = baseline["perplexity"]
        for key, result in report["results"].items():
            result["relative_perplexity_reduction_pct"] = round(
                100 * (baseline_ppl - result["perplexity"]) / baseline_ppl, 6
            )
    atomic_json(OUT, report)
    print(f"  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
