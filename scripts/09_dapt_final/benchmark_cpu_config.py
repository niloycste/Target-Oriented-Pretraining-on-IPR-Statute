"""Benchmark scientifically neutral, stage-safe CPU settings for final DAPT."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "experiments" / "dapt" / "final_cpu"
OUT = FINAL / "cpu_benchmark.json"
REPORT = FINAL / "CPU_OPTIMIZATION_REPORT.md"
BASE_MODEL = "law-ai/InLegalBERT"
EFFECTIVE_BATCH = 32
QA_STEPS = 6
MLM_STEPS = 4
QA_PARTITIONS = [(8, 4), (16, 2), (32, 1)]
MLM_PARTITIONS = [(4, 8), (8, 4), (16, 2)]
OBSERVED_SQUAD_STEPS = 1274


def bench(
    stage: str, micro: int, accum: int, threads: int, workers: int, steps: int,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForMaskedLM, AutoModelForQuestionAnswering

    started = time.time()
    result = {
        "stage": stage,
        "micro_batch": micro,
        "grad_accum": accum,
        "effective_batch": micro * accum,
        "threads": threads,
        "dataloader_workers": workers,
    }
    try:
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        if stage == "mlm":
            sequence_length = 256
            model = AutoModelForMaskedLM.from_pretrained(
                BASE_MODEL, local_files_only=True
            )
        else:
            sequence_length = 384
            model = AutoModelForQuestionAnswering.from_pretrained(
                BASE_MODEL, local_files_only=True
            )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        count = micro * accum * steps
        ids = torch.randint(1000, 20000, (count, sequence_length))
        attention = torch.ones_like(ids)
        if stage == "mlm":
            labels = torch.full_like(ids, -100)
            labels[:, ::7] = ids[:, ::7]
            dataset = TensorDataset(ids, attention, labels)
        else:
            starts = torch.randint(0, sequence_length // 2, (count,))
            ends = starts + torch.randint(0, sequence_length // 2, (count,))
            ends.clamp_(max=sequence_length - 1)
            dataset = TensorDataset(ids, attention, starts, ends)
        loader = DataLoader(dataset, batch_size=micro, num_workers=workers)

        optimizer.zero_grad()
        timed = time.time()
        completed = 0
        for index, batch in enumerate(loader):
            if stage == "mlm":
                batch_ids, batch_attention, batch_labels = batch
                output = model(
                    input_ids=batch_ids,
                    attention_mask=batch_attention,
                    labels=batch_labels,
                )
            else:
                batch_ids, batch_attention, batch_start, batch_end = batch
                output = model(
                    input_ids=batch_ids,
                    attention_mask=batch_attention,
                    start_positions=batch_start,
                    end_positions=batch_end,
                )
            (output.loss / accum).backward()
            if (index + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                completed += 1
        elapsed = time.time() - timed
        result.update(
            {
                "optimizer_steps": completed,
                "seconds": round(elapsed, 3),
                "sec_per_step": round(elapsed / completed, 6),
                "examples_per_sec": round(count / elapsed, 6),
                "status": "ok",
            }
        )
        del model, optimizer, loader, dataset, ids, attention
        gc.collect()
        return result
    except Exception as exc:
        gc.collect()
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "seconds_before_failure": round(time.time() - started, 3),
            }
        )
        return result


def run_and_print(
    runs: list[dict], stage: str, micro: int, accum: int,
    threads: int, workers: int, steps: int,
) -> dict:
    result = bench(stage, micro, accum, threads, workers, steps)
    runs.append(result)
    label = f"{stage:5s} {micro:2d}x{accum} threads={threads:2d} workers={workers}"
    if result["status"] == "ok":
        print(f"  {label}  {result['sec_per_step']:7.3f} s/step", flush=True)
    else:
        print(f"  {label}  FAILED: {result['error']}", flush=True)
    return result


def count_features(path: Path, tokenizer) -> int | None:
    if not path.exists():
        return None
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    count = 0
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        encoded = tokenizer(
            [row["question"].strip() for row in batch],
            [row["context"] for row in batch],
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
        )
        count += len(encoded["input_ids"])
    return count


def optimizer_steps(features: int, micro: int, accum: int, epochs: int) -> int:
    batches_per_epoch = math.ceil(features / micro)
    return epochs * math.ceil(batches_per_epoch / accum)


def runtime_projection(qa: dict, mlm: dict) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    feature_counts = {
        "ipr_100": count_features(ROOT / "data" / "splits" / "train_squad.jsonl", tokenizer),
        "ipr_25": count_features(FINAL / "train_squad_25.jsonl", tokenizer),
        "ipr_50": count_features(FINAL / "train_squad_50.jsonl", tokenizer),
    }
    if any(value is None for value in feature_counts.values()):
        return {
            "status": "pending",
            "reason": "low-resource materialisations were unavailable during benchmark",
        }

    qa_steps = {
        name: optimizer_steps(int(features), qa["micro_batch"], qa["grad_accum"], 3)
        for name, features in feature_counts.items()
    }
    # Twenty-three full pipelines: 20 A + 3 new B. Experiment C adds twelve
    # models per low-resource fraction. Eighteen of the full pipelines use MLM.
    squad_hours = 23 * OBSERVED_SQUAD_STEPS * qa["sec_per_step"] / 3600
    full_ipr_hours = 23 * qa_steps["ipr_100"] * qa["sec_per_step"] / 3600
    low_ipr_hours = 12 * (
        qa_steps["ipr_25"] + qa_steps["ipr_50"]
    ) * qa["sec_per_step"] / 3600
    mlm_steps_per_model = optimizer_steps(
        900, mlm["micro_batch"], mlm["grad_accum"], 3
    )
    mlm_hours = 18 * mlm_steps_per_model * mlm["sec_per_step"] / 3600
    training_hours = squad_hours + full_ipr_hours + low_ipr_hours + mlm_hours
    wall_low = training_hours * 1.10
    wall_high = training_hours * 1.30
    return {
        "status": "estimated",
        "basis": "measured optimizer-step pilots; observed 1,274-step SQuAD stage",
        "qa_feature_counts": feature_counts,
        "optimizer_steps_per_model": {
            **qa_steps,
            "squad": OBSERVED_SQUAD_STEPS,
            "mlm_approx": mlm_steps_per_model,
        },
        "components_hours": {
            "squad_23_models": round(squad_hours, 2),
            "full_ipr_23_models": round(full_ipr_hours, 2),
            "low_resource_ipr_24_models": round(low_ipr_hours, 2),
            "mlm_18_models": round(mlm_hours, 2),
        },
        "training_compute_hours": round(training_hours, 2),
        "estimated_wall_hours_range": [round(wall_low, 2), round(wall_high, 2)],
        "estimated_wall_days_continuous": [round(wall_low / 24, 1), round(wall_high / 24, 1)],
        "overhead_allowance": "10-30% for evaluation, I/O, tokenisation and model loading",
    }


def main() -> int:
    import torch

    FINAL.mkdir(parents=True, exist_ok=True)
    cores = os.cpu_count() or 1
    default_threads = min(24, cores)
    runs: list[dict] = []

    print("\nQA micro-batch partitioning")
    for micro, accum in QA_PARTITIONS:
        run_and_print(runs, "qa", micro, accum, default_threads, 0, QA_STEPS)
    qa_success = [row for row in runs if row["stage"] == "qa" and row["status"] == "ok"]
    if not qa_success:
        raise RuntimeError("every QA CPU benchmark configuration failed")
    qa_best = min(qa_success, key=lambda row: row["sec_per_step"])

    print("\nQA thread count")
    thread_candidates = sorted(
        {max(1, min(8, cores)), min(16, cores), default_threads, cores}
    )
    for threads in thread_candidates:
        if threads == qa_best["threads"]:
            continue
        run_and_print(
            runs, "qa", qa_best["micro_batch"], qa_best["grad_accum"],
            threads, 0, QA_STEPS,
        )
    qa_success = [row for row in runs if row["stage"] == "qa" and row["status"] == "ok"]
    qa_best = min(qa_success, key=lambda row: row["sec_per_step"])

    # Generated training scripts execute at module scope. On Windows, spawned
    # DataLoader workers would re-import and recursively execute the script, so
    # workers=0 is the only safe setting without changing the scientific code.
    qa_winner = qa_best

    print("\nMLM-safe micro-batch partitioning")
    for micro, accum in MLM_PARTITIONS:
        run_and_print(
            runs, "mlm", micro, accum, qa_winner["threads"], 0, MLM_STEPS
        )
    mlm_success = [row for row in runs if row["stage"] == "mlm" and row["status"] == "ok"]
    if not mlm_success:
        raise RuntimeError("every MLM CPU benchmark configuration failed")
    mlm_winner = min(mlm_success, key=lambda row: row["sec_per_step"])

    qa_baseline = next(
        row for row in qa_success
        if row["micro_batch"] == 8 and row["grad_accum"] == 4
        and row["threads"] == default_threads and row["dataloader_workers"] == 0
    )
    qa_speedup = qa_baseline["sec_per_step"] / qa_winner["sec_per_step"]
    projection = runtime_projection(qa_winner, mlm_winner)
    report = {
        "description": "Stage-safe CPU pilots with scientific settings held fixed",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "logical_cores": cores,
            "mkl_available": torch.backends.mkl.is_available(),
            "mkldnn_available": torch.backends.mkldnn.is_available(),
        },
        "invariant": (
            "effective batch=32, examples, model, optimizer, learning rate, loss, "
            "epochs and selection criterion are unchanged"
        ),
        "dataloader_policy": "workers=0 on Windows; avoids recursive module execution",
        "baseline": qa_baseline,
        "recommended": qa_winner,
        "recommended_by_stage": {
            "squad": qa_winner,
            "ipr": qa_winner,
            "mlm": mlm_winner,
        },
        "speedup_vs_baseline": round(qa_speedup, 6),
        "estimated_squad_stage_hours": round(
            OBSERVED_SQUAD_STEPS * qa_winner["sec_per_step"] / 3600, 3
        ),
        "runtime_projection": projection,
        "runs": runs,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf8")

    lines = [
        "# CPU optimization report",
        "",
        "QA and MLM were benchmarked separately because MLM's vocabulary head has a different memory profile.",
        "Effective batch size and every scientific training choice remained fixed.",
        "",
        "| Stage | Micro | Accum | Threads | Workers | sec/step | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in runs:
        seconds = f"{row['sec_per_step']:.3f}" if row["status"] == "ok" else "—"
        lines.append(
            f"| {row['stage']} | {row['micro_batch']} | {row['grad_accum']} | "
            f"{row['threads']} | {row['dataloader_workers']} | {seconds} | {row['status']} |"
        )
    lines += [
        "",
        f"QA recommendation: **{qa_winner['micro_batch']}×{qa_winner['grad_accum']}**, "
        f"{qa_winner['threads']} threads, {qa_winner['dataloader_workers']} workers.",
        f"MLM recommendation: **{mlm_winner['micro_batch']}×{mlm_winner['grad_accum']}**, "
        f"{mlm_winner['threads']} threads, {mlm_winner['dataloader_workers']} workers.",
        f"Measured QA speed-up versus 8×4 baseline: **{qa_speedup:.3f}×**.",
        f"Projected 1,274-step SQuAD stage: **{report['estimated_squad_stage_hours']:.2f} hours**.",
    ]
    if projection.get("status") == "estimated":
        low, high = projection["estimated_wall_hours_range"]
        low_days, high_days = projection["estimated_wall_days_continuous"]
        lines += [
            "",
            f"Projected full 47-reader study: **{low:.0f}-{high:.0f} wall hours** "
            f"(**{low_days:.1f}-{high_days:.1f} continuous days**).",
            "This includes a 10-30% allowance beyond measured training compute for evaluation and I/O.",
        ]
    lines += [
        "",
        "The final runner reads `cpu_benchmark.json` and applies the stage-specific configuration.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")
    print(f"  wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
