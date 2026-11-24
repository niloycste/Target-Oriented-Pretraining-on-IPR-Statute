"""Additional-encoder replication of D; default prepares and checks, --execute trains.

All 20 readers are newly trained per encoder. Existing InLegalBERT results are
never reused as the no-DAPT baseline. Run one process per encoder directory.
"""
from __future__ import annotations

import argparse
import ast
import gc
import importlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
d = importlib.import_module("scripts.dapt_equal_update.run_experiment_d")
b = d.base
MODELS = {"bert": "bert-base-uncased", "legalbert": "nlpaueb/legal-bert-base-uncased"}
SEEDS = [7, 42, 99, 123, 2026]
CONDITIONS = ["N", "A", "S", "M"]


def paths(encoder):
    return (ROOT / "experiments/dapt/encoder_extension" / encoder,
            ROOT / "artifacts/models/encoder_extension" / encoder)


def sources(condition, seed, snapshot, exp, work):
    tag = f"E_{condition}_s{seed}"
    mlm = work / f"{tag}_mlm"
    squad = work / f"{tag}_squad"
    ipr = work / f"{tag}_ipr"
    stages = {}
    if condition != "N":
        text = d.make_mlm_source(d.CORPORA[condition], mlm, seed)
        text = b.set_const(text, "BASE_MODEL", snapshot)
        # The untied-weight workaround is specific to InLegalBERT. Preserve
        # the additional encoder's own tying configuration instead.
        text = text.replace("model.config.tie_word_embeddings = False",
                            "model.config.tie_word_embeddings = (model.get_input_embeddings().weight.data_ptr() "
                            "== model.get_output_embeddings().weight.data_ptr())")
        text = text.replace(
            "from_pretrained(BASE_MODEL, ignore_mismatched_sizes=True)",
            "from_pretrained(BASE_MODEL, local_files_only=True)")
        stages["mlm"] = (text, mlm)
    stages["squad"] = (b.make_squad_source(
        snapshot if condition == "N" else mlm / "final", squad, seed), squad)
    stages["ipr"] = (b.make_ipr_source(squad / "final", ipr, seed), ipr)
    return {stage: (b.set_const(text, "CACHE_DIR", exp / "cache" / stage), output)
            for stage, (text, output) in stages.items()}


def freeze(path, payload):
    if path.exists():
        if json.loads(path.read_text(encoding="utf8")) != payload:
            raise RuntimeError(f"Frozen protocol/input/code mismatch: {path}")
    else:
        d.atomic_json(path, payload)


def prepare(encoder):
    from huggingface_hub import snapshot_download
    from datasets import config
    from transformers import AutoConfig, AutoTokenizer
    exp, work = paths(encoder)
    snapshot = Path(snapshot_download(MODELS[encoder], local_files_only=True))
    AutoConfig.from_pretrained(snapshot, local_files_only=True)
    AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    inputs = [*d.CORPORA.values(), *[ROOT / "data/splits" / name for name in
              ("train_squad.jsonl", "val_squad.jsonl", "val.jsonl", "test.jsonl")]]
    cached = sorted((Path(config.HF_DATASETS_CACHE) / "rajpurkar___squad/plain_text").glob(
        "**/squad-train.arrow"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cached:
        raise FileNotFoundError("SQuAD Arrow cache missing")
    inputs += [cached[0], cached[0].with_name("squad-validation.arrow")]
    inputs += sorted(snapshot.glob("*.json")) + sorted(snapshot.glob("*.txt"))
    weights = sorted(snapshot.glob("*.safetensors")) or sorted(snapshot.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"Model weights missing: {snapshot}")
    inputs += weights
    generated = {}
    checks = 0
    for seed in SEEDS:
        for condition in CONDITIONS:
            for stage, (text, output) in sources(condition, seed, snapshot, exp, work).items():
                ast.parse(text)
                if str(exp / "cache" / stage) not in text:
                    raise AssertionError("cache isolation failed")
                if stage == "mlm":
                    for expected in ("MAX_STEPS       = 100", "max_steps=MAX_STEPS",
                                     "WARMUP_STEPS = 10", "TokenCountingTrainer"):
                        if expected not in text:
                            raise AssertionError(expected)
                else:
                    if 'metric_for_best_model="eval_loss"' not in text:
                        raise AssertionError("checkpoint selection changed")
                path = exp / "generated_scripts" / f"E_{condition}_s{seed}_{stage}.py"
                generated[path] = text
                checks += 1
    # Freeze the actual source dependencies as well as generated code.
    dependencies = [Path(__file__), Path(d.__file__), Path(b.__file__), Path(b.base.__file__),
                    ROOT / "scripts/qa_evaluation.py",
                    ROOT / "scripts/dapt_equal_update/telemetry.py"]
    for folder in ("02_synthetic_dapt", "03_finetuning"):
        dependencies += list((ROOT / "scripts" / folder).glob("*original_only.py"))
    dependencies += [ROOT / "scripts/09_dapt_final" / name for name in
                     ("evaluate_final_experiments.py", "analyse_final_experiments.py")]
    import hashlib
    protocol = {"encoder": encoder, "model": MODELS[encoder], "snapshot": str(snapshot),
                "seeds": SEEDS, "conditions": CONDITIONS, "updates": 100, "warmup": 10,
                "design": "post-hoc encoder extension; full-data gold-context QA",
                "analysis": "within-encoder seed-blocked ANOVA; paired A/S/M minus N t intervals and Holm tests",
                "limitations": "same corpus records, not equal tokenizer-specific token exposure; non-detection is not equivalence",
                "inputs": {str(p): d.sha256(p) for p in inputs + dependencies},
                "scripts": {str(p): hashlib.sha256(t.encode()).hexdigest() for p,t in generated.items()}}
    freeze(exp / "PROTOCOL.json", protocol)
    for path, text in generated.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf8")
    print(f"Prepared {encoder}: {checks} stage scripts checked; snapshot {snapshot.name}", flush=True)
    return exp, work, snapshot


def smoke(snapshot):
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForQuestionAnswering
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    for cls in (AutoModelForMaskedLM, AutoModelForQuestionAnswering):
        model = cls.from_pretrained(snapshot, local_files_only=True)
        batch = tok("What is protected?", "The Act protects registered designs.", return_tensors="pt")
        if cls == AutoModelForMaskedLM:
            labels = torch.full_like(batch["input_ids"], -100)
            labels[0, 2] = batch["input_ids"][0, 2]
            batch["input_ids"][0, 2] = tok.mask_token_id
            batch["labels"] = labels
        else:
            batch.update(start_positions=torch.tensor([7]), end_positions=torch.tensor([8]))
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite smoke loss")
        loss.backward()
        if not any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("missing gradients")
        print(f"Smoke {cls.__name__}: forward/backward PASS, loss={loss.item():.4f}", flush=True)
        del model, batch, loss
        gc.collect()


def execute(exp, work, snapshot):
    e = importlib.import_module("scripts.09_dapt_final.evaluate_final_experiments")
    stats = importlib.import_module("scripts.09_dapt_final.analyse_final_experiments")
    import torch
    torch.set_num_threads(b.cpu_config("ipr")["threads"])
    b.FINAL, b.WORK, b.LEDGER = exp, work, exp / "RUN_LEDGER.csv"
    e.FINAL = exp
    rows = b.read_ledger()
    config = e.QADecodingConfig(max_seq_length=384, doc_stride=128, batch_size=16, max_answer_tokens=None)
    for seed in SEEDS:
        for condition in CONDITIONS:
            if shutil.disk_usage(ROOT).free < 8 * 1024**3:
                raise RuntimeError("Less than 8 GiB free; stopped before next pipeline")
            for stage, (source, output) in sources(condition, seed, snapshot, exp, work).items():
                b.execute_stage(rows, "E", condition, seed, stage, output, output / "final",
                                lambda s=source: s, False, 0)
            checkpoint = work / f"E_{condition}_s{seed}_ipr/final"
            key = f"E_{condition}_s{seed}_100"
            target = dict(experiment="E", condition=condition, seed=seed, fraction=100, checkpoint=checkpoint)
            for split in ("val", "test"):
                result = e.evaluate_one(key, target, e.load_records(split), split,
                                        torch.device("cpu"), config, e.checkpoint_sha256(checkpoint))
                e.atomic_json(exp / "evaluations" / split / f"{key}.json", result)
                e.update_aggregate(split, {key: result}, config)
    for split, filename in (("test", "results.json"), ("val", "validation_results.json")):
        runs = json.loads((exp / filename).read_text())["runs"]
        table = {c: {s: runs[f"E_{c}_s{s}_100"] for s in SEEDS} for c in CONDITIONS}
        report = {}
        for metric in ("f1", "em"):
            contrasts = {c: stats.paired([table[c][s][metric] for s in SEEDS],
                         [table["N"][s][metric] for s in SEEDS], f"{c}-N", SEEDS) for c in "ASM"}
            stats.holm(contrasts)
            report[metric] = dict(blocked_anova=stats.blocked_anova(table, metric),
                contrasts=contrasts, descriptives={c: stats.mean_sd_ci([table[c][s][metric] for s in SEEDS]) for c in CONDITIONS})
        d.atomic_json(exp / f"{split}_statistics.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=MODELS, default="bert")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    exp, work, snapshot = prepare(args.encoder)
    if args.smoke:
        smoke(snapshot)
    if args.execute:
        execute(exp, work, snapshot)


if __name__ == "__main__":
    main()
