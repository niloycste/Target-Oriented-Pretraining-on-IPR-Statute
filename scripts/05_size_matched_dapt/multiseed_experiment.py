"""Multi-seed rerun of the final IPR fine-tuning stage for A and C'.

Only the last stage is repeated. The DAPT and SQuAD-transfer checkpoints are
reused unchanged, so this measures fine-tuning variance, not variance of the
whole pipeline -- state it that way in the paper.

Each run is produced by copying the ORIGINAL training script and patching only
BASE_DIR, SEED and OUTPUT_DIR, so the training configuration is identical to
the seed-42 runs already reported (A: lr 3e-5, warmup 50, best-by-eval_f1;
C': lr 2e-5, warmup 100, best-by-eval_loss).

Evaluation calls the same sliding-window, exhaustive constrained decoder as
``evaluate_qa.py`` and returns literal source-context spans.

Run:  python multiseed_experiment.py
Resumable: finished runs are skipped.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root: data lives here
RESULTS_DIR = ROOT / "results" / "experiments" / "size_matched_dapt"
RUNS = ROOT / "artifacts" / "models" / "multiseed"
OUT_JSON = RESULTS_DIR / "multiseed_results.json"
TEST_FILE = ROOT / "data" / "splits" / "test.jsonl"
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
EVAL_BATCH_SIZE = 64
SEEDS = [7, 99]                      # 42 already reported

VARIANTS = {
    "A": {
        "script": "finetune_baseline_a.py",
        "desc": "InLegalBERT -> IPR direct",
        "seed42_dir": ROOT / "artifacts" / "models" / "direct" / "baseline_a" / "final",
    },
    "C'": {
        "script": "finetune_ipr_modelc_v2.py",
        "desc": "InLegalBERT + DAPT(Orig+Synth) -> SQuAD v1 -> IPR",
        "seed42_dir": ROOT / "artifacts" / "models" / "squad_v1" / "squad_v1_modelc_v2" / "ipr_final" / "final",
    },
}


def patch_script(src: Path, dst: Path, seed: int, outdir: Path):
    s = src.read_text(encoding="utf-8")
    # replacements contain Windows paths, so use callables: a plain string
    # replacement would treat backslashes as regex escapes
    s, n1 = re.subn(r"^BASE_DIR\s*=.*$",
                    lambda _m: f'BASE_DIR   = Path(r"{ROOT}")', s,
                    count=1, flags=re.M)
    s, n2 = re.subn(r"^SEED\s*=\s*\d+.*$",
                    lambda _m: f"SEED           = {seed}", s,
                    count=1, flags=re.M)
    s, n3 = re.subn(r"^OUTPUT_DIR\s*=.*$",
                    lambda _m: f'OUTPUT_DIR = Path(r"{outdir}")', s,
                    count=1, flags=re.M)
    if not (n1 and n2 and n3):
        raise RuntimeError(f"patch failed for {src.name}: "
                           f"BASE_DIR={n1} SEED={n2} OUTPUT_DIR={n3}")
    dst.write_text(s, encoding="utf-8")


def train_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    for code, cfg in VARIANTS.items():
        for seed in SEEDS:
            tag = f"{code.replace(chr(39), 'prime')}_seed{seed}"
            outdir = RUNS / tag
            final = outdir / "final"
            if final.exists():
                print(f"[skip] {tag} already trained")
                continue
            script = RUNS / f"{tag}.py"
            patch_script(ROOT / cfg["script"], script, seed, outdir)
            print(f"\n{'='*70}\n[train] {code} seed={seed}  ->  {outdir}\n{'='*70}",
                  flush=True)
            t0 = time.time()
            r = subprocess.run([sys.executable, "-X", "utf8", str(script)],
                               cwd=str(ROOT))
            if r.returncode != 0:
                print(f"[FAIL] {tag} exited {r.returncode}")
                continue
            print(f"[done] {tag} in {(time.time()-t0)/3600:.2f} h", flush=True)


# ------------------------------------------------------------------ evaluation
def evaluate(model_dir):
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.qa_evaluation import (
        QADecodingConfig,
        compute_em_f1,
        predict_answers,
        resolve_device,
    )

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    mdl = AutoModelForQuestionAnswering.from_pretrained(str(model_dir),
                                                        local_files_only=True)
    device = resolve_device("auto")
    mdl.to(device)
    mdl.eval()
    recs = [json.loads(l) for l in TEST_FILE.open(encoding="utf-8") if l.strip()]
    config = QADecodingConfig(
        max_seq_length=MAX_SEQ_LENGTH,
        doc_stride=DOC_STRIDE,
        batch_size=EVAL_BATCH_SIZE,
        max_answer_tokens=None,
    )
    decoded = predict_answers(
        tok,
        mdl,
        device,
        [rec["question"] for rec in recs],
        [rec["context"] for rec in recs],
        config,
    )
    ems, f1s = [], []
    for rec, (pred, _) in zip(recs, decoded):
        a, b = compute_em_f1(pred, rec["answer"])
        ems.append(a)
        f1s.append(b)
    n = len(ems)
    return {"n": n, "em": round(100 * sum(ems) / n, 4),
            "f1": round(100 * sum(f1s) / n, 4),
            "inference_protocol": config.as_dict()}


def evaluate_all():
    res = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for code, cfg in VARIANTS.items():
        res.setdefault(code, {"description": cfg["desc"], "seeds": {}})
        targets = [("42", cfg["seed42_dir"])]
        for seed in SEEDS:
            tag = f"{code.replace(chr(39), 'prime')}_seed{seed}"
            targets.append((str(seed), RUNS / tag / "final"))
        for seed, d in targets:
            if not Path(d).exists():
                print(f"  [missing] {code} seed {seed}: {d}")
                continue
            print(f"  [eval] {code} seed {seed} …", flush=True)
            res[code]["seeds"][seed] = evaluate(d)
            print(f"         {res[code]['seeds'][seed]}", flush=True)
            OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # aggregate
    import statistics as st
    for code in res:
        s = res[code]["seeds"]
        if len(s) >= 2:
            for m in ("em", "f1"):
                vals = [s[k][m] for k in sorted(s)]
                res[code][f"{m}_mean"] = round(st.mean(vals), 2)
                res[code][f"{m}_sd"] = round(st.pstdev(vals) if len(vals) < 3
                                             else st.stdev(vals), 2)
                res[code][f"{m}_min"] = min(vals)
                res[code][f"{m}_max"] = max(vals)
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"{'variant':6s} {'seed':>5s} {'EM':>8s} {'F1':>8s}")
    print("-" * 70)
    for code in res:
        for seed in sorted(res[code]["seeds"]):
            v = res[code]["seeds"][seed]
            print(f"{code:6s} {seed:>5s} {v['em']:8.2f} {v['f1']:8.2f}")
        if "f1_mean" in res[code]:
            print(f"{code:6s} {'mean':>5s} {res[code]['em_mean']:8.2f} "
                  f"{res[code]['f1_mean']:8.2f}"
                  f"   (F1 sd {res[code]['f1_sd']}, "
                  f"range {res[code]['f1_min']}-{res[code]['f1_max']})")
        print("-" * 70)
    if "A" in res and "C'" in res and "f1_mean" in res["A"] and "f1_mean" in res["C'"]:
        d = res["C'"]["f1_mean"] - res["A"]["f1_mean"]
        print(f"\nmean F1 difference  C' - A = {d:+.2f}")
    print(f"\nwritten to {OUT_JSON}")


if __name__ == "__main__":
    train_all()
    print("\n\nEVALUATING ALL CHECKPOINTS\n")
    evaluate_all()
