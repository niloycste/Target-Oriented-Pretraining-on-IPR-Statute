"""Size-matched DAPT ablation with a unified checkpoint-selection rule.

Addresses the two acknowledged confounds in the DAPT comparison:

  (1) corpus scale     901 authentic vs 13,211 synthetic passages
  (2) checkpoint rule  best-eval_f1 for A/H', lowest-eval_loss for C'/I'

Every run below trains on exactly 901 passages and selects the checkpoint with
the LOWEST validation loss, so scale and selection are held fixed and only
corpus composition varies.

Runs produced
  H_prime_lossrule  re-runs only H's final IPR stage from its saved SQuAD
                    checkpoint under the loss rule (stages 1-2 reused)
  auth901           MLM -> SQuAD v1 -> IPR, 901 authentic passages
  synth901          MLM -> SQuAD v1 -> IPR, 901 synthetic passages
  mixed901          MLM -> SQuAD v1 -> IPR, 450 authentic + 451 synthetic

Each stage is produced by copying the ORIGINAL script and rewriting only path
constants and the checkpoint rule, so all other hyperparameters are identical to
the published runs by construction.

Resumable: a stage whose final/ directory exists is skipped.
Run:  python size_matched_dapt_experiment.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[2]   # repo root: data lives here
RESULTS_DIR = ROOT / "results" / "experiments" / "size_matched_dapt"
WORK = ROOT / "artifacts" / "models" / "size_matched"
MERGE = ROOT / "data" / "dapt"
OUT_JSON = RESULTS_DIR / "size_matched_results.json"

VARIANTS = ["auth901", "synth901", "mixed901"]


# --------------------------------------------------------------- patching
def _sub(text, pattern, replacement, label):
    new, n = re.subn(pattern, lambda _m: replacement, text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"patch failed [{label}]: {pattern!r} matched {n} times")
    return new


def loss_rule(text):
    """Force lowest-validation-loss checkpoint selection."""
    if "metric_for_best_model" in text:
        text = _sub(text, r'^\s*metric_for_best_model=.*$',
                    '    metric_for_best_model="eval_loss",\n    greater_is_better=False,',
                    "metric_for_best_model")
    else:
        text = _sub(text, r'^\s*load_best_model_at_end=True,\s*$',
                    '    load_best_model_at_end=True,\n'
                    '    metric_for_best_model="eval_loss",\n'
                    '    greater_is_better=False,',
                    "load_best_model_at_end")
    return text


def make_mlm(variant, outdir):
    s = (ROOT / "scripts" / "02_synthetic_dapt" / "pretrain_mlm_original_only.py").read_text(encoding="utf-8")
    s = _sub(s, r"^INPUT_FILE\s*=.*$",
             f'INPUT_FILE = Path(r"{MERGE / f"pretraining_contexts_{variant}.jsonl"}")', "mlm in")
    s = _sub(s, r"^OUTPUT_DIR\s*=.*$", f'OUTPUT_DIR = Path(r"{outdir}")', "mlm out")
    return s


def make_squad(mlm_final, outdir):
    s = (ROOT / "scripts" / "03_finetuning" / "finetune_squad_v1_original_only.py").read_text(encoding="utf-8")
    s = _sub(s, r"^MODEL_DIR\s*=.*$", f'MODEL_DIR  = Path(r"{mlm_final}")', "squad in")
    s = _sub(s, r"^OUTPUT_DIR\s*=.*$", f'OUTPUT_DIR = Path(r"{outdir}")', "squad out")
    return loss_rule(s)


def make_ipr(squad_final, outdir):
    s = (ROOT / "scripts" / "03_finetuning" / "finetune_ipr_original_only.py").read_text(encoding="utf-8")
    s = _sub(s, r"^MODEL_DIR\s*=.*$", f'MODEL_DIR  = Path(r"{squad_final}")', "ipr in")
    s = _sub(s, r"^OUTPUT_DIR\s*=.*$", f'OUTPUT_DIR = Path(r"{outdir}")', "ipr out")
    s = _sub(s, r"^BASE_DIR\s*=.*$", f'BASE_DIR   = Path(r"{ROOT}")', "ipr base")
    return loss_rule(s)


def run_stage(tag, source, final_dir):
    if Path(final_dir).exists():
        print(f"[skip] {tag} already done", flush=True)
        return True
    WORK.mkdir(parents=True, exist_ok=True)
    script = WORK / f"{tag}.py"
    script.write_text(source, encoding="utf-8")
    print(f"\n{'='*72}\n[run ] {tag}\n{'='*72}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-X", "utf8", str(script)], cwd=str(ROOT))
    ok = r.returncode == 0
    print(f"[{'done' if ok else 'FAIL'}] {tag} in {(time.time()-t0)/3600:.2f} h", flush=True)
    return ok


# --------------------------------------------------------------- pipeline
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    targets = {}

    # (b) H' final stage only, under the loss rule
    h_squad = ROOT / "artifacts" / "models" / "squad_v1" / "original_only_squadv1" / "squad_v1" / "final"
    h_out = WORK / "Hprime_lossrule_ipr"
    if h_squad.exists():
        if run_stage("Hprime_lossrule_ipr", make_ipr(h_squad, h_out), h_out / "final"):
            targets["H' (loss rule)"] = h_out / "final"
    else:
        print(f"[warn] missing {h_squad}; skipping H' re-run")

    # (c) three size-matched variants, full pipeline
    for v in VARIANTS:
        mlm_dir = WORK / f"{v}_mlm"
        sq_dir = WORK / f"{v}_squad"
        ipr_dir = WORK / f"{v}_ipr"
        if not run_stage(f"{v}_mlm", make_mlm(v, mlm_dir), mlm_dir / "final"):
            continue
        if not run_stage(f"{v}_squad", make_squad(mlm_dir / "final", sq_dir), sq_dir / "final"):
            continue
        if run_stage(f"{v}_ipr", make_ipr(sq_dir / "final", ipr_dir), ipr_dir / "final"):
            targets[v] = ipr_dir / "final"

    # ------------------------------------------------------------ evaluate
    from multiseed_experiment import evaluate
    res = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for name, path in targets.items():
        print(f"[eval] {name}", flush=True)
        res[name] = evaluate(path)
        print(f"       {res[name]}", flush=True)
        OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"{'run':22s}{'n':>6s}{'EM':>9s}{'F1':>9s}")
    print("-" * 60)
    for k, v in res.items():
        print(f"{k:22s}{v['n']:6d}{v['em']:9.2f}{v['f1']:9.2f}")
    print("=" * 60)
    print(f"\nwritten to {OUT_JSON}")


if __name__ == "__main__":
    main()
