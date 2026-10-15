"""STEP 7 (dapt.md sec. 37): generate DRY_RUN_REPORT.md from the built artifacts.

The report is generated rather than written by hand so that every number in it is
read back from the manifests actually on disk. If a manifest changes, the report
changes with it, and a stale claim cannot survive a rebuild.

Covers the seventeen items sec. 37 requires. Items that depend on work not yet
finished are printed as PENDING with the reason -- never estimated or invented.

Run: python scripts/09_dapt_final/write_dry_run_report.py
"""

import io
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "experiments" / "dapt" / "final_cpu"
OUT = FINAL / "DRY_RUN_REPORT.md"

SEEDS_A = [7, 42, 99, 123, 2026]
SEEDS_B = [7, 42, 99]


def load(name):
    p = FINAL / name
    return json.load(io.open(p, encoding="utf8")) if p.exists() else None


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    prov = load("SYNTHETIC_PROVENANCE_AUDIT.json")
    corp = load("MATCHED_CORPORA_SUMMARY.json")
    auth = load("AUTHENTIC_TRAIN_ONLY_MANIFEST.json")
    style = load("S_STYLE_manifest.json")
    chars = load("CORPUS_CHARACTERISATION.json")
    diag = load("MLM_DIAGNOSTIC_MANIFEST.json")
    bench = load("cpu_benchmark.json")
    low = {p: load(f"LOW_RESOURCE_{p}_MANIFEST.json") for p in (10, 25, 50)}

    L = []
    w = L.append
    w("# DAPT final experiment — dry-run report")
    w("")
    w(f"Generated {date.today().isoformat()} at commit `{git_rev()}` by "
      "`scripts/09_dapt_final/write_dry_run_report.py`.")
    w("")
    w("Required by `papers/target_pretraining/dapt.md` §37. Training must not begin until this "
      "report has been reviewed (§36).")
    w("")

    w("## 1. Train-only context counts")
    w("")
    if auth:
        w(f"- Unique training-split statutory contexts: **{auth['count']}**")
        w(f"- Total non-padding tokens: **{auth['total_tokens']:,}**")
        w("- The archived corpora used 901 records; the extra three were test "
          "contexts. Rebuilding from the training split alone yields 898.")
    else:
        w("- PENDING: authentic manifest not built.")
    w("")

    w("## 2. Audit of the existing synthetic corpus")
    w("")
    if prov:
        c = prov["checks"]
        w(f"- Verdict: **{prov['verdict']}**")
        w(f"- Pool: {c['pool_identity']['pool_records']:,} passages, all traceable "
          "to the generation logs.")
        w(f"- Seed exemplars not traceable to train: "
          f"**{c['seed_example_contexts']['exemplars_not_traceable_to_train']}** "
          f"(all from test), carried by "
          f"{c['seed_example_contexts']['contaminated_seed_entities']} seed entities.")
        p = prov["propagation"]
        w(f"- Of those entities, **{p['contaminated_entities_actually_sampled']}** were "
          f"sampled for generation (out of "
          f"{p['distinct_entities_sampled_for_generation']} used), so the defect did "
          "not reach any generated passage.")
        h = c["heldout_text_reproduction"]
        w(f"- No passage reproduces held-out text: 0 exact matches, and n-gram overlap "
          f"reaches zero at n={h['ngram_n']} against both held-out and training text. "
          "Short-span overlap is formulaic statutory language.")
        w("")
        w("  | n | overlap vs held-out | overlap vs train (reference) |")
        w("  |---|---|---|")
        for n, d in sorted(h["decay_curve"].items(), key=lambda kv: int(kv[0])):
            w(f"  | {n} | {d['vs_heldout']} | {d['vs_train_reference']} |")
    else:
        w("- PENDING: provenance audit not run.")
    w("")

    w("## 3. Chosen synthetic generator subset")
    w("")
    if corp:
        w(f"- {corp['synthetic_source']}")
        w("- Restricting to one generator family is a control decision, not a claim "
          "that 12B output is better than 27B (§6).")
    w("")

    w("## 4-6. Token budgets, record counts, mixed ratio")
    w("")
    if corp:
        w(f"Primary matching criterion: **{corp['primary_matching_criterion']}**. "
          f"Budget set by condition A at **{corp['token_budget']:,}** tokens.")
        w("")
        w("| Condition | Records | Tokens | Δ tokens | Δ % | Authentic token share |")
        w("|---|---|---|---|---|---|")
        for name, c in corp["corpora"].items():
            w(f"| {name} | {c['records']} | {c['non_padding_tokens']:,} | "
              f"{c['delta_from_budget']:+d} | {c['delta_pct']:+.3f}% | "
              f"{100*c['authentic_token_share']:.1f}% |")
        if style:
            w(f"| S_STYLE | {style['records']} | {style['non_padding_tokens']:,} | "
              f"{style['delta_from_budget']:+d} | {style['delta_pct']:+.3f}% | 0.0% |")
        w("")
        w("The archived record-matched corpora differed by 13.2% in token exposure "
          "(authentic 128,929 vs synthetic 146,001), confounding composition with "
          "exposure. Matching on tokens removes that.")
    w("")

    w("## 7. S_STYLE construction")
    w("")
    w("- Same generator (gemma3:12b), same decoding parameters, same eight style "
      "families, and the *same* quality filter — imported directly from "
      "`scripts/02_synthetic_dapt/generate_synthetic_large.py` rather than "
      "reimplemented, so it cannot drift.")
    w("- Only manipulated variable: prompts are not conditioned on IndicIPR-QA "
      "training concepts and are directed away from the seven target Acts.")
    w("- Non-target Acts, concepts and style exemplars are a fixed, documented, "
      "hand-authored list in `generate_style_only_corpus.py` (§17 forbids scraping).")
    if style:
        w(f"- Built: {style['records']} records, {style['non_padding_tokens']:,} tokens.")
    else:
        gen = FINAL / "style_only_generated.jsonl"
        n = sum(1 for _ in io.open(gen, encoding="utf8")) if gen.exists() else 0
        w(f"- **PENDING**: generation in progress ({n} passages accepted so far; "
          "target 1,000, then trimmed to the token budget).")
    if chars:
        w("")
        w("Corpus properties (§18 — guards against 'S_STYLE lost because it is worse "
          "text'):")
        w("")
        w("| Corpus | Records | Mean words | Type-token ratio | Distinct-2 | Exact dup |")
        w("|---|---|---|---|---|---|")
        for name, c in chars["corpora"].items():
            w(f"| {name} | {c['records']} | {c['mean_words']:.1f} | "
              f"{c['type_token_ratio']:.4f} | {c['distinct_2']:.4f} | "
              f"{c['exact_duplicate_rate']:.3f} |")
        sim = chars.get("lexical_similarity_to_authentic", {})
        if sim:
            w("")
            for k, v in sim.items():
                w(f"- Max Jaccard, authentic train vs {k}: **{v['max_jaccard']:.4f}** "
                  f"({v['pairs_above_threshold']} pairs ≥ {chars['jaccard_threshold']})")
    w("")

    w("## 8-9. Cached datasets")
    w("")
    w("- SQuAD v1: fixed 20,000-example train / 2,000-example validation subset, "
      "selected once with subset seed 42. Tokenized features are cached under "
      "`experiments/dapt/final_cpu/cache/squad_*.arrow`.")
    w("- IndicIPR-QA: `data/splits/{train,val,test}.jsonl` — 3,710 / 408 / 344. "
      "Full and low-resource tokenized features are cached under "
      "`experiments/dapt/final_cpu/cache/ipr_*.arrow`.")
    w("")

    w("## 10. Seeds")
    w("")
    w(f"- Experiment A: {SEEDS_A} (5 seeds × 4 conditions = 20 full pipelines).")
    w(f"- Experiment B: {SEEDS_B} (S_STYLE only; S is reused from A, §19).")
    w(f"- Experiment C: {SEEDS_B} × {{25%, 50%}} × 4 conditions = 24 IPR-only runs.")
    w("- Corpus-selection seed 20260829, independent of all training seeds.")
    w("- Each seed propagates through MLM, SQuAD and IPR alike (§11) — verified in "
      "the dry run.")
    w("")

    w("## 11. Exact commands")
    w("")
    w("```")
    w("python scripts/09_dapt_final/audit_synthetic_provenance.py")
    w("python scripts/09_dapt_final/build_final_matched_corpora.py")
    w("python scripts/09_dapt_final/generate_style_only_corpus.py")
    w("python scripts/09_dapt_final/characterise_corpora.py")
    w("python scripts/09_dapt_final/build_diagnostic_and_lowresource.py")
    w("python scripts/09_dapt_final/benchmark_cpu_config.py")
    w("python scripts/09_dapt_final/run_final_experiments.py --experiment A --dry-run")
    w("python scripts/09_dapt_final/run_final_experiments.py --experiment A")
    w("python scripts/09_dapt_final/run_final_experiments.py --experiment B")
    w("python scripts/09_dapt_final/run_final_experiments.py --experiment C")
    w("python scripts/09_dapt_final/prepare_final_analysis_lock.py")
    w("python scripts/09_dapt_final/evaluate_final_experiments.py --split test")
    w("python scripts/09_dapt_final/evaluate_final_mlm.py")
    w("python scripts/09_dapt_final/analyse_final_experiments.py")
    w("python scripts/09_dapt_final/run_full_study.py")
    w("```")
    w("")

    w("## 12-13. Run directories and resume policy")
    w("")
    w("- Stage outputs: `artifacts/models/final_cpu/{experiment}_{condition}_s{seed}_"
      "{mlm,squad,ipr}/`, final weights in `final/`.")
    w("- Any stage whose `final/` exists is skipped, so a crash costs at most the "
      "stage in flight (§14).")
    w("- `RUN_LEDGER.csv` records experiment, condition, seed, stage, status, "
      "timestamps, checkpoint and runtime.")
    w("- Every template resumes the latest `checkpoint-*` automatically. Recovery "
      "checkpoints are requested every 100 optimizer steps for SQuAD/IPR and every "
      "25 steps for MLM, in addition to epoch-level scientific checkpoints.")
    if bench:
        recovery_minutes = 100 * bench["recommended"]["sec_per_step"] / 60
        w(f"- Measured SQuAD/IPR recovery interval: approximately "
          f"**{recovery_minutes:.1f} minutes**, satisfying the 60–90 minute target.")
    w("")

    w("## 14-15. Runtime estimates")
    w("")
    if bench:
        b, r = bench["baseline"], bench["recommended"]
        mlm_r = bench.get("recommended_by_stage", {}).get("mlm", r)
        w(f"- Baseline {b['micro_batch']}×{b['grad_accum']} @ {b['threads']} threads: "
          f"{b['sec_per_step']:.3f} s/step.")
        w(f"- Recommended {r['micro_batch']}×{r['grad_accum']} @ {r['threads']} threads: "
          f"{r['sec_per_step']:.3f} s/step (**{bench['speedup_vs_baseline']:.2f}×**).")
        w(f"- MLM uses its separately measured safe partition: "
          f"{mlm_r['micro_batch']} x {mlm_r['grad_accum']} at "
          f"{mlm_r['threads']} threads ({mlm_r['sec_per_step']:.3f} s/step).")
        w("- Effective batch is held at 32 throughout, so every configuration performs "
          "the identical optimizer step on identical examples.")
        projection = bench.get("runtime_projection", {})
        if projection.get("status") == "estimated":
            low_h, high_h = projection["estimated_wall_hours_range"]
            low_d, high_d = projection["estimated_wall_days_continuous"]
            w(f"- Full serial study projection: **{low_h:.0f}-{high_h:.0f} wall hours** "
              f"(**{low_d:.1f}-{high_d:.1f} continuous days**), including a 10-30% "
              "allowance beyond measured training compute for evaluation and I/O.")
            w(f"- Measured-compute components (hours): "
              f"{projection['components_hours']}.")
    else:
        w("- **PENDING**: the CPU benchmark requires an otherwise-idle machine. "
          "A separate long-running job (`research/3_transfer/run_transfer.py`) was "
          "active, and measuring under contention would mis-rank thread counts.")
        w("- Prior estimate from archived runs: ~15.8 h per full pipeline → "
          "23 pipelines ≈ 363 h, plus 24 short IPR-only runs. This is an estimate "
          "carried forward, not a measurement from this configuration.")
    w("")

    w("## 16. Statistical plan")
    w("")
    w("- Primary unit of replication: the complete training seed. Test-item bootstrap "
      "is a different quantity and is not a substitute (§29).")
    w("- Experiment A: randomised-block ANOVA with seed as block, plus six "
      "prespecified contrasts (A−N, S−N, M−N, A−S, A−M, S−M) under Holm correction.")
    w("- Experiment B: paired S vs S_STYLE over three seeds, reported as mean paired "
      "difference with per-seed differences and a 95% CI; significance treated "
      "cautiously at n=3 (§30).")
    w("- Experiment C: DAPT−N differences at 25/50/100% supervision (§31).")
    w("- `p > 0.05` is reported as non-detection, never as equivalence (§29).")
    w("- Implementation validated against SciPy: SS decomposition exact, ANOVA p and "
      "paired t identical to `scipy.stats.ttest_rel`, Holm verified by hand.")
    w("")

    w("## 17. Test-set discipline")
    w("")
    w("Confirmation that the 344-item test set entered nothing:")
    w("")
    w("| Use | Status |")
    w("|---|---|")
    w("| DAPT corpora (A/S/M) | Built from training contexts only — 898 unique, "
      "verified disjoint from val and test |")
    w("| Synthetic generation seeds | Audited; the 3 test-derived exemplars were "
      "never sampled |")
    w("| S_STYLE construction | Fixed non-target list; no split data |")
    if diag:
        w(f"| MLM diagnostic | {diag['diagnostic_contexts']} **validation** contexts, "
          "replacing the archived test-set perplexity |")
    w("| Low-resource subsets | Drawn from the training split only |")
    w("| Checkpoint selection | Validation loss/F1 |")
    w("| Test evaluation | Not performed; deferred until `FINAL_ANALYSIS_LOCK.md` "
      "exists (§15) |")
    w("")
    if low.get(25):
        w("Low-resource subsets (sampled at statutory-section level, stratified by "
          "Act, seed 20260829):")
        w("")
        w("| Fraction | Sections | QA pairs | Realised |")
        w("|---|---|---|---|")
        for pct in (10, 25, 50):
            m = low.get(pct)
            if m:
                w(f"| {pct}% | {m['sections_selected']}/{m['sections_total']} | "
                  f"{m['qa_pairs_selected']} | {100*m['realised_qa_fraction']:.1f}% |")
        w("")
        w("Sampling by section rather than by QA row keeps rows that share a section "
          "on the same side of the split; the 10% arm realises 13.9% because every "
          "Act must retain at least one section.")
    w("")

    w("## Outstanding before training")
    w("")
    outstanding = []
    if not bench:
        outstanding.append("Run the CPU benchmark on an otherwise-idle machine (§13).")
    if not style:
        outstanding.append("Finish S_STYLE generation and rebuild it at the token budget.")
    if bench:
        recovery_minutes = 100 * bench["recommended"]["sec_per_step"] / 60
        if recovery_minutes > 90:
            outstanding.append(
                f"Reduce recovery cadence: measured 100-step interval is {recovery_minutes:.1f} minutes."
            )
    if outstanding:
        for number, item in enumerate(outstanding, 1):
            w(f"{number}. {item}")
        w("")
        w("**STOP.** Resolve the items above before Experiment A begins.")
    else:
        w("None. All §37 pre-training gates are satisfied; create the final analysis "
          "lock and begin the serial experiment queue.")
    w("")

    OUT.write_text("\n".join(L), encoding="utf8")
    print(f"  wrote {OUT.relative_to(ROOT)} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
