# DAPT final experiment — dry-run report

Generated 2026-09-01 at commit `85d0a12` by `scripts/09_dapt_final/write_dry_run_report.py`.

Required by `papers/target_pretraining/dapt.md` §37. Training must not begin until this report has been reviewed (§36).

## 1. Train-only context counts

- Unique training-split statutory contexts: **898**
- Total non-padding tokens: **128,727**
- The archived corpora used 901 records; the extra three were test contexts. Rebuilding from the training split alone yields 898.

## 2. Audit of the existing synthetic corpus

- Verdict: **REUSE the archived synthetic pool -- the seed-pool defect did not propagate to any generated passage**
- Pool: 13,211 passages, all traceable to the generation logs.
- Seed exemplars not traceable to train: **3** (all from test), carried by 19 seed entities.
- Of those entities, **0** were sampled for generation (out of 1282 used), so the defect did not reach any generated passage.
- No passage reproduces held-out text: 0 exact matches, and n-gram overlap reaches zero at n=25 against both held-out and training text. Short-span overlap is formulaic statutory language.

  | n | overlap vs held-out | overlap vs train (reference) |
  |---|---|---|
  | 12 | 204 | 517 |
  | 15 | 61 | 99 |
  | 20 | 16 | 6 |
  | 25 | 0 | 0 |
  | 30 | 0 | 0 |
  | 40 | 0 | 0 |

## 3. Chosen synthetic generator subset

- gemma3:12b only (dapt.md sec. 6 single-generator control)
- Restricting to one generator family is a control decision, not a claim that 12B output is better than 27B (§6).

## 4-6. Token budgets, record counts, mixed ratio

Primary matching criterion: **total non-padding MLM tokens**. Budget set by condition A at **128,727** tokens.

| Condition | Records | Tokens | Δ tokens | Δ % | Authentic token share |
|---|---|---|---|---|---|
| A | 898 | 128,727 | +0 | +0.000% | 100.0% |
| S | 861 | 128,713 | -14 | -0.011% | 0.0% |
| M | 879 | 128,663 | -64 | -0.050% | 50.0% |
| S_STYLE | 858 | 128,723 | -4 | -0.003% | 0.0% |

The archived record-matched corpora differed by 13.2% in token exposure (authentic 128,929 vs synthetic 146,001), confounding composition with exposure. Matching on tokens removes that.

## 7. S_STYLE construction

- Same generator (gemma3:12b), same decoding parameters, same eight style families, and the *same* quality filter — imported directly from `scripts/02_synthetic_dapt/generate_synthetic_large.py` rather than reimplemented, so it cannot drift.
- Only manipulated variable: prompts are not conditioned on IndicIPR-QA training concepts and are directed away from the seven target Acts.
- Non-target Acts, concepts and style exemplars are a fixed, documented, hand-authored list in `generate_style_only_corpus.py` (§17 forbids scraping).
- Built: 858 records, 128,723 tokens.

Corpus properties (§18 — guards against 'S_STYLE lost because it is worse text'):

| Corpus | Records | Mean words | Type-token ratio | Distinct-2 | Exact dup |
|---|---|---|---|---|---|
| A | 898 | 125.5 | 0.0367 | 0.2227 | 0.000 |
| S | 861 | 130.5 | 0.0331 | 0.2653 | 0.000 |
| M | 879 | 128.2 | 0.0398 | 0.2788 | 0.000 |
| S_STYLE | 858 | 131.5 | 0.0351 | 0.2871 | 0.000 |

- Max Jaccard, authentic train vs S: **0.0427** (0 pairs ≥ 0.8)
- Max Jaccard, authentic train vs S_STYLE: **0.0448** (0 pairs ≥ 0.8)

## 8-9. Cached datasets

- SQuAD v1: fixed 20,000-example train / 2,000-example validation subset, selected once with subset seed 42. Tokenized features are cached under `experiments/dapt/final_cpu/cache/squad_*.arrow`.
- IndicIPR-QA: `data/splits/{train,val,test}.jsonl` — 3,710 / 408 / 344. Full and low-resource tokenized features are cached under `experiments/dapt/final_cpu/cache/ipr_*.arrow`.

## 10. Seeds

- Experiment A: [7, 42, 99, 123, 2026] (5 seeds × 4 conditions = 20 full pipelines).
- Experiment B: [7, 42, 99] (S_STYLE only; S is reused from A, §19).
- Experiment C: [7, 42, 99] × {25%, 50%} × 4 conditions = 24 IPR-only runs.
- Corpus-selection seed 20260829, independent of all training seeds.
- Each seed propagates through MLM, SQuAD and IPR alike (§11) — verified in the dry run.

## 11. Exact commands

```
python scripts/09_dapt_final/audit_synthetic_provenance.py
python scripts/09_dapt_final/build_final_matched_corpora.py
python scripts/09_dapt_final/generate_style_only_corpus.py
python scripts/09_dapt_final/characterise_corpora.py
python scripts/09_dapt_final/build_diagnostic_and_lowresource.py
python scripts/09_dapt_final/benchmark_cpu_config.py
python scripts/09_dapt_final/run_final_experiments.py --experiment A --dry-run
python scripts/09_dapt_final/run_final_experiments.py --experiment A
python scripts/09_dapt_final/run_final_experiments.py --experiment B
python scripts/09_dapt_final/run_final_experiments.py --experiment C
python scripts/09_dapt_final/prepare_final_analysis_lock.py
python scripts/09_dapt_final/evaluate_final_experiments.py --split test
python scripts/09_dapt_final/evaluate_final_mlm.py
python scripts/09_dapt_final/analyse_final_experiments.py
python scripts/09_dapt_final/run_full_study.py
```

## 12-13. Run directories and resume policy

- Stage outputs: `artifacts/models/final_cpu/{experiment}_{condition}_s{seed}_{mlm,squad,ipr}/`, final weights in `final/`.
- Any stage whose `final/` exists is skipped, so a crash costs at most the stage in flight (§14).
- `RUN_LEDGER.csv` records experiment, condition, seed, stage, status, timestamps, checkpoint and runtime.
- Every template resumes the latest `checkpoint-*` automatically. Recovery checkpoints are requested every 100 optimizer steps for SQuAD/IPR and every 25 steps for MLM, in addition to epoch-level scientific checkpoints.
- Measured SQuAD/IPR recovery interval: approximately **38.3 minutes**, satisfying the 60–90 minute target.

## 14-15. Runtime estimates

- Baseline 8×4 @ 24 threads: 24.990 s/step.
- Recommended 16×2 @ 24 threads: 22.964 s/step (**1.09×**).
- MLM uses its separately measured safe partition: 8 x 4 at 24 threads (16.493 s/step).
- Effective batch is held at 32 throughout, so every configuration performs the identical optimizer step on identical examples.
- Full serial study projection: **299-353 wall hours** (**12.4-14.7 continuous days**), including a 10-30% allowance beyond measured training compute for evaluation and I/O.
- Measured-compute components (hours): {'squad_23_models': 186.91, 'full_ipr_23_models': 55.02, 'low_resource_ipr_24_models': 22.27, 'mlm_18_models': 7.17}.

## 16. Statistical plan

- Primary unit of replication: the complete training seed. Test-item bootstrap is a different quantity and is not a substitute (§29).
- Experiment A: randomised-block ANOVA with seed as block, plus six prespecified contrasts (A−N, S−N, M−N, A−S, A−M, S−M) under Holm correction.
- Experiment B: paired S vs S_STYLE over three seeds, reported as mean paired difference with per-seed differences and a 95% CI; significance treated cautiously at n=3 (§30).
- Experiment C: DAPT−N differences at 25/50/100% supervision (§31).
- `p > 0.05` is reported as non-detection, never as equivalence (§29).
- Implementation validated against SciPy: SS decomposition exact, ANOVA p and paired t identical to `scipy.stats.ttest_rel`, Holm verified by hand.

## 17. Test-set discipline

Confirmation that the 344-item test set entered nothing:

| Use | Status |
|---|---|
| DAPT corpora (A/S/M) | Built from training contexts only — 898 unique, verified disjoint from val and test |
| Synthetic generation seeds | Audited; the 3 test-derived exemplars were never sampled |
| S_STYLE construction | Fixed non-target list; no split data |
| MLM diagnostic | 104 **validation** contexts, replacing the archived test-set perplexity |
| Low-resource subsets | Drawn from the training split only |
| Checkpoint selection | Validation loss/F1 |
| Test evaluation | Not performed; deferred until `FINAL_ANALYSIS_LOCK.md` exists (§15) |

Low-resource subsets (sampled at statutory-section level, stratified by Act, seed 20260829):

| Fraction | Sections | QA pairs | Realised |
|---|---|---|---|
| 10% | 55/541 | 516 | 13.9% |
| 25% | 135/541 | 942 | 25.4% |
| 50% | 270/541 | 1852 | 49.9% |

Sampling by section rather than by QA row keeps rows that share a section on the same side of the split; the 10% arm realises 13.9% because every Act must retain at least one section.

## Outstanding before training

None. All §37 pre-training gates are satisfied; create the final analysis lock and begin the serial experiment queue.
