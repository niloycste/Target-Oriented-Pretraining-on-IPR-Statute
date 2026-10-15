# Statutory DAPT experiment protocol

This file is the run contract for the standalone DAPT paper in
`papers/target_pretraining/`. It separates existing archived evidence from the
clean train-only experiments that should be run before final submission claims.

## Current archived evidence

The current manuscript draft may cite the following artifacts as audited
archived evidence:

| Evidence | Artifact |
|---|---|
| Gold-context reader inventory | `results/evaluation/qa_eval_report.json` |
| Paired bootstrap comparisons for C-prime | `results/evaluation/bootstrap_significance_report.json` |
| Archived DAPT corpus manifest | `combined_paper/Springer_Nature_LaTeX_Template/dapt_corpus_manifest.csv` |
| Archived MLM perplexity | `results/evaluation/mlm_eval_report.json` |
| Record-matched single-seed results | `results/experiments/size_matched_dapt/size_matched_results.json` |
| Record-matched three-seed results | `results/experiments/size_matched_dapt/size_matched_multiseed_results.json` |
| Record-matched statistics | `results/experiments/size_matched_dapt/size_matched_multiseed_stats.json` |
| Unified archived and record-matched MLM diagnostic | `results/evaluation/mlm_eval_report.json` |
| Record/token/split audit | `results/experiments/size_matched_dapt/size_matched_corpus_audit.json` |

These artifacts are useful but not final submission evidence for corpus-causal
claims. The historical H-prime/C-prime corpora contain three test contexts. In
the record-matched study, auth901 contains three test contexts covering six test
questions, mixed901 contains one covering two, and synth901 contains none. No
test questions or answer labels are used during MLM pretraining, but the final
Paper A experiment should remove the exposure rather than rely on an audit. The
901-record corpora are also not token-matched.

## Final train-only rerun

### RQ1: Does statutory DAPT help beyond the same SQuAD-transfer route?

Run the following conditions with all DAPT corpora built strictly from the
training split:

1. No DAPT: encoder → SQuAD v1 → IndicIPR-QA.
2. Authentic DAPT: train-only authentic statutory contexts → SQuAD v1 →
   IndicIPR-QA.
3. Synthetic DAPT: train-seed-generated statutory-style contexts → SQuAD v1 →
   IndicIPR-QA.
4. Mixed DAPT: train-only authentic + train-seed-generated synthetic contexts →
   SQuAD v1 → IndicIPR-QA.

### RQ2: Which corpus property matters?

For the core comparison:

- match record counts;
- match token budgets;
- keep the encoder fixed;
- keep MLM probability, max sequence length, batch size, learning rate, epoch
  count, SQuAD subset, final QA fine-tuning settings, and decoder fixed;
- repeat the full pipeline with at least three seeds per condition.

Recommended extension:

- run small, medium, and large synthetic-scale conditions;
- include one target-aligned synthetic condition and one off-target or weakened
  synthetic control;
- repeat the core comparison for a second encoder family.

### RQ3: Does MLM fit predict QA performance?

For every DAPT checkpoint:

- evaluate MLM loss and perplexity on held-out statutory contexts that are not
  present in the DAPT corpus;
- report perplexity as a secondary diagnostic only;
- do not select the downstream reader by test-set perplexity.

## Required outputs

Write final outputs under `experiments/dapt/`:

```text
experiments/dapt/
  configs/
  manifests/
  runs/
  results/
  stats/
```

At minimum, the final archive should contain:

- exact corpus manifest with counts, token counts, unique normalized records,
  split eligibility, and SHA-256 hashes;
- per-run training config;
- per-run evaluation JSON;
- multi-seed summary JSON;
- statistical comparison JSON;
- MLM diagnostic JSON;
- generated figures.

## Reporting rules

- Do not call any archived DAPT route a single-factor ablation.
- Do not describe a non-significant result as equivalence.
- Report effect sizes with p-values.
- State whether confidence intervals are paired or unpaired.
- Use token-level F1 and EM as manuscript metrics.
- Keep character-F1, if used, as a diagnostic only.
