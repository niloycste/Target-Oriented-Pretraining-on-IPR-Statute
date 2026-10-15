# Final DAPT analysis lock

Locked: **2026-09-01T20:03:07+05:30** at Git revision `85d0a12`.

This lock freezes all corpus, split, training, evaluation and analysis inputs before final test evaluation.
Any hash mismatch causes the canonical test evaluator to refuse execution.

## Frozen design

- Experiment A: N/A/S/M × seeds 7, 42, 99, 123, 2026.
- Experiment B: S versus S_STYLE × seeds 7, 42, 99.
- Experiment C: N/A/S/M × seeds 7, 42, 99 × 25%, 50%, 100% supervision.
- Primary metrics: token-level EM and F1 under the canonical sliding-window decoder.
- MLM loss/perplexity: validation-only secondary diagnostic.

## Corpus lock

| Condition | Records | Tokens | Delta |
|---|---:|---:|---:|
| A | 898 | 128,727 | +0.000% |
| S | 861 | 128,713 | -0.011% |
| M | 879 | 128,663 | -0.050% |
| S_STYLE | 858 | 128,723 | -0.003% |

Validation MLM diagnostic: 104 contexts.

Machine-readable hashes: `experiments/dapt/final_cpu/FINAL_ANALYSIS_LOCK.json`.
