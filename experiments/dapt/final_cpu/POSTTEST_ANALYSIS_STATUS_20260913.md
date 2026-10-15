# Post-test analysis status — 13 September 2026

## Completed

- All 47 test evaluations are saved in results.json.
- The exact A/B/C condition, seed and supervision-fraction design was checked.
- All 16,168 prediction rows (47 x 344) have aligned, unique question IDs. Their EM/F1 averages reproduce the saved run summaries within rounding tolerance.
- analyse_final_experiments.py completed without --allow-partial and wrote FINAL_STATISTICS.json and QA_DIFFICULTY_DIAGNOSTICS.json.
- Independent checks reproduced Experiment A descriptive means/SDs and paired-test p-values for both EM and F1.
- The original analysis lock and approved pre-test amendment passed verification before evaluation. Neither was modified.

## MLM diagnostic stopped: checkpoint-loading issue

The diagnostic was started using the frozen evaluate_final_mlm.py. Loading emitted a missing cls.predictions.decoder.bias warning. Inspection of base_encoder confirmed:

- config.json has tie_word_embeddings=false (Transformers 5.3.0).
- model.safetensors contains cls.predictions.bias and cls.predictions.decoder.weight, but not cls.predictions.decoder.bias.
- The loaded prediction bias matches the saved tensor (norm approximately 244.79), while the decoder bias is a separate zero tensor.
- Calling model.tie_weights() did not restore the bias alias under this configuration.
- The A_A_s7_mlm checkpoint has the same absent decoder-bias key and configuration.

The run was stopped. Its partial results were preserved as MLM_DIAGNOSTIC_RESULTS_incomplete_loader_issue_20260913.json and must not be used in the manuscript. FINAL_STATISTICS.json does not incorporate them.

This observation establishes a diagnostic loading problem; it does not establish that restoring an alias alone faithfully recovers every trained MLM head. The save/load behavior and training-time bias need review before choosing a correction. No training checkpoints, frozen scripts, or manuscript results were changed. No retraining was launched.

## Full-data test summary (five seeds)

### Completion update — 14 September 2026

The approved bias-loading investigation and correction are complete. See
MLM_BIAS_CORRECTION_20260913.md and its hashed JSON manifest. The corrected
evaluator exited successfully after all 19 checkpoint evaluations and tensor/hash
checks. The original baseline and restored baseline had identical tensors and
logits. All 19 diagnostics use 104 validation contexts and 2,092 masked tokens.
The earlier stopped-run observations above are retained as history, not the
current status. No checkpoints or frozen evaluation scripts were changed.
FINAL_STATISTICS.json has been regenerated to reference the completed diagnostic.

| Condition | Mean F1 | Sample SD |
|---|---:|---:|
| No DAPT | 85.862624 | 1.082675 |
| Authentic | 85.746362 | 0.585905 |
| Synthetic | 85.391397 | 0.669018 |
| Mixed | 85.478109 | 0.554225 |

The seed-blocked condition test gives F(3,12)=0.635163, p=0.60651601. No statistically significant condition effect was detected; this is not evidence of equivalence. The completed MLM-head loading review is disclosed in the correction note above.
