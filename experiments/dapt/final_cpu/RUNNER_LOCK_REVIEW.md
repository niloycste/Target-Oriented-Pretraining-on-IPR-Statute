# DAPT runner review before final test evaluation

Reviewed on 13 September 2026. This review does not modify the original
analysis lock, evaluator, training scripts, checkpoints, or test results.

## Verified evidence

- 44 of the 45 locked files still match their original SHA-256 hashes.
  This includes the corpora, splits, training templates, CPU benchmark,
  canonical decoder, test evaluator, and final analysis script.
- The only mismatch is `scripts/09_dapt_final/run_final_experiments.py`.
- The current runner's source builders reproduce the executable syntax trees
  of all 88 saved stage scripts: 18 MLM, 23 SQuAD, and 47 IPR scripts.
  This comparison checks saved code, not a new training run.
- The earlier completion audit found all 47 reader checkpoint hashes consistent
  with their validation records; saved per-question scores reproduce the
  aggregate validation EM/F1 values.
- Final `results.json` and `FINAL_STATISTICS.json` are absent.

## Current runner behaviour inspected

The runner implements serial execution, retries, recovery-checkpoint validation,
completed-stage skipping and cleanup, ledger reconciliation, local base-encoder
materialisation, low-resource split materialisation, and validation evaluation.
The base-encoder materialisation sets `tie_word_embeddings=False`; Windows
data-loader workers are set to zero. These are observations about the current
runner, not verified differences from its original version. The saved stage
scripts agree with the current source builders, including their training
settings. The runner invokes validation evaluation, not final test evaluation.

## Unresolved provenance limitation

The original runner is absent from Git revision `85d0a12`, and no original
source copy was found in the workspace. The original lock stores only its hash.
Consequently, an exact before/after comparison is not possible from the available
evidence. Agreement with saved stage scripts cannot prove that every change was
operational only, reconstruct overwritten earlier scripts, or establish that
resuming interrupted training is numerically identical to uninterrupted training.

The current source builder also imports
`scripts/05_size_matched_dapt/size_matched_dapt_experiment.py`, which was not in
the original 45-file hash list. Saved generated-script checks provide evidence
about its present output, but a later amendment should freeze this dependency
explicitly as well as the reviewed runner and checkpoint inventory.

## Decision and next step

The available-artifact review is complete, but equivalence to the original
runner is not established. The original strict test gate remains in place.
Do not regenerate the original lock and describe it as the September 1 lock.

Either recover the original runner for an exact comparison, or obtain explicit
author approval for a dated, disclosed pre-test amendment that preserves the
original lock, records this limitation, freezes current inputs and checkpoints,
and permits only the reviewed hash changes. Any further unreviewed change must
still fail validation. No final test evaluation was launched by this review.

## Recent manuscript results

`papers/target_pretraining/target-pretraining.tex`, Section 9, Table 11, includes
all 47 recent readers as 13 validation-summary rows: Experiment A (20 readers),
Experiment B (3 additional style-only readers), and Experiment C (24 readers).
These are results on 408 validation questions, not the 344-question test set.
Archived test results and figures remain separate. Final test statistics and
MLM diagnostics have not been represented as completed results.

Reproduce the review with `python scripts/09_dapt_final/review_runner_lock.py`.
Machine-readable evidence is in `RUNNER_LOCK_REVIEW.json`; checkpoint and score
checks are in `TRAINING_COMPLETION_AUDIT.json`.
