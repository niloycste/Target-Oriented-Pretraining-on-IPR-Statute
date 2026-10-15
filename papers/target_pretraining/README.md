# Statutory DAPT paper

Updated: 2 October 2026.

The manuscript is `target-pretraining.tex`; the compiled output is
`target-pretraining.pdf`. This paper studies additional domain-adaptive
pretraining for gold-context statutory extractive QA. Retrieval experiments
belong to the separate retrieval/reference paper.

## Current status

The A/B/C package and the post-hoc equal-update Experiment D are complete.
The evidence contains 62 unique readers: 47 from A--C and 15 new adapted
readers from D. Experiment D also reuses the five matched no-DAPT readers from
Experiment A, giving 20 D condition--seed cells. Every reader has saved
validation and test predictions. The paper leads with Experiment D, treats
Experiment A as a three-epoch sensitivity analysis, and reports validation
scores separately from test scores. Appendix C preserves older, more
confounded experiments.

The A--C reporting audit independently re-scores 16,168 test and 19,176
validation answers and verifies 47 reader checkpoint fingerprints. The
Experiment D audit separately re-scores 6,880 test and 8,160 validation
answers, checks its 20-cell inventory, verifies the 15 new checkpoint
fingerprints and training telemetry, and reproduces its statistics and
manuscript tables. The audit JSON files record no failures in the implemented
checks. These checks verify retained evidence, not independent replication of
training or legal correctness.

See `REVISION_NOTES_20260920.md` for corrections and remaining author actions.
This is not yet an upload-ready submission package: declarations, reviewer
access and confidential companion-paper details need author confirmation.

## Reproduce the reporting checks

From the repository root:

```powershell
python scripts/09_dapt_final/audit_submission_evidence.py
python scripts/dapt_equal_update/audit_experiment_d.py
python scripts/08_figures/make_final_dapt_figures.py
```

None of these commands trains models or runs fresh QA/MLM inference. The
audits read saved predictions and checkpoints and write separate audit JSON
files. The plotting command writes `07_final_full_data`,
`08_final_supervision`, and `09_equal_update_full_data` as PDF/PNG files in
`plots/`. Earlier plots remain on disk but are not in the revised manuscript.

From this paper directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error target-pretraining.tex
bibtex target-pretraining
pdflatex -interaction=nonstopmode -halt-on-error target-pretraining.tex
pdflatex -interaction=nonstopmode -halt-on-error target-pretraining.tex
```

## Evidence map

Paths are relative to the repository root.

| Evidence | Source |
|---|---|
| Experiment D test / validation predictions | `experiments/dapt/equal_update/results.json`, `validation_results.json` |
| Experiment D statistics | `experiments/dapt/equal_update/EXPERIMENT_D_TEST_STATISTICS.json`, `EXPERIMENT_D_VAL_STATISTICS.json` |
| Experiment D frozen test plan | `experiments/dapt/equal_update/EXPERIMENT_D_TEST_LOCK.json` |
| Experiment D training telemetry / models | `artifacts/models/equal_update/` |
| Final test / validation predictions | `experiments/dapt/final_cpu/results.json`, `validation_results.json` |
| A/B/C statistics and interactions | `experiments/dapt/final_cpu/FINAL_STATISTICS.json` |
| Answer-property diagnostics | `experiments/dapt/final_cpu/QA_DIFFICULTY_DIAGNOSTICS.json` |
| Corpus properties | `experiments/dapt/final_cpu/CORPUS_CHARACTERISATION.json` |
| Token budgets / generation provenance | `experiments/dapt/final_cpu/MATCHED_CORPORA_SUMMARY.json`, `SYNTHETIC_PROVENANCE_AUDIT.json` |
| Low-supervision manifests | `experiments/dapt/final_cpu/LOW_RESOURCE_25_MANIFEST.json`, `LOW_RESOURCE_50_MANIFEST.json` |
| Corrected validation-only MLM diagnostic | `experiments/dapt/final_cpu/MLM_DIAGNOSTIC_RESULTS.json` |
| Saved stage scripts / models | `artifacts/models/final_cpu/` |
| Training inventory / amendment | `experiments/dapt/final_cpu/TRAINING_COMPLETION_AUDIT.json`, `PRETEST_AMENDMENT_20260913.json` |
| Archived reader scores | `results/evaluation/qa_eval_report.json` |
| Archived paired-question bootstrap | `results/evaluation/bootstrap_significance_report.json` |
| Archived record-matched study | `results/experiments/size_matched_dapt/` |
| Archived test-context MLM diagnostic | `results/evaluation/mlm_eval_report.json` |

## Interpretation and provenance

Experiment D was added after the A--C results as a post-hoc sensitivity check.
All adapted D routes use 100 optimiser updates, 10 warmup updates, the same
peak learning rate, and the same linear decay. Its validation results were
inspected before a separate test lock was created. The locked evaluation did
not retrain models or change checkpoints. The reporting audits do not
overwrite frozen statistics.

Experiment A matches the initial corpora at approximately 129,000 non-padding
tokens, but its MLM routes take 81--87 updates inside a 100-update warmup.
Experiment D matches 100 updates and the learning-rate schedule, but processed
token totals still differ because the fixed corpora have different record
lengths. Reduced supervision changes final answer-label fine-tuning while
broader task-informed adaptation text and SQuAD transfer remain available.
These limitations are disclosed rather than treated as fully controlled.

The raw seed file includes held-out exemplars; the propagation audit finds that
their associated entities were not used for generated passages. The original
runner is unavailable. The pre-test amendment and post-test MLM loading
correction are explicitly described. Neither the whole raw seed file nor the
entire execution history is described as an untouched, training-only
preregistered experiment.

`dapt.md` remains the historical plan. The 19 September audit is a pre-revision
record; its manuscript section/line numbers refer to an older draft.
