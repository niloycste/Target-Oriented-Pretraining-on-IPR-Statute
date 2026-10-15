# Experiment D: equal-update DAPT sensitivity

Status: complete as of 2 October 2026. All 15 adapted routes finished training
and validation, the test analysis was frozen, and all 20 condition--seed cells
(including five reused no-DAPT readers) were evaluated on the test split.

This workflow is a post-hoc sensitivity analysis. It leaves the frozen A/B/C
artifacts untouched and writes only to:

- `experiments/dapt/equal_update/`
- `artifacts/models/equal_update/`

Design: authentic, synthetic, and mixed DAPT are each trained with five seeds
(`7, 42, 99, 123, 2026`), exactly 100 MLM optimizer updates, 10 warmup updates,
and the frozen downstream SQuAD and IndicIPR-QA settings. The no-DAPT results
are reused from Experiment A because no MLM stage exists for that condition.

## 1. Safe dry run

```powershell
python scripts/dapt_equal_update/run_experiment_d.py
python scripts/dapt_equal_update/smoke_test.py
```

The first command creates the protocol, generated scripts, and a machine-
readable audit. It never trains unless `--execute` is supplied.

## 2. Execute training and validation

```powershell
python scripts/dapt_equal_update/run_experiment_d.py --execute
```

The driver is serial, resumable, and records a run ledger. Each MLM route
writes `training_telemetry.json` containing completed optimizer updates, exact
non-padding tokens processed, and the loss/learning-rate trajectory.

## 3. Inspect validation, then freeze the test analysis

```powershell
python scripts/dapt_equal_update/analyse_experiment_d.py --split val
python scripts/dapt_equal_update/freeze_test_analysis.py --confirm
```

Do not freeze until all 15 adapted validation runs are present and reviewed.

## 4. Canonical test evaluation and statistics

```powershell
python scripts/dapt_equal_update/evaluate_experiment_d.py --split test
python scripts/dapt_equal_update/analyse_experiment_d.py --split test
```

The final report contains per-seed EM/F1, a seed-blocked ANOVA, paired A/N,
S/N, and M/N differences with 95% t intervals, and Holm-adjusted p-values.
Non-detection must not be described as equivalence.

## 5. Read-only evidence audit

```powershell
python scripts/dapt_equal_update/audit_experiment_d.py
```

The audit performs no training or inference. It re-scores saved validation and
test answers, verifies the frozen inputs, checkpoints, training telemetry,
statistics, and Experiment D manuscript tables, and writes
`papers/target_pretraining/submission_audit_equal_update_20261002.json`.
