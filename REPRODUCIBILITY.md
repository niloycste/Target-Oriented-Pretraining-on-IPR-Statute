# Reproducibility guide

This guide separates checks of saved evidence from expensive model training.
Run all commands from the repository root. The paper reports results from saved
artifacts in this workspace; a command that only checks those artifacts does not
retrain a model.

## 1. Environment

Create an isolated Python environment and install the pinned packages:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The completed LegalQAEval audit used Python 3.13.7 with `torch 2.10.0`,
`transformers 5.3.0`, `datasets 4.7.0`, `huggingface-hub 1.6.0`,
`numpy 2.4.2`, and `scipy 1.17.1`. The full pinned environment is in
`requirements.txt`.

## 2. Fast checks that do not train models

```powershell
python -X utf8 verify_release.py
python -X utf8 -m unittest discover -s experiments/legalqaeval -p "test_*.py" -v
python -X utf8 experiments/legalqaeval/independent_audit.py
```

The first command checks the released IndicIPR-QA evidence. The tests cover
data exclusions, saved-output validation, scoring, reporting, and overflow
decoding. The independent LegalQAEval audit reconstructs the subsets and
rescales all 72,300 saved predictions without using the production scoring
functions. A passing audit checks artifact and calculation consistency; it is
not independent retraining or legal validation.

## 3. Checks that require retained checkpoints

On the machine used for this study, the original InLegalBERT reader weights are
retained under the parent workspace `artifacts/models/` directory. BERT and
LEGAL-BERT extension weights are under
`artifacts/models/encoder_extension/`. These weights are not all part of the
versioned repository.

```powershell
python -X utf8 scripts/dapt_equal_update/audit_experiment_d.py
python -X utf8 experiments/legalqaeval/verify_readers.py --encoder bert
python -X utf8 experiments/legalqaeval/verify_readers.py --encoder legalbert
python -X utf8 experiments/legalqaeval/integration_check.py
```

These commands verify checkpoint fingerprints, saved predictions, stage
ledgers, MLM telemetry, and reported statistics. They do not train models.

## 4. Main result files

- Equal-update InLegalBERT statistics:
  `experiments/dapt/equal_update/EXPERIMENT_D_TEST_STATISTICS.json`
- BERT extension statistics:
  `experiments/dapt/encoder_extension/bert/test_statistics.json`
- LEGAL-BERT extension statistics:
  `experiments/dapt/encoder_extension/legalbert/test_statistics.json`
- LegalQAEval statistics:
  `experiments/legalqaeval/run_v1/outputs/<encoder>/test_statistics.json`
- Independent external audit:
  `experiments/legalqaeval/INDEPENDENT_AUDIT.json`
- Experiment D manuscript audit:
  `papers/target_pretraining/submission_audit_equal_update_20261002.json`

Validation statistics are stored beside the matching test statistics. Per-item
external predictions are under
`experiments/legalqaeval/run_v1/outputs/<encoder>/<split>/`.

## 5. Frozen model and dataset identities

The extension protocols record exact file hashes and local snapshots:

- BERT: `bert-base-uncased`, snapshot revision
  `86b5e0934494bd15c9632b12f734a8a67f723594`.
- LEGAL-BERT: `nlpaueb/legal-bert-base-uncased`, snapshot revision
  `15b570cbf88259610b082a167dacc190124f60f6`.
- Main encoder: `law-ai/InLegalBERT`; frozen inputs and checkpoint hashes are
  recorded in the original and equal-update experiment records.
- External dataset: `isaacus/LegalQAEval`, commit
  `f6c7aafcc176b25c32e1e052d02ad9bc06c96bcb`.

The external protocol is in
`experiments/legalqaeval/run_v1/PROTOCOL.json`. It records dataset hashes,
overlap rules, decoder settings, input hashes, and the 333 validation and 329
test examples used for the primary subsets.

## 6. Re-run external inference

The wrapper below first checks all 60 final reader fingerprints. It then resumes
or verifies the 120 encoder-by-condition-by-seed-by-split evaluations and
regenerates six statistics reports. It performs inference only; it does not
train the readers.

```powershell
python -X utf8 experiments/legalqaeval/run_all.py
```

The wrapper needs the retained checkpoints and the cached SQuAD reference files
whose hashes are listed in the protocol. For a new external work directory,
download the pinned dataset and prepare the audit inputs explicitly:

```powershell
python -X utf8 experiments/legalqaeval/run.py --work experiments/legalqaeval/run_v2 download --revision f6c7aafcc176b25c32e1e052d02ad9bc06c96bcb
python -X utf8 experiments/legalqaeval/run.py --work experiments/legalqaeval/run_v2 prepare --squad-train-arrow "<squad-train.arrow>" --squad-val-arrow "<squad-validation.arrow>"
```

Review the generated audit and protocol before inference. Do not change the
subset rules after observing scores.

## 7. Training commands

Training is not needed to verify the reported calculations. To repeat or resume
the equal-update and encoder-extension training routes, first review the
protocols and disk requirements:

```powershell
python -X utf8 scripts/dapt_equal_update/run_experiment_d.py --execute
python -X utf8 scripts/dapt_equal_update/encoder_extension.py --encoder bert --execute
python -X utf8 scripts/dapt_equal_update/encoder_extension.py --encoder legalbert --execute
```

These jobs are serial, expensive, and resumable. Do not run two processes for
the same encoder and output directory.

## 8. Build the manuscript

The bibliography file is already generated. After changing the LaTeX source,
run two PDF passes from `papers/target_pretraining/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error target-pretraining.tex
pdflatex -interaction=nonstopmode -halt-on-error target-pretraining.tex
```

The final source and PDF are `target-pretraining.tex` and
`target-pretraining.pdf` in that directory.
