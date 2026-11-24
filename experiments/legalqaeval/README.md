# LegalQAEval external transfer experiment

Status: scripts prepared and tested; the pinned dataset has been downloaded.
Evaluation completed on 24 November 2026 at 13:33 IST: 60 readers, both splits,
120 evaluations and six verified statistics reports. A separate independent
audit rescored all 72,300 predictions and reconstructed the overlap exclusions,
metrics and statistics. See `INDEPENDENT_AUDIT.json` and run it again with:

```powershell
python -X utf8 experiments/legalqaeval/independent_audit.py
```

The numerical external section has been inserted into the manuscript.
`run_v1/EXECUTION_STATUS.json` records the completed 120 evaluations, and the
timestamped logs retain the execution trace.
Pinned revision: `f6c7aafcc176b25c32e1e052d02ad9bc06c96bcb`.
The primary subsets contain **333 validation and 329 test examples**. The test
source counts are 234 MS MARCO, 54 HotpotQA and 41 Natural Questions examples.
See `run_v1/AUDIT.json` for all counts and item-level prepared files for flags.
This experiment performs inference using existing readers. It does not train
models or modify completed training checkpoints.

The completed LEGAL-BERT and BERT readers were rescored and their fingerprints,
telemetry and statistical reports verified. See `legalbert_verification.json`
and `bert_verification.json`. `INTEGRATION_CHECK.json` records strict model
loading, valid source spans and independent metric checks on four validation
examples per encoder. This is an implementation check, not a performance result.

The original InLegalBERT checkpoints are retained at the parent workspace's
`artifacts/models/`, rather than the Git repository's model directory. The
execution wrapper resolves that location and checks every reader against its
original evaluation fingerprint. This pre-scoring location amendment is saved
in `run_v1/EXECUTION_AMENDMENT.json`. No retraining or score-based selection is
involved. Use the wrapper for this machine's complete execution:

```powershell
python -X utf8 experiments/legalqaeval/run_all.py
```

Do not launch a second wrapper while one is active. It verifies all 60 original
reader fingerprints, runs the integration check, evaluates each encoder on
validation then test, and regenerates each statistics report. The raw pipeline
commands below remain useful on a layout where all expected model paths exist.
An interrupted wrapper can be rerun; completed external reader outputs are
verified before skipping. `EXECUTION_STATUS.json` becomes `COMPLETE` only after
all 120 split evaluations and six statistical reports pass validation.

`finish_report.py` is the separately launched reporting step. It waits for
complete evaluation, validates all outputs again and uses `apply_patch` to
replace only the marked LegalQAEval section in the manuscript. It refuses if
that section was edited after scheduling, preserving a generated section for
review instead. Its status is in `run_v1/MANUSCRIPT_STATUS.json`. Model/data
outputs are never changed by reporting; the PDF needs rebuilding after the
final numerical tables have been inserted.

## Research question and interpretation

Do the effects of Indian statutory DAPT persist when the resulting readers
answer other legal-related questions with supplied contexts?

Evaluate N/A/S/M at seeds 7, 42, 99, 123, 2026 for each of:

| Encoder | Checkpoints |
| --- | --- |
| InLegalBERT | Experiment D A/S/M; matching Experiment A N |
| BERT | Completed encoder extension |
| LEGAL-BERT | Encoder extension, once all 20 readers finish |

These are the final readers **after SQuAD and IPR fine-tuning**, not the MLM
checkpoints. There are 60 readers if all three encoder cohorts are evaluated.
No best-seed or best-condition selection is permitted. Each encoder is analysed
separately. Do not attribute differences between encoders causally to legal
pretraining or pool encoder means as independent replications.

This is a post hoc external transfer evaluation, not a new within-dataset DAPT
training replication. Good scores or nonsignificant differences do not prove
generalisation or equivalence. Low transfer performance must also be reported.
This experiment cannot establish transfer across all legal domains or
jurisdictions, or independently verify legal correctness.

## Dataset and safeguards

Source: https://huggingface.co/datasets/isaacus/LegalQAEval

The dataset card describes 1,204 validation and 1,206 test examples from SQuAD
2.0, MS MARCO, HotpotQA and Natural Questions. It has no training split and
contains both answerable and unanswerable examples. It includes legal-related
texts selected from broader QA resources, not exclusively statutory provisions.
Download saves the README, dataset commit, and SHA-256 hashes. Follow the
dataset and underlying source licences, including attribution and applicable
share-alike/research-use terms; see the saved dataset card. Cite Butler (2025),
LegalQAEval, and report the exact downloaded revision.

Primary subset (fixed before external scoring):

- Answerable examples from **non-SQuAD sources**.
- Exclude normalized question **or** context matches against all cached SQuAD
  training and validation records, all IPR question splits, and A/S/M adaptation
  corpora. This deliberately uses a conservative superset of the actual SQuAD
  20k/2k subsets, including examples used for checkpoint selection.
- NFKC normalization, case folding, word tokenization and collapsed whitespace
  are used for overlap checks. Full normalized text equality is checked, not
  semantic similarity, containment or fuzzy matches. Undetected contamination
  and unknown encoder pretraining exposure remain limitations.

In this downloaded revision, every SQuAD-derived example matches the full
cached SQuAD validation context pool and is excluded from clean scoring. The
25 cross-split shared normalized contexts affect only SQuAD-derived examples
(112 validation and 136 test records, including unanswerable records), so
excluding those overlaps does not further reduce the non-SQuAD primary subset.

Secondary subsets are clean answerable examples including SQuAD, all answerable
examples (explicitly exploratory and potentially exposed), and clean
answerable examples by original source. Do not present the broader subset as
the primary external result. Counts and exclusions must accompany every score.

Validation and test remain separate; shared normalized questions or contexts
are flagged item by item and excluded from both splits' primary and clean
subsets. They remain only in the all-answerable exploratory subset. Initial
preparation found 25 shared normalized context texts and stopped; before any
external scoring, this exclusion rule was specified and tested. No raw
annotations were changed. Invalid offsets, unknown sources,
duplicate IDs and duplicate question/context pairs also cause explicit failure.
Do not silently repair annotations or relax rules after observing results.
If an input defect requires a protocol amendment, document it and use a fresh
`--work` directory before new scoring.

The current decoder always returns a span. Unanswerable examples are counted
in the audit and excluded from this experiment. No answerability F1, abstention
accuracy or full-dataset benchmark score is claimed. An abstention experiment
would require its own validation-only threshold-selection protocol.

All annotated answer spans are retained. EM and F1 independently take the
maximum over the accepted answer texts. This treats the listed spans as
alternative references; it does not require jointly extracting all spans.
Report that limitation, particularly for multi-answer questions.

Scoring uses the project's existing normalization (articles retained), rather
than claiming exact comparability with official SQuAD or other published
LegalQAEval scores. Context-only sliding windows use 384 tokens, overlap 128,
unrestricted span length within a window, and maximum summed reader logits.
Questions that cannot be tokenized under this setting fail visibly rather
than being silently shortened or dropped.

## Commands from the repository root

Use the current project Python environment. Dependencies already used here:
`torch`, `transformers`, `datasets`, `huggingface_hub`, `numpy`, `scipy`.
Do not upgrade the training environment while LEGAL-BERT is running.

Run the local tests first:

```powershell
python -X utf8 -m unittest discover -s experiments/legalqaeval -p "test_*.py" -v
```

Download the small dataset (network needed, no model inference):

```powershell
python -X utf8 experiments/legalqaeval/run.py download
```

`main` is resolved once to an immutable commit. You can specify an exact
commit with `download --revision <commit>`. If rerunning an existing download,
use the revision recorded in `run_v1/DOWNLOAD.json`.

Locate the SQuAD Arrow files recorded in the encoder extension's
`PROTOCOL.json` input hashes. Pass those exact files explicitly:

```powershell
python -X utf8 experiments/legalqaeval/run.py prepare --squad-train-arrow "<absolute path to squad-train.arrow>" --squad-val-arrow "<absolute path to squad-validation.arrow>"
```

Preparation writes `AUDIT.json`, item-level inclusion/exclusion flags in
`prepared/val.json` and `prepared/test.json`, and `PROTOCOL.json`. Review the
counts before inference; no score-based filtering or tuning is allowed.
All input, code and reference hashes are checked at evaluation/analysis time.

Wait for the active training to finish before external CPU inference. Each
encoder must have all 20 saved checkpoints available before any of its scores
are produced. For each encoder, run validation then test, then analysis:

```powershell
python -X utf8 experiments/legalqaeval/run.py evaluate --encoder bert --split val
python -X utf8 experiments/legalqaeval/run.py evaluate --encoder bert --split test
python -X utf8 experiments/legalqaeval/run.py analyse --encoder bert --split val
python -X utf8 experiments/legalqaeval/run.py analyse --encoder bert --split test
```

Repeat with `--encoder inlegalbert` and `--encoder legalbert`. Validation is
descriptive here: do not select checkpoints, conditions, seeds or a new recipe
using these external scores. The scripts load models only from local paths.
They reject missing QA weights and missing/mismatched tensors during loading.
They do not fetch models or launch training.

Use `--work <new directory>` **before the subcommand** for a separate run:

```powershell
python -X utf8 experiments/legalqaeval/run.py --work experiments/legalqaeval/run_v2 download --revision <commit>
```

## Outputs, recovery and statistics

Outputs default to `experiments/legalqaeval/run_v1/`:

- `raw/`, `DOWNLOAD.json`: pinned dataset and source documentation.
- `prepared/`, `AUDIT.json`, `PROTOCOL.json`: audited inputs, counts and rules.
- `outputs/<encoder>/COHORT.json`: all 20 checkpoint hashes, CPU settings and
  protocol identity, fixed before the encoder's first external evaluation.
- `outputs/<encoder>/<split>/<condition>_s<seed>.json`: per-item predictions,
  references, source labels, exclusions and EM/F1; atomic write per complete
  reader, with progress printed during inference.
- `outputs/<encoder>/<split>_statistics.json`: rescored, complete 20-cell
  analysis, provenance hashes, means, sample SDs, paired t intervals,
  seed-blocked ANOVA and Holm-corrected A/S/M versus N comparisons.

If interrupted, rerun the identical command. Completed reader outputs are
rescored and verified before being skipped. An interrupted reader restarts
inference; no partial reader is treated as complete. Do not run simultaneous
commands in the same work directory. Changed protocol/code/data/checkpoints
cause refusal instead of silently reusing stale results. Analysis rejects
missing or duplicate condition/seed cells and validates saved answers again.
Analysis can be rerun to regenerate a report from unchanged saved predictions.

Seed intervals condition on the fixed external questions and corpora; they do
not capture question sampling or source-document uncertainty. Holm correction
is within encoder, subset, split and metric, not study-wide. Source and
secondary analyses are exploratory. Nondetection is not evidence of equivalence.

## What to add to the paper after completion

Report the primary external subset counts and source mix, the dataset revision,
overlap removals, all 20 cells per encoder, mean EM/F1 with seed SD, and paired
DAPT-minus-N intervals. State that all readers were trained on Indian IPR and
evaluated externally without further training. Include secondary results and
failures rather than selecting favourable sources or encoders. Keep external
results separate from the existing IndicIPR-QA tables.

This can strengthen the paper by testing a clearly specified transfer claim;
the experiment cannot guarantee a favourable outcome or publication acceptance.
COVID-QA is not part of this protocol and has not been run for this study.
