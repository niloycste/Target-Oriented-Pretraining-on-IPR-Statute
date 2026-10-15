PROJECT:
Finalize the standalone DAPT paper under a CPU-only compute constraint.

CURRENT PAPER:
“Authentic, Synthetic, or Mixed Legal Text?
A Multi-Seed Audit of Domain-Adaptive Pretraining
for Statutory Extractive Question Answering”

CRITICAL CONSTRAINT:
NO GPU is available.

The goal is NOT to run every possible experiment.

The goal is to obtain the largest scientific improvement per CPU-hour
while fixing the actual weaknesses of the current manuscript.

DO NOT modify or rerun the separate retrieval/reference-resolution paper.

============================================================
1. FINAL SCIENTIFIC STRATEGY
============================================================

Prioritize only three experimental questions:

PRIMARY QUESTION A:
Under a clean train-only and token-controlled design, does additional
statutory DAPT improve downstream extractive QA, and does corpus
composition matter?

PRIMARY QUESTION B:
Does TARGET-ALIGNED synthetic statutory text perform differently from
generic/style-only synthetic legal text under equal compute?

PRIMARY QUESTION C:
Does DAPT become more useful when downstream supervised QA data are
scarcer?

These three questions should become the scientific center of the final paper.

Do NOT make second-encoder robustness mandatory under the present CPU
constraint.

Second-encoder generalization should be listed as future work / limitation
unless substantial compute becomes available later.

============================================================
2. REVISED FINAL RESEARCH QUESTIONS
============================================================

Rewrite the final paper around:

RQ1 — DAPT efficacy
Does additional statutory DAPT improve gold-context extractive QA over
the otherwise identical InLegalBERT → SQuAD → IndicIPR-QA route?

RQ2 — Corpus composition
With clean train-only construction and matched MLM exposure, do authentic,
target-aligned synthetic, and mixed DAPT corpora produce reliably different
QA performance?

RQ3 — Target alignment
Does seed-guided target-aligned synthetic text provide more useful adaptation
than synthetic text that matches statutory/legal STYLE but is not aligned to
IndicIPR-QA concepts?

RQ4 — Low-resource behavior
Does the contribution of DAPT change when only a fraction of the downstream
IndicIPR-QA supervised training data is available?

MLM perplexity should become a SECONDARY DIAGNOSTIC rather than a main RQ.

============================================================
3. DATA HYGIENE — NON-NEGOTIABLE
============================================================

Freeze the released IndicIPR-QA split:

Train = 3710 QA pairs
Validation = 408
Test = 344

For ALL new experiments:

- DAPT corpus construction must use TRAIN material only.
- Validation contexts must not enter DAPT.
- Test contexts must not enter DAPT.
- Validation/test questions must not enter seed extraction.
- Validation/test answers must not enter seed extraction.
- Test data must not be used for:
  prompt choice,
  corpus choice,
  checkpoint selection,
  hyperparameter selection,
  model selection,
  stopping decisions.

Validation data may be used for model/checkpoint selection.

The final test evaluation must occur only after protocol lock.

============================================================
4. DO NOT REGENERATE DATA UNNECESSARILY
============================================================

The existing synthetic pool contains 13,211 accepted passages and appears
to have been generated from training-split seed information.

Before regenerating anything:

AUDIT its provenance.

Verify programmatically that:

- every source seed comes from the FINAL training split;
- no validation/test question IDs occur;
- no validation/test context IDs occur;
- no validation/test answer text was explicitly used as generation input.

If the existing 13,211-passage synthetic pool passes this audit:

REUSE IT.

Do NOT regenerate 13,211 passages.

This saves substantial CPU time.

Create:

experiments/dapt/final_cpu/
    SYNTHETIC_PROVENANCE_AUDIT.json

If provenance cannot be established, stop and report the problem before
training.

============================================================
5. REBUILD THE AUTHENTIC CORPUS CLEANLY
============================================================

Reconstruct authentic DAPT data from FINAL TRAINING-SPLIT statutory contexts
only.

Deduplicate by the canonical statutory-context identifier and normalized text.

Do not assume the final unique-context count is 901.

Calculate the true count from the released split.

Save:

AUTHENTIC_TRAIN_ONLY_MANIFEST.json

For each authentic record store:

context_id
Act
section
word count
token count
SHA256 / content hash

============================================================
6. FINAL SYNTHETIC CORPUS STANDARD
============================================================

For the NEW causal experiment, standardize synthetic data as much as possible.

Prefer Gemma 3:12B outputs only if the existing 12B pool contains enough
clean target-aligned text for the matched experiment.

Reason:
using one generator family makes the target-alignment control cleaner.

Do not mix 12B and 27B in the new causal experiment unless necessary.

The archived 12B+27B corpus remains valid legacy evidence and must not be
overwritten.

If using 12B-only final synthetic data, document this as a deliberate
experimental-control decision rather than a claim that 12B is superior.

============================================================
7. EXPERIMENT A — CLEAN PRIMARY COMPARISON
MANDATORY
============================================================

Conditions:

N = No DAPT
A = Authentic DAPT
S = Target-aligned Synthetic DAPT
M = Mixed DAPT

Primary encoder:
law-ai/InLegalBERT

Downstream route for ALL four:

→ SQuAD v1
→ IndicIPR-QA
→ gold-context evaluation

N receives:
ZERO additional MLM adaptation.

A/S/M receive matched MLM exposure.

IMPORTANT:
Do NOT “token-match” N to DAPT.
N has no MLM stage by definition.

============================================================
8. MATCH A/S/M PROPERLY
============================================================

The manipulated factor should be corpus composition.

Control as much exposure as possible.

For A/S/M match:

- total non-padding MLM tokens;
- sequence length;
- MLM masking probability;
- effective batch size;
- optimizer;
- learning rate;
- warmup;
- number of processed MLM tokens;
- checkpoint-selection rule.

Also attempt to match record count where possible.

However:

TOKEN EXPOSURE IS THE PRIMARY MATCHING CRITERION.

Do not distort corpora merely to obtain mathematically identical record
counts if token matching is cleaner.

Report:

number of unique records
total raw tokens
total tokenizer tokens
total non-padding tokens
mean tokens/record
number of repeated records
number of MLM optimization steps

============================================================
9. MIXED CORPUS DEFINITION
============================================================

Define Mixed as approximately:

50% authentic MLM tokens
+
50% target-aligned synthetic MLM tokens.

The ratio should be based primarily on TOKEN EXPOSURE rather than simply
450/451 records.

Choose the exact algorithm BEFORE training.

Save the selected records and token counts.

Do not optimize the mixture based on downstream F1.

============================================================
10. CORPUS SELECTION MUST BE REPRODUCIBLE
============================================================

Create a script such as:

scripts/dapt/build_final_matched_corpora.py

Use a fixed corpus-construction seed independent of training seeds.

Example:
corpus_selection_seed = 20260829

The script should construct:

A_manifest.json
S_manifest.json
M_manifest.json

and print:

records
tokens
difference from target token budget
authentic/synthetic token ratio

Do NOT manually choose favorable passages.

============================================================
11. PRIMARY RANDOM SEEDS
============================================================

Run FIVE complete-pipeline seeds for Experiment A:

7
42
99
123
2026

Therefore:

N × 5
A × 5
S × 5
M × 5

TOTAL = 20 full pipelines.

A complete pipeline means:

for N:
Base → SQuAD → IPR

for A/S/M:
Base → DAPT → SQuAD → IPR

Every seed must propagate through all stochastic stages.

Do NOT:

train one SQuAD model and reuse it five times while claiming five
full-pipeline seeds.

============================================================
12. PRESERVE THE ORIGINAL SQuAD STAGE
============================================================

DO NOT reduce SQuAD from 20k to 5k merely to save time.

Use the existing:

20,000 SQuAD training examples
2,000 validation examples

for the primary experiment.

Reason:
SQuAD transfer is already known to materially affect reader performance.
Changing it simultaneously with the DAPT design introduces another variable.

Optimize implementation instead of changing the scientific stage.

============================================================
13. CPU OPTIMIZATION BEFORE TRAINING
============================================================

The SQuAD stage is the CPU bottleneck.

Before final training, inspect the implementation and optimize only
scientifically neutral overhead.

Allowed optimizations:

- tokenize the fixed SQuAD 20k/2k subset ONCE;
- cache tokenized SQuAD features;
- tokenize IndicIPR-QA splits ONCE;
- cache DAPT tokenization for each corpus;
- reuse deterministic preprocessing artifacts;
- avoid repeatedly parsing raw JSON/PDF/text;
- use available CPU threads efficiently;
- inspect BLAS/MKL/oneDNN availability;
- benchmark dataloader worker count;
- avoid unnecessary evaluation every few steps;
- avoid excessive checkpoint writes;
- preserve enough checkpoints for crash recovery.

These optimizations must NOT change:

training examples
optimizer
learning rate
model architecture
loss
number of epochs
selection criterion

Create:

CPU_OPTIMIZATION_REPORT.md

Report baseline runtime versus optimized runtime using validation/pilot jobs.

============================================================
14. CRASH / RESUME PROTECTION
============================================================

CPU jobs may run for many hours.

Every stage must be resumable.

Save checkpoints during long SQuAD training frequently enough that a crash
does not lose many hours.

Determine checkpoint frequency from actual step runtime.

Target:
no more than approximately 60–90 minutes of work lost after interruption.

Use:
resume_from_checkpoint
or the framework-equivalent mechanism.

Maintain:

RUN_LEDGER.csv

Columns:

experiment
condition
seed
stage
status
start_time
end_time
checkpoint
runtime_hours
notes

Status values:

NOT_STARTED
RUNNING
MLM_DONE
SQUAD_DONE
IPR_DONE
EVALUATED
FAILED
RESUMABLE

If a completed MLM or SQuAD checkpoint is valid, do not rerun that completed
stage after a crash.

============================================================
15. TEST DISCIPLINE
============================================================

During pipeline debugging:

DO NOT evaluate on the 344-item test set.

Use train/validation only.

After:
corpora,
commands,
hyperparameters,
seeds,
statistics

are finalized, create:

FINAL_ANALYSIS_LOCK.md

Then perform final test evaluation.

============================================================
16. EXPERIMENT B — TARGET ALIGNMENT
MANDATORY FOR NOVELTY
============================================================

This is the most important experiment after A.

Question:

Does target concept alignment matter beyond simply generating legal-looking text?

Reuse the target-aligned synthetic condition S from Experiment A.

Create ONE additional synthetic control:

S_STYLE

S_STYLE should contain:
generic statutory/legal-style synthetic passages,
but should NOT be conditioned on IndicIPR-QA target concepts.

Preferred design:

Use the SAME generator model used for S.

Use the SAME:
decoding parameters
length requirements
eight statutory style families
filtering pipeline
token budget
number/length distribution as closely as possible.

Difference:

S uses IndicIPR-QA training-derived legal seeds.

S_STYLE does NOT receive those target seeds.

For S_STYLE prompts, ask for generic Indian statutory-style legal language
that does not concern the seven target IPR Acts.

Do NOT deliberately generate nonsense.

The control must still be plausible legal/statutory-style language.

============================================================
17. TARGET-ALIGNMENT CONTROL SAFETY
============================================================

Do not use validation/test data to build S_STYLE.

Do not scrape a new external corpus automatically.

If an already available general legal seed list exists in the repository,
it may be used after provenance inspection.

Otherwise use a predefined generic legal-style generation procedure that
does not use IndicIPR-QA concepts.

Document the exact prompt.

============================================================
18. MATCH TARGET-ALIGNED AND STYLE-ONLY SYNTHETIC CORPORA
============================================================

S versus S_STYLE must be matched for:

generator family
generation settings
MLM token budget
DAPT steps
sequence length
training recipe

Also compare corpus-level properties:

mean length
sentence count
legal-style marker rate
type-token ratio
exact duplicate rate

This prevents a reviewer from arguing that S_STYLE failed simply because
it was lower-quality text.

============================================================
19. TARGET-ALIGNMENT TRAINING RUNS
============================================================

Do NOT rerun S unnecessarily.

Reuse S results from Experiment A for seeds:

7
42
99

Run only:

S_STYLE × seeds 7, 42, 99

Additional full pipelines:
3

Compare:

S_target vs S_STYLE

and optionally:
S_STYLE vs N

using the corresponding primary runs.

Thus Experiment B adds only THREE expensive full pipelines.

============================================================
20. EXPERIMENT C — LOW-RESOURCE DATA-EFFICIENCY
HIGH VALUE / LOW COMPUTE
============================================================

This experiment is designed specifically for the CPU constraint.

It should NOT repeat DAPT or SQuAD.

Reuse the already trained post-SQuAD checkpoints from Experiment A.

Question:

Does DAPT become more useful when fewer supervised IndicIPR-QA examples
are available?

Use:

N
A
S
M

Use seeds:
7
42
99

Create fixed low-resource IndicIPR-QA training subsets:

25%
50%
100%

100% already exists from Experiment A.

Therefore only 25% and 50% require new fine-tuning.

Optionally add 10% if runtime is low.

============================================================
21. BUILD LOW-RESOURCE SUBSETS CAREFULLY
============================================================

Do NOT sample random QA rows independently if avoidable.

Prefer sampling at the TRAINING STATUTORY-SECTION level.

Stratify approximately by Act.

For each fraction:

select a fixed subset of training sections
include QA pairs associated with those sections
use the SAME subset for every model condition and seed

Keep the existing validation split unchanged.

Save:

LOW_RESOURCE_25_MANIFEST.json
LOW_RESOURCE_50_MANIFEST.json

If adding 10%:
LOW_RESOURCE_10_MANIFEST.json

Use one fixed sampling seed.

Do not choose the subset based on model performance.

============================================================
22. LOW-RESOURCE TRAINING
============================================================

Start each run from its already-existing post-SQuAD checkpoint.

Therefore no additional:

DAPT
or
SQuAD

training is required.

Only repeat the final IndicIPR-QA fine-tuning stage.

Conditions:
N / A / S / M

Seeds:
7 / 42 / 99

Fractions:
25%
50%

Additional IPR-only runs:
4 × 3 × 2 = 24

Because datasets are smaller, these should cost much less than 24 full
pipelines.

============================================================
23. LOW-RESOURCE ANALYSIS
============================================================

Report F1/EM curves against supervised-data fraction.

Main question:

Are DAPT gains larger at 25% or 50% training data than at 100%?

Possible scientifically valuable outcomes include:

- DAPT matters only when supervision is very scarce.
- DAPT does not help even under reduced supervision.
- synthetic DAPT helps more than authentic DAPT at low data.
- authentic DAPT becomes more useful as labels decrease.
- composition differences remain smaller than seed variance.

Do not assume any outcome.

============================================================
24. CLEAN MLM DIAGNOSTIC
============================================================

Stop using test contexts as the PRIMARY final perplexity diagnostic.

Use unique statutory contexts from the VALIDATION split that are absent from
all DAPT corpora.

Freeze this diagnostic set.

For every A/S/M run compute:

MLM loss
perplexity
relative reduction from InLegalBERT

No test data required.

MLM perplexity is SECONDARY.

Do not select QA models using PPL.

============================================================
25. CHEAP CORPUS ANALYSIS — NO MODEL TRAINING
============================================================

Add a stronger corpus characterization section.

For authentic, target synthetic, mixed-source components, and S_STYLE report:

number of records
total tokens
mean/median length
length distribution
type-token ratio
distinct-1 / distinct-2 if useful
exact duplicate rate
near-duplicate estimate
seed/concept coverage
legal-style marker frequency

Also calculate lexical similarity between:

authentic TRAIN contexts
and
target synthetic corpus

versus:

authentic TRAIN contexts
and
S_STYLE corpus.

Do not use test performance to choose these metrics.

============================================================
26. NEAR-DUPLICATE AUDIT
============================================================

The current paper only guarantees exact normalized-text deduplication.

Add a near-duplicate audit.

Use a reproducible method such as:

MinHash
SimHash
or normalized n-gram Jaccard

with a predefined similarity threshold.

Report:

number and percentage of near-duplicate synthetic passages
within synthetic corpus

and, importantly:

near matches between synthetic passages and authentic TRAIN contexts.

This is analysis only.

Do not delete records after seeing downstream results.

If deduplication is changed for the final corpus, do it BEFORE training and
record the rule in the protocol.

============================================================
27. CHEAP QA DIAGNOSTICS — LEGAL INTERPRETABILITY
============================================================

Using final per-question predictions, compute performance by answer-span
difficulty.

No new training.

Suggested bins:

gold answer length:
short
medium
long

semicolon present / absent

enumerated-clause answer / non-enumerated

where these properties can be defined deterministically.

Question:

Does DAPT mainly help difficult statutory boundary extraction?

For example:
long answers,
enumerated clauses,
provisos,
multi-clause spans.

Keep this analysis exploratory unless categories are predefined reliably.

============================================================
28. OPTIONAL MANUAL ERROR AUDIT
============================================================

Only if human time is available.

Take a small fixed sample of cases where:

S improves substantially over N
and
N outperforms S.

Inspect errors for:

boundary truncation
over-extension
adjacent clause inclusion
wrong proviso/exception
wrong enumerated item
institutional-role boundary

Do not claim inter-annotator reliability unless more than one independent
annotator is actually used.

This is optional and requires no GPU.

============================================================
29. STATISTICS — PRIMARY EXPERIMENT
============================================================

Primary unit:
complete training seed.

For Experiment A:

conditions:
N A S M

seeds:
7 42 99 123 2026

Report:

individual seed F1
mean
SD
95% CI

Primary overall comparison:
randomized-block ANOVA with seed as block
or statistically equivalent blocked analysis.

Prespecified pairwise contrasts:

A-N
S-N
M-N
A-S
A-M
S-M

Use multiple-comparison correction.

Report effect estimates and intervals.

Do NOT interpret:
p > 0.05
as equivalence.

============================================================
30. STATISTICS — TARGET ALIGNMENT
============================================================

Compare paired seeds:

S_TARGET versus S_STYLE

Seeds:
7
42
99

Report:

mean paired difference
individual paired differences
95% CI
effect direction

With only three seeds, treat statistical significance cautiously.

The main value is that this is a controlled mechanistic comparison.

============================================================
31. STATISTICS — LOW-RESOURCE CURVE
============================================================

For each fraction report:

N
A
S
M

mean and SD across seeds 7/42/99.

Report DAPT-minus-N differences at:

25%
50%
100%

The key analysis is the interaction pattern:

does DAPT advantage grow as supervised data decreases?

Do not overcomplicate the inferential model given only three seeds.

A clear data-efficiency curve with paired seed differences is sufficient.

============================================================
32. DO NOT RUN THE SECOND ENCODER NOW
============================================================

Under CPU-only constraints:

DO NOT spend approximately nine additional complete pipelines on a second
encoder before the primary questions are resolved.

In the final manuscript state clearly:

“The controlled experiments use InLegalBERT as the common starting encoder.
Whether the observed composition and target-alignment effects transfer to
other base encoders remains an external-validity question.”

This is an acceptable scoped limitation.

If GPU or substantially more compute becomes available later, second-encoder
robustness can be added.

============================================================
33. DO NOT RUN A LARGE SCALE GRID
============================================================

Do NOT run:

900
3000
7000
13000

× multiple seeds

at this stage.

The CPU cost is disproportionate to its scientific value relative to
Experiments A/B/C.

Synthetic-scale analysis becomes future work unless all primary experiments
finish early.

============================================================
34. DO NOT SHRINK SQuAD YET
============================================================

Do NOT change:

20k train / 2k validation SQuAD

for the primary experiment.

First optimize preprocessing and caching.

Only propose a reduced SQuAD stage if the optimized runtime remains
impossible.

If proposing such a change:
STOP and request approval before training.

============================================================
35. EXPECTED CPU PACKAGE
============================================================

Expensive full pipelines:

Experiment A:
20

Experiment B / alignment:
3 additional

TOTAL FULL PIPELINES:
23

Do not run Experiment C as full pipelines.
It reuses post-SQuAD checkpoints and repeats only IPR fine-tuning.

Approximate current-runtime estimate:

23 × ~15.8 h
≈ 363 CPU-hours
≈ 15 continuous days before optimization.

Low-resource additional IPR-only runs:
approximately 24
but substantially shorter because DAPT and SQuAD are not repeated and
training sets are only 25%/50%.

Actual time must be recomputed after preprocessing/cache optimization.

============================================================
36. ORDER OF EXECUTION
============================================================

DO NOT start all jobs simultaneously.

STEP 1:
Repository + data audit.

STEP 2:
CPU preprocessing/runtime optimization.

STEP 3:
Build final train-only matched A/S/M corpora.

STEP 4:
Build S_STYLE corpus.

STEP 5:
Create validation MLM diagnostic.

STEP 6:
Create low-resource manifests.

STEP 7:
Produce DRY_RUN_REPORT.md.

STOP.

Do not start final expensive training until the dry-run report is reviewed.

After approval:

STEP 8:
Run Experiment A.

STEP 9:
Run Experiment B.

STEP 10:
Run low-resource IPR-only Experiment C.

STEP 11:
Generate statistics and diagnostics.

STEP 12:
Only then rewrite manuscript results.

============================================================
37. DRY-RUN REPORT REQUIREMENTS
============================================================

Create:

DRY_RUN_REPORT.md

It must show:

1. exact final train-only context counts;
2. audit status of existing synthetic corpus;
3. chosen synthetic generator subset;
4. exact A/S/M token budgets;
5. record counts;
6. mixed token ratio;
7. exact S_STYLE construction;
8. SQuAD cached dataset path;
9. IPR cached dataset paths;
10. seeds;
11. exact shell commands;
12. run directory naming convention;
13. checkpoint/resume policy;
14. estimated per-stage runtime after optimization;
15. estimated total CPU time;
16. statistical plan;
17. confirmation that the test set has not entered corpus construction.

STOP AFTER THIS DOCUMENT.

============================================================
38. FINAL MANUSCRIPT AFTER RESULTS
============================================================

Once experiments are complete:

REMOVE:
“required future protocol”
“stronger version should include”
“submission-status checklist”
“should be updated after”
“future causal experiment”

Replace those sections with actual final experimental methods and results.

The final manuscript should tell this story:

1. Archived DAPT checkpoints suggested a possible benefit but were confounded.

2. A clean five-seed train-only experiment tests whether DAPT itself helps
   and whether authentic/synthetic/mixed composition matters.

3. A controlled style-only synthetic baseline tests whether TARGET ALIGNMENT
   matters beyond synthetic legal language.

4. A low-resource supervised-data experiment tests when DAPT becomes most
   useful.

5. MLM perplexity and corpus diagnostics explain what adaptation changes
   without treating PPL as downstream evidence.

6. The conclusions follow the final results even if the expected hypothesis
   fails.

============================================================
39. STOPPING RULE
============================================================

The paper is experimentally sufficient once these are complete:

MANDATORY:
- clean train-only N/A/S/M comparison;
- token-matched A/S/M;
- five complete primary seeds;
- no-DAPT control;
- target-aligned versus style-only synthetic control;
- clean validation MLM diagnostic;
- low-resource 25%/50%/100% downstream analysis;
- corpus provenance / duplicate audit;
- canonical statistics and result provenance.

NOT REQUIRED BEFORE SUBMISSION:
- second encoder;
- another generator model;
- external benchmark;
- synthetic scale grid;
- many mixing ratios;
- retrieval experiments;
- RAG;
- practitioner study.

STOP once the mandatory CPU package is complete.

============================================================
40. IMPORTANT SCIENTIFIC RULE
============================================================

Do not optimize the experiment to make synthetic DAPT win.

The paper is successful if it establishes any well-controlled result, including:

- DAPT helps;
- DAPT does not help at full supervision;
- DAPT helps only in low-resource supervision;
- authentic and synthetic text are indistinguishable;
- target alignment matters;
- target alignment does not matter;
- mixed data helps;
- mixed data hurts;
- MLM fit improves without QA gains.

The scientific contribution is the controlled attribution of the effect,
not a preferred leaderboard outcome.