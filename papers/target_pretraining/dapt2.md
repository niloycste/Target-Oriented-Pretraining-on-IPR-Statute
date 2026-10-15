Detailed Issues in the Paper
A. Critical methodological issues
Issue 1 — The DAPT training budget is not actually matched

Severity: Major

The paper repeatedly frames the adaptation corpora as being approximately matched by token count. The token counts are indeed extremely close:

Authentic: 128,727
Synthetic: 128,713
Mixed: 128,663
Style: 128,723

However, the optimizer update counts are:

Authentic: 87
Synthetic: 81
Mixed: 84
Style: 81

The paper itself acknowledges this.

Why this matters

The model does not learn from “tokens” directly in an equivalent way. It learns through optimizer updates.

Because the batch size and number of epochs are fixed while the number of records differs, the conditions receive different numbers of parameter updates.

More importantly, all conditions finish before the 100-step warmup ends.

Therefore, the effective learning-rate trajectory differs across conditions.

For example, the synthetic condition finishes at roughly 81% of the warmup schedule, while authentic finishes at roughly 87%.

So the experiment is not purely:

Authentic text vs synthetic text

It is closer to:

Authentic text + 87 updates vs synthetic text + 81 updates.

Why a reviewer will care

If synthetic DAPT performs differently, you cannot confidently attribute the difference entirely to corpus composition.

How to fix

Ideally run a new experiment with:

equal number of optimizer updates,
equal effective batch size,
equal token exposure,
identical LR schedule,
identical number of warmup steps relative to total steps.

Then the comparison becomes much cleaner.

If you cannot rerun

Explicitly change the claim to:

“The study compares approximately token-matched adaptation corpora under a fixed three-epoch training recipe, rather than exactly update-matched DAPT.”

That would substantially improve the methodological precision.

Issue 2 — The 100-update warmup is longer than the entire DAPT training

Severity: Major

This is particularly unusual.

The paper reports:

81–87 optimizer updates

but:

100 warmup updates

Therefore, the entire MLM training run occurs during warmup.

Why this matters

Normally, with:

warmup → full learning rate → decay

the model eventually reaches the intended peak learning rate.

Here the model apparently does:

0 → 0.81/0.84/0.87 × LR

and stops.

So the nominal learning rate of the experiment is never actually reached.

Consequence

The paper's negative DAPT result could partly mean:

“DAPT under this extremely short/incomplete learning-rate schedule did not improve QA.”

rather than:

“DAPT does not improve QA.”

The paper correctly limits its claim, but this issue deserves much more prominence.

Recommended fix

At minimum, add a table:

Condition	Total updates	Warmup	Final LR relative to peak
Authentic	87	100	~87%
Synthetic	81	100	~81%
Mixed	84	100	~84%

Then explicitly explain why 100 warmup steps were chosen.

Better: rerun with a sensible warmup fraction, e.g. 5–10% of total steps.

Issue 3 — The synthetic-vs-authentic comparison does not isolate corpus composition

Severity: Major

The paper's central comparison is:

Authentic vs Synthetic vs Mixed.

But the corpora differ in many properties beyond origin.

Table 7 shows differences in:

average word count,
median length,
lexical diversity,
bigram diversity,
frequency of legal markers such as “shall,”
“provided that,”
“subject to.”

For example, synthetic text contains substantially more instances of some legal markers.

This is not necessarily a flaw.

Actually, these differences may be the thing you're trying to study.

But then the paper needs to state exactly what “composition” means.

Currently, “authentic vs synthetic” could mean:

source authenticity,
lexical style,
sentence structure,
legal terminology,
discourse structure,
redundancy,
semantic content,
generation artifacts.
Recommended clarification

Define the causal factor explicitly:

“Corpus provenance/composition” rather than simply “text type.”

Then say:

“The experiment does not identify which linguistic property within the synthetic corpus causes any observed difference.”

Issue 4 — Reduced supervision is not actually reduced access to task information

Severity: Major

This is one of the most important issues.

The 25% and 50% conditions reduce the final supervised IPR fine-tuning examples.

But the DAPT corpus is still generated using the full eligible training resource.

The paper explicitly says that the adaptation corpora remain derived from the full training resource, including question-derived generation seeds.

Therefore:

25% condition does NOT mean:

The model only has access to 25% of the benchmark's training information.

Instead it means approximately:

The model has 25% of the answer-label fine-tuning data while still receiving a DAPT corpus derived from the full training resource.

Why this matters

If synthetic DAPT gives +0.43 F1 at 50%, that could reflect access to information derived from the omitted training sections/questions.

So calling this a:

“low-resource”

or:

“25%/50% supervision”

experiment without qualification could mislead readers.

Recommended terminology

Change:

Reduced supervision

to:

Reduced answer-label fine-tuning

and explicitly say:

“These are not strict low-resource settings because the DAPT stage retains access to the full task-informed training resource.”

Issue 5 — The 50% result is statistically fragile

Severity: Major

The paper reports:

Synthetic − No DAPT = +0.43 F1

with:

95% CI [0.30, 0.56]

and:

Holm-adjusted p = .014.

The three paired differences are approximately:

+0.463
+0.369
+0.452

This looks impressively consistent.

But there are only:

3 seeds

Therefore:

df = 2

for the paired t-test.

The more important issue

The paper also performs an exact sign-flip sensitivity analysis:

p = .25

The paper itself notes this.

That creates two very different inferential stories:

Analysis	Result
Paired t-test	significant
Exact sign-flip	p = .25
What this means

You cannot treat the +0.43 as a robustly established effect.

The t-test is extremely sensitive to the assumption that the paired differences can reasonably be modeled by a t distribution with only three observations.

Recommended wording

Instead of:

“Synthetic DAPT improves F1 by 0.43.”

Use:

“A +0.43 F1 contrast was observed for synthetic DAPT at 50% answer-label fine-tuning; this contrast is significant under the paired t-test but is not supported by the exact sign-flip sensitivity analysis and should therefore be considered preliminary.”

This is much safer.

Issue 6 — The 50% result does not establish that DAPT becomes more effective with less supervision

Severity: Major

This is a very important statistical distinction.

The paper asks whether the DAPT effect changes as supervision decreases.

The correct statistical test for that is the interaction.

For synthetic DAPT at 50%, the paper reports:

interaction = +0.39
95% CI [−0.59, 1.36]
p = .230

At 25%:

+0.31
CI [−3.84, 4.45]
p = .781.

So:

What you have evidence for

Synthetic DAPT differs from N within the 50% condition under the paired t-test.

What you do NOT have evidence for

Synthetic DAPT becomes more beneficial because supervision is reduced.

That second claim requires an interaction.

Recommended conclusion

The answer to RQ4 should primarily be:

No supervision-level interaction was detected.

Then separately mention:

“A preliminary within-50% contrast was observed.”

This distinction should appear in the abstract and conclusion.

Issue 7 — Multiple-comparison correction is defensible but potentially vulnerable to reviewer criticism

Severity: Moderate–Major

The paper applies Holm correction within analysis families.

That's good.

However, the paper conducts many analyses:

full-data ANOVA,
six pairwise contrasts,
EM contrasts,
style comparison,
25% comparisons,
50% comparisons,
interaction tests,
difficulty analyses,
MLM analyses,
etc.

The +0.43 result is significant under one particular correction family.

Reviewer criticism could be:

Why was the entire study not treated as one family of confirmatory hypotheses?

You don't necessarily need to do that statistically, because these are different research questions.

But the paper should make the distinction explicit.

Recommended solution

Call the 50% finding:

“a within-fraction, multiplicity-adjusted exploratory contrast”

rather than presenting it as a central confirmatory result.

Issue 8 — The seed count is too small for the strength of some conclusions

Severity: Major

The main experiment uses:

5 seeds.

The style and reduced-supervision experiments use:

3 seeds.

Five seeds is better than one seed, but still small.

Three seeds is especially problematic.

Why?

Training-seed variation is one of the central things you're trying to measure.

With only three observations, you cannot reliably estimate the distribution of training randomness.

Recommended fix

If resources allow:

10 seeds minimum for the principal comparison.
Ideally 10+ for the reduced-supervision experiment.

If not, make the paper explicitly a:

small-seed empirical audit

rather than a definitive benchmark study.

Issue 9 — One synthetic corpus is being treated as representative of synthetic DAPT

Severity: Major

The paper fixes:

corpus selection seed = 20260829.

This is useful for reproducibility.

But it means you have:

one synthetic corpus realization.

The study therefore estimates:

performance of this particular synthetic corpus,

not necessarily:

performance of synthetic legal text generation in general.

Why this matters

Another synthetic sample might contain:

different legal terminology,
different repetition,
different semantic coverage,
different generation artifacts,
different passage quality.
Better design

Generate multiple independent synthetic corpora and treat:

training seed,
corpus realization

as separate sources of variation.

That would be substantially stronger.

Issue 10 — The synthetic generation process is target-conditioned

Severity: Moderate–Major

Synthetic passages are generated using:

statutory context,
question text,
target concepts/keyphrases.

The paper deliberately does this.

That's interesting.

But it means the synthetic corpus is task-informed.

Even though answer strings and offsets are not explicitly provided, the question itself can reveal what the answer is about.

Why this matters

A reviewer could say:

The synthetic corpus is not merely “legal text.” It is text generated in response to the QA task.

That makes the low-supervision interpretation especially complicated.

Recommended terminology

Call it:

task-conditioned synthetic statutory text

instead of simply:

synthetic legal text.

That makes the methodology clearer.

Issue 11 — No human/legal validation of synthetic text

Severity: Moderate

The paper explicitly states that the generated passages were not validated for:

legal correctness,
doctrinal consistency,
authority,
factual correctness.

This is honest.

But it creates a major interpretive limitation.

The model could be learning:

stylistic patterns of generated legal-looking language

rather than:

correct legal knowledge.

Recommended addition

Include a small human evaluation, perhaps:

100 synthetic passages,
two legal/NLP annotators,
correctness,
legal plausibility,
grammaticality,
statutory style,
factual consistency.

Even a small audit would strengthen the paper considerably.

B. Data leakage and provenance issues
Issue 12 — The raw seed artifact contained test contexts

Severity: Major provenance concern

The paper states that the raw seed artifact contained:

three test-split contexts.

The final generation audit found no propagation into the final corpus.

This is good forensic auditing.

But from a methodological standpoint, the pipeline should never have had access to test examples in the first place.

What you have proven

You have evidence that the test examples did not propagate into the final synthetic training corpus.

What you have NOT proven

You cannot claim the pipeline was inherently leakage-free.

Recommended wording

“The raw seed artifact contained three test contexts; a subsequent provenance audit found no propagation of those contexts into the accepted synthetic corpus.”

That is precise.

Do not say:

“No test data ever entered the pipeline.”

because that would be false.

Issue 13 — Exact overlap testing is insufficient to rule out semantic leakage

Severity: Moderate

The paper uses:

exact normalized matching,
25-token overlap.

That is useful.

But synthetic generation can paraphrase a held-out passage without sharing 25 consecutive tokens.

Therefore:

no 25-token overlap ≠ no semantic reproduction.

The paper acknowledges this.

Potential improvement

Use semantic similarity:

sentence embeddings,
cosine similarity,
nearest-neighbor retrieval,
high-threshold semantic duplicate detection.

Even if you don't use it as a hard quarantine rule, it would make the provenance audit stronger.

Issue 14 — Possible generator pretraining contamination is unresolved

Severity: Moderate

The synthetic generator is Gemma 3.

Even if your prompts contain only training data, the generator itself has been pretrained on a massive corpus.

You cannot demonstrate that it never encountered the source statute or benchmark.

This is probably impossible to fully establish.

Recommendation

Keep the limitation but phrase it clearly:

“The audit establishes corpus-level non-reproduction relative to the supplied benchmark material, but cannot establish independence from the generator's external pretraining corpus.”

That is scientifically precise.

C. Statistical and evaluation issues
Issue 15 — The metric is not exactly official SQuAD F1

Severity: Moderate

The paper says the normalization:

lowercases,
removes punctuation,
normalizes whitespace,

but does not exactly follow official SQuAD normalization.

You later acknowledge this.

Problem

Calling the result simply:

“F1”

can cause readers to assume official SQuAD F1.

Recommendation

Use:

SQuAD-style token F1

throughout.

And specify the exact normalization algorithm.

Issue 16 — No independent legal-correctness evaluation

Severity: Moderate

The model is judged against benchmark annotations.

Therefore:

F1 measures agreement with the benchmark answer.

It does not establish:

legal correctness.

The paper correctly says this.

But because the domain is legal QA, this limitation deserves more prominence.

Example

A benchmark answer could omit an important qualification while still being considered correct by the benchmark.

The model could receive high F1 for reproducing that annotation while still being legally incomplete.

Recommendation

Add:

“Our evaluation measures annotation-relative extraction accuracy rather than independent legal correctness.”

This should appear in the abstract or conclusion.

Issue 17 — Gold-context setup limits real-world significance

Severity: Moderate

The model receives the correct statutory paragraph.

Therefore retrieval is excluded.

Real legal QA often requires:

question
   ↓
retrieve relevant provision
   ↓
identify answer span
   ↓
reason over legal context

Your experiment evaluates only:

question + correct context
        ↓
answer span

This is perfectly legitimate, but the practical implications must remain narrow.

Issue 18 — Windowing details need more reproducibility

You specify:

384 tokens,
128 overlap,
start/end logits,
no answer-length cap.

Good.

But a reviewer may still ask:

How are offsets converted?
How are special tokens treated?
Are impossible spans masked?
Are spans allowed to cross feature boundaries?
How are duplicate answers across windows handled?
How are null/no-answer cases handled?

If these details exist in code but not paper, add them briefly.

D. Experimental-control issues
Issue 19 — Style control is not a perfect control

Severity: Moderate

The Style condition is useful.

But it differs from S in more than target alignment.

The paper itself says it uses:

generic concepts,
non-target-domain exemplars,
eight style families,
same generator,
same filtering.

But there is no human validation of semantic quality.

Therefore:

S vs Style

does not isolate target alignment perfectly.

It compares two different generation processes.

Better interpretation

Call it:

construction-matched style control

rather than:

target-alignment control.

Issue 20 — Token matching does not equal information matching

Severity: Moderate

Two corpora can contain:

128,700 tokens

but have very different information content.

Synthetic text may repeat concepts.

Authentic statutes may contain unique provisions.

Therefore token matching controls quantity, not information content.

This is not a flaw in itself, but it should be explicitly stated.

Issue 21 — Corpus length distributions are different

The synthetic corpus has a much narrower length distribution.

Table 7 shows:

Authentic: 10th–90th percentile approximately 33–229 words
Synthetic: 113–148 words
Style: 114–148 words.

That's a substantial difference.

Why it matters

The authentic corpus contains many very short and very long passages.

Synthetic passages are much more standardized.

Thus the model receives different sequence-length distributions.

Recommendation

Add:

length-matched sampling,
or a controlled experiment where authentic passages are sampled to match synthetic length distribution.

That would strengthen the composition analysis.

Issue 22 — Synthetic text has different lexical distribution

The “shall”, “provided that”, and “subject to” frequencies differ substantially.

This could mean synthetic data is more stereotypically legal.

Therefore, if synthetic DAPT performs differently, you won't know whether the effect comes from:

semantic coverage,
legal terminology,
repetition,
stylistic markers,
passage length,
or generator artifacts.

This should be discussed as a mechanism-identification limitation.

E. Reproducibility issues
Issue 23 — Original runner is unavailable

Severity: Major reproducibility issue

The paper says the original runner cannot be recovered, although the current runner reproduces the final scripts.

That's transparent, but it means:

independent reproduction of the original execution is impossible.

The current code can reproduce the reported stage structure, but that is not the same as independently reproducing the original training.

Recommendation

Release:

exact current code,
environment file,
model revision,
tokenizer revision,
configuration files,
hashes,
seeds,
checkpoint identifiers,
predictions.
Issue 24 — Model revision/hash should be specified

The paper identifies InLegalBERT, but a reproducibility-grade paper should ideally specify:

exact model repository revision / commit hash.

Because model repositories can change.

The same applies to:

Gemma,
tokenizer,
Transformers,
PyTorch.

If these are not already in the omitted supplementary material, add them.

Issue 25 — Software/hardware environment needs stronger specification

The paper says MLM was run on CPU.

But reproducibility would benefit from:

CPU model,
RAM,
OS,
Python version,
PyTorch version,
Transformers version,
CUDA status if relevant,
tokenizer version,
deterministic settings.

This is particularly important because the paper is framed as an audit.

Issue 26 — Fixed SQuAD subset needs a manifest

The paper says:

20k training / 2k validation

with seed 42.

That's reproducible if the exact sampling script and dataset version are available.

But for a paper emphasizing auditability, I recommend providing:

a fixed list of example IDs.

Then anyone can verify exactly what was used.

F. Post-hoc / analysis integrity issues
Issue 27 — The analysis amendment occurred after validation results were observed

Severity: Major transparency issue

The appendix states that the analysis record was amended after validation results were observed.

Even if the amendment was legitimate, this introduces a potential researcher-degree-of-freedom concern.

Reviewer question

Were any hypotheses, contrasts, exclusion criteria, metrics, or statistical procedures changed after seeing validation results?

The paper needs to answer this explicitly.

Recommended statement

Add something like:

“The amendment changed execution/reproducibility infrastructure only and did not alter the predefined hypotheses, conditions, evaluation set, metrics, contrast definitions, or multiplicity procedure.”

Only say this if it is factually true.

Issue 28 — Post-test MLM loader correction

Severity: Moderate

The MLM loading correction occurred after test evaluation.

You explain that:

QA predictions were unchanged,
checkpoints were unchanged,
only MLM diagnostic loading was corrected.

That is good.

But a reviewer may still ask:

Was the final MLM diagnostic a post-hoc corrected analysis?

Answer: yes.

Therefore label it explicitly as:

corrected post-test diagnostic

rather than presenting it as though it was generated through the original pipeline without intervention.

Issue 29 — Archived and final experiments have different evidentiary status

This is very important.

The paper has:

Primary experiment

Cleaned/audited/token-matched.

Archived experiments

Different corpus sizes, test exposure, old runner, etc.

The paper distinguishes them, but the distinction should be visually stronger.

Otherwise reviewers may think the manuscript is combining multiple incompatible experiments.

G. Claims and interpretation
Issue 30 — “No benefit” needs to remain “no detected benefit”

The paper mostly handles this correctly.

The distinction should be maintained everywhere:

Bad:

“DAPT does not improve QA.”

Better:

“We did not detect a downstream QA improvement under the tested configuration.”

Because the experiment is underpowered for tiny effects and uses one model/dataset/schedule.

Issue 31 — Don't generalize from InLegalBERT

The result applies to:

InLegalBERT → SQuAD → IndicIPR-QA.

It does not establish what happens with:

BERT,
RoBERTa,
Legal-BERT,
DeBERTa,
modern encoder architectures,
decoder-only models.

The discussion acknowledges this, but the abstract should also make the scope clear.

Issue 32 — Don't generalize from 129k tokens to DAPT generally

129k tokens is tiny relative to typical pretraining corpora.

The paper is essentially testing:

short-budget continued pretraining.

Therefore the result should not be generalized to:

DAPT at large scale.

This is perhaps the most important scope limitation after the seed issue.

Issue 33 — The phrase “synthetic adaptation” can sound broader than what was tested

You are testing:

Gemma 3-generated statutory-style passages with a specific prompting/filtering pipeline.

You are not testing:

synthetic legal data in general.

So use:

“Gemma-generated synthetic statutory-style adaptation”

where precision matters.

H. Presentation and writing issues
Issue 34 — “Reader” is confusing terminology

Use:

run

or:

training run

instead of:

reader.

“Reader” normally implies:

human reader,
reading-comprehension model,
QA architecture.

For example:

“47 reader runs”

is confusing.

Use:

“47 training/evaluation runs.”

Issue 35 — Too much methodological caveat is buried in appendices

The paper's most important caveats are sometimes only fully explained later.

For example:

test-context contamination in raw seed artifact,
post-test amendment,
loader correction,
unavailable original runner.

Because these affect interpretation, they should be summarized in the main methodology section.

The appendix can contain the audit details.

Issue 36 — Appendix C may distract from the primary contribution

The archived experiments are interesting but potentially confusing.

I'd reduce them considerably or explicitly title the section:

Historical experiments excluded from primary inference

Then summarize their purpose in one paragraph.

Issue 37 — Answer-difficulty analysis is not strongly connected to the RQs

It is useful, but it is exploratory.

I'd move most of Table 8 to supplementary material unless the venue specifically values extensive diagnostics.

Issue 38 — The seven Acts should be named

Instead of only:

seven Indian IP Acts

give the actual names somewhere in the paper.

That improves:

reproducibility,
domain interpretation,
external validity assessment.

A legal NLP reviewer will want to know exactly which legislation is represented.

Issue 39 — Need clearer distinction between statutory text and benchmark-generated text

The paper uses:

authentic statutory provisions,
synthetic statutory-style passages,
machine-generated questions.

These are three distinct data sources.

A simple provenance diagram would help:

Indian Acts
    │
    ├── authentic provisions
    │
    └── benchmark questions
            │
            ▼
       Gemma prompts
            │
            ▼
 synthetic adaptation corpus

Then show which information is available at each stage.

This would make the leakage argument much easier to understand.

I. What I consider the top 10 issues

If you don't want to fix everything immediately, prioritize these:

Priority	Issue	Severity
1	81–87 updates vs 100 warmup	Critical
2	50% result based on only 3 seeds	Critical
3	t-test significant vs exact sign-flip p=.25	Critical
4	Reduced supervision still uses full task-informed DAPT resource	Major
5	Token matching does not equal update matching	Major
6	50% result does not establish a supervision interaction	Major
7	Raw seed artifact contained test contexts	Major provenance issue
8	One fixed synthetic corpus realization	Major
9	Post-validation amendment / post-test loader correction	Major transparency issue
10	Archived experiments can confuse primary inference	Moderate–Major
What I would change in the paper's main conclusion

The current paper is strongest if its conclusion is essentially:

Under a short, approximately token-matched DAPT recipe applied to InLegalBERT, additional authentic, synthetic, or mixed statutory adaptation substantially improved masked-language-model fit but did not produce a detectable improvement in full-data extractive QA. A preliminary +0.43 F1 synthetic-DAPT contrast appeared under 50% answer-label fine-tuning, but the interaction with supervision level was not detected and the finding was not supported by an exact sign-flip sensitivity test.

That conclusion is much more defensible than a conclusion that synthetic DAPT is beneficial under reduced supervision.

The core evidence for the full-data conclusion is quite solid within the stated experimental scope. The weak point is specifically the reduced-supervision inference.

Bottom line

I would characterize the paper as:

Methodologically thoughtful + highly transparent + interesting null result + promising audit framework, but statistically underpowered for the most interesting positive claim and not fully controlled for optimization budget.

If you fix Issues 1–7, the paper's scientific defensibility improves substantially.