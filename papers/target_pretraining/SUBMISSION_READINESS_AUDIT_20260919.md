# DAPT submission-readiness audit

> Historical pre-revision audit. The manuscript was subsequently revised on
> 20 September 2026; see `REVISION_NOTES_20260920.md` and
> `submission_audit_20260920.json` for current verification and remaining author
> actions. Section/line numbers below refer to the earlier draft.

Audit date: 19 September 2026. Target: `target-pretraining.tex`, its bibliography and PDF, the executed CPU study, and supporting code/results.

## Verdict

**The mandatory, frozen CPU experiment package is complete. The principal numerical results are reproducible from the saved predictions. The manuscript is not yet submission-ready.**

This is not a finding that the experiments need to be started again. It is a finding that the completed study needs a more accurate methods account, fuller reporting of existing analyses, a different main-paper emphasis, and finished submission materials. A separate optimisation-budget limitation also constrains what can be concluded from the negative result.

No manuscript, bibliography, checkpoint, corpus, frozen analysis file, or experiment result was changed in this review. No training, generation, or fresh model inference was run. The new artifacts are this report, `submission_audit_20260919.json`, and `scripts/09_dapt_final/audit_submission_evidence.py`. PDF inspection files are under `tmp/pdfs/dapt_audit_20260919/`.

## 1. What was actually verified

The fresh audit script:

- Checks the exact expected inventory of 47 readers on both validation and test splits.
- Independently re-scores **16,168 test predictions and 19,176 validation predictions**, using the saved answer strings and the released references. All 35,344 scores agree with the saved results within serialization precision.
- Checks question IDs, question text, answer references, and literal source-paragraph extraction. All 47 test runs use the same 344 questions; validation uses the same 408 questions.
- Recomputes all 47 reader checkpoint fingerprints from the actual weight/configuration/tokenizer files and compares them with both evaluation records. All match.
- Checks the original analysis lock and accepted pre-test amendment, including their additional file hashes. No unapproved mismatch was found.
- Independently recomputes the main seed-blocked ANOVA and the synthetic-minus-no-DAPT contrast at 50% supervision. Re-executes the frozen pure analysis functions in memory for all A/B/C statistics and difficulty aggregates; these reproduce the saved reports.
- Recounts corpus records and tokenizer-defined non-padding token budgets, and checks for exact normalized held-out contexts in the final DAPT corpora. No exact validation/test context match was found.
- Checks materialized reduced-supervision IDs against their manifests.
- Checks the 19-entry MLM diagnostic inventory, common mask counts, and loss-to-perplexity conversion. This is an artifact/arithmetic check, not new MLM inference.
- Compares the actual manuscript validation/test table entries and archived reader table entries against their result files.
- Checks LaTeX labels/citations and reviews the existing 26-page PDF, including all six figures, the final study tables, and bibliography.

There are no detected numerical-integrity failures in these checks. This does **not** certify annotation correctness, historical generator pretraining provenance, legal validity of synthetic text, or independent replication of model training.

Reproduce the local numerical audit from the repository root:

```powershell
python scripts/09_dapt_final/audit_submission_evidence.py
```

It reads sizeable checkpoint files and writes only the separate audit JSON. Its `errors` list concerns the implemented integrity checks; an empty list does not mean that the methodological and editorial findings below disappear.

## 2. Experiment completion and verified results

| Component | Executed package | Status |
|---|---|---|
| A: full supervision | N/A/S/M, seeds 7, 42, 99, 123, 2026 | 20 readers complete |
| B: target alignment | Three style-only readers; reuse the matching three S readers from A | 3 additional readers complete |
| C: reduced supervision | N/A/S/M at 25% and 50%, seeds 7, 42, 99 | 24 additional readers complete |
| Validation/test evaluation | 47 readers on each split | Complete |
| Final MLM diagnostic | Base encoder + 15 A/S/M + 3 style checkpoints | 19 complete; 104 validation contexts; 2,092 fixed masked positions each |
| Corpus/provenance analyses | Matching manifests, source audit, corpus characterization, duplicate checks | Saved and available |
| Difficulty/interaction analyses | Item tags, group scores, paired supervision-level interaction contrasts | Saved and available, but incompletely reported |

The saved training audit records 88 stage scripts: 18 MLM, 23 SQuAD, and 47 IPR. This is consistent with the intended checkpoint reuse. The original plan in `dapt.md` explicitly defers a second encoder under the CPU constraint and makes 10% supervision optional. Their absence is **not an unfinished mandatory run**. Multiple independently sampled DAPT corpora, external-domain replication, and a synthetic-data scaling curve were not executed; do not describe them as completed.

### Full-data test performance

All values are on the 0–100 scale; SD is across five complete training seeds.

| Condition | Mean EM | Mean F1 ± SD | Paired F1 difference from N, 95% CI |
|---|---:|---:|---|
| No DAPT (N) | 65.76 | 85.86 ± 1.08 | Reference |
| Authentic (A) | 65.58 | 85.75 ± 0.59 | −0.12 [−1.60, 1.37] |
| Synthetic (S) | 65.81 | 85.39 ± 0.67 | −0.47 [−1.41, 0.47] |
| Mixed (M) | 65.35 | 85.48 ± 0.55 | −0.38 [−2.01, 1.24] |

Independent calculation: F(3,12) = 0.635163, p = 0.606516, partial eta-squared = 0.137031. The manuscript's rounded values are correct. This is failure to detect an effect under the executed recipe, **not equivalence and not proof that DAPT is generally ineffective**.

### Target-alignment and reduced-supervision results

- The correctly paired three-seed comparison is S = 85.34 versus style-only = 85.60 F1, difference −0.26, CI [−1.32, 0.80], unadjusted p = 0.404. Comparing the five-seed S mean with the three-seed style mean as if paired would be wrong; the current prose correctly avoids that.
- At 25%, F1 means are N 83.54, A 83.73, S 83.89, M 83.73. None of the DAPT-versus-N intervals excludes zero.
- At 50%, F1 means are N 84.17, A 84.35, S 84.60, M 84.53. S−N = 0.427997 points, paired-t CI [0.300074, 0.555921], raw p = 0.004791, within-fraction Holm-adjusted p = 0.014373. The reported +0.43 and p = 0.014 are correct.
- Validation MLM perplexities are N 14.301; A 2.938 ± 0.022; S 3.235 ± 0.020; M 3.027 ± 0.008; style-only 3.393 ± 0.020. The lower MLM perplexities do not establish better full-data answer extraction.

The archived C′ 87.04, B′ 86.02, and E2′ 87.37 F1 values agree with their archived result file. They are not replacements for the multi-seed no-DAPT control: E2′ uses a different encoder and the archived routes have the disclosed design differences.

Primary evidence: `experiments/dapt/final_cpu/results.json`, `validation_results.json`, `FINAL_STATISTICS.json`, `MLM_DIAGNOSTIC_RESULTS.json`, and `results/evaluation/qa_eval_report.json`.

## 3. Required methodological corrections

### P1. Token matching does not match optimiser updates; all final MLM runs remain in warmup

Relevant manuscript: Section 4.3/Table 3 and Section 9/Table 10. Code: `artifacts/models/final_cpu/A_A_s2026_mlm.py:43` and `:139`; corresponding S/M/style scripts; `scripts/09_dapt_final/build_final_matched_corpora.py`.

The saved recipe uses three epochs, batch size 8, gradient accumulation 4, a nominal learning rate of 2e−5, a linear scheduler, and **100 warmup updates**.

| Corpus | Records | Non-padding tokens per epoch | Optimiser updates over three epochs | Warmup updates |
|---|---:|---:|---:|---:|
| A | 898 | 128,727 | 87 | 100 |
| S | 861 | 128,713 | 81 | 100 |
| M | 879 | 128,663 | 84 | 100 |
| Style-only | 858 | 128,723 | 81 | 100 |

The update counts follow the saved batching recipe and are corroborated by completed progress traces: 87/87 and 81/81 in `resume_20260906_clean_restart.stderr.log`, and 84/84 in `resume_20260908_174537.stderr.log` / `resume_20260909_102711.stderr.log`. The final study matches token totals to within 0.05%, not update counts or integrated learning-rate exposure. Every MLM run finishes before the scheduled warmup completes.

**Why it matters:** This does not invalidate the recorded scores or prove that a different schedule would improve QA. It limits attribution to composition and makes a general negative conclusion about DAPT especially weak. Batch composition, partial accumulation groups, and scheduler exposure still differ.

**Required correction without retraining:** Add the executed update counts, warmup duration, scheduler, and terminal-MLM-checkpoint rule to the final methods. State that the result concerns this short, token-matched three-epoch adaptation recipe and is not an equal-update or fully tuned DAPT comparison.

**Optional strengthening requiring an explicit new protocol:** A matched-update/schedule sensitivity study would directly address this concern. It would require downstream reader evaluation, not merely new MLM loss plots. Preserve the frozen original results and label any such extension as post-hoc; do not silently overwrite them. A second encoder is less directly targeted at this problem.

### P1. The stated MLM checkpoint-selection rule is inaccurate

At `target-pretraining.tex:469` and Table 3 around `:490`, the record-matched runs are said to use the lowest validation loss at every MLM, SQuAD, and IPR stage.

In `scripts/05_size_matched_dapt/size_matched_dapt_experiment.py:70`, `make_mlm` substitutes paths in the MLM script but does not apply the validation-loss rule. `make_squad` and `make_ipr` do apply that rule. The MLM Trainer has a training dataset but no evaluation dataset and saves the final model after its fixed training duration. The final CPU MLM scripts follow the same terminal-checkpoint pattern.

Replace the blanket statement with:

> MLM adaptation uses the terminal checkpoint after three epochs; SQuAD transfer and IndicIPR-QA fine-tuning select checkpoints by lowest validation loss.

The same inaccurate generalization appears in legacy script docstrings. Do not modify frozen source artifacts merely to make their comments agree; document the discrepancy in the release notes or maintain a clearly separated corrected version.

### P1. Distinguish the raw seed-file defect from downstream corpus eligibility

Relevant manuscript: abstract, Sections 4.1–4.2, Table 2, Appendix A/Table 14 (`:1157`), and Appendix B's first safeguard (`:1181`).

The saved `SYNTHETIC_PROVENANCE_AUDIT.json` reports:

- 5,859 raw seed records, with 881 distinct exemplar contexts.
- Three test-context exemplars associated with 19 seed entities; the raw seed-exemplar check fails.
- None of those 19 entities among the 1,282 distinct entities actually sampled for generation.
- No generated passage removed by the propagation quarantine; 13,211 passages remain eligible, including the 9,294 Gemma-3-12B passages used as the final sampling pool.
- No exact held-out reproduction and no 25-token held-out overlap in that audit. Shorter matches exist; absence of a 25-token match is not proof of semantic independence.

Thus, “all 5,859 seeds were training-only” and “held-out contexts were excluded before seed extraction” are not accurate historical descriptions. The available evidence supports **audited non-propagation into the generated pool**, not a flawless original seed file. The final corpora have no exact held-out contexts in the fresh checks. Do not misreport this finding as proven test leakage into the final synthetic-DAPT models.

The archived authentic/record-matched corpora have separate, already disclosed test-context exposure. Keep those historical limitations distinct from the seed-file issue and the final train-only corpus audit.

### P1. Define what 25% and 50% supervision mean

Relevant manuscript: Table 10 and Section 9.3. Code: `build_diagnostic_and_lowresource.py`, `run_final_experiments.py:495`; evidence: `LOW_RESOURCE_25_MANIFEST.json` and `LOW_RESOURCE_50_MANIFEST.json`.

Sampling is by section, stratified by Act, using one fixed sampling seed (20260829):

| Requested fraction | Selected training sections | Selected QA pairs | Actual QA fraction |
|---|---:|---:|---:|
| 25% | 135 / 541 | 942 / 3,710 | 25.39% |
| 50% | 270 / 541 | 1,852 / 3,710 | 49.92% |

The fractions limit final supervised IPR fine-tuning. The DAPT corpus and question-derived generation seeds still come from the broader eligible training resource, and SQuAD transfer remains available. These are **reduced answer-label fine-tuning experiments**, not experiments in which only 25%/50% of the entire task-resource or question-collection budget exists. Three training seeds do not provide three independently sampled low-resource datasets.

This is a defensible design if stated clearly. It cannot by itself establish proportional reductions in total annotation, question acquisition, or end-to-end data costs.

### P2. Clarify the lock-review chronology

Section 9.2 (`:960`) states that 44 of 45 originally locked files are unchanged. This is true of the saved runner review immediately before amendment. The final accepted amendment authorizes **two** changed files: runner and evaluator. The evaluator change adds amendment/checkpoint enforcement, not a different scoring rule.

Use dated wording: “At runner review, 44/45 original hashes matched; the pre-test amendment subsequently authorized the reviewed runner and evaluator, with the remaining 43 original hashes unchanged.” All current hashes match their applicable approved records. Preserve the existing disclosures that validation results had already been seen and that the original runner source cannot be recovered.

## 4. Completed analyses that still need to appear

### Supervision-level interaction

`FINAL_STATISTICS.json` already contains the difference-in-differences analysis using matching seeds. For synthetic DAPT:

- (S−N) at 25% minus (S−N) at 100%: +0.3053, CI [−3.8393, 4.4498], raw p = 0.7813.
- (S−N) at 50% minus (S−N) at 100%: +0.3874, CI [−0.5886, 1.3633], raw p = 0.2298.

All six A/S/M interaction intervals include zero. Report this, at least in an appendix. A significant contrast at 50% and a non-significant contrast at 100% do not establish that the effects differ between supervision levels. No new training is needed.

### Difficulty diagnostics

`QA_DIFFICULTY_DIAGNOSTICS.json` is complete but not substantively integrated. Selected full-data F1 means:

| Mechanically defined group | Questions | N | A | S | M |
|---|---:|---:|---:|---:|---:|
| Enumerated answer | 70 | 71.16 | 70.32 | 69.89 | 69.47 |
| Semicolon in answer | 41 | 81.68 | 79.82 | 78.86 | 80.00 |
| Proviso/exception phrase | 50 | 85.35 | 83.61 | 83.52 | 83.71 |
| Answer length 31+ words | 187 | 85.22 | 84.77 | 84.47 | 84.47 |

These groups are tagged from the gold answer by lexical/formatting rules, not practitioner annotations of legal complexity. They overlap. The short-answer group is also relatively difficult, so the saved results do not support a simple monotonically increasing answer-length difficulty claim. Present the full bins, counts, and seed variation; avoid cherry-picking or causal legal interpretations.

### Corpus characterization and alignment

`CORPUS_CHARACTERISATION.json` provides useful already-computed evidence:

- Mean words per record: A 125.55, S 130.46, M 128.17, style 131.53.
- A has much wider length variation (10th–90th percentiles 33–229 words) than S (113–148) or style (114–148).
- Final S is single-generator Gemma-3-12B, not the two-generator mixture described for the archive.
- Legal markers, vocabulary diversity, concept/Act coverage, and generator metadata are available.
- The near-duplicate audit samples 400 records per corpus: one A pair and zero S/M/style pairs meet 5-gram Jaccard >= 0.8. This is sampled evidence, not a full-corpus near-duplicate guarantee.
- Sampled authentic-to-S and authentic-to-style comparisons each use 90,000 pairs, with maximum Jaccard 0.0427 and 0.0448. This lexical diagnostic does not demonstrate a large target-alignment separation.

Add a compact main-text corpus table and an appendix for the details. Define “target-aligned” by its construction, not as an empirically established semantic superiority. The alignment comparison still depends on particular generated corpora and cannot isolate every generator or concept-selection property.

### Three-seed inferential sensitivity

The +0.43 result is numerically correct under the prespecified paired-t analysis. Its three seed differences are +0.462722, +0.368832, and +0.452438. They are unusually consistent but provide only two degrees of freedom.

As a **new, post-hoc sensitivity check**, the exact two-sided paired sign-flip p-value is 0.25 (only eight sign assignments are possible). This does not make the t-test calculation wrong and should not replace it retrospectively. It makes clear how strongly the narrow t interval depends on assumptions with n=3. Report the seed-level differences and retain the manuscript's caution about the small effect.

The reported seed intervals condition on one fixed test sample, one corpus selection, and fixed reduced-supervision subsets. They are not simultaneous estimates of seed, corpus, source-section, and dataset uncertainty. If adding a source-cluster or hierarchical sensitivity analysis from saved predictions, label it post-hoc and define its estimand; do not treat 47×344 predictions as independent test examples.

## 5. Manuscript organization and claim scope

The final study starts in **Section 9 after the main discussion in Section 8**. All six existing figures emphasize archived evidence. The introduction's three RQs still give MLM fit a primary place, whereas `dapt.md` specifies efficacy, composition, target alignment, and low-resource behavior as four primary questions, with MLM diagnostic evidence secondary.

Recommended structure:

1. Introduction: four questions tied explicitly to A, B, and C; contribution is a bounded empirical audit, not a new architecture.
2. Related work: distinguish TOP-inspired seed-guided generation, DAPT, target adaptation, and supervised synthetic QA.
3. Benchmark, task, split provenance, metrics, and source-relative interpretation.
4. Final corpus construction, executed training recipe, checkpoint selection, budgets, reduced-supervision subsets, and amendments.
5. Full-data efficacy/composition results, style control, reduced-supervision effects and interactions.
6. Corpus and answer-difficulty diagnostics; validation-only MLM fit as supporting evidence.
7. Legal interpretation, limitations, and conclusion.
8. Appendices: archived checkpoints/record-matched results, full seed tables, audit trail, and detailed diagnostics.

The defensible central finding is: **Under this short, token-matched adaptation recipe on one Indian statutory QA benchmark, additional DAPT improves MLM fit but gives no detected full-data QA gain; one small synthetic-DAPT contrast appears at 50% supervised fine-tuning, without a demonstrated supervision-level interaction or target-alignment benefit.**

Keep “non-detection, not equivalence.” Do not claim that the no-DAPT model is significantly best, that synthetic and authentic corpora are equally effective, or that DAPT cannot work. Do not use the archived cross-encoder-family best-reader comparison as a causal control.

Replace “annotated governing paragraph” / “known governing provision” around `target-pretraining.tex:268` and `:286` with “annotated source paragraph/provision.” The experiment does not establish that this is the uniquely governing legal authority. Explain that token overlap and exact match do not directly validate preservation of obligations, exceptions, or legal completeness. Existing human dataset validation is not a new practitioner audit of these reader outputs or synthetic passages.

## 6. References and citation verification

All **27 bibliography keys are used**. No undefined citation keys, undefined cross-references, or duplicate labels were found. Public identities and the main cited themes were checked against publisher/proceedings/author sources where accessible. This is not a full-text re-review of every cited paper, and it does not independently authenticate the anonymous accepted companion paper.

| Entry | Verification / required action | Primary source |
|---|---|---|
| Devlin et al., BERT (2019) | Identity, venue, year and pages match | [ACL](https://aclanthology.org/N19-1423/) |
| Gururangan et al. (2020) | DAPT/TAPT precedent and metadata match; full author list would improve BibTeX completeness | [ACL](https://aclanthology.org/2020.acl-main.740/) |
| Sengupta et al., TOP (2025) | Correct procedural precedent; keep clear that it is not this method's name | [ACL](https://aclanthology.org/2025.coling-main.469/) |
| Chalkidis et al., LEGAL-BERT (2020) | Identity and legal adaptation scope match | [ACL](https://aclanthology.org/2020.findings-emnlp.261/) |
| Paul et al., InLegalBERT (2023) | Identity and Indian-law scope verified; author lists ICAIL 2023. ACM full-page access failed, so exact proceedings pages were not independently rechecked there | [Author publication list](https://cse.iitkgp.ac.in/~pawang/Publications.html), [preprint](https://arxiv.org/abs/2209.06049) |
| Yang et al., EntiGraph (2025) | ICLR 2025 identity and synthetic continued-pretraining scope verified | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html) |
| Kang et al. (2025) | Metadata and conditional synthetic-data findings match | [ACL](https://aclanthology.org/2025.emnlp-main.544/) |
| Chen et al. (2025) | **Keep 2025**: official PMLR entry gives ACML 2025, volume 304, pages 543–557 despite `chen26a` in the URL | [PMLR](https://proceedings.mlr.press/v304/chen26a.html) |
| Jeong et al. (2024) | Metadata and cautionary medical-adaptation comparison match; it is supporting cross-domain context, not legal evidence | [ACL](https://aclanthology.org/2024.emnlp-main.677/) |
| LeXFiles/LegalLAMA (2023) | Identity and multinational-English corpus scope match | [ACL](https://aclanthology.org/2023.acl-long.865/) |
| MultiLegalPile (2024) | Identity, multilingual scope and 689GB title match | [ACL](https://aclanthology.org/2024.acl-long.805/) |
| Nair and Modani (2023) | Identity and legal-characteristic pretraining theme match | [ACL](https://aclanthology.org/2023.findings-eacl.190/) |
| Mosbach et al. (2021) | ICLR paper supports attention to fine-tuning variation | [Primary paper](https://openreview.net/pdf?id=nzpLWnVAyah) |
| Han et al., AusLaw (2026) | Real 2026 article; jurisdiction-specific finding is task-specific. Remove `pages={1--35}` unless official journal pagination is assigned | [Publisher](https://link.springer.com/article/10.1007/s10506-026-09506-9) |
| SQuAD (2016) | Metadata and span-extraction role match. Your normalization retains articles, so do not call it identical to the official SQuAD scorer | [ACL](https://aclanthology.org/D16-1264/) |
| KeyBERT (2020) | Author repository supplies the same year/DOI citation | [Author repository](https://github.com/MaartenGr/KeyBERT#citation) |
| Gemma Team (2025) | Technical report identity verified; actual local model variants/versions must also be documented as experimental metadata | [Report](https://arxiv.org/abs/2503.19786) |
| SciBERT (2019) | Metadata and scientific-domain encoder role match | [ACL](https://aclanthology.org/D19-1371/) |
| BioBERT (2020) | Volume 36(4), pages 1234–1240, issue year 2020 confirmed; online publication in 2019 is not a reason to change the issue year | [Publisher](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506) |
| LexGLUE (2022) | Metadata and benchmark scope match | [ACL](https://aclanthology.org/2022.acl-long.297/) |
| CUAD (2021) | **Correct booktitle** to the NeurIPS Datasets and Benchmarks track proceedings; the current generic Advances in Neural Information Processing Systems entry is imprecise | [Official proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/6ea9ab1baa0efb9e19094440c317e21b-Abstract-round1.html) |
| AILQA (2026) | Publisher confirms 24 August 2026; remove `pages={1--33}` unless assigned journal pagination becomes available | [Publisher](https://link.springer.com/article/10.1007/s10506-026-09537-2) |
| Italiani et al. (2025) | Generation/distillation paper exists; remove `pages={1--26}` unless assigned journal pagination becomes available | [Publisher](https://link.springer.com/article/10.1007/s10506-025-09463-9) |
| Yuan et al. (2024) | Volume 32, pages 769–805, 2024 issue year is consistent with earlier online publication | [Publisher](https://link.springer.com/article/10.1007/s10506-023-09367-6) |
| DiscoLQA (2025) | Volume 33, pages 323–359, 2025 issue year confirmed despite 2024 online publication | [Publisher](https://link.springer.com/article/10.1007/s10506-023-09387-2) |
| EQUALS (2023) | Author repository confirms real Chinese-law extractive QA and retrieval scope. ACM page access failed; exact ACM pagination/complete bibliographic fields remain a final metadata check | [Author repository](https://github.com/andongBlue/EQUALS) |
| Anonymous companion benchmark (2026) | Not externally verifiable while anonymized; provide accepted-paper details and overlap disclosure confidentially to the editor | Author action required |

The related-work claims are generally cautious and appropriately distinguish synthetic pretraining from supervised data generation. No nonexistent public paper was identified in this check. That is narrower than certifying every sentence against every cited full text or proving novelty against all current literature.

## 7. Figures, tables and PDF

The existing PDF has 26 pages. Visual review found no obvious clipped tables, missing figure assets, or visibly unresolved references. The existing build log has no reported undefined references/citations or overfull boxes. This review did not regenerate the manuscript PDF.

Required improvements:

- Add explicit prose references to `tab:generation_settings`, `alg:dapt`, `tab:training_recipe`, `fig:reader_inventory`, `tab:paired_contrasts`, `fig:ppl_bars`, and `tab:final_mlm`. These seven floats currently have no explicit `ref` use.
- Figure 1 is an archived design diagram, with small labels at printed width. A final-study overview should show A/B/C, token budgets, seed counts, and checkpoint reuse instead.
- Figures 2 and 6 use truncated QA bar axes, visually magnifying subpoint differences. Prefer paired points/effect intervals or clearly justified axes. This is a presentation issue, not a numerical error.
- Add figures for the final five-seed comparison and the 25/50/100% paired effects; include seed points and uncertainty. Archived plots can move to the appendix.
- Table 10 currently floats before the heading introducing the final study on page 17. Reordering the manuscript should fix this; avoid interpreting LaTeX float placement as a missing citation.
- There is substantial whitespace before the conclusion, and a single bibliography entry occupies page 26. These are secondary layout issues, best fixed after substantive restructuring.

The PDF skill was used to render and visually inspect the existing output. No image-generation or figure-editing tool was needed for this review.

## 8. Submission readiness and journal fit

The current journal guidance calls for a 150–250-word abstract, 4–6 keywords, relevant declarations, and permits LaTeX for mathematical content. The abstract is approximately **255 whitespace-counted words**, with undefined QA/MLM abbreviations; reduce it comfortably below 250 and define or avoid abbreviations. Six keywords are already present. Add author-confirmed funding/competing-interest and other applicable declarations. [Submission guidelines](https://link.springer.com/journal/10506/submission-guidelines)

The current data/code availability paragraph at `target-pretraining.tex:1209` is a draft instruction (“should accompany”, “should contain”), not a finished availability statement. Replace it with the actual anonymous-review access mechanism, contents, version, and eventual release policy. Do not promise an archive or public license that has not been prepared/approved. Follow the journal's current anonymized-submission workflow; keep identity-bearing metadata out of review artifacts.

The journal explicitly includes evaluation/auditing of legal AI, but also expects substantive technical, legal, and empirical depth and warns against straightforward applications without a strong novel contribution. The audit framing is a plausible fit, not an acceptance prediction. The strongest improvements are accurate protocol reporting, the final controlled study at the center of the paper, and legally interpretable error analysis. [Aims and scope](https://link.springer.com/journal/10506/aims-and-scope)

“Q1” depends on ranking database, year and subject category; this audit does not certify a particular current quartile or guarantee acceptance. External validation or a second encoder could strengthen generality but was not required to finish the frozen CPU plan. A practitioner error audit would strengthen legal depth, but must be genuinely performed and documented rather than inferred from token scores.

Provide the editor the companion resource paper and explain the distinct contribution of this DAPT paper and the separate retrieval paper. Reuse of the same benchmark is not itself a problem; duplicated claims, tables, or results must be identified and appropriately attributed. A full publication-overlap investigation was not performed here.

## 9. Recommended next revision, in order

1. Correct the checkpoint rule, raw-seed provenance, update/warmup account, reduced-supervision definition, and amendment chronology.
2. Recenter the paper on the completed A/B/C study; move most archived material to appendices.
3. Integrate the existing interaction, corpus-characterization and difficulty results, with explicit exploratory/operational qualifications.
4. Add final-study plots and complete the cross-references. Preserve all original statistics and clearly label any new post-hoc analysis.
5. Correct the bibliography metadata noted above, shorten the abstract, and supply real declarations and availability information.
6. Decide explicitly whether to keep the narrowly scoped executed-recipe claim or authorize a separate matched-update/schedule sensitivity study. Do not automatically expand into expensive retraining or a second encoder.
7. Rebuild the revised LaTeX, inspect every page at publication scale, and re-run the evidence audit against the new tables.

**Bottom line:** experiments complete within the agreed CPU scope; principal reported numbers verified; analysis integration incomplete; substantive revisions required before submission.
