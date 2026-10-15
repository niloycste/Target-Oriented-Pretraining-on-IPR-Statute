# Evidence-based manuscript revision

Date: 20 September 2026. Scope: the DAPT/target-pretraining paper only.

## Outcome

The manuscript now leads with the completed 47-reader study, without inventing
experiments or changing saved results. Existing analyses are reported in the
main results/diagnostics or supporting appendices. Earlier record-matched
experiments are explicitly historical evidence.

This completes the evidence-based reporting revision, not every factual step
needed for submission. Resolve the author actions below before submission.
An internal audit cannot guarantee Q1-journal acceptance.

## Corrections made

- Rebuilt the abstract, questions, main results, discussion and conclusion
  around experiments A/B/C, rather than the strongest archived run.
- Corrected the training account: terminal MLM checkpoints, validation-selected
  supervised checkpoints, actual token budgets and 81--87 optimiser updates
  inside the 100-update warmup.
- Distinguished raw seed contamination from the saved non-propagation result;
  disclosed original-runner uncertainty, the pre-test amendment and subsequent
  MLM loading correction.
- Reported the target-style control, all supervision-level interactions, all
  nine answer-property groups, final corpus properties and validation-only MLM
  diagnostics.
- Retained the +0.43 F1 synthetic-minus-no-DAPT contrast at 50% supervision,
  its paired interval and within-fraction correction. Explained that the
  interaction is not established and three paired seeds limit inference.
  Added a labelled post-hoc sign-flip sensitivity calculation, not a new
  experimental result or replacement for frozen statistics.
- Generated two vector figures from saved test scores. Old figures and results
  remain on disk but are not presented as final experimental evidence.
- Corrected CUAD's proceedings venue and removed unsupported journal page
  ranges for Han, AILQA and Italiani. Removed the unverified assertion that
  companion-paper details had already been supplied to the editor.
- Added an accurate AI-assisted manuscript-preparation disclosure for author
  review. Did not invent funding, conflicts, ethics approval, access links,
  public releases or practitioner validation.

## Verification

The current report is `submission_audit_20260920.json`. Reproduce it with
`python scripts/09_dapt_final/audit_submission_evidence.py` from the root.

- Independently rescored 16,168 test and 19,176 validation answers; checked
  question/reference alignment and literal extraction from supplied contexts.
- Recomputed all 47 actual reader fingerprints and compared them with the
  evaluation records; checked the original lock and authorised amendment.
- Reproduced A/B/C statistics and answer-property aggregates from saved runs.
- Checked corpus counts, non-padding tokens, exact held-out-context matches,
  reduced-supervision manifests and the 19-checkpoint MLM inventory.
- Checked displayed test, validation, archived-reader, difficulty, corpus-
  property, budget, interaction and final-MLM tables against saved files.
- All 27 bibliography keys are cited; no missing references, duplicate labels
  or unreferenced figure/table/algorithm labels were detected.
- Built the PDF with BibTeX and repeated pdfLaTeX passes; inspected rendered
  pages and corrected the opening-heading layout. No undefined-reference or
  overfull-box warning remains. The class still emits underfull-box notices;
  these are not evidence of missing content.

This does not independently replicate training, re-run MLM inference,
establish semantic non-contamination or certify legal correctness. No model
checkpoint, training corpus, frozen statistical report or experimental
prediction was modified during this revision.

## Author actions before submission

1. **Confirm funding and competing interests.** Supply actual statements;
   do not default to "no funding" or "no conflicts" without confirmation.
   Complete authorship/contribution and applicable ethics declarations.
2. **Arrange reviewer access.** Confirm an anonymised supplementary archive or
   anonymous repository. Add the actual access mechanism and release policy
   to Data and Code Availability. Confirm redistribution rights/licences
   before sharing models, data or generated corpora. No archive upload or
   public release has been performed by this revision.
3. **Confirm the companion paper.** Verify accepted status and exact identity,
   supply the details confidentially to the editor, and disclose shared
   benchmark/model material and the separate retrieval paper. The anonymous
   citation cannot be independently authenticated from public metadata.
4. **Approve the scientific text and AI-use disclosure.** The disclosure
   describes assistance actually used; final responsibility and approval
   remain with the authors.
5. **Finish the EQUALS metadata check.** Its identity and Chinese-law QA scope
   were checked against the authors' repository, but direct ACM access returned
   HTTP 403. Do not claim that every bibliographic field is independently
   publisher-verified. Consult the publisher export when accessible. The
   existing citation is retained; this access limit is not evidence of a
   nonexistent paper.

## Scientific limitations that editing cannot remove

The study supports a recipe-specific non-detection of full-data DAPT gains,
not general ineffectiveness. Matching tokens does not match updates or
learning-rate exposure; every MLM run ends during warmup. One encoder,
one benchmark, fixed corpus selections and fixed reduced-supervision subsets
limit generalisation. No human assessment establishes legal adequacy.

A prospectively specified matched-update/schedule study, more seeds, another
encoder or independent benchmark, and practitioner assessment would strengthen
the evidence. None has been performed or claimed here. Such extensions would
be new experiments, not retrospective changes to the frozen study.

## Reference and journal checks

The 27-entry source ledger is in Section 6 of the historical
`SUBMISSION_READINESS_AUDIT_20260919.md`. Correction requests are superseded
where implemented above. This was an identity, metadata and claim-scope check
where primary sources were accessible, not full-text re-review of every paper.

Additional checks during revision:

- [Official CUAD proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/6ea9ab1baa0efb9e19094440c317e21b-Abstract-round1.html)
  support the corrected Datasets and Benchmarks venue.
- [ACM's InLegalBERT record](https://doi.org/10.1145/3594536.3595165)
  confirms Paul et al., ICAIL 2023, pages 187--196, resolving the earlier
  pagination-access limitation.
- [The EQUALS author repository](https://github.com/andongBlue/EQUALS)
  confirms Chinese-law extractive QA with an article-retrieval component.
- [Artificial Intelligence and Law submission guidance](https://link.springer.com/journal/10506/submission-guidelines)
  was checked for abstract/keywords, declarations and AI-use disclosure.
  The revised abstract has approximately 201 words and six keywords.

No guaranteed acceptance or current database-specific quartile is claimed.
