# Author-approved pre-test amendment: 13 September 2026

The author explicitly approved this amendment and final test evaluation in the
conversation on 13 September 2026. The original September 1 analysis lock is
preserved byte-for-byte; this is a later amendment, not a replacement or a
claim of unchanged original code. Validation results had already been observed.
No final-study test prediction files existed when this amendment was created.

## Review and limitations

The original runner source could not be recovered. Its equivalence to the
current runner therefore cannot be proved. Forty-four of 45 original file
hashes match, and the current builder reproduces all 88 saved training-stage
scripts at the executable syntax-tree level. The 47 final reader checkpoint
hashes match the validation records. These checks do not establish numerical
equivalence between interrupted and uninterrupted training.

## Permitted changes

1. Accept the specifically reviewed current training runner hash.
2. Accept the test evaluator change that verifies this amendment and checks
   each checkpoint against the frozen inventory. The scoring, decoder,
   test data, and statistical analysis were not changed.

The amendment additionally freezes the source-builder dependency missing from
the original lock, review evidence, validation aggregate, saved stage scripts,
amendment verifier, and all 47 reader checkpoint fingerprints. Any unreviewed
file or checkpoint change continues to block final test evaluation.

Exact original and reviewed hashes are recorded in
`PRETEST_AMENDMENT_20260913.json`. Safety tests confirmed rejection of expanded
override scope, an incorrect runner hash, and an incorrect dependency hash.

This amendment authorises evaluation of the existing trained readers, not
test-driven model selection, retraining, or changes to the analysis design.
