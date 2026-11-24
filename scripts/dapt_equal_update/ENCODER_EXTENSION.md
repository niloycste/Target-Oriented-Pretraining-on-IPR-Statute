# Additional encoder experiment

Added 6 November 2026 as a post-hoc extension. Primary additional encoder:
`bert-base-uncased`; optional replication: `nlpaueb/legal-bert-base-uncased`.
The cached snapshot revision and file hashes are frozen in each PROTOCOL.json
before training. This extension does not establish that prior legal exposure
causes an encoder difference: pretraining sources and tokenizers also differ.

Each encoder has 20 NEW readers: N/A/S/M by seeds 7,42,99,123,2026.
N is trained from that encoder, never copied from InLegalBERT. A/S/M use the
existing fixed corpus records and 100 updates, 10 warmup updates. Downstream
SQuAD (20,000 examples, two epochs) and IPR training retain the existing recipe
and validation-loss checkpoint selection. Token counts need not match across
tokenizers. Token caches are isolated by encoder and stage. MLM telemetry is
in the individual model directory.

Commands from the repository root:

```powershell
python scripts/dapt_equal_update/test_encoder_extension.py
python scripts/dapt_equal_update/encoder_extension.py --encoder bert --smoke
python scripts/dapt_equal_update/encoder_extension.py --encoder bert --execute
```

Replace bert with legalbert for the optional second encoder. Do not run two
processes for the same encoder. Training is serial and resumes checkpoints;
completed stages are skipped using the inherited model completeness check.
Keep the machine awake. CPU execution may require several days for 20 routes.
At least 8 GiB free disk is checked before each route; allow substantially more
space for all retained checkpoints. No existing study artifacts are modified.

Outputs: `experiments/dapt/encoder_extension/<encoder>/`; model checkpoints:
`artifacts/models/encoder_extension/<encoder>/`. RUN_LEDGER.csv records training
status. Each completed reader produces validation and test predictions with the
canonical sliding-window decoder. All conditions and analyses are fixed before
new test evaluation; test results must not be used to change this recipe.
The existing test set has already been studied, so this is not an independent
unseen-data confirmation. At completion, val_statistics.json and
test_statistics.json report seed-blocked ANOVA, descriptive scores, paired
A/S/M minus N contrasts with t intervals and Holm corrections within metric.

Current verification covers generated-route isolation, unchanged downstream
logic, protocol mutation rejection, and real MLM/QA forward/backward passes.
It does not prove that the implementation is error-free. Full training,
telemetry and saved prediction audits remain necessary before manuscript use.
The draft protocol files preserve the preparation state before smoke-review
corrections; PROTOCOL.json is the actual execution protocol.
