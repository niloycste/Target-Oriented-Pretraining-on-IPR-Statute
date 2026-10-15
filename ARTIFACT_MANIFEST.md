# Artifact manifest

This folder was assembled from the research workspace by
`scripts/utils/build_paper_repositories.ps1` in the parent workspace.

- `papers/`: submission manuscript and rendered figures.
- `scripts/`: final, equal-update, archived-control, evaluation, and plotting
  code.
- `data/`: fixed IndicIPR-QA data and the archived DAPT text corpora.
- `experiments/`: final corpora, manifests, locks, ledgers, predictions, and
  statistical outputs. Cache directories and logs are omitted.
- `results/`: archived reader, MLM, bootstrap, and matched-corpus evidence.
- `docs/`: experiment protocol, paper ownership, and provenance.

Small tokenizer/configuration metadata and training telemetry are included for
provenance. No credential file, trained model weight, optimizer state, or
tokenization cache is included.
