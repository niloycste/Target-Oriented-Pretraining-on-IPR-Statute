# DAPT paper protocol pointer

The executable protocol for the standalone statutory DAPT paper
is maintained at:

`experiments/dapt/EXPERIMENT_PROTOCOL.md`

The split-paper policy is maintained at:

`docs/PAPER_SPLIT_OWNERSHIP.md`

Current DAPT result provenance is listed in:

`docs/RESULT_PROVENANCE.md`

## Final CPU-only experiment (Experiments A, B, C)

The plan is `papers/target_pretraining/dapt.md`. Its build steps (§36 STEP 1–7) are implemented
in `scripts/09_dapt_final/` and their outputs land in
`experiments/dapt/final_cpu/`:

| Artifact | Produced by | Records |
|---|---|---|
| `SYNTHETIC_PROVENANCE_AUDIT.json` | `audit_synthetic_provenance.py` | verdict: reuse the 13,211-passage pool |
| `AUTHENTIC_TRAIN_ONLY_MANIFEST.json` | `build_final_matched_corpora.py` | 898 train-only contexts |
| `corpus_{A,S,M}.jsonl` + manifests | `build_final_matched_corpora.py` | token-matched to 128,727 |
| `corpus_S_STYLE.jsonl` | `generate_style_only_corpus.py` → `characterise_corpora.py` | pending generation |
| `CORPUS_CHARACTERISATION.json` | `characterise_corpora.py` | §18/§25/§26 |
| `MLM_DIAGNOSTIC_MANIFEST.json` | `build_diagnostic_and_lowresource.py` | 104 validation contexts |
| `LOW_RESOURCE_{10,25,50}_MANIFEST.json` | `build_diagnostic_and_lowresource.py` | section-level, Act-stratified |
| `RUN_LEDGER.csv` | `run_final_experiments.py` | per-stage status |
| `DRY_RUN_REPORT.md` | `write_dry_run_report.py` | §37 checklist |

Two differences from the archived study are the reason it exists:

1. **Train-only.** The archived corpora carried three test contexts. The new
   authentic corpus is 898 unique training contexts, verified disjoint from
   validation and test.
2. **Token-matched.** The archived corpora were matched on record count, leaving
   authentic at 128,929 tokens against synthetic at 146,001 — a 13.2% exposure
   advantage confounded with composition. The new corpora agree within 0.05%.

Training has **not** started. §36 requires review of `DRY_RUN_REPORT.md` first.
