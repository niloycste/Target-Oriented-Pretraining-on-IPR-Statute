# Approved MLM diagnostic loading correction

Date: 13 September 2026. This is a post-test diagnostic correction, not a retrospective change to the original analysis lock. The author explicitly approved investigating and correcting the loading issue before this run.

## Cause and evidence

The installed Transformers version is 5.3.0. The locally cached original law-ai/InLegalBERT model loads distinct embedding and decoder weight matrices, but its predictions.bias and predictions.decoder.bias share storage. The training source loads this original model, sets tie_word_embeddings=false to preserve the distinct matrices, trains, and saves it. Safetensors serialization retains only one copy of the shared bias. Loading the saved model with that configuration creates an independent zero decoder bias because the alias is not restored.

The original baseline bias alias was inspected directly. The saved baseline contains all model tensors except the duplicate decoder-bias key; legacy LayerNorm gamma/beta names are mapped to weight/bias. Restoring decoder.bias = predictions.bias in memory produces exactly the same tensors and logits as the original baseline (maximum logit difference 0 on the verification input). This is not a new learned bias or a change to the separate decoder weights.

## Scope

The new evaluate_mlm_bias_corrected.py wrapper imports the frozen evaluator and replaces only its model-loading entry point. It asserts the expected key pattern, restores the bias alias, and checks every loaded tensor against the serialized checkpoint. It preserves the original masking, contexts, batching default, loss computation and perplexity computation. Original checkpoint files, training scripts, QA predictions, original analysis lock and pre-test amendment are unchanged. No retraining is performed.

MLM_BIAS_CORRECTION_20260913.json records approval, timing, baseline verification and hashes of all 19 checkpoint weights/configurations and the diagnostic dependencies before corrected scoring. Hashes are checked again after scoring. The result is accepted only when all 19 targets are present with the same masked-token count on 104 validation contexts.

The earlier partial diagnostic with an unconnected zero bias remains archived as MLM_DIAGNOSTIC_RESULTS_incomplete_loader_issue_20260913.json and must not be used as scientific evidence. The corrected output includes explicit loading-correction provenance.

## Reproduction

Run from the repository root with the existing offline environment:

    python -m scripts.09_dapt_final.evaluate_mlm_bias_corrected

The wrapper refuses to overwrite an existing output or reuse unverified partial results. The original frozen evaluator alone does not apply this correction.

The library implementation inspected for this review is https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/bert/modeling_bert.py .
