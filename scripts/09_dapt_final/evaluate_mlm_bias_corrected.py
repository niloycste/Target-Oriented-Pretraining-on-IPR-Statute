"""Approved post-test diagnostic loading correction; frozen evaluator is unchanged."""
import argparse
import gc
import importlib
import json
from datetime import datetime
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForMaskedLM, AutoTokenizer

frozen = importlib.import_module('scripts.09_dapt_final.evaluate_final_mlm')
lockcheck = importlib.import_module('scripts.09_dapt_final.analysis_amendment')
FINAL = frozen.FINAL
MANIFEST = FINAL / 'MLM_BIAS_CORRECTION_20260913.json'
REAL_LOADER = AutoModelForMaskedLM


def corrected_model(path):
    """Recover only the shared bias removed by safetensors serialization."""
    path = Path(path)
    with safe_open(str(path / 'model.safetensors'), framework='pt') as saved:
        keys = set(saved.keys())
        assert 'cls.predictions.bias' in keys
        assert 'cls.predictions.decoder.bias' not in keys
        assert 'cls.predictions.decoder.weight' in keys
        bias = saved.get_tensor('cls.predictions.bias')
    model = REAL_LOADER.from_pretrained(str(path), local_files_only=True)
    assert model.config.tie_word_embeddings is False
    head = model.cls.predictions
    assert torch.equal(head.bias.detach(), bias)
    # Legacy InLegalBERT serialization uses gamma/beta for LayerNorm.
    names = {k.replace('.LayerNorm.gamma', '.LayerNorm.weight').replace(
        '.LayerNorm.beta', '.LayerNorm.bias'): k for k in keys}
    assert set(model.state_dict()) - set(names) == {'cls.predictions.decoder.bias'}
    assert not (set(names) - set(model.state_dict()))
    # Do not tie decoder weights: their independence is intentional.
    assert head.decoder.weight.data_ptr() != model.bert.embeddings.word_embeddings.weight.data_ptr()
    head.decoder.bias = head.bias
    assert head.decoder.bias.data_ptr() == head.bias.data_ptr()
    # Confirm that every serialized tensor survives loading unchanged.
    with safe_open(str(path / 'model.safetensors'), framework='pt') as saved:
        for key, value in model.state_dict().items():
            source = 'cls.predictions.bias' if key == 'cls.predictions.decoder.bias' else names[key]
            assert torch.equal(value.detach(), saved.get_tensor(source)), key
    return model


class CorrectedLoader:
    @staticmethod
    def from_pretrained(path, **kwargs):
        return corrected_model(path)


def fingerprint():
    paths = [Path(__file__), Path(frozen.__file__), FINAL / 'mlm_diagnostic.jsonl',
             FINAL / 'FINAL_ANALYSIS_LOCK.json', FINAL / 'PRETEST_AMENDMENT_20260913.json']
    for path in frozen.targets().values():
        paths.extend([path / 'config.json', path / 'model.safetensors'])
    return {p.relative_to(frozen.ROOT).as_posix(): lockcheck.sha256(p) for p in paths}


def verify_baseline():
    original = REAL_LOADER.from_pretrained('law-ai/InLegalBERT', local_files_only=True)
    assert original.cls.predictions.bias.data_ptr() == original.cls.predictions.decoder.bias.data_ptr()
    restored = corrected_model(frozen.WORK / 'base_encoder')
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    original.eval()
    restored.eval()
    ids = torch.tensor([[101, 1996, 103, 102]])
    with torch.no_grad():
        difference = (original(ids).logits - restored(ids).logits).abs().max().item()
    assert difference == 0, difference
    del original, restored
    gc.collect()
    return {'all_baseline_tensors_identical': True, 'maximum_logit_difference': difference,
            'original_bias_alias_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    lockcheck.verify_amendment(json.loads((FINAL / 'FINAL_ANALYSIS_LOCK.json').read_text()))
    if args.prepare:
        if MANIFEST.exists():
            raise RuntimeError('Correction manifest already exists; refusing overwrite')
        evidence = verify_baseline()
        frozen.atomic_json(MANIFEST, {
            'created_at': datetime.now().astimezone().isoformat(),
            'approved_by': 'author: explicit conversation approval to investigate and correct MLM loading, 2026-09-13',
            'timing': 'post-test; before corrected validation-only MLM evaluation',
            'scope': 'Restore decoder.bias alias to saved predictions.bias in memory only. No weight tying, retraining, checkpoint edits, mask changes or QA changes.',
            'baseline_verification': evidence, 'sha256': fingerprint(),
        })
        print('Correction manifest saved; baseline tensors and logits identical.', flush=True)
        return
    manifest = json.loads(MANIFEST.read_text())
    assert fingerprint() == manifest['sha256'], 'Correction inputs changed'
    if frozen.OUT.exists():
        raise RuntimeError('Output already exists; refusing unverified reuse or overwrite')
    frozen.AutoModelForMaskedLM = CorrectedLoader
    # Execute the original mask construction, inference and scoring unchanged.
    assert frozen.main() == 0
    assert fingerprint() == manifest['sha256'], 'Inputs changed during evaluation'
    result = json.loads(frozen.OUT.read_text())
    assert set(result['results']) == set(frozen.targets())
    assert result['contexts'] == 104
    assert len({r['masked_tokens'] for r in result['results'].values()}) == 1
    result['loading_correction'] = {
        'manifest': MANIFEST.relative_to(frozen.ROOT).as_posix(),
        'manifest_sha256': lockcheck.sha256(MANIFEST),
        'all_serialized_tensors_verified': True,
        'checkpoints_unchanged_after_evaluation': True,
    }
    result['updated_at'] = datetime.now().astimezone().isoformat()
    frozen.atomic_json(frozen.OUT, result)
    print('All 19 diagnostics complete; serialized tensors and input hashes verified.', flush=True)


if __name__ == '__main__':
    main()
