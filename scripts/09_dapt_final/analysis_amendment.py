"""Strict verification of the author-approved, dated pre-test amendment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'experiments/dapt/final_cpu'
ORIGINAL_LOCK_HASH = 'b05afa430cd0846b714157de1da6b66bc508eae2b40bf8e1e6659fd1f8d2cfc3'
ALLOWED_OVERRIDES = {
    'scripts/09_dapt_final/run_final_experiments.py',
    'scripts/09_dapt_final/evaluate_final_experiments.py',
}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def verify_files(lock, amendment, root=ROOT):
    if amendment.get('original_lock_sha256') != ORIGINAL_LOCK_HASH:
        raise RuntimeError('Amendment does not reference the preserved original lock')
    if amendment.get('approved_by') != 'author: explicit conversation approval, 2026-09-13':
        raise RuntimeError('Missing amendment approval record')
    overrides = amendment.get('overrides', {})
    if set(overrides) != ALLOWED_OVERRIDES:
        raise RuntimeError('Amendment override scope differs from the reviewed two files')
    required = {
        'scripts/09_dapt_final/analysis_amendment.py',
        'scripts/05_size_matched_dapt/size_matched_dapt_experiment.py',
        'experiments/dapt/final_cpu/RUNNER_LOCK_REVIEW.json',
        'experiments/dapt/final_cpu/RUNNER_LOCK_REVIEW.md',
        'experiments/dapt/final_cpu/TRAINING_COMPLETION_AUDIT.json',
    }
    if not required.issubset(amendment.get('additional_sha256', {})):
        raise RuntimeError('Amendment omits required dependencies or evidence')
    for relative, original in lock['sha256'].items():
        expected = original
        if relative in overrides:
            item = overrides[relative]
            if item['original_sha256'] != original:
                raise RuntimeError(f'Original hash mismatch in amendment: {relative}')
            expected = item['reviewed_sha256']
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Unreviewed file change; test evaluation refused: {relative}')
    for relative, expected in amendment['additional_sha256'].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Amendment dependency changed: {relative}')

def verify_amendment(lock):
    if sha256(FINAL / 'FINAL_ANALYSIS_LOCK.json') != ORIGINAL_LOCK_HASH:
        raise RuntimeError('Original analysis lock changed; evaluation refused')
    amendment = json.loads((FINAL / 'PRETEST_AMENDMENT_20260913.json').read_text(encoding='utf8'))
    verify_files(lock, amendment)
    print('Original analysis lock preserved; dated pre-test amendment verified.', flush=True)
    return amendment
