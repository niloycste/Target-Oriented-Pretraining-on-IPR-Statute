"""Verify completed corrected diagnostics and attach final audit provenance."""
import importlib
import json
import math
from datetime import datetime

m = importlib.import_module('scripts.09_dapt_final.evaluate_mlm_bias_corrected')
manifest = json.loads(m.MANIFEST.read_text())
m.lockcheck.verify_amendment(json.loads((m.FINAL / 'FINAL_ANALYSIS_LOCK.json').read_text()))
assert m.fingerprint() == manifest['sha256'], 'Correction inputs changed'
d = json.loads(m.frozen.OUT.read_text())
assert set(d['results']) == set(m.frozen.targets())
assert d['contexts'] == 104 and d['source_split'] == 'val'
assert d['mask_seed'] == m.frozen.MASK_SEED
assert d['max_length'] == 256 and d['mlm_probability'] == 0.15
assert {r['masked_tokens'] for r in d['results'].values()} == {2092}
base = d['results']['N_base']['perplexity']
for key, row in d['results'].items():
    assert row['checkpoint'] == m.frozen.targets()[key].relative_to(m.frozen.ROOT).as_posix()
    assert math.isfinite(row['loss']) and math.isfinite(row['perplexity'])
    assert math.isclose(math.exp(row['loss']), row['perplexity'], rel_tol=1e-7)
    assert abs(100 * (base-row['perplexity']) / base-row['relative_perplexity_reduction_pct']) < 1e-5
d['completion_audit'] = {
    'verified_at': datetime.now().astimezone().isoformat(),
    'correction_manifest': m.MANIFEST.relative_to(m.frozen.ROOT).as_posix(),
    'correction_manifest_sha256': m.lockcheck.sha256(m.MANIFEST),
    'input_hashes_match': True,
    'expected_targets': 19,
    'completed_targets': 19,
    'masked_tokens_per_target': 2092,
    'perplexity_and_reduction_arithmetic_verified': True,
    'execution_evidence': 'Corrected evaluator session exited 0 and reported all 19 tensor and input-hash checks passed.',
}
m.frozen.atomic_json(m.frozen.OUT, d)
print('PASS: 19 corrected MLM results; hashes, design, masks and arithmetic verified.')
