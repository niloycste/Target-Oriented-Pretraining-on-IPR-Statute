"""Audit saved final-study checkpoints and validation results without inference."""
import collections
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'experiments/dapt/final_cpu'

def checkpoint_hash(path):
    files = set()
    for pattern in ('config.json', '*.safetensors', 'pytorch_model*.bin', '*.index.json',
                    'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json'):
        files.update(p for p in path.glob(pattern) if p.is_file())
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda p: p.name):
        digest.update(file.name.encode('utf8'))
        with file.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()

def main():
    aggregate = json.loads((FINAL / 'validation_results.json').read_text(encoding='utf8'))
    runs = aggregate['runs']
    expected = {f'A_{c}_s{s}_100' for c in 'NASM' for s in (7,42,99,123,2026)}
    expected |= {f'B_S_STYLE_s{s}_100' for s in (7,42,99)}
    expected |= {f'C_{c}_s{s}_{f}' for c in 'NASM' for s in (7,42,99) for f in (25,50)}
    errors = []
    if set(runs) != expected:
        errors.append('Run inventory differs from the 47 planned readers')
    groups = collections.defaultdict(list)
    for key, run in sorted(runs.items()):
        saved = json.loads((ROOT / run['predictions_file']).read_text(encoding='utf8'))
        predictions = saved['predictions']
        if run['split'] != 'val' or saved['split'] != 'val' or len(predictions) != 408:
            errors.append(f'{key}: split/count mismatch')
        for metric in ('em', 'f1'):
            recomputed = 100 * statistics.mean(p[metric] for p in predictions)
            if abs(recomputed - run[metric]) > 0.000002 or saved[metric] != run[metric]:
                errors.append(f'{key}: {metric} mismatch')
        model = ROOT / run['checkpoint']
        if not (model / 'config.json').exists() or not list(model.glob('*.safetensors')):
            errors.append(f'{key}: missing checkpoint')
        elif checkpoint_hash(model) != run['checkpoint_sha256']:
            errors.append(f'{key}: checkpoint hash mismatch')
        groups[(run['experiment'], run['condition'], run['fraction'])].append(run)
        print(f'Checked {key}', flush=True)
    lock = json.loads((FINAL / 'FINAL_ANALYSIS_LOCK.json').read_text(encoding='utf8'))
    mismatches = [p for p,h in lock['sha256'].items() if not (ROOT/p).exists()
                  or hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h]
    summaries = [dict(experiment=k[0], condition=k[1], fraction=k[2], seeds=len(rs),
                      em_mean=statistics.mean(r['em'] for r in rs),
                      f1_mean=statistics.mean(r['f1'] for r in rs),
                      f1_sd=statistics.stdev(r['f1'] for r in rs))
                 for k,rs in sorted(groups.items())]
    report = dict(checked_at=datetime.now().astimezone().isoformat(), expected_readers=47,
                  saved_validation_runs=len(runs), validation_errors=errors,
                  analysis_lock_mismatches=mismatches, validation_summaries=summaries,
                  missing_final_outputs=[p for p in ('results.json', 'FINAL_STATISTICS.json',
                    'MLM_DIAGNOSTIC_RESULTS.json', 'QA_DIFFICULTY_DIAGNOSTICS.json')
                    if not (FINAL/p).exists()],
                  scope='Saved checkpoint hashes and saved validation-score aggregation; no retraining or inference')
    (FINAL/'TRAINING_COMPLETION_AUDIT.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
