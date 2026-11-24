"""Rescore completed encoder extensions, verify training telemetry and weights."""
from __future__ import annotations
import argparse
import collections
import importlib
import hashlib
import statistics
from pathlib import Path
import run as external


def close(actual, expected, location='root'):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f'Keys differ: {location}')
        for key in expected:
            close(actual[key], expected[key], f'{location}/{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(location)
        for a, b in zip(actual, expected):
            close(a, b, location)
    elif isinstance(expected, (float, int)):
        if abs(actual - expected) > 0.000003:
            raise ValueError(f'{location}: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(location)


def verify(encoder):
    import csv
    from scripts.qa_evaluation import compute_em_f1
    canonical = importlib.import_module('scripts.09_dapt_final.evaluate_final_experiments')
    stats = importlib.import_module('scripts.09_dapt_final.analyse_final_experiments')
    exp = external.ROOT / 'experiments/dapt/encoder_extension' / encoder
    model_dir = external.ROOT / 'artifacts/models/encoder_extension' / encoder
    with (exp / 'RUN_LEDGER.csv').open(encoding='utf8', newline='') as handle:
        ledger = list(csv.DictReader(handle))
    expected_stages = {(c, str(s), stage) for c in external.CONDITIONS for s in external.SEEDS
                       for stage in (['squad', 'ipr'] if c == 'N' else ['mlm', 'squad', 'ipr'])}
    if len(ledger) != 55 or {(r['condition'], r['seed'], r['stage']) for r in ledger} != expected_stages:
        raise ValueError('Incomplete/duplicate stage ledger')
    if any(r['status'] != r['stage'].upper() + '_DONE' for r in ledger):
        raise ValueError('Unfinished training stage')
    protocol = external.read(exp / 'PROTOCOL.json')
    for name, digest in protocol['inputs'].items():
        if external.sha(name) != digest:
            raise ValueError(f'Training input/code changed: {name}')
    for name, digest in protocol['scripts'].items():
        # The runner hashes generated source strings before Windows write_text
        # translates LF to CRLF. Compare the recorded source representation.
        normalized_digest = hashlib.sha256(Path(name).read_text(encoding='utf8').encode('utf8')).hexdigest()
        if normalized_digest != digest:
            raise ValueError(f'Generated training script changed: {name}')
    telemetry = []
    for c in 'ASM':
        for seed in external.SEEDS:
            path = model_dir / f'E_{c}_s{seed}_mlm/training_telemetry.json'
            data = external.read(path)
            if (data['completed_optimizer_updates'] != 100 or data['configured']['warmup_steps'] != 10
                    or data['configured']['seed'] != seed):
                raise ValueError(f'Telemetry mismatch: {path}')
            telemetry.append({'path': str(path), 'sha256': external.sha(path),
                              'nonpadding_tokens': data['total_non_padding_tokens_processed']})
    fingerprints, checked = {}, 0
    for split, filename in [('test', 'results.json'), ('val', 'validation_results.json')]:
        records = external.jsonl(external.ROOT / f'data/splits/{split}.jsonl')
        runs = external.read(exp / filename)['runs']
        if set(runs) != {f'E_{c}_s{s}_100' for c in external.CONDITIONS for s in external.SEEDS}:
            raise ValueError('Incomplete evaluation cells')
        for key, run in runs.items():
            path = external.ROOT / run['predictions_file']
            payload = external.read(path)
            predictions = payload['predictions']
            if [p['id'] for p in predictions] != [r['id'] for r in records]:
                raise ValueError(f'IDs differ: {key}')
            values = []
            for pred, record in zip(predictions, records):
                if pred['question'] != record['question'] or pred['gold'] != record['answer']:
                    raise ValueError(f'Input mismatch: {key}/{record["id"]}')
                if pred['prediction'] and pred['prediction'] not in record['context']:
                    raise ValueError(f'Non-source prediction: {key}')
                pair = compute_em_f1(pred['prediction'], record['answer'])
                close(pair[0], pred['em'])
                close(pair[1], pred['f1'])
                values.append(pair)
            for index, metric in enumerate(('em', 'f1')):
                value = 100 * statistics.mean(v[index] for v in values)
                close(value, run[metric])
                close(value, payload[metric])
            checkpoint = external.ROOT / run['checkpoint']
            fingerprint = fingerprints.setdefault(str(checkpoint), canonical.checkpoint_sha256(checkpoint))
            if fingerprint != run['checkpoint_sha256'] or fingerprint != payload['checkpoint_sha256']:
                raise ValueError(f'Checkpoint fingerprint mismatch: {key}')
            checked += len(predictions)
        table = {c: {s: runs[f'E_{c}_s{s}_100'] for s in external.SEEDS} for c in external.CONDITIONS}
        recomputed = {}
        for metric in ('f1', 'em'):
            contrasts = {c: stats.paired([table[c][s][metric] for s in external.SEEDS],
                         [table['N'][s][metric] for s in external.SEEDS], f'{c}-N', external.SEEDS)
                         for c in 'ASM'}
            stats.holm(contrasts)
            recomputed[metric] = {'blocked_anova': stats.blocked_anova(table, metric),
                                 'contrasts': contrasts,
                                 'descriptives': {c: stats.mean_sd_ci([table[c][s][metric] for s in external.SEEDS])
                                                  for c in external.CONDITIONS}}
        close(recomputed, external.read(exp / f'{split}_statistics.json'))
    report = {'status': 'PASS', 'encoder': encoder, 'predictions_rescored': checked,
              'readers_fingerprinted': len(fingerprints), 'completed_training_stages': len(ledger),
              'training_inputs_and_generated_code_match': True,
              'generated_script_hash_convention': 'UTF-8 source with universal-newline reading; runner hashes pre-write strings',
              'test_and_validation_statistics_reproduced': True, 'mlm_telemetry': telemetry,
              'checkpoint_fingerprints': fingerprints,
              'scope': 'Artifact consistency verification, not independent retraining or legal assessment'}
    external.atomic(external.HERE / f'{encoder}_verification.json', report)
    print(f'PASS {encoder}: {checked} predictions; {len(fingerprints)} readers; 55 stages; stats reproduced', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=['bert', 'legalbert'], required=True)
    verify(parser.parse_args().encoder)
