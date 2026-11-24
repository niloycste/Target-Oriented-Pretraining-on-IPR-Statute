"""Verified serial external evaluation across all three encoder cohorts."""
import importlib
from pathlib import Path
import traceback
from datetime import datetime
import run as external

ORIGINAL_TARGETS = external.targets


def resolved_targets(encoder):
    targets = ORIGINAL_TARGETS(encoder)
    for target in targets:
        checkpoint = Path(target['checkpoint'])
        if not checkpoint.is_dir() and encoder == 'inlegalbert':
            legacy = external.ROOT.parents[1] / checkpoint.relative_to(external.ROOT)
            if not legacy.is_dir():
                raise FileNotFoundError(f'Neither original nor workspace checkpoint exists: {checkpoint}')
            target['checkpoint'] = str(legacy)
    return targets


def preflight(work):
    canonical = importlib.import_module('scripts.09_dapt_final.evaluate_final_experiments')
    external.verify_protocol(work)
    inventory = {}
    for encoder in external.ENCODERS:
        file = (external.ROOT / 'experiments/dapt/equal_update/results.json' if encoder == 'inlegalbert'
                else external.ROOT / f'experiments/dapt/encoder_extension/{encoder}/results.json')
        saved = external.read(file)['runs']
        prefix = 'D' if encoder == 'inlegalbert' else 'E'
        inventory[encoder] = []
        for target in resolved_targets(encoder):
            key = f'{prefix}_{target["condition"]}_s{target["seed"]}_100'
            digest = canonical.checkpoint_sha256(Path(target['checkpoint']))
            if digest != saved[key]['checkpoint_sha256']:
                raise ValueError(f'Original fingerprint mismatch: {encoder}/{key}')
            inventory[encoder].append({**target, 'original_reader_sha256': digest})
        print(f'Preflight PASS: {encoder}, all 20 original reader fingerprints', flush=True)
    amendment = {
        'reason': 'Original InLegalBERT readers are retained in the parent workspace, not the Git repository.',
        'timing': 'Recorded before any formal external evaluation; no training or selection changes.',
        'unchanged': 'dataset, subsets, scoring, seeds, conditions, decoder and inference settings',
        'protocol_sha256': external.sha(work / 'PROTOCOL.json'),
        'inventory': inventory,
        'additional_code': {p.name: external.sha(p) for p in
                            [Path(__file__), external.HERE / 'integration_check.py',
                             external.HERE / 'verify_readers.py']},
    }
    external.freeze(work / 'EXECUTION_AMENDMENT.json', amendment)


def main():
    import integration_check
    import verify_readers
    work = external.HERE / 'run_v1'
    status_path = work / 'EXECUTION_STATUS.json'
    def status(stage, **details):
        external.atomic(status_path, {'status': stage, 'updated_at': datetime.now().astimezone().isoformat(), **details})
    try:
        status('VERIFYING')
        verify_readers.verify('legalbert')
        verify_readers.verify('bert')
        preflight(work)
        external.targets = resolved_targets
        integration_check.main()
        for encoder in external.ENCODERS:
            for split in ('val', 'test'):
                status('EVALUATING', encoder=encoder, split=split)
                external.evaluate(work, encoder, split, threads=4)
                external.analyse(work, encoder, split)
        status('COMPLETE', evaluations=120, readers=60,
               note='All primary and secondary outputs verified; update manuscript from final statistical reports.')
        print('COMPLETE: all 60 readers evaluated on both external splits and all six statistics reports verified.', flush=True)
    except Exception:
        status('FAILED', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
