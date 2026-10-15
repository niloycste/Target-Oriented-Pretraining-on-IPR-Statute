"""One-time amendment creation after explicit author approval; never relock."""
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
gate = importlib.import_module('scripts.09_dapt_final.analysis_amendment')
evaluator = importlib.import_module('scripts.09_dapt_final.evaluate_final_experiments')
FINAL = gate.FINAL

def main():
    dest = FINAL / 'PRETEST_AMENDMENT_20260913.json'
    if dest.exists():
        raise RuntimeError('Amendment already exists; refusing to overwrite')
    if (FINAL/'results.json').exists() or list((FINAL/'evaluations/test').glob('*.json')):
        raise RuntimeError('Test results already exist; cannot create a pre-test amendment')
    lock = json.loads((FINAL/'FINAL_ANALYSIS_LOCK.json').read_text(encoding='utf8'))
    assert gate.sha256(FINAL/'FINAL_ANALYSIS_LOCK.json') == gate.ORIGINAL_LOCK_HASH
    for p,h in lock['sha256'].items():
        if p not in gate.ALLOWED_OVERRIDES and gate.sha256(ROOT/p) != h:
            raise RuntimeError(f'Unexpected locked-file change: {p}')
    audit = json.loads((FINAL/'TRAINING_COMPLETION_AUDIT.json').read_text(encoding='utf8'))
    assert not audit['validation_errors'] and audit['saved_validation_runs'] == 47
    review = json.loads((FINAL/'RUNNER_LOCK_REVIEW.json').read_text(encoding='utf8'))
    assert len(review['saved_stage_checks']) == 88
    assert all(r['executable_ast_matches_current_builder'] for r in review['saved_stage_checks'])
    assert review['runner_sha256'] == gate.sha256(ROOT/'scripts/09_dapt_final/run_final_experiments.py')
    runs = json.loads((FINAL/'validation_results.json').read_text(encoding='utf8'))['runs']
    targets = evaluator.target_table()
    assert set(runs) == set(targets) and len(targets) == 47
    checkpoints = {}
    for key, target in targets.items():
        current = evaluator.checkpoint_sha256(target['checkpoint'])
        if current != runs[key]['checkpoint_sha256']:
            raise RuntimeError(f'Checkpoint changed: {key}')
        checkpoints[key] = dict(path=target['checkpoint'].relative_to(ROOT).as_posix(), sha256=current)
    additional = [
        'scripts/09_dapt_final/analysis_amendment.py',
        'scripts/09_dapt_final/freeze_pretest_amendment.py',
        'scripts/05_size_matched_dapt/size_matched_dapt_experiment.py',
        'experiments/dapt/final_cpu/RUNNER_LOCK_REVIEW.json',
        'experiments/dapt/final_cpu/RUNNER_LOCK_REVIEW.md',
        'experiments/dapt/final_cpu/TRAINING_COMPLETION_AUDIT.json',
        'experiments/dapt/final_cpu/validation_results.json',
    ] + [r['path'] for r in review['saved_stage_checks']]
    document = dict(
        created_at=datetime.now().astimezone().isoformat(),
        approved_by='author: explicit conversation approval, 2026-09-13',
        original_lock_sha256=gate.ORIGINAL_LOCK_HASH,
        limitations='Original runner source unavailable; equivalence to the original runner cannot be proved. Validation results were already observed before this amendment. No final test results exist at amendment creation.',
        overrides={p:dict(original_sha256=lock['sha256'][p], reviewed_sha256=gate.sha256(ROOT/p),
                         reason='Reviewed current runner and saved stage scripts' if p.endswith('run_final_experiments.py') else 'Add strict dated-amendment verification and frozen-checkpoint enforcement; scoring and decoding unchanged')
                   for p in sorted(gate.ALLOWED_OVERRIDES)},
        additional_sha256={p:gate.sha256(ROOT/p) for p in additional},
        checkpoints=checkpoints,
    )
    gate.verify_files(lock, document)
    with dest.open('x', encoding='utf8') as stream:
        json.dump(document, stream, indent=2)
    print(f'Preserved original lock; created {dest}', flush=True)

if __name__ == '__main__':
    main()
