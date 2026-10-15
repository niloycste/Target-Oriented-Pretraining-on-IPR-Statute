"""Read-only runner review; write evidence without changing the analysis lock."""
import ast
import collections
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'experiments/dapt/final_cpu'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    runner = ROOT / 'scripts/09_dapt_final/run_final_experiments.py'
    lock_path = FINAL / 'FINAL_ANALYSIS_LOCK.json'
    lock = json.loads(lock_path.read_text(encoding='utf8'))
    spec = importlib.util.spec_from_file_location('reviewed_runner', runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []
    for exp, condition, seed, fraction in module.experiment_plan('all'):
        tag = f'{exp}_{condition if fraction == 100 else condition + str(fraction)}_s{seed}'
        stages = ['ipr'] if exp == 'C' else (['squad', 'ipr'] if condition == 'N' else ['mlm', 'squad', 'ipr'])
        for stage in stages:
            output = module.WORK / f'{tag}_{stage}'
            saved = module.WORK / f'{tag}_{stage}.py'
            if stage == 'mlm':
                source = module.make_mlm_source(module.FINAL / module.CORPUS_FILE[condition], output, seed)
            elif stage == 'squad':
                previous = module.WORK / 'base_encoder' if condition == 'N' else module.WORK / f'{tag}_mlm' / 'final'
                source = module.make_squad_source(previous, output, seed)
            else:
                previous = module.WORK / (f'A_{condition}_s{seed}_squad' if exp == 'C' else f'{tag}_squad') / 'final'
                train = module.FINAL / f'train_squad_{fraction}.jsonl' if exp == 'C' else None
                source = module.make_ipr_source(previous, output, seed, train)
            match = saved.exists() and ast.dump(ast.parse(saved.read_text(encoding='utf8')), include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False)
            checks.append(dict(path=saved.relative_to(ROOT).as_posix(), stage=stage,
                               saved_sha256=digest(saved) if saved.exists() else None,
                               executable_ast_matches_current_builder=match))
    locked_files = [dict(path=p, expected=h, current=digest(ROOT/p) if (ROOT/p).exists() else None)
                    for p,h in lock['sha256'].items()]
    mismatches = [r for r in locked_files if r['expected'] != r['current']]
    git_check = subprocess.run(['git', 'cat-file', '-e', f"{lock['git_revision']}:scripts/09_dapt_final/run_final_experiments.py"],
                               cwd=ROOT, capture_output=True, text=True)
    report = dict(reviewed_at=datetime.now().astimezone().isoformat(),
                  original_lock_sha256=digest(lock_path), runner_sha256=digest(runner),
                  locked_file_count=len(locked_files), locked_file_mismatches=mismatches,
                  original_runner_in_recorded_git_revision=git_check.returncode == 0,
                  stage_counts=dict(collections.Counter(r['stage'] for r in checks)),
                  saved_stage_checks=checks,
                  pending_test_outputs=[p for p in ('results.json','FINAL_STATISTICS.json') if not (FINAL/p).exists()],
                  decision='REVIEW_INCOMPLETE_FOR_ORIGINAL_RUNNER_DIFF',
                  limitation='Saved scripts match the current builder, but do not establish equivalence to the unavailable original runner or prove interruption-free numerical equivalence.',
                  action='Original lock and evaluator unchanged; obtain original runner or explicit approval for a disclosed pre-test amendment.')
    (FINAL/'RUNNER_LOCK_REVIEW.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k != 'saved_stage_checks'}, indent=2))
    print('All saved stage scripts match:', all(r['executable_ast_matches_current_builder'] for r in checks))

if __name__ == '__main__':
    main()
