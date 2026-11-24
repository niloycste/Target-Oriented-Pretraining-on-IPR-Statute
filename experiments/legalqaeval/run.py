"""External, answerable-only transfer evaluation. No training is performed."""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = [7, 42, 99, 123, 2026]
CONDITIONS = ['N', 'A', 'S', 'M']
ENCODERS = ['inlegalbert', 'bert', 'legalbert']
DATASET = 'isaacus/LegalQAEval'
SOURCE_NAMES = ('stanford--squad-v2', 'microsoft--ms-marco',
                'yang--hotpotqa', 'google--natural-questions')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def jsonl(path):
    with Path(path).open(encoding='utf8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(temp, path)


def freeze(path, data):
    if Path(path).exists():
        if read(path) != data:
            raise ValueError(f'Frozen artifact differs: {path}; use a new --work directory')
    else:
        atomic(path, data)


def normalized(text):
    """Conservative overlap check, separate from answer scoring."""
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def source_of(identifier):
    sources = [s for s in SOURCE_NAMES if f'/{s}/' in identifier]
    if len(sources) != 1:
        raise ValueError(f'Unknown or ambiguous source: {identifier}')
    return sources[0]


def validate_rows(rows):
    ids = set()
    for row in rows:
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids:
            raise ValueError('Missing or duplicate ID')
        ids.add(row['id'])
        source_of(row['id'])
        for field in ('text', 'question'):
            if not isinstance(row.get(field), str) or not normalized(row[field]):
                raise ValueError(f'Empty/invalid {field}: {row["id"]}')
        if not isinstance(row.get('answers'), list):
            raise ValueError(f'Invalid answers: {row["id"]}')
        for answer in row['answers']:
            text, start, end = answer.get('text'), answer.get('start'), answer.get('end')
            if (not isinstance(text, str) or not text.strip() or
                    type(start) is not int or type(end) is not int or
                    not 0 <= start < end <= len(row['text']) or
                    row['text'][start:end] != text):
                raise ValueError(f'Answer offset mismatch: {row["id"]}')


def download(work, revision):
    from huggingface_hub import HfApi, hf_hub_download
    # Resolve a branch/tag to a commit once, then fetch both splits at that commit.
    commit = HfApi().dataset_info(DATASET, revision=revision).sha
    payload = {'dataset': DATASET, 'revision': commit, 'files': {}}
    for name in ('val.jsonl', 'test.jsonl', 'README.md'):
        path = Path(hf_hub_download(DATASET, name, repo_type='dataset', revision=commit))
        payload['files'][name] = {'sha256': sha(path)}
        destination = work / 'raw' / name
        if destination.exists() and sha(destination) != sha(path):
            raise ValueError(f'Download would overwrite different data: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            # Copy only the explicitly selected dataset files.
            import shutil
            shutil.copyfile(path, destination)
    freeze(work / 'DOWNLOAD.json', payload)
    print(f'Downloaded {DATASET} at {commit}')


def build_indexes(reference_groups):
    questions, texts = collections.defaultdict(set), collections.defaultdict(set)
    for name, rows in reference_groups.items():
        for row in rows:
            question = normalized(row.get('question', ''))
            context = normalized(row.get('context', row.get('text', '')))
            if question:
                questions[question].add(name)
            if context:
                texts[context].add(name)
    return questions, texts


def audit_splits(splits, references):
    questions, texts = build_indexes(references)
    for rows in splits.values():
        validate_rows(rows)
    val, test = splits['val'], splits['test']
    if set(r['id'] for r in val) & set(r['id'] for r in test):
        raise ValueError('IDs shared between validation and test')
    shared = {field: {normalized(r[field]) for r in val} & {normalized(r[field]) for r in test}
              for field in ('question', 'text')}
    output = {}
    for split, rows in splits.items():
        audited = []
        seen_pairs = set()
        for row in rows:
            pair = normalized(row['question']), normalized(row['text'])
            if pair in seen_pairs:
                raise ValueError(f'Duplicate normalized question/context: {row["id"]}')
            seen_pairs.add(pair)
            overlap = sorted(questions.get(pair[0], set()) | texts.get(pair[1], set()))
            cross_split = [field for field in ('question', 'text') if normalized(row[field]) in shared[field]]
            source = source_of(row['id'])
            answerable = bool(row['answers'])
            audited.append({**row, 'source': source, 'overlap_resources': overlap,
                            'cross_split_overlap': cross_split,
                            'answerable': answerable,
                            'primary': answerable and not overlap and not cross_split and source != 'stanford--squad-v2',
                            'clean_answerable': answerable and not overlap and not cross_split})
        output[split] = audited
    return output


def prepare(work, train_arrow, val_arrow):
    from datasets import Dataset
    receipt = read(work / 'DOWNLOAD.json')
    for name, metadata in receipt['files'].items():
        if sha(work / 'raw' / name) != metadata['sha256']:
            raise ValueError(f'Raw download changed: {name}')
    reference_paths = {
        'ipr_train': ROOT / 'data/splits/train_squad.jsonl',
        'ipr_val': ROOT / 'data/splits/val.jsonl',
        'ipr_test': ROOT / 'data/splits/test.jsonl',
        **{f'corpus_{c}': ROOT / f'experiments/dapt/final_cpu/corpus_{c}.jsonl'
           for c in 'ASM'},
    }
    references = {name: jsonl(path) for name, path in reference_paths.items()}
    # Audit against ALL cached SQuAD train/validation examples, a conservative
    # superset of the actual 20k/2k subsets; do not reconstruct a new shuffle.
    for name, path in [('squad_train_all', train_arrow), ('squad_val_all', val_arrow)]:
        reference_paths[name] = Path(path).resolve()
        references[name] = list(Dataset.from_file(str(path)))
    splits = {split: jsonl(work / 'raw' / f'{split}.jsonl') for split in ('val', 'test')}
    audited = audit_splits(splits, references)
    protocol = {
        'schema': 1, 'dataset': receipt, 'design': 'post-hoc external transfer; no training',
        'seeds': SEEDS, 'conditions': CONDITIONS, 'encoders': ENCODERS,
        'primary': 'answerable; non-SQuAD source; no normalized question OR context overlap with '
                   'study resources or opposite evaluation split; cross-split matches excluded on both sides',
        'secondary': ['clean_answerable_including_squad', 'all_answerable_exploratory'],
        'unanswerable': 'counted in audit; excluded from forced-span evaluation',
        'metrics': 'max independently over annotated answers; project EM/F1; articles retained',
        'inference': {'max_seq_length': 384, 'doc_stride': 128, 'batch_size': 16,
                      'max_answer_tokens': None, 'query_batch_size': 8},
        'references': {name: {'path': str(path.resolve()), 'sha256': sha(path)}
                       for name, path in reference_paths.items()},
        'code': {str(p.relative_to(ROOT)): sha(p) for p in
                 (Path(__file__), ROOT / 'scripts/qa_evaluation.py',
                  ROOT / 'scripts/09_dapt_final/analyse_final_experiments.py')},
        'limitations': ['same already-studied IPR checkpoints',
                        'exact normalized overlap audit is not a semantic contamination audit',
                        'unknown original model pretraining exposure',
                        'multiple listed spans treated as accepted alternatives, not jointly required spans',
                        'seed intervals conditional on fixed evaluation examples',
                        'no causal comparison of encoder pretraining histories'],
    }
    for split, rows in audited.items():
        freeze(work / 'prepared' / f'{split}.json', rows)
    protocol['prepared'] = {s: sha(work / 'prepared' / f'{s}.json') for s in audited}
    freeze(work / 'PROTOCOL.json', protocol)
    report = {}
    for split, rows in audited.items():
        report[split] = {group: dict(collections.Counter(r['source'] for r in rows if predicate(r)))
                        for group, predicate in {
                            'all': lambda r: True, 'answerable': lambda r: r['answerable'],
                            'unanswerable': lambda r: not r['answerable'],
                            'primary': lambda r: r['primary'],
                            'clean_answerable': lambda r: r['clean_answerable'],
                            'overlap': lambda r: bool(r['overlap_resources']),
                            'cross_split_overlap': lambda r: bool(r['cross_split_overlap']),
                        }.items()}
        if not any(r['primary'] for r in rows):
            raise ValueError(f'No primary examples remain in {split}')
    report['pre_scoring_preparation_note'] = (
        'Initial preparation stopped on normalized cross-split contexts. Before any external '
        'model evaluation or protocol freeze, the handling rule was specified as exclusion '
        'on both sides in primary/clean subsets, retaining all-answerable exploratory outputs.')
    freeze(work / 'AUDIT.json', report)
    print(json.dumps(report, indent=2))


def verify_protocol(work):
    protocol = read(work / 'PROTOCOL.json')
    for name, metadata in protocol['dataset']['files'].items():
        if sha(work / 'raw' / name) != metadata['sha256']:
            raise ValueError(f'Raw input changed: {name}')
    for metadata in protocol['references'].values():
        if sha(metadata['path']) != metadata['sha256']:
            raise ValueError(f'Overlap reference changed: {metadata["path"]}')
    for relative, expected in protocol['code'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'Code changed after preparation: {relative}')
    for split, expected in protocol['prepared'].items():
        if sha(work / 'prepared' / f'{split}.json') != expected:
            raise ValueError(f'Prepared {split} changed')
    return protocol


def targets(encoder):
    if encoder not in ENCODERS:
        raise ValueError(encoder)
    result = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            if encoder == 'inlegalbert':
                path = (ROOT / f'artifacts/models/final_cpu/A_N_s{seed}_ipr/final'
                        if condition == 'N' else
                        ROOT / f'artifacts/models/equal_update/D_{condition}_s{seed}_ipr/final')
            else:
                path = ROOT / f'artifacts/models/encoder_extension/{encoder}/E_{condition}_s{seed}_ipr/final'
            result.append({'condition': condition, 'seed': seed, 'checkpoint': str(path)})
    return result


def checkpoint_hash(path):
    path = Path(path)
    if not (path / 'config.json').is_file():
        raise ValueError(f'Checkpoint unfinished/missing: {path}')
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.suffix in ('.json', '.safetensors', '.bin', '.txt', '.model')))
    if not any(p.suffix in ('.safetensors', '.bin') and 'training_args' not in p.name for p in files):
        raise ValueError(f'Checkpoint weights missing: {path}')
    return hashlib.sha256(''.join(f'{p.name}:{sha(p)}\n' for p in files).encode()).hexdigest()


def score(prediction, answers):
    from scripts.qa_evaluation import compute_em_f1
    if not answers:
        raise ValueError('Unanswerable example passed to forced-span scoring')
    scores = [compute_em_f1(prediction, a['text']) for a in answers]
    return {'em': max(s[0] for s in scores), 'f1': max(s[1] for s in scores)}


def summarize(rows):
    result = {}
    for group, predicate in {
        'primary': lambda r: r['primary'],
        'clean_answerable_including_squad': lambda r: r['clean_answerable'],
        'all_answerable_exploratory': lambda r: True,
        **{f'source/{s}': lambda r, s=s: r['source'] == s and r['clean_answerable']
           for s in SOURCE_NAMES},
    }.items():
        selected = [r for r in rows if predicate(r)]
        result[group] = {'n': len(selected), **{
            metric: 100 * statistics.mean(r[metric] for r in selected) if selected else None
            for metric in ('em', 'f1')}}
    return result


def evaluate(work, encoder, split, threads):
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from scripts.qa_evaluation import QADecodingConfig, predict_answers
    protocol = verify_protocol(work)
    torch.set_num_threads(threads)
    # Require the complete encoder cohort BEFORE producing any external scores.
    models = [{**t, 'checkpoint_sha256': checkpoint_hash(t['checkpoint'])} for t in targets(encoder)]
    lock = {'protocol_sha256': sha(work / 'PROTOCOL.json'), 'models': models,
            'device': 'cpu', 'threads': threads}
    freeze(work / 'outputs' / encoder / 'COHORT.json', lock)
    rows = [r for r in read(work / 'prepared' / f'{split}.json') if r['answerable']]
    if not rows:
        raise ValueError('No answerable examples')
    config = QADecodingConfig(**{k: v for k, v in protocol['inference'].items()
                               if k != 'query_batch_size'})
    for target in models:
        output = work / 'outputs' / encoder / split / f'{target["condition"]}_s{target["seed"]}.json'
        identity = {'cohort_sha256': sha(work / 'outputs' / encoder / 'COHORT.json'),
                    'split': split, **target}
        if output.exists():
            existing = read(output)
            validate_result(existing, rows, identity)
            print(f'Verified existing {output.name}', flush=True)
            continue
        tokenizer = AutoTokenizer.from_pretrained(target['checkpoint'], local_files_only=True, use_fast=True)
        model, info = AutoModelForQuestionAnswering.from_pretrained(
            target['checkpoint'], local_files_only=True, output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError(f'Checkpoint failed strict loading: {info}')
        model.to('cpu').eval()
        predictions = []
        batch_size = protocol['inference']['query_batch_size']
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset+batch_size]
            decoded = predict_answers(tokenizer, model, torch.device('cpu'),
                                      [r['question'] for r in batch], [r['text'] for r in batch], config)
            if len(decoded) != len(batch):
                raise ValueError('Prediction count mismatch')
            for row, (text, span_score) in zip(batch, decoded):
                if not math.isfinite(span_score) or not text or text not in row['text']:
                    raise ValueError(f'Invalid prediction: {row["id"]}')
                predictions.append({**row, 'prediction': text, 'span_score': span_score,
                                    **score(text, row['answers'])})
            print(f'{encoder} {target["condition"]} seed {target["seed"]} {split}: '
                  f'{min(offset+batch_size,len(rows))}/{len(rows)}', flush=True)
        if checkpoint_hash(target['checkpoint']) != target['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during inference')
        payload = {'identity': identity, 'predictions': predictions, 'metrics': summarize(predictions)}
        validate_result(payload, rows, identity)
        atomic(output, payload)
        del model, tokenizer
        gc.collect()


def validate_result(payload, expected_rows, identity):
    if payload['identity'] != identity:
        raise ValueError('Result identity mismatch')
    predictions = payload['predictions']
    if [r['id'] for r in predictions] != [r['id'] for r in expected_rows]:
        raise ValueError('Result IDs missing, duplicated, or reordered')
    for row, expected in zip(predictions, expected_rows):
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Result input changed: {row["id"]}')
        if not row['prediction'] or row['prediction'] not in row['text'] or not math.isfinite(row['span_score']):
            raise ValueError('Invalid saved prediction')
        for metric, value in score(row['prediction'], row['answers']).items():
            if not math.isfinite(row[metric]) or abs(row[metric] - value) > 1e-10:
                raise ValueError('Saved score differs from rescoring')
    if payload['metrics'] != summarize(predictions):
        raise ValueError('Saved summary differs from rescoring')


def analyse(work, encoder, split):
    verify_protocol(work)
    cohort_path = work / 'outputs' / encoder / 'COHORT.json'
    cohort = read(cohort_path)
    if cohort['protocol_sha256'] != sha(work / 'PROTOCOL.json'):
        raise ValueError('Cohort protocol mismatch')
    if len(cohort['models']) != 20:
        raise ValueError('Incomplete cohort')
    if [{k: t[k] for k in ('condition', 'seed', 'checkpoint')} for t in cohort['models']] != targets(encoder):
        raise ValueError('Cohort target mapping changed')
    rows = [r for r in read(work / 'prepared' / f'{split}.json') if r['answerable']]
    results = []
    for target in cohort['models']:
        identity = {'cohort_sha256': sha(cohort_path), 'split': split, **target}
        path = work / 'outputs' / encoder / split / f'{target["condition"]}_s{target["seed"]}.json'
        if not path.exists():
            raise ValueError(f'Incomplete experiment; missing {path.name}')
        payload = read(path)
        validate_result(payload, rows, identity)
        results.append(payload)
    report = statistical_report(results)
    report['cohort_sha256'] = sha(cohort_path)
    report['evaluation_sha256'] = {
        f'{t["condition"]}_s{t["seed"]}.json': sha(
            work / 'outputs' / encoder / split / f'{t["condition"]}_s{t["seed"]}.json')
        for t in cohort['models']}
    atomic(work / 'outputs' / encoder / f'{split}_statistics.json', report)
    print(f'Validated 20 cells and wrote {split}_statistics.json for {encoder}')


def statistical_report(results):
    stats = importlib.import_module('scripts.09_dapt_final.analyse_final_experiments')
    cells = [(r['identity']['condition'], r['identity']['seed']) for r in results]
    if len(cells) != 20 or set(cells) != {(c, s) for c in CONDITIONS for s in SEEDS}:
        raise ValueError('Incomplete or duplicate statistical cells')
    report = {'interpretation': 'Seed uncertainty conditional on fixed examples; no equivalence test. '
              'Holm within encoder, subset, split and metric; secondary subsets exploratory.',
              'subsets': {}}
    for group in results[0]['metrics']:
        table = {c: {} for c in CONDITIONS}
        for result in results:
            target = result['identity']
            table[target['condition']][target['seed']] = result['metrics'][group]
        sizes = {table[c][s]['n'] for c in CONDITIONS for s in SEEDS}
        if len(sizes) != 1:
            raise ValueError('Unequal evaluation subsets across cells')
        n = sizes.pop()
        report['subsets'][group] = {'n': n}
        if n == 0:
            continue
        for metric in ('em', 'f1'):
            contrasts = {c: stats.paired([table[c][s][metric] for s in SEEDS],
                                        [table['N'][s][metric] for s in SEEDS], f'{c}-N', SEEDS)
                         for c in 'ASM'}
            stats.holm(contrasts)
            report['subsets'][group][metric] = {
                'descriptives': {c: stats.mean_sd_ci([table[c][s][metric] for s in SEEDS])
                                 for c in CONDITIONS},
                'blocked_anova': stats.blocked_anova(table, metric), 'contrasts': contrasts}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=HERE / 'run_v1')
    sub = parser.add_subparsers(dest='command', required=True)
    dl = sub.add_parser('download')
    dl.add_argument('--revision', default='main', help='Resolved to a pinned commit before download')
    prep = sub.add_parser('prepare')
    prep.add_argument('--squad-train-arrow', type=Path, required=True)
    prep.add_argument('--squad-val-arrow', type=Path, required=True)
    for command in ('evaluate', 'analyse'):
        child = sub.add_parser(command)
        child.add_argument('--encoder', choices=ENCODERS, required=True)
        child.add_argument('--split', choices=['val', 'test'], default='test')
        if command == 'evaluate':
            child.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    work = args.work.resolve()
    if args.command == 'download':
        download(work, args.revision)
    elif args.command == 'prepare':
        prepare(work, args.squad_train_arrow, args.squad_val_arrow)
    elif args.command == 'evaluate':
        if args.threads < 1:
            parser.error('--threads must be positive')
        evaluate(work, args.encoder, args.split, args.threads)
    else:
        analyse(work, args.encoder, args.split)


if __name__ == '__main__':
    main()
