"""Independent rescoring/statistics audit; no inference or checkpoint changes."""
import collections
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import statistics
import string
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
WORK = HERE / 'run_v1'
ENCODERS = ['inlegalbert', 'bert', 'legalbert']
CONDITIONS = ['N', 'A', 'S', 'M']
SEEDS = [7, 42, 99, 123, 2026]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def overlap_key(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def independent_scores(prediction, answers):
    def normalized(value):
        return ' '.join(value.lower().translate(str.maketrans('', '', string.punctuation)).split())
    prediction = normalized(prediction)
    p = prediction.split()
    values = []
    for answer in answers:
        gold = normalized(answer['text'])
        g = gold.split()
        common = sum((collections.Counter(p) & collections.Counter(g)).values())
        values.append((float(prediction == gold), 2*common/(len(p)+len(g)) if p and g else 0.0))
    return max(v[0] for v in values), max(v[1] for v in values)


def near(a, b, label):
    if not math.isfinite(float(a)) or abs(a-b) > 0.000004:
        raise ValueError(f'{label}: {a} != {b}')


def main():
    import numpy as np
    from scipy import stats
    from datasets import Dataset
    protocol = load(WORK / 'PROTOCOL.json')
    for name, metadata in protocol['dataset']['files'].items():
        if digest(WORK / 'raw' / name) != metadata['sha256']:
            raise ValueError('Raw dataset checksum mismatch')
    q_seen, c_seen = set(), set()
    for metadata in protocol['references'].values():
        path = Path(metadata['path'])
        if digest(path) != metadata['sha256']:
            raise ValueError('Reference checksum mismatch')
        records = Dataset.from_file(str(path)) if path.suffix == '.arrow' else lines(path)
        for record in records:
            question = overlap_key(record.get('question', ''))
            context = overlap_key(record.get('context', record.get('text', '')))
            if question:
                q_seen.add(question)
            if context:
                c_seen.add(context)
    raw = {s: lines(WORK / 'raw' / f'{s}.jsonl') for s in ('val', 'test')}
    shared = {field: {overlap_key(r[field]) for r in raw['val']} &
                    {overlap_key(r[field]) for r in raw['test']} for field in ('question', 'text')}
    all_records = {}
    for split in raw:
        all_records[split] = {}
        for row in raw[split]:
            q, c = overlap_key(row['question']), overlap_key(row['text'])
            source = row['id'].split('/')[1]
            clean = bool(row['answers']) and q not in q_seen and c not in c_seen and q not in shared['question'] and c not in shared['text']
            all_records[split][row['id']] = (row, source, clean, clean and source != 'stanford--squad-v2')
    cells_checked, predictions_checked = 0, 0
    for encoder in ENCODERS:
        cohort_path = WORK / 'outputs' / encoder / 'COHORT.json'
        cohort = load(cohort_path)
        if cohort['protocol_sha256'] != digest(WORK / 'PROTOCOL.json'):
            raise ValueError('Cohort protocol mismatch')
        for split in ('val', 'test'):
            summary = load(WORK / 'outputs' / encoder / f'{split}_statistics.json')
            if summary['cohort_sha256'] != digest(cohort_path):
                raise ValueError('Statistics cohort mismatch')
            measurements = collections.defaultdict(dict)
            expected_ids = [r['id'] for r in raw[split] if r['answers']]
            for condition in CONDITIONS:
                for seed in SEEDS:
                    name = f'{condition}_s{seed}.json'
                    path = WORK / 'outputs' / encoder / split / name
                    if summary['evaluation_sha256'][name] != digest(path):
                        raise ValueError('Output checksum mismatch')
                    result = load(path)
                    if result['identity']['cohort_sha256'] != digest(cohort_path):
                        raise ValueError('Result cohort mismatch')
                    if result['identity']['condition'] != condition or result['identity']['seed'] != seed or result['identity']['split'] != split:
                        raise ValueError('Result cell mismatch')
                    if [r['id'] for r in result['predictions']] != expected_ids:
                        raise ValueError('Wrong prediction IDs/order/count')
                    groups = collections.defaultdict(list)
                    for prediction in result['predictions']:
                        original, source, clean, primary = all_records[split][prediction['id']]
                        for field in ('question', 'text', 'answers'):
                            if prediction[field] != original[field]:
                                raise ValueError('Prediction inputs differ from raw dataset')
                        if prediction['source'] != source or prediction['primary'] != primary or prediction['clean_answerable'] != clean:
                            raise ValueError('Reconstructed inclusion rules disagree')
                        text = prediction['prediction']
                        if not text or text not in original['text']:
                            raise ValueError('Answer is not a literal source span')
                        em, f1 = independent_scores(text, original['answers'])
                        near(em, prediction['em'], 'per-item EM')
                        near(f1, prediction['f1'], 'per-item F1')
                        groups['all_answerable_exploratory'].append((em, f1))
                        if clean:
                            groups['clean_answerable_including_squad'].append((em, f1))
                            groups[f'source/{source}'].append((em, f1))
                        if primary:
                            groups['primary'].append((em, f1))
                    for group, saved in result['metrics'].items():
                        pairs = groups[group]
                        if saved['n'] != len(pairs):
                            raise ValueError('Subset count mismatch')
                        if not pairs:
                            if saved['em'] is not None or saved['f1'] is not None:
                                raise ValueError('Empty subset has scores')
                            continue
                        values = [100*statistics.mean(p[i] for p in pairs) for i in (0, 1)]
                        near(values[0], saved['em'], 'cell EM')
                        near(values[1], saved['f1'], 'cell F1')
                        measurements[group][condition, seed] = values
                    cells_checked += 1
                    predictions_checked += len(result['predictions'])
            for group, data in summary['subsets'].items():
                if not data['n']:
                    continue
                for index, metric in enumerate(('em', 'f1')):
                    matrix = np.array([[measurements[group][c,s][index] for s in SEEDS] for c in CONDITIONS])
                    for ci, condition in enumerate(CONDITIONS):
                        near(float(matrix[ci].mean()), data[metric]['descriptives'][condition]['mean'], 'mean')
                        near(float(matrix[ci].std(ddof=1)), data[metric]['descriptives'][condition]['sd'], 'SD')
                    # Fit the seed-blocked additive model by least squares,
                    # independently of the production sum-of-squares routine.
                    design = np.array([[1, *(float(ci==j) for j in range(1,4)),
                                        *(float(si==j) for j in range(1,5))]
                                       for ci in range(4) for si in range(5)])
                    y = matrix.flatten()
                    residual = y - design @ np.linalg.lstsq(design,y,rcond=None)[0]
                    sse = float(residual @ residual)
                    ss_condition = 5*float(np.sum((matrix.mean(axis=1)-matrix.mean())**2))
                    f_value = (ss_condition/3)/(sse/12) if sse else 0
                    near(f_value, data[metric]['blocked_anova']['F_condition'], 'blocked F')
                    near(float(stats.f.sf(f_value,3,12)), data[metric]['blocked_anova']['p_condition'], 'blocked p')
                    pvalues = []
                    for ci, condition in enumerate(CONDITIONS[1:], start=1):
                        delta = matrix[ci]-matrix[0]
                        row = data[metric]['contrasts'][condition]
                        near(float(delta.mean()), row['mean_difference'], 'paired difference')
                        half = float(stats.t.ppf(.975,4))*float(delta.std(ddof=1))/math.sqrt(5)
                        near(float(delta.mean())-half,row['ci95'][0],'CI low')
                        near(float(delta.mean())+half,row['ci95'][1],'CI high')
                        p = float(stats.ttest_rel(matrix[ci],matrix[0]).pvalue)
                        near(p,row['p_uncorrected'],'paired p')
                        pvalues.append((condition,p))
                    previous = 0
                    for rank,(condition,p) in enumerate(sorted(pvalues,key=lambda t:t[1])):
                        previous = min(1,max(previous,(3-rank)*p))
                        near(previous,data[metric]['contrasts'][condition]['p_holm'],'Holm p')
    report = {'status': 'PASS', 'split_evaluations': cells_checked, 'predictions_independently_rescored': predictions_checked,
              'primary_counts': {s: sum(v[3] for v in all_records[s].values()) for s in raw},
              'checks': ['raw reference hashes', 'independent overlap/subset reconstruction',
                         'literal extracted spans', 'independent multireference EM/F1', 'cell means/SD',
                         'least-squares seed-blocked ANOVA', 'paired t intervals/tests', 'Holm corrections',
                         'result and cohort checksums'],
              'python': sys.version, 'packages': {p: importlib.metadata.version(p)
                         for p in ('torch','transformers','datasets','huggingface_hub','numpy','scipy')},
              'scope': 'Artifact and calculation consistency; not independent retraining or legal validation'}
    # Generated audit artifact, kept separate from the frozen evaluation outputs.
    (HERE / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'PASS: {cells_checked} evaluations; {predictions_checked} predictions; subsets and statistics independently reproduced')


if __name__ == '__main__':
    main()
