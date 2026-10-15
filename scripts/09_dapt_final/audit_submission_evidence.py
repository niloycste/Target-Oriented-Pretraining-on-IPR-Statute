"""Read-only audit of frozen DAPT evidence; writes only a separate review report.

No training, inference, corpus changes, or edits to locked experiment artifacts.
Run from the repository root: python scripts/09_dapt_final/audit_submission_evidence.py
"""
import ast
import collections
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import statistics as st
import string

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'experiments/dapt/final_cpu'
PAPER = ROOT / 'papers/target_pretraining'
OUT = PAPER / 'submission_audit_20260920.json'
ERRORS = []


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def check(ok, message):
    if not ok:
        ERRORS.append(message)


def norm(s):
    return ' '.join(s.lower().translate(str.maketrans('', '', string.punctuation)).split())


def score(pred, gold):
    p, g = norm(pred), norm(gold)
    a, b = p.split(), g.split()
    common = sum((collections.Counter(a) & collections.Counter(b)).values())
    return float(p == g), 2 * common / (len(a) + len(b)) if a and b else 0.0


def mean_sd(values):
    return dict(mean=st.mean(values), sd=st.stdev(values) if len(values) > 1 else 0)


def paired(a, b):
    d = np.array(a) - np.array(b)
    ci = stats.t.interval(.95, len(d)-1, loc=d.mean(), scale=stats.sem(d))
    p = stats.ttest_rel(a, b).pvalue
    permutations = [abs(np.mean(d * signs)) for signs in itertools.product((-1, 1), repeat=len(d))]
    return dict(differences=d.tolist(), mean=float(d.mean()), ci95=list(map(float, ci)),
                p=float(p), exact_sign_flip_p=sum(v >= abs(d.mean())-1e-12 for v in permutations)/len(permutations))


def main():
    expected = {f'A_{c}_s{s}_100' for c in 'NASM' for s in (7, 42, 99, 123, 2026)}
    expected |= {f'B_S_STYLE_s{s}_100' for s in (7, 42, 99)}
    expected |= {f'C_{c}_s{s}_{f}' for c in 'NASM' for s in (7, 42, 99) for f in (25, 50)}
    report = {'scope': 'Fresh rescoring of saved answers, statistical recomputation, provenance hashes, corpus and manuscript audit. No new inference.', 'errors': ERRORS}
    all_runs = {}
    for split, file, count in [('test', 'results.json', 344), ('val', 'validation_results.json', 408)]:
        runs = read(FINAL / file)['runs']
        check(set(runs) == expected, f'{split}: inventory mismatch')
        gold = {str(r['id']): r for r in lines(ROOT / f'data/splits/{split}.jsonl')}
        groups = collections.defaultdict(list)
        for key, run in runs.items():
            saved = read(ROOT / run['predictions_file'])
            ps = saved['predictions']
            check(len(ps) == count and {p['id'] for p in ps} == set(gold), f'{split}:{key}: ID mismatch')
            scores = []
            for p in ps:
                g = gold[p['id']]
                check(p['gold'] == g['answer'] and p['question'] == g['question'], f'{split}:{key}:{p["id"]}: gold mismatch')
                check(not p['prediction'] or p['prediction'] in g['context'], f'{split}:{key}:{p["id"]}: nonliteral span')
                em, f1 = score(p['prediction'], g['answer'])
                check(abs(em-p['em']) < 1e-8 and abs(f1-p['f1']) < 6e-9, f'{split}:{key}:{p["id"]}: score mismatch')
                scores.append((em, f1))
            for i, metric in enumerate(('em', 'f1')):
                fresh = 100 * st.mean(v[i] for v in scores)
                check(abs(fresh-run[metric]) < 2e-6 and abs(fresh-saved[metric]) < 2e-6, f'{split}:{key}: aggregate {metric} mismatch')
            check(run['checkpoint_sha256'] == saved['checkpoint_sha256'], f'{split}:{key}: fingerprint mismatch')
            groups[f'{run["experiment"]}:{run["condition"]}:{run["fraction"]}'].append(run)
        report[split] = {'runs': len(runs), 'prediction_rows': count*len(runs), 'summaries': {
            key: {'n': len(rs), 'em': mean_sd([r['em'] for r in rs]), 'f1': mean_sd([r['f1'] for r in rs]),
                  'seed_f1': {str(r['seed']): r['f1'] for r in rs}} for key, rs in sorted(groups.items())}}
        all_runs[split] = runs
        print(f'Rescored {split}: {len(runs)*count} saved answers', flush=True)

    lock = read(FINAL / 'FINAL_ANALYSIS_LOCK.json')
    amend = read(FINAL / 'PRETEST_AMENDMENT_20260913.json')
    mismatches, accepted_overrides = [], []
    for p, original in lock['sha256'].items():
        actual = digest(ROOT/p) if (ROOT/p).exists() else 'MISSING'
        if actual != original:
            if amend.get('overrides', {}).get(p, {}).get('reviewed_sha256') == actual:
                accepted_overrides.append(p)
            else:
                mismatches.append(p)
    for p, h in amend['additional_sha256'].items():
        check((ROOT/p).exists() and digest(ROOT/p) == h, f'Amendment dependency mismatch: {p}')
    check(not mismatches, f'Unapproved lock mismatch: {mismatches}')
    report['lock'] = dict(original_files=len(lock['sha256']), reviewed_overrides=accepted_overrides, unapproved_mismatches=mismatches)

    hashes = {}
    for key, run in all_runs['test'].items():
        folder = ROOT/run['checkpoint']
        files = set()
        for pattern in ('config.json', '*.safetensors', 'pytorch_model*.bin', '*.index.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json'):
            files.update(folder.glob(pattern))
        h = hashlib.sha256()
        for f in sorted(files, key=lambda f: f.name):
            h.update(f.name.encode())
            with f.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    h.update(chunk)
        hashes[key] = h.hexdigest() == run['checkpoint_sha256'] == all_runs['val'][key]['checkpoint_sha256']
        check(hashes[key], f'Checkpoint hash mismatch: {key}')
    report['checkpoint_hashes'] = hashes
    print('Checked all 47 reader checkpoint fingerprints', flush=True)

    # Independently recompute paired statistics and the complete-block ANOVA.
    runs = all_runs['test']
    saved_stats = read(FINAL/'FINAL_STATISTICS.json')
    seeds = (7, 42, 99, 123, 2026)
    independent = {}
    for metric in ('f1', 'em'):
        matrix = np.array([[runs[f'A_{c}_s{s}_100'][metric] for c in 'NASM'] for s in seeds])
        residual = matrix - matrix.mean(axis=0) - matrix.mean(axis=1)[:,None] + matrix.mean()
        ssc = len(seeds)*np.sum((matrix.mean(axis=0)-matrix.mean())**2)
        sse = np.sum(residual**2)
        f = (ssc/3)/(sse/12)
        p = stats.f.sf(f, 3, 12)
        check(abs(p-saved_stats[f'experiment_A_{metric}']['blocked_anova']['p_condition'])<1e-7, f'{metric}: ANOVA mismatch')
        independent[metric] = dict(F=float(f), p=float(p), partial_eta_sq=float(ssc/(ssc+sse)))
    report['independent_anova'] = independent
    report['S_minus_N_50'] = paired([runs[f'C_S_s{s}_50']['f1'] for s in (7,42,99)], [runs[f'C_N_s{s}_50']['f1'] for s in (7,42,99)])

    # Re-run pure analysis functions in memory only; do not invoke their main/writers.
    spec = importlib.util.spec_from_file_location('frozen_analysis', ROOT/'scripts/09_dapt_final/analyse_final_experiments.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = module.table_from_runs(runs, 'A', 100, list('NASM'))
    b = module.table_from_runs(runs, 'B', 100, ['S_STYLE'])
    c = {f: module.table_from_runs(runs, 'C', f, list('NASM')) for f in (25,50)}
    for metric in ('em','f1'):
        for exp, fresh in [('A',module.experiment_a(a,metric)),('B',module.experiment_b(a,b,metric)),('C',module.experiment_c(a,c,metric))]:
            check(fresh == saved_stats[f'experiment_{exp}_{metric}'], f'{exp}:{metric}: saved statistics differ from recomputation')
    check(module.difficulty_analysis(runs) == read(FINAL/'QA_DIFFICULTY_DIAGNOSTICS.json'), 'Difficulty aggregates differ from saved predictions')
    report['interaction_f1'] = saved_stats['experiment_C_f1']['interaction_pattern']
    report['difficulty_full_data'] = {k:v for k,v in read(FINAL/'QA_DIFFICULTY_DIAGNOSTICS.json')['aggregate'].items() if k.startswith('A:')}

    norm_context = lambda s: ' '.join(s.lower().split())
    split_rows = {s: lines(ROOT/f'data/splits/{s}.jsonl') for s in ('train','val','test')}
    contexts = {s:{norm_context(r['context']) for r in rows} for s,rows in split_rows.items()}
    report['splits'] = {s:dict(qa=len(rows),unique_contexts=len(contexts[s])) for s,rows in split_rows.items()}
    report['split_context_intersections'] = {f'{a}/{b}':len(contexts[a]&contexts[b]) for a,b in itertools.combinations(contexts,2)}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT/'artifacts/models/final_cpu/base_encoder'), local_files_only=True)
    report['corpora'] = {}
    for c in ('A','S','M','S_STYLE'):
        rows = lines(FINAL/f'corpus_{c}.jsonl')
        texts = [r['context'] for r in rows]
        tokens = sum(len(ids) for ids in tok(texts,truncation=True,max_length=256)['input_ids'])
        held = {s:len({norm_context(t) for t in texts}&contexts[s]) for s in ('val','test')}
        check(not any(held.values()), f'{c}: held-out exact context exposure')
        steps = 3*math.ceil(math.ceil(len(rows)/8)/4)
        report['corpora'][c] = dict(records=len(rows), unique=len(set(map(norm_context,texts))), tokens=tokens, heldout_exact_matches=held,
                                  steps_implied_by_saved_recipe=steps, warmup_steps=100, nominal_tokens_across_three_epochs=tokens*3)
    report['low_resource'] = {}
    for frac in (25,50):
        manifest = read(FINAL/f'LOW_RESOURCE_{frac}_MANIFEST.json')
        report['low_resource'][str(frac)] = {k:v for k,v in manifest.items() if k != 'question_ids'}
        actual = lines(FINAL/f'train_squad_{frac}.jsonl')
        check({str(r['id']) for r in actual} == set(manifest['question_ids']), f'{frac}: subset IDs mismatch')

    mlm = read(FINAL/'MLM_DIAGNOSTIC_RESULTS.json')
    report['mlm'] = dict(checkpoints=len(mlm['results']),contexts=mlm['contexts'],masked_tokens=sorted({r['masked_tokens'] for r in mlm['results'].values()}))
    check(len(mlm['results']) == 19, 'MLM inventory incomplete')
    for k,r in mlm['results'].items():
        check(abs(math.exp(r['loss'])-r['perplexity'])<1e-6, f'{k}: PPL mismatch')

    tex = (PAPER/'target-pretraining.tex').read_text(encoding='utf8')
    bib = (PAPER/'target-pretraining-references.bib').read_text(encoding='utf8')
    # Compare the actual typeset table entries, not just the machine summaries.
    condition_codes = {
        'No DAPT': 'N',
        'Authentic': 'A',
        'Synthetic': 'S',
        'Mixed': 'M',
        'Style-only': 'S_STYLE',
        'Style Only': 'S_STYLE',
    }
    report['manuscript_table_rows_checked'] = {}
    for split, label in [('val', 'tab:final_validation'), ('test', 'tab:final_test')]:
        table = tex.split('\\label{' + label + '}', 1)[1].split('\\end{table}', 1)[0]
        matched = re.findall(r'^([ABC])\s*&\s*([^&]+?)\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*([\d.]+)\s*&\s*\$([\d.]+)\\pm([\d.]+)\$', table, re.M)
        check(len(matched) == 13, f'{label}: expected 13 rows')
        for exp, name, fraction, n, em, f1, sd in matched:
            group = report[split]['summaries'][f'{exp}:{condition_codes[name]}:{fraction}']
            check(int(n) == group['n'], f'{label}:{name}:{fraction}: seed count')
            for displayed, actual in [(em, group['em']['mean']), (f1, group['f1']['mean']), (sd, group['f1']['sd'])]:
                check(displayed == f'{actual:.2f}', f'{label}:{name}:{fraction}: {displayed} != {actual:.2f}')
        report['manuscript_table_rows_checked'][label] = len(matched)
    archived = {r['code']: r for r in read(ROOT/'results/evaluation/qa_eval_report.json')['results_summary']}
    table = tex.split('\\label{tab:reader_results}', 1)[1].split('\\end{table}', 1)[0]
    matched = re.findall(r"^([A-Z0-9]+'?)\s*&[^\n]*?&\s*([\d.]+)\s*&\s*([\d.]+)\s*\\\\", table, re.M)
    check(len(matched) == 6, 'Archived reader table: expected 6 rows')
    for code, em, f1 in matched:
        for displayed, metric in [(em, 'em'), (f1, 'f1')]:
            check(displayed == f'{archived[code][metric]:.2f}', f'Archived reader table: {code} {metric}')
    report['manuscript_table_rows_checked']['tab:reader_results'] = len(matched)
    def table_rows(label):
        body = tex.split('\\label{' + label + '}', 1)[1].split('\\end{table}', 1)[0]
        return {parts[0]: parts[1:] for line in body.splitlines() if '&' in line and line.rstrip().endswith('\\\\')
                for parts in [[s.strip().rstrip('\\').strip() for s in line.split('&')]]}

    # New manuscript diagnostic tables are checked directly against frozen JSON.
    rows = table_rows('tab:difficulty')
    groups = {'1--10 words':'short_1_10', '11--30 words':'medium_11_30', '31+ words':'long_31_plus',
              'Enumerated':'enumerated_yes', 'Not enumerated':'enumerated_no', 'Semicolon':'semicolon_yes',
              'No semicolon':'semicolon_no', 'Proviso/exception':'proviso_yes', 'No tagged phrase':'proviso_no'}
    for name, group in groups.items():
        for i, condition in enumerate('NASM'):
            value = report['difficulty_full_data'][f'A:{condition}:100'][group]
            check(int(rows[name][0]) == value['items_per_run'], f'Difficulty count: {name}')
            expected_cell = f"${value['f1']['mean']:.2f}\\pm{value['f1']['sd']:.2f}$"
            check(rows[name][i+1] == expected_cell, f'Difficulty table: {name} {condition}')
    report['manuscript_table_rows_checked']['tab:difficulty'] = len(groups)

    budget_rows = table_rows('tab:budgets')
    for condition, values in report['corpora'].items():
        name = 'Style' if condition == 'S_STYLE' else condition
        actual = [int(s.replace(',','')) for s in budget_rows[name]]
        check(actual == [values['records'], values['tokens'], values['steps_implied_by_saved_recipe'], 100], f'Budget table: {name}')
    report['manuscript_table_rows_checked']['tab:budgets'] = 4

    body = tex.split('\\label{tab:interactions}', 1)[1].split('\\end{table}', 1)[0]
    for fraction, condition, delta, lo, hi, p in re.findall(r'(25|50)\\% & ([ASM]) & \$([+-][\d.]+)\$ & \$\[([+-]?[\d.]+),([+-]?[\d.]+)\]\$ & ([\d.]+)', body):
        expected_value = report['interaction_f1'][fraction][condition]
        check(delta == f"{expected_value['mean_difference']:+.2f}" and lo == f"{expected_value['ci95'][0]:.2f}" and
              hi == f"{expected_value['ci95'][1]:.2f}" and p == f"{expected_value['p_uncorrected']:.3f}", f'Interaction table: {fraction} {condition}')
    report['manuscript_table_rows_checked']['tab:interactions'] = len(re.findall(r'(?:25|50)\\% & [ASM]', body))
    check(report['manuscript_table_rows_checked']['tab:interactions'] == 6, 'Expected six interaction rows')

    mlm_rows = table_rows('tab:final_mlm')
    for name, prefix in [('Authentic','A_A_'),('Synthetic','A_S_'),('Mixed','A_M_'),('Style Only','B_S_STYLE_')]:
        values = [r['perplexity'] for key,r in mlm['results'].items() if key.startswith(prefix)]
        check(int(mlm_rows[name][0]) == len(values), f'MLM table count: {name}')
        numbers = re.findall(r'\d+\.\d+', mlm_rows[name][1])
        check(numbers == [f'{st.mean(values):.3f}', f'{st.stdev(values):.3f}'], f'MLM table: {name}')
    check(mlm_rows['No DAPT'][1] == f"{mlm['results']['N_base']['perplexity']:.3f}", 'Base MLM table')
    report['manuscript_table_rows_checked']['tab:final_mlm'] = 5
    corpus_rows = table_rows('tab:corpus_characterisation')
    properties = read(FINAL/'CORPUS_CHARACTERISATION.json')['corpora']
    for i, condition in enumerate(['A', 'S', 'M', 'S_STYLE']):
        value = properties[condition]
        expected = {
            'Mean words': f"{value['mean_words']:.2f}",
            'Median words': f"{value['median_words']:.0f}",
            '10th--90th percentiles': '--'.join(str(n) for n in value['p10_p90_words']),
            'Type--token ratio': f"{value['type_token_ratio']:.4f}",
            'Distinct bigram ratio': f"{value['distinct_2']:.4f}",
            'Exact duplicate rate': f"{value['exact_duplicate_rate']:.0f}",
        }
        for marker in ['shall', 'provided that', 'subject to']:
            expected['\\emph{' + marker + '}'] = f"{value['legal_marker_rate_per_100w'][marker]:.3f}"
        for name, cell in expected.items():
            check(corpus_rows[name][i] == cell, f'Corpus characterisation table: {name} {condition}')
    report['manuscript_table_rows_checked']['tab:corpus_characterisation'] = 9
    labels = re.findall(r'\\label\{([^}]+)\}',tex)
    refs = re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}',tex)
    citations = {s.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for s in group.split(',')}
    keys = re.findall(r'@\w+\{([^,]+),',bib)
    abstract = tex.split('\\abstract{',1)[1].split('\\keywords',1)[0]
    abstract = re.sub(r'\\[a-zA-Z]+', '', abstract)
    abstract = re.sub(r'[{}\\]', '', abstract)
    report['manuscript'] = dict(abstract_whitespace_words=len(abstract.split()),missing_refs=sorted(set(refs)-set(labels)), duplicate_labels=[k for k,v in collections.Counter(labels).items() if v>1],
                               missing_citations=sorted(citations-set(keys)),unused_bibkeys=sorted(set(keys)-citations),bib_entries=len(keys),
                               uncited_float_labels=[s for s in labels if s.startswith(('tab:','fig:','alg:')) and s not in refs])
    check(not report['manuscript']['missing_refs'] and not report['manuscript']['missing_citations'], 'Undefined manuscript references')
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:report[k] for k in ('errors','independent_anova','S_minus_N_50','corpora','manuscript')},indent=2))
    print(f'Wrote {OUT.relative_to(ROOT)}',flush=True)
    return bool(ERRORS)


if __name__ == '__main__':
    raise SystemExit(main())
