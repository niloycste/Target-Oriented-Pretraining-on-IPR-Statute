"""Wait for verified external outputs, then apply a numerical manuscript patch."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import subprocess
import time
import run as external
from run_all import resolved_targets

BEGIN = '% LEGALQAEVAL_EXTERNAL_BEGIN'
END = '% LEGALQAEVAL_EXTERNAL_END'
MANUSCRIPT = external.ROOT / 'papers/target_pretraining/target-pretraining.tex'
LABELS = {'inlegalbert': 'InLegalBERT', 'bert': 'BERT', 'legalbert': 'LEGAL-BERT'}
CONDITIONS = {'N': 'No DAPT', 'A': 'Authentic', 'S': 'Synthetic', 'M': 'Mixed'}


def region(text):
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError('External section markers are missing or duplicated')
    return text[text.index(BEGIN):text.index(END)+len(END)]


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def generated_section(reports):
    lines = [BEGIN, r'\section{External Transfer Evaluation: LegalQAEval}',
             r'\label{sec:external_legalqaeval}', '',
             'We evaluated all 60 fixed readers on LegalQAEval after SQuAD transfer',
             'and IndicIPR-QA fine-tuning, without external training or checkpoint',
             'selection. N/A/S/M are compared within each encoder across seeds 7, 42,',
             '99, 123, and 2026. Each reader was evaluated separately on the external',
             'validation and test splits; no best seed or condition was selected.', '',
             'LegalQAEval contains legal-related examples drawn from SQuAD~2.0,',
             'MS MARCO, HotpotQA, and Natural Questions. The downloaded revision is',
             r'\texttt{f6c7aafcc176b25c32e1e052d02ad9bc06c96bcb}.',
             'It provides 1,204 validation and 1,206 test examples, including',
             'unanswerable questions. The dataset and source documentation are at',
             r'\url{https://huggingface.co/datasets/isaacus/LegalQAEval}.', '',
             'The primary subsets retain 333 validation and 329 test examples.',
             'They contain answerable, non-SQuAD questions without normalized',
             'question or context matches against the IPR splits, adaptation',
             'corpora, full cached SQuAD training and validation pools, or the',
             'opposite external split. The primary test subset contains 234',
             'MS MARCO, 54 HotpotQA, and 41 Natural Questions examples. All',
             'SQuAD-derived examples matched the conservative reference pool.',
             'The audit also identified 25 normalized context texts shared across',
             'external splits; these occur only among SQuAD-derived examples and',
             'are excluded on both sides from clean scoring. Raw annotations',
             'were retained. These checks do not rule out semantic overlap or',
             'unknown encoder pretraining exposure.', '',
             'The fixed decoder uses the same 384-token windows, 128-token overlap,',
             'and maximum summed start/end logits. It always extracts a span.',
             'Unanswerable examples are counted in the audit but excluded from',
             'scoring. EM and F1 independently take the maximum over listed answers,',
             'treating them as accepted alternatives rather than jointly required',
             'spans. Normalization retains articles. The results therefore concern',
             'an audited answerable subset and the project scoring convention,',
             'rather than the complete official benchmark.', '']
    for split, n in [('test', 329), ('val', 333)]:
        title = 'test' if split == 'test' else 'validation'
        lines += [r'\begin{table}[!htbp]',
                  f'\\caption{{External transfer: primary LegalQAEval {title} subset ({n} questions).',
                  r'EM/F1 use the 0--100 scale. SD is across five seeds; intervals',
                  r'use paired seed differences. Validation is descriptive.}',
                  f'\\label{{tab:legalqaeval_{split}}}', r'\centering\scriptsize',
                  r'\setlength{\tabcolsep}{3pt}',
                  r'\begin{tabular}{@{}llrr>{\raggedright\arraybackslash}p{3.8cm}@{}}',
                  r'\toprule',
                  r'Encoder & Condition & EM & F1 $\pm$ SD & DAPT--N F1 (95\% interval) \\',
                  r'\midrule']
        for encoder in external.ENCODERS:
            data = reports[encoder][split]['subsets']['primary']
            if data['n'] != n:
                raise ValueError('Primary sample count changed')
            for condition in external.CONDITIONS:
                em = data['em']['descriptives'][condition]
                f1 = data['f1']['descriptives'][condition]
                contrast = '---'
                if condition != 'N':
                    c = data['f1']['contrasts'][condition]
                    lo, hi = c['ci95']
                    contrast = f'${c["mean_difference"]:+.2f}\\;[{lo:.2f},{hi:.2f}]$'
                lines.append(f'{LABELS[encoder]} & {CONDITIONS[condition]} & {em["mean"]:.2f} & '
                             f'${f1["mean"]:.2f}\\pm{f1["sd"]:.2f}$ & {contrast} ' + r'\\')
            lines.append(r'\addlinespace')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    lines += ['The primary test comparisons use seed-blocked ANOVA and paired',
              'DAPT versus N tests with Holm correction within encoder and metric.',
              'This correction is not study-wide. The F1 condition tests are:']
    for encoder in external.ENCODERS:
        data = reports[encoder]['test']['subsets']['primary']['f1']
        anova = data['blocked_anova']
        f_value = anova['F_condition']
        f_text = f'{f_value:.3f}' if isinstance(f_value, (float, int)) else str(f_value)
        ps = ', '.join(f'{c}: {data["contrasts"][c]["p_holm"]:.3f}' for c in 'ASM')
        lines.append(f'{LABELS[encoder]} $F(3,12)={f_text}$, $p={anova["p_condition"]:.3f}$ '
                     f'(Holm-adjusted paired $p$: {ps}).')
    significant = [(LABELS[e], c, reports[e]['test']['subsets']['primary']['f1']['contrasts'][c])
                   for e in external.ENCODERS for c in 'ASM'
                   if reports[e]['test']['subsets']['primary']['f1']['contrasts'][c]['p_holm'] < .05]
    lines += ['']
    if not significant:
        lines += ['None of the nine primary paired F1 comparisons rejects its null',
                  'under the specified within-encoder corrections. This does not',
                  'establish equivalence; the intervals show the range of effects',
                  'compatible with the observed seed variation.']
    else:
        lines += ['The following paired F1 contrasts meet the specified within-encoder',
                  'Holm criterion; they do not constitute study-wide corrected findings:']
        for name, condition, row in significant:
            lines.append(f'{name}, {CONDITIONS[condition]} minus N: '
                         f'${row["mean_difference"]:+.2f}$ points, $p={row["p_holm"]:.3f}$.')
    lines += ['', r'\begin{table}[!htbp]',
              r'\caption{Exploratory external test F1 means by source after overlap exclusions.',
              r'Columns give N/A/S/M means across five seeds. These analyses are secondary.}',
              r'\label{tab:legalqaeval_sources}', r'\centering\scriptsize',
              r'\begin{tabular}{@{}llrrrrr@{}}', r'\toprule',
              r'Encoder & Source & $n$ & N & A & S & M \\', r'\midrule']
    for encoder in external.ENCODERS:
        for source, label in [('microsoft--ms-marco', 'MS MARCO'), ('yang--hotpotqa', 'HotpotQA'),
                              ('google--natural-questions', 'Natural Questions')]:
            data = reports[encoder]['test']['subsets'][f'source/{source}']
            values = ' & '.join(f'{data["f1"]["descriptives"][c]["mean"]:.2f}' for c in external.CONDITIONS)
            lines.append(f'{LABELS[encoder]} & {label} & {data["n"]} & {values} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '',
              r'\begin{table}[!htbp]',
              r'\caption{Exploratory F1 on all 603 answerable external test examples,',
              r'including potentially exposed SQuAD examples. These are not primary transfer estimates.}',
              r'\label{tab:legalqaeval_all_answerable}', r'\centering\small',
              r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule', r'Encoder & N & A & S & M \\', r'\midrule']
    for encoder in external.ENCODERS:
        data = reports[encoder]['test']['subsets']['all_answerable_exploratory']
        values = ' & '.join(f'{data["f1"]["descriptives"][c]["mean"]:.2f}' for c in external.CONDITIONS)
        lines.append(f'{LABELS[encoder]} & {values} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '',
              'These results assess transfer of the final Indian-IPR readers to other',
              'legal-related reading examples. They do not replicate DAPT training',
              'on a second statutory corpus, evaluate abstention, establish legal',
              'correctness, or isolate causal effects of encoder pretraining history.',
              'The five-seed intervals condition on fixed external questions and',
              'corpus samples; they do not include question-sampling uncertainty.',
              'All per-item outputs and secondary analyses are retained.',
              r'\FloatBarrier', END]
    return '\n'.join(lines)


def finish(executable, expected_region, wait):
    work = external.HERE / 'run_v1'
    status_file = work / 'MANUSCRIPT_STATUS.json'
    def state(value, **details):
        external.atomic(status_file, {'status': value,
                       'updated_at': datetime.now().astimezone().isoformat(), **details})
    state('WAITING_FOR_COMPLETE_EVALUATION')
    try:
        while True:
            status = external.read(work / 'EXECUTION_STATUS.json')
            if status['status'] == 'FAILED':
                raise RuntimeError('Evaluation failed; manuscript numerical patch is withheld')
            if status['status'] == 'COMPLETE':
                break
            if not wait:
                raise RuntimeError('Evaluation is not complete')
            time.sleep(30)
        external.targets = resolved_targets
        reports = {}
        for encoder in external.ENCODERS:
            reports[encoder] = {}
            for split in ('val', 'test'):
                external.analyse(work, encoder, split)
                reports[encoder][split] = external.read(work / 'outputs' / encoder / f'{split}_statistics.json')
        new = generated_section(reports)
        snippet = work / 'external_results_section.tex'
        # A generated report artifact; the manuscript itself is edited with apply_patch.
        snippet.write_text(new + '\n', encoding='utf8')
        text = MANUSCRIPT.read_text(encoding='utf8')
        old = region(text)
        if digest(old) != expected_region:
            raise ValueError('External manuscript section was edited while evaluation ran; snippet retained for review')
        patch = '\n'.join(['*** Begin Patch', f'*** Update File: {MANUSCRIPT.as_posix()}', '@@',
                           *['-' + line for line in old.splitlines()],
                           *['+' + line for line in new.splitlines()], '*** End Patch'])
        state('APPLYING_VERIFIED_TABLES')
        subprocess.run([str(executable), '--codex-run-as-apply-patch', patch],
                       cwd=external.ROOT, check=True, encoding='utf8')
        if region(MANUSCRIPT.read_text(encoding='utf8')) != new:
            raise ValueError('Post-patch manuscript verification failed')
        state('TEX_UPDATED_PDF_PENDING', manuscript=str(MANUSCRIPT),
              section_sha256=digest(new), note='All 120 external evaluations verified before table insertion.')
        print('Updated manuscript with verified external test/validation and secondary tables.', flush=True)
    except Exception:
        import traceback
        state('FAILED', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-patch-exe', type=Path, required=True)
    parser.add_argument('--expected-region-sha256', required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    finish(args.apply_patch_exe, args.expected_region_sha256, args.wait)
