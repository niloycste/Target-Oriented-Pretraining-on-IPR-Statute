"""Small real-reader check; writes diagnostics, never the primary results."""
import collections
import gc
import math
import string
import time
import run as external


def independent_score(prediction, references):
    def tokens(value):
        return value.lower().translate(str.maketrans('', '', string.punctuation)).split()
    predicted = tokens(prediction)
    scores = []
    for reference in references:
        gold = tokens(reference['text'])
        overlap = sum((collections.Counter(predicted) & collections.Counter(gold)).values())
        scores.append((float(predicted == gold), 2 * overlap / (len(predicted)+len(gold))
                       if predicted and gold else 0.0))
    return {'em': max(s[0] for s in scores), 'f1': max(s[1] for s in scores)}


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from scripts.qa_evaluation import QADecodingConfig, predict_answers
    torch.set_num_threads(4)
    work = external.HERE / 'run_v1'
    external.verify_protocol(work)
    from run_all import resolved_targets
    rows = [r for r in external.read(work / 'prepared/val.json') if r['primary']]
    # Selection depends only on source and context length, never predictions.
    sample = [next(r for r in rows if r['source'] == source)
              for source in external.SOURCE_NAMES if any(r['source'] == source for r in rows)]
    longest = max(rows, key=lambda r: len(r['text']))
    if longest['id'] not in {r['id'] for r in sample}:
        sample.append(longest)
    results = {}
    for encoder in external.ENCODERS:
        target = resolved_targets(encoder)[0]  # N, seed 7 fixed in advance
        tokenizer = AutoTokenizer.from_pretrained(target['checkpoint'], local_files_only=True, use_fast=True)
        model, info = AutoModelForQuestionAnswering.from_pretrained(
            target['checkpoint'], local_files_only=True, output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError(f'Strict checkpoint loading failed: {encoder}: {info}')
        started = time.monotonic()
        decoded = predict_answers(tokenizer, model, torch.device('cpu'),
                                  [r['question'] for r in sample], [r['text'] for r in sample],
                                  QADecodingConfig())
        predictions = []
        for row, (text, span_score) in zip(sample, decoded):
            if not text or text not in row['text'] or not math.isfinite(span_score):
                raise ValueError('Invalid extracted source span')
            score = external.score(text, row['answers'])
            independent = independent_score(text, row['answers'])
            if any(abs(score[k]-independent[k]) > 1e-10 for k in score):
                raise ValueError('Independent metric calculation disagrees')
            predictions.append({'id': row['id'], 'question': row['question'],
                                'prediction': text, 'answers': row['answers'], **score})
        results[encoder] = {'checkpoint': target['checkpoint'], 'strict_loading': 'PASS',
                            'source_spans': 'PASS', 'independent_scores': 'PASS',
                            'seconds': time.monotonic()-started, 'predictions': predictions}
        print(f'PASS {encoder}: {len(predictions)} actual examples; strict QA loading and independent scoring', flush=True)
        del model, tokenizer
        gc.collect()
    external.atomic(external.HERE / 'INTEGRATION_CHECK.json',
                    {'status': 'PASS', 'selection': 'validation only; source coverage plus longest context',
                     'purpose': 'implementation check, not an external performance estimate', 'readers': results})


if __name__ == '__main__':
    main()
