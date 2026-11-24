"""Offline tests; no downloads, pretrained model loading, or training."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('legalqaeval_run', Path(__file__).with_name('run.py'))
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def example(key='1', text='The law protects designs.', question='What is protected?',
            source='microsoft--ms-marco', answer='designs'):
    start = text.index(answer) if answer else None
    return {'id': f'isaacus--legalqaeval/{source}/validation/{key}',
            'text': text, 'question': question,
            'answers': [{'text': answer, 'start': start, 'end': start+len(answer)}] if answer else []}


class DataTests(unittest.TestCase):
    def splits(self):
        return {'val': [example()], 'test': [example('2', 'Acts protect marks.', 'What do Acts protect?', answer='marks')]}

    def test_valid_offsets_and_unanswerable(self):
        pipeline.validate_rows([example(), example('2', answer=None)])

    def test_unicode_offsets(self):
        pipeline.validate_rows([example(text='Law protects café.', answer='café')])

    def test_invalid_offsets(self):
        row = example()
        row['answers'][0]['end'] -= 1
        with self.assertRaisesRegex(ValueError, 'offset'):
            pipeline.validate_rows([row])

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):
            pipeline.validate_rows([example(), example()])

    def test_unknown_source(self):
        with self.assertRaises(ValueError):
            pipeline.validate_rows([example(source='unknown')])

    def test_empty_question(self):
        with self.assertRaises(ValueError):
            pipeline.validate_rows([example(question='  ')])

    def test_cross_split_context(self):
        splits = self.splits()
        splits['test'] = [example('2', question='A different question?')]
        audited = pipeline.audit_splits(splits, {})
        for split in ('val', 'test'):
            self.assertFalse(audited[split][0]['primary'])
            self.assertEqual(audited[split][0]['cross_split_overlap'], ['text'])

    def test_cross_split_question(self):
        splits = self.splits()
        splits['test'][0]['question'] = 'WHAT is protected !'
        audited = pipeline.audit_splits(splits, {})
        for split in ('val', 'test'):
            self.assertFalse(audited[split][0]['clean_answerable'])
            self.assertEqual(audited[split][0]['cross_split_overlap'], ['question'])

    def test_question_or_context_overlap(self):
        for reference in ({'question': 'WHAT is protected?'}, {'context': 'THE LAW protects designs!'}):
            audited = pipeline.audit_splits(self.splits(), {'train': [reference]})
            self.assertFalse(audited['val'][0]['primary'])
            self.assertEqual(audited['val'][0]['overlap_resources'], ['train'])
            self.assertTrue(audited['test'][0]['primary'])

    def test_squad_excluded_without_exact_overlap(self):
        splits = self.splits()
        splits['val'][0] = example(source='stanford--squad-v2')
        row = pipeline.audit_splits(splits, {})['val'][0]
        self.assertFalse(row['primary'])
        self.assertTrue(row['clean_answerable'])

    def test_unanswerable_excluded(self):
        splits = self.splits()
        splits['val'][0]['answers'] = []
        row = pipeline.audit_splits(splits, {})['val'][0]
        self.assertFalse(row['primary'])
        self.assertFalse(row['clean_answerable'])


class EvaluationTests(unittest.TestCase):
    def result(self):
        row = pipeline.audit_splits(DataTests().splits(), {})['test'][0]
        prediction = {**row, 'prediction': 'marks', 'span_score': 1.0,
                      **pipeline.score('marks', row['answers'])}
        return {'identity': {'test': True}, 'predictions': [prediction],
                'metrics': pipeline.summarize([prediction])}, [row]

    def test_multiple_accepted_answers(self):
        self.assertEqual(pipeline.score('registered designs', [{'text': 'marks'}, {'text': 'registered designs'}]),
                         {'em': 1.0, 'f1': 1.0})

    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.score('something', [])

    def test_project_articles_retained(self):
        self.assertEqual(pipeline.score('the law', [{'text': 'law'}])['em'], 0)

    def test_saved_result_rescored(self):
        payload, rows = self.result()
        pipeline.validate_result(payload, rows, payload['identity'])
        payload['predictions'][0]['f1'] = .4
        with self.assertRaises(ValueError):
            pipeline.validate_result(payload, rows, payload['identity'])

    def test_saved_inputs_checked(self):
        payload, rows = self.result()
        payload['predictions'][0]['question'] = 'Altered'
        with self.assertRaises(ValueError):
            pipeline.validate_result(payload, rows, payload['identity'])

    def test_missing_predictions_rejected(self):
        payload, rows = self.result()
        payload['predictions'] = []
        with self.assertRaises(ValueError):
            pipeline.validate_result(payload, rows, payload['identity'])

    def test_nan_rejected(self):
        payload, rows = self.result()
        payload['predictions'][0]['f1'] = math.nan
        with self.assertRaises(ValueError):
            pipeline.validate_result(payload, rows, payload['identity'])

    def test_freeze_rejects_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'frozen.json'
            pipeline.freeze(path, {'x': 1})
            pipeline.freeze(path, {'x': 1})
            with self.assertRaises(ValueError):
                pipeline.freeze(path, {'x': 2})

    def test_missing_weights_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline.atomic(Path(folder)/'config.json', {})
            with self.assertRaisesRegex(ValueError, 'weights missing'):
                pipeline.checkpoint_hash(folder)

    def test_target_routes(self):
        for encoder in pipeline.ENCODERS:
            self.assertEqual(len(pipeline.targets(encoder)), 20)
        self.assertIn('final_cpu/A_N_', pipeline.targets('inlegalbert')[0]['checkpoint'].replace('\\', '/'))
        self.assertIn('equal_update/D_A_', pipeline.targets('inlegalbert')[5]['checkpoint'].replace('\\', '/'))

    def test_complete_statistics_and_missing_cell(self):
        fixture, _ = self.result()
        results = []
        for condition in pipeline.CONDITIONS:
            for seed in pipeline.SEEDS:
                cell = copy.deepcopy(fixture)
                cell['identity'] = {'condition': condition, 'seed': seed}
                results.append(cell)
        report = pipeline.statistical_report(results)
        self.assertEqual(report['subsets']['primary']['f1']['contrasts']['A']['mean_difference'], 0)
        self.assertEqual(report['subsets']['primary']['f1']['contrasts']['A']['p_holm'], 1)
        with self.assertRaises(ValueError):
            pipeline.statistical_report(results[:-1])
        results[-1]['identity'] = results[0]['identity']
        with self.assertRaises(ValueError):
            pipeline.statistical_report(results)

    def test_real_decoder_overflow_with_random_tiny_model(self):
        # Exercise actual Transformers and canonical windows using a tiny random
        # model. It is a correctness smoke check, not a performance experiment.
        import torch
        from transformers import BertConfig, BertForQuestionAnswering, BertTokenizerFast
        from scripts.qa_evaluation import QADecodingConfig, predict_answers
        with tempfile.TemporaryDirectory() as folder:
            vocab = Path(folder)/'vocab.txt'
            vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwhat\nis\nprotected\n?\nlaw\nmarks\n.\n', encoding='utf8')
            tokenizer = BertTokenizerFast(vocab_file=str(vocab))
            torch.manual_seed(17)
            model = BertForQuestionAnswering(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
                    num_hidden_layers=1, num_attention_heads=2, intermediate_size=24))
            context = 'law marks . ' * 30
            decoded = predict_answers(tokenizer, model, torch.device('cpu'), ['what is protected ?'],
                                      [context], QADecodingConfig(max_seq_length=24, doc_stride=8, batch_size=2))
            self.assertEqual(len(decoded), 1)
            self.assertTrue(decoded[0][0])
            self.assertIn(decoded[0][0], context)
            self.assertTrue(math.isfinite(decoded[0][1]))


if __name__ == '__main__':
    unittest.main()
