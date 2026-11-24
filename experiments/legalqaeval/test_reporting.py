import copy
import unittest
import finish_report
import run as pipeline


class ReportingTests(unittest.TestCase):
    def test_section_markers_are_unique(self):
        with self.assertRaises(ValueError):
            finish_report.region('no markers')
        text = 'before\n' + finish_report.BEGIN + '\nbody\n' + finish_report.END + '\nafter'
        self.assertEqual(finish_report.region(text), finish_report.BEGIN + '\nbody\n' + finish_report.END)
        with self.assertRaises(ValueError):
            finish_report.region(text + finish_report.BEGIN)

    def test_report_uses_all_encoders_and_both_splits(self):
        # Fixture statistics only; never stored as research outputs.
        from test_pipeline import EvaluationTests
        fixture, _ = EvaluationTests().result()
        cells = []
        for condition in pipeline.CONDITIONS:
            for seed in pipeline.SEEDS:
                cell = copy.deepcopy(fixture)
                cell['identity'] = {'condition': condition, 'seed': seed}
                cells.append(cell)
        base = pipeline.statistical_report(cells)
        # Source fixture has empty groups; fill their statistical schema solely
        # to exercise the numeric report formatter.
        for key in list(base['subsets']):
            base['subsets'][key] = copy.deepcopy(base['subsets']['primary'])
        reports = {}
        for encoder in pipeline.ENCODERS:
            reports[encoder] = {}
            for split, n in [('val', 333), ('test', 329)]:
                report = copy.deepcopy(base)
                report['subsets']['primary']['n'] = n
                reports[encoder][split] = report
        text = finish_report.generated_section(reports)
        self.assertEqual(text.count('\\begin{table}'), 4)
        self.assertEqual(text.count('\\end{table}'), 4)
        for name in finish_report.LABELS.values():
            self.assertIn(name, text)
        self.assertIn('329 test', text)
        self.assertNotIn('in progress', text)
        self.assertIn('not establish equivalence', ' '.join(text.split()))
        reports['bert']['test']['subsets']['primary']['n'] = 1
        with self.assertRaises(ValueError):
            finish_report.generated_section(reports)


if __name__ == '__main__':
    unittest.main()
