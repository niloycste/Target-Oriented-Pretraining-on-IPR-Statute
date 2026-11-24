"""Regression checks for encoder isolation and preservation of the D recipe."""
import ast
import importlib
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
x = importlib.import_module("scripts.dapt_equal_update.encoder_extension")


class ExtensionTests(unittest.TestCase):
    def test_all_routes_and_exact_downstream_logic(self):
        for encoder in x.MODELS:
            exp, work = x.paths(encoder)
            snapshot = Path("C:/model-cache") / encoder
            for seed in x.SEEDS:
                for condition in x.CONDITIONS:
                    stages = x.sources(condition, seed, snapshot, exp, work)
                    self.assertEqual(set(stages), {"squad", "ipr"} if condition == "N"
                                     else {"mlm", "squad", "ipr"})
                    for stage, (source, output) in stages.items():
                        ast.parse(source)
                        self.assertTrue(output.is_relative_to(work))
                        self.assertIn(str(exp / "cache" / stage), source)
                        assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)]
                        self.assertFalse(any('law-ai/InLegalBERT' in ast.unparse(n) for n in assignments))
                        if stage == "mlm":
                            self.assertNotIn("ignore_mismatched_sizes=True", source)
                            self.assertNotIn("model.config.tie_word_embeddings = False", source)
                            self.assertIn(str(x.d.CORPORA[condition]), source)
                        else:
                            maker = x.b.make_squad_source if stage == "squad" else x.b.make_ipr_source
                            expected = maker(snapshot, output, seed)
                            normalise = lambda t: re.sub(r"^(MODEL_DIR|CACHE_DIR)\s*=.*$",
                                                        r"\1 = <PATH>", t, flags=re.M)
                            self.assertEqual(normalise(source), normalise(expected))
                    self.assertIn(str(snapshot if condition == "N" else
                        work / f"E_{condition}_s{seed}_mlm/final"), stages["squad"][0])

    def test_protocol_refuses_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.json"
            x.freeze(path, {"seed": 7})
            x.freeze(path, {"seed": 7})
            with self.assertRaises(RuntimeError):
                x.freeze(path, {"seed": 42})


if __name__ == "__main__":
    unittest.main()
