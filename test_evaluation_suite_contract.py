from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parent


class EvaluationSuiteContractTests(unittest.TestCase):
    def test_runner_loads_engine_once_and_forwards_frozen_seed(self) -> None:
        source = (ROOT / "scripts" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertEqual(source.count("engine = F5TTS("), 1)
        self.assertIn('seed_everything(seed)', source)
        self.assertIn('infer_sequential(engine, args, row["text"], int(row["seed"]))', source)
        self.assertIn("generation-observations.json", source)

    def test_multi_chunk_inference_is_sequential(self) -> None:
        source = (ROOT / "scripts" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        self.assertIn("for chunk in chunks:", source)
        self.assertIn("inference_chunk_count", source)
        self.assertIn("concurrent multi-chunk path", source)

    def test_instruction_limit_is_not_hidden(self) -> None:
        source = (ROOT / "scripts" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        self.assertIn('"instruction_applied": False', source)
        self.assertIn("has no separate instruction input", source)


if __name__ == "__main__":
    unittest.main()
