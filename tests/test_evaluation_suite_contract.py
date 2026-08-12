from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class EvaluationSuiteContractTests(unittest.TestCase):
    def test_runner_records_all_or_none_runtime_artifact_binding(self) -> None:
        source = (ROOT / "scripts" / "run_evaluation_suite.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertEqual(source.count("engine = F5TTS("), 1)
        self.assertIn("artifact set id and sha256 must be provided together", source)
        self.assertIn('"runtime_id": args.runtime_id', source)
        self.assertIn('"artifact_set_sha256": args.artifact_set_sha256', source)
        self.assertIn('"observation_schema_version": "1.0.0"', source)


if __name__ == "__main__":
    unittest.main()
