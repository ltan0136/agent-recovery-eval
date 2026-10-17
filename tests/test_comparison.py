"""Checks that paired comparisons cannot silently mix evaluation conditions."""

import json
from pathlib import Path
import tempfile
import unittest

from recovery_eval.comparison import compare_files
from recovery_eval.core import build_cases, run_suite


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.rows = run_suite(build_cases(1, 1))
        self.file = self.path / "episodes.jsonl"
        self.file.write_text("".join(json.dumps(r) + "\n" for r in self.rows))

    def tearDown(self):
        self.temp.cleanup()

    def test_self_comparison_has_no_flips(self):
        result = compare_files(str(self.file), str(self.file), "bounded_retry", "bounded_retry", str(self.path / "out"))
        self.assertEqual((result["regressions"], result["improvements"]), (0, 0))

    def test_known_policy_change_reports_the_actual_case_ids(self):
        result = compare_files(str(self.file), str(self.file), "idempotent_retry", "verify_then_retry", str(self.path / "out"))
        regressions = [p for p in result["pairs"] if p["change"] == "regression"]
        self.assertTrue(any("stale_read" in p["case_id"] for p in regressions))
        self.assertFalse(any("clean" in p["case_id"] for p in regressions))

    def test_changed_frozen_conditions_are_rejected(self):
        self.rows[0]["condition_hash"] = "different"
        changed = self.path / "changed.jsonl"
        changed.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        with self.assertRaisesRegex(ValueError, "conditions differ"):
            compare_files(str(self.file), str(changed), "bounded_retry", "bounded_retry", str(self.path / "out"))

    def test_unpaired_cases_are_rejected(self):
        changed = self.path / "changed.jsonl"
        changed.write_text("".join(json.dumps(r) + "\n" for r in self.rows[1:]))
        with self.assertRaisesRegex(ValueError, "Case sets differ"):
            compare_files(str(self.file), str(changed), "bounded_retry", "bounded_retry", str(self.path / "out"))
