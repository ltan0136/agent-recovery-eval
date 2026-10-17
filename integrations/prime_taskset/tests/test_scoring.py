"""Check strict scoring and prompt/reference separation."""

import json
import unittest

from recovery_diagnosis.scoring import build_prompt, exact_score


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"duplicates": 1, "missing": 0, "success": False}

    def test_exact_json_passes(self):
        self.assertEqual(exact_score(json.dumps(self.expected), self.expected), 1.0)

    def test_invalid_answers_fail(self):
        replies = [
            '```json\n{"duplicates":1,"missing":0,"success":false}\n```',
            '{"duplicates":1,"missing":0,"success":false,"extra":0}',
            '{"duplicates":true,"missing":0,"success":false}',
            '{"duplicates":1.0,"missing":0,"success":false}',
            '{"duplicates":"1","missing":0,"success":false}',
            '{"duplicates":1,"missing":0,"success":0}',
            '{"duplicates":1,"missing":0,"success":false,"success":false}',
            '{"duplicates":-1,"missing":0,"success":false}',
            '{"duplicates":0,"missing":0,"success":true}',
            '{"duplicates":1,"success":false}',
            '[]',
            None,
        ]
        for reply in replies:
            with self.subTest(reply=reply):
                self.assertEqual(exact_score(reply, self.expected), 0.0)

    def test_nested_reference_labels_rejected(self):
        with self.assertRaises(ValueError):
            build_prompt({"operations": []}, [{"result": {"ground_truth": self.expected}}])

    def test_prompt_contains_only_public_case_fields(self):
        prompt = build_prompt({"operations": [{"operation_id": "op-1"}]}, [])
        self.assertIn('"operation_id":"op-1"', prompt)
        self.assertNotIn('"expected"', prompt)
        self.assertNotIn('"ground_truth"', prompt)


if __name__ == "__main__":
    unittest.main()
