import unittest
from app.grouping import group_sums


class HiddenGroupingTests(unittest.TestCase):
    def test_skips_malformed_and_boolean_values(self):
        rows = [{"g": "x", "v": 1.5}, {"g": "x", "v": True}, {"v": 9}, {"g": "y", "v": -2}]
        self.assertEqual(group_sums(rows, "g", "v"), {"x": 1.5, "y": -2})

    def test_preserves_inputs_and_first_seen_order(self):
        rows = [{"g": "z", "v": 1}, {"g": "a", "v": 2}]
        snapshot = [dict(row) for row in rows]
        result = group_sums(rows, "g", "v")
        self.assertEqual(list(result), ["z", "a"])
        self.assertEqual(rows, snapshot)
