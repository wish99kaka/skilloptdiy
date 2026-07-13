import unittest
from app.intervals import coalesce_ranges


class HiddenIntervalTests(unittest.TestCase):
    def test_normalizes_and_merges_touching_intervals(self):
        self.assertEqual(coalesce_ranges([(5, 3), (5, 7), (10, 9), (7, 9)]), [(3, 10)])

    def test_empty_and_input_preservation(self):
        self.assertEqual(coalesce_ranges([]), [])
        values = [(3, 1), (8, 9)]
        coalesce_ranges(values)
        self.assertEqual(values, [(3, 1), (8, 9)])
