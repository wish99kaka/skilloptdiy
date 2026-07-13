import unittest
from app.intervals import merge_windows


class HiddenIntervalTests(unittest.TestCase):
    def test_normalizes_and_merges_touching_intervals(self):
        self.assertEqual(merge_windows([(5, 3), (5, 7), (10, 9), (7, 9)]), [(3, 10)])

    def test_empty_and_input_preservation(self):
        self.assertEqual(merge_windows([]), [])
        values = [(3, 1), (8, 9)]
        merge_windows(values)
        self.assertEqual(values, [(3, 1), (8, 9)])
