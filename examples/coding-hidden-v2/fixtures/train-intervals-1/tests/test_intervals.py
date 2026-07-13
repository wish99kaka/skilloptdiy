import unittest
from app.intervals import merge_windows


class IntervalTests(unittest.TestCase):
    def test_merges_unsorted_overlaps(self):
        values = [(5, 8), (1, 3), (2, 6)]
        self.assertEqual(merge_windows(values), [(1, 8)])


if __name__ == "__main__":
    unittest.main()
