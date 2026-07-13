import unittest

from app.intervals import merge_intervals


class IntervalTests(unittest.TestCase):
    def test_merges_overlapping_and_touching_intervals(self):
        self.assertEqual(
            merge_intervals([(5, 7), (1, 3), (2, 4), (8, 8), (7, 8)]),
            [(1, 4), (5, 8)],
        )


if __name__ == "__main__":
    unittest.main()

