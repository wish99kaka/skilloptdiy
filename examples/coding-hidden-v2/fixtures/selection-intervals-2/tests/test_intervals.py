import unittest
from app.intervals import coalesce_ranges


class IntervalTests(unittest.TestCase):
    def test_merges_unsorted_overlaps(self):
        values = [(7, 10), (3, 5), (4, 8)]
        self.assertEqual(coalesce_ranges(values), [(3, 10)])


if __name__ == "__main__":
    unittest.main()
