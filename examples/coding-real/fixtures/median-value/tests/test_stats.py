import unittest

from app.stats import median


class StatsTests(unittest.TestCase):
    def test_median_for_odd_and_even_lengths(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([10, 2, 4, 8]), 6)


if __name__ == "__main__":
    unittest.main()

