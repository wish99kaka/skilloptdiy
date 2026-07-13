import unittest
from app.averages import rolling_mean


class HiddenMovingAverageTests(unittest.TestCase):
    def test_width_one_and_width_larger_than_input(self):
        self.assertEqual(rolling_mean([2, 4], 1), [2.0, 4.0])
        self.assertEqual(rolling_mean([2, 4], 3), [])

    def test_rejects_non_positive_width_and_preserves_input(self):
        with self.assertRaises(ValueError):
            rolling_mean([1], 0)
        values = [1, 2, 3]
        rolling_mean(values, 2)
        self.assertEqual(values, [1, 2, 3])
