import unittest
from app.averages import moving_average


class HiddenMovingAverageTests(unittest.TestCase):
    def test_width_one_and_width_larger_than_input(self):
        self.assertEqual(moving_average([2, 4], 1), [2.0, 4.0])
        self.assertEqual(moving_average([2, 4], 3), [])

    def test_rejects_non_positive_width_and_preserves_input(self):
        with self.assertRaises(ValueError):
            moving_average([1], 0)
        values = [1, 2, 3]
        moving_average(values, 2)
        self.assertEqual(values, [1, 2, 3])
