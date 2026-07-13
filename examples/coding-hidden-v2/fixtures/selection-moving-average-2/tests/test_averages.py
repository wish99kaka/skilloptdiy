import unittest
from app.averages import rolling_mean


class MovingAverageTests(unittest.TestCase):
    def test_returns_complete_window_averages(self):
        self.assertEqual(rolling_mean([2, 4, 6, 8], 2), [3.0, 5.0, 7.0])


if __name__ == "__main__":
    unittest.main()
