import unittest
from app.averages import moving_average


class MovingAverageTests(unittest.TestCase):
    def test_returns_complete_window_averages(self):
        self.assertEqual(moving_average([1, 3, 5, 7], 2), [2.0, 4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
