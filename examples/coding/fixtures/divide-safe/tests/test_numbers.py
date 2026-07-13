import unittest

from app.numbers import divide_safe


class NumberTests(unittest.TestCase):
    def test_division_by_zero_returns_zero(self):
        self.assertEqual(divide_safe(10, 0), 0)
        self.assertEqual(divide_safe(10, 2), 5)


if __name__ == "__main__":
    unittest.main()

