import unittest

from app.numbers import clamp_negative


class NumberTests(unittest.TestCase):
    def test_negative_values_become_zero(self):
        self.assertEqual(clamp_negative(-3), 0)
        self.assertEqual(clamp_negative(4), 4)


if __name__ == "__main__":
    unittest.main()

