import unittest

from app.numbers import absolute_value


class NumberTests(unittest.TestCase):
    def test_returns_absolute_value(self):
        self.assertEqual(absolute_value(-5), 5)
        self.assertEqual(absolute_value(3), 3)


if __name__ == "__main__":
    unittest.main()

