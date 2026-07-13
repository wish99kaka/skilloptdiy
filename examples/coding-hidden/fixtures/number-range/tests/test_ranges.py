import unittest

from app.ranges import number_range


class NumberRangeTests(unittest.TestCase):
    def test_returns_inclusive_integer_range(self) -> None:
        self.assertEqual(number_range(1, 3), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
