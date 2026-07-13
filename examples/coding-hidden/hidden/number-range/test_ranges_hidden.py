import unittest

from app.ranges import number_range


class HiddenNumberRangeTests(unittest.TestCase):
    def test_accepts_reversed_bounds(self) -> None:
        self.assertEqual(number_range(3, 1), [1, 2, 3])

    def test_handles_singleton_range(self) -> None:
        self.assertEqual(number_range(2, 2), [2])


if __name__ == "__main__":
    unittest.main()
