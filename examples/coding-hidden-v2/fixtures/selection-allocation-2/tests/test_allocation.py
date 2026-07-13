import unittest
from app.allocation import split_capacity


class AllocationTests(unittest.TestCase):
    def test_allocates_proportionally(self):
        self.assertEqual(split_capacity(10, [1, 2, 1]), [3, 5, 2])


if __name__ == "__main__":
    unittest.main()
