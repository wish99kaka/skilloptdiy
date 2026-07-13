import unittest
from app.allocation import allocate_units


class AllocationTests(unittest.TestCase):
    def test_allocates_proportionally(self):
        self.assertEqual(allocate_units(9, [1, 2, 1]), [2, 5, 2])


if __name__ == "__main__":
    unittest.main()
