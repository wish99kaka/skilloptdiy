import unittest
from app.allocation import allocate_units


class HiddenAllocationTests(unittest.TestCase):
    def test_distributes_tied_remainders_by_index(self):
        self.assertEqual(allocate_units(2, [1, 1, 1]), [1, 1, 0])

    def test_zero_weights_and_invalid_inputs(self):
        self.assertEqual(allocate_units(5, [0, 0]), [0, 0])
        with self.assertRaises(ValueError):
            allocate_units(-1, [1])
        with self.assertRaises(ValueError):
            allocate_units(3, [1, -1])
