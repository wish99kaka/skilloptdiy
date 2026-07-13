import unittest
from app.allocation import split_capacity


class HiddenAllocationTests(unittest.TestCase):
    def test_distributes_tied_remainders_by_index(self):
        self.assertEqual(split_capacity(2, [1, 1, 1]), [1, 1, 0])

    def test_zero_weights_and_invalid_inputs(self):
        self.assertEqual(split_capacity(5, [0, 0]), [0, 0])
        with self.assertRaises(ValueError):
            split_capacity(-1, [1])
        with self.assertRaises(ValueError):
            split_capacity(3, [1, -1])
