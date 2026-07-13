import unittest
from app.batches import partition_items


class HiddenBatchingTests(unittest.TestCase):
    def test_empty_and_oversized_chunks(self):
        self.assertEqual(partition_items([], 3), [])
        self.assertEqual(partition_items([1, 2], 9), [[1, 2]])

    def test_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            partition_items([1], 0)

    def test_does_not_mutate_input(self):
        values = [1, 2, 3]
        partition_items(values, 2)
        self.assertEqual(values, [1, 2, 3])
