import unittest
from app.batches import chunk_records


class HiddenBatchingTests(unittest.TestCase):
    def test_empty_and_oversized_chunks(self):
        self.assertEqual(chunk_records([], 3), [])
        self.assertEqual(chunk_records([1, 2], 9), [[1, 2]])

    def test_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            chunk_records([1], 0)

    def test_does_not_mutate_input(self):
        values = [1, 2, 3]
        chunk_records(values, 2)
        self.assertEqual(values, [1, 2, 3])
