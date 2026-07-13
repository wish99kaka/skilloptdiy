import unittest
from app.batches import chunk_records


class BatchingTests(unittest.TestCase):
    def test_splits_consecutive_items(self):
        self.assertEqual(chunk_records([1, 2, 3, 4, 5, 6], 3), [[1, 2, 3], [4, 5, 6]])


if __name__ == "__main__":
    unittest.main()
