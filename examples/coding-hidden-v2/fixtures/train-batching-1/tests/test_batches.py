import unittest
from app.batches import partition_items


class BatchingTests(unittest.TestCase):
    def test_splits_consecutive_items(self):
        self.assertEqual(partition_items([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])


if __name__ == "__main__":
    unittest.main()
