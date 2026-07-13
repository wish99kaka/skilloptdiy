import unittest

from app.collections import unique_items


class CollectionTests(unittest.TestCase):
    def test_removes_duplicates_preserving_order(self):
        self.assertEqual(unique_items(["a", "b", "a"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()

