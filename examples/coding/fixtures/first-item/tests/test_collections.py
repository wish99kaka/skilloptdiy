import unittest

from app.collections import first_item


class CollectionTests(unittest.TestCase):
    def test_returns_first_item(self):
        self.assertEqual(first_item(["a", "b"]), "a")


if __name__ == "__main__":
    unittest.main()

