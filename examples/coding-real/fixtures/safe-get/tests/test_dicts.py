import unittest

from app.dicts import safe_get


class DictTests(unittest.TestCase):
    def test_returns_default_for_missing_key(self):
        self.assertEqual(safe_get({"a": 1}, "a", 0), 1)
        self.assertEqual(safe_get({"a": 1}, "b", 0), 0)


if __name__ == "__main__":
    unittest.main()

