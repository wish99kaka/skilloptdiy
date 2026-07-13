import unittest

from app.data import nested_get


class DataTests(unittest.TestCase):
    def test_reads_dot_path_and_returns_default_for_missing_path(self):
        payload = {"user": {"profile": {"name": "Ada"}}}
        self.assertEqual(nested_get(payload, "user.profile.name"), "Ada")
        self.assertEqual(nested_get(payload, "user.profile.age", 0), 0)


if __name__ == "__main__":
    unittest.main()

