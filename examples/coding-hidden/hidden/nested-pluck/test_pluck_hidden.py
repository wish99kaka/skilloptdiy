import unittest

from app.pluck import pluck


class HiddenPluckTests(unittest.TestCase):
    def test_skips_records_missing_path(self) -> None:
        records = [{"user": {"name": "Ada"}}, {"user": {}}, {}]
        self.assertEqual(pluck(records, "user.name"), ["Ada"])

    def test_supports_list_indexes(self) -> None:
        records = [{"users": [{"name": "Ada"}]}, {"users": [{"name": "Grace"}]}]
        self.assertEqual(pluck(records, "users.0.name"), ["Ada", "Grace"])


if __name__ == "__main__":
    unittest.main()
