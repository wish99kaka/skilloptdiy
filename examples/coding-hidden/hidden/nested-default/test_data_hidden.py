import unittest

from app.data import get_path


class HiddenGetPathTests(unittest.TestCase):
    def test_returns_default_for_missing_path(self) -> None:
        data = {"user": {"profile": {}}}
        self.assertEqual(get_path(data, "user.profile.name", "unknown"), "unknown")

    def test_supports_list_indexes(self) -> None:
        data = {"users": [{"name": "Ada"}, {"name": "Grace"}]}
        self.assertEqual(get_path(data, "users.1.name"), "Grace")


if __name__ == "__main__":
    unittest.main()
