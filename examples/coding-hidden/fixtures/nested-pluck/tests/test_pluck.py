import unittest

from app.pluck import pluck


class PluckTests(unittest.TestCase):
    def test_plucks_nested_values(self) -> None:
        records = [{"user": {"name": "Ada"}}, {"user": {"name": "Grace"}}]
        self.assertEqual(pluck(records, "user.name"), ["Ada", "Grace"])


if __name__ == "__main__":
    unittest.main()
