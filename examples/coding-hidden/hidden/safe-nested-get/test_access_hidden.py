import unittest

from app.access import safe_get


class HiddenSafeGetTests(unittest.TestCase):
    def test_returns_default_for_missing_nested_value(self) -> None:
        self.assertEqual(safe_get({"order": {}}, "order.total", 0), 0)

    def test_supports_list_indexes(self) -> None:
        data = {"orders": [{"total": 12}, {"total": 34}]}
        self.assertEqual(safe_get(data, "orders.1.total"), 34)


if __name__ == "__main__":
    unittest.main()
