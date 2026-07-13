import unittest

from app.access import safe_get


class SafeGetTests(unittest.TestCase):
    def test_reads_nested_value(self) -> None:
        data = {"order": {"total": 42}}
        self.assertEqual(safe_get(data, "order.total"), 42)


if __name__ == "__main__":
    unittest.main()
