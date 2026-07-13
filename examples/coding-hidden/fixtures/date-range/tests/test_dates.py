import unittest

from app.dates import date_range


class DateRangeTests(unittest.TestCase):
    def test_returns_inclusive_date_strings(self) -> None:
        self.assertEqual(
            date_range("2026-01-01", "2026-01-03"),
            ["2026-01-01", "2026-01-02", "2026-01-03"],
        )


if __name__ == "__main__":
    unittest.main()
