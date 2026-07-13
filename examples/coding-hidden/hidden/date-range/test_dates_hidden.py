import unittest

from app.dates import date_range


class HiddenDateRangeTests(unittest.TestCase):
    def test_accepts_reversed_bounds(self) -> None:
        self.assertEqual(
            date_range("2026-01-03", "2026-01-01"),
            ["2026-01-01", "2026-01-02", "2026-01-03"],
        )

    def test_handles_leap_day(self) -> None:
        self.assertEqual(
            date_range("2024-02-28", "2024-03-01"),
            ["2024-02-28", "2024-02-29", "2024-03-01"],
        )


if __name__ == "__main__":
    unittest.main()
