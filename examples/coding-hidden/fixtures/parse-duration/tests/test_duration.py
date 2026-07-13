import unittest

from app.duration import parse_duration


class ParseDurationTests(unittest.TestCase):
    def test_parses_minutes_and_hours(self) -> None:
        self.assertEqual(parse_duration("5m"), 300)
        self.assertEqual(parse_duration("2h"), 7200)


if __name__ == "__main__":
    unittest.main()
