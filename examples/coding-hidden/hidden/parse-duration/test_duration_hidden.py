import unittest

from app.duration import parse_duration


class HiddenParseDurationTests(unittest.TestCase):
    def test_accepts_seconds_and_whitespace(self) -> None:
        self.assertEqual(parse_duration(" 45s "), 45)

    def test_returns_zero_for_malformed_input(self) -> None:
        self.assertEqual(parse_duration("soon"), 0)
        self.assertEqual(parse_duration(""), 0)


if __name__ == "__main__":
    unittest.main()
