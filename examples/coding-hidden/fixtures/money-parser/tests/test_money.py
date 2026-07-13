import unittest

from app.money import parse_money


class MoneyTests(unittest.TestCase):
    def test_parses_dollar_strings_as_cents(self) -> None:
        self.assertEqual(parse_money("$3.50"), 350)
        self.assertEqual(parse_money("$0.99"), 99)


if __name__ == "__main__":
    unittest.main()
