import unittest

from app.money import parse_money


class HiddenMoneyTests(unittest.TestCase):
    def test_accepts_commas_and_currency_prefix(self) -> None:
        self.assertEqual(parse_money("USD 1,200.05"), 120005)
        self.assertEqual(parse_money(" $10.00 "), 1000)

    def test_preserves_negative_sign(self) -> None:
        self.assertEqual(parse_money("-$2.75"), -275)


if __name__ == "__main__":
    unittest.main()
