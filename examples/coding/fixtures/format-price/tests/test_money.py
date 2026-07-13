import unittest

from app.money import format_price


class MoneyTests(unittest.TestCase):
    def test_formats_price_with_two_decimals(self):
        self.assertEqual(format_price(3), "$3.00")


if __name__ == "__main__":
    unittest.main()

