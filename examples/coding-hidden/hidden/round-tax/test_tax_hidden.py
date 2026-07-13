import unittest

from app.tax import add_tax


class HiddenAddTaxTests(unittest.TestCase):
    def test_rounds_half_up(self) -> None:
        self.assertEqual(add_tax(199, 0.075), 214)

    def test_zero_rate_returns_original(self) -> None:
        self.assertEqual(add_tax(999, 0), 999)


if __name__ == "__main__":
    unittest.main()
