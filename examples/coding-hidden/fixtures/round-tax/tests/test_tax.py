import unittest

from app.tax import add_tax


class AddTaxTests(unittest.TestCase):
    def test_adds_tax_in_cents(self) -> None:
        self.assertEqual(add_tax(1000, 0.075), 1075)


if __name__ == "__main__":
    unittest.main()
