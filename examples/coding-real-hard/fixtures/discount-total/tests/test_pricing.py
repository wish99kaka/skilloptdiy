import unittest

from app.pricing import order_total


class PricingTests(unittest.TestCase):
    def test_applies_quantities_discount_and_tax(self):
        items = [
            {"price": 10.0, "quantity": 2},
            {"price": 5.0, "quantity": 1},
        ]
        self.assertEqual(order_total(items, discount_percent=10, tax_percent=8), 24.3)


if __name__ == "__main__":
    unittest.main()

