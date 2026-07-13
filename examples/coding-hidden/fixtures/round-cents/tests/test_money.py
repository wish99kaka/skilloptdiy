import unittest

from app.money import to_cents


class ToCentsTests(unittest.TestCase):
    def test_converts_decimal_amount_to_cents(self) -> None:
        self.assertEqual(to_cents("1.23"), 123)
        self.assertEqual(to_cents("2.50"), 250)


if __name__ == "__main__":
    unittest.main()
