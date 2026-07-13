import unittest

from app.money import to_cents


class HiddenToCentsTests(unittest.TestCase):
    def test_rounds_half_up(self) -> None:
        self.assertEqual(to_cents("1.005"), 101)

    def test_preserves_negative_sign(self) -> None:
        self.assertEqual(to_cents("-2.505"), -251)


if __name__ == "__main__":
    unittest.main()
