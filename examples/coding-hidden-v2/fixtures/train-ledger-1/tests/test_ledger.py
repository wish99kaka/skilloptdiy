import unittest
from app.ledger import running_balance


class LedgerTests(unittest.TestCase):
    def test_applies_positive_and_negative_amounts(self):
        events = [{"amount": 5}, {"amount": -2}, {"note": "skip"}]
        self.assertEqual(running_balance(10, events), 13)


if __name__ == "__main__":
    unittest.main()
