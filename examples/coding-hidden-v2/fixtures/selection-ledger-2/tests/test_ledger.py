import unittest
from app.ledger import apply_ledger


class LedgerTests(unittest.TestCase):
    def test_applies_positive_and_negative_amounts(self):
        events = [{"amount": 5}, {"amount": -2}, {"note": "skip"}]
        self.assertEqual(apply_ledger(20, events), 23)


if __name__ == "__main__":
    unittest.main()
