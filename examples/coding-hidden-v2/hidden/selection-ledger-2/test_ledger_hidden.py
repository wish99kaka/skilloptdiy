import unittest
from app.ledger import apply_ledger


class HiddenLedgerTests(unittest.TestCase):
    def test_skips_boolean_and_malformed_amounts(self):
        events = [{"amount": True}, {"amount": "5"}, {}, {"amount": 1.5}]
        self.assertEqual(apply_ledger(2, events), 3.5)

    def test_does_not_mutate_events(self):
        events = [{"amount": -3}, {"amount": 8}]
        snapshot = [dict(item) for item in events]
        apply_ledger(4, events)
        self.assertEqual(events, snapshot)
