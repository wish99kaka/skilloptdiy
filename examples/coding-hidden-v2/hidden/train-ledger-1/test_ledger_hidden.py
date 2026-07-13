import unittest
from app.ledger import running_balance


class HiddenLedgerTests(unittest.TestCase):
    def test_skips_boolean_and_malformed_amounts(self):
        events = [{"amount": True}, {"amount": "5"}, {}, {"amount": 1.5}]
        self.assertEqual(running_balance(2, events), 3.5)

    def test_does_not_mutate_events(self):
        events = [{"amount": -3}, {"amount": 8}]
        snapshot = [dict(item) for item in events]
        running_balance(4, events)
        self.assertEqual(events, snapshot)
