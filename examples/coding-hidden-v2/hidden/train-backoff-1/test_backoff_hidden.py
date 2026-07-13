import unittest
from app.backoff import backoff_delays


class HiddenBackoffTests(unittest.TestCase):
    def test_empty_attempts_and_cap_below_base(self):
        self.assertEqual(backoff_delays(2, 0, 8), [])
        self.assertEqual(backoff_delays(5, 3, 2), [2, 2, 2])

    def test_rejects_negative_base_or_cap(self):
        with self.assertRaises(ValueError):
            backoff_delays(-1, 2, 4)
        with self.assertRaises(ValueError):
            backoff_delays(1, 2, -4)
