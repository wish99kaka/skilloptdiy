import unittest
from app.backoff import retry_schedule


class BackoffTests(unittest.TestCase):
    def test_caps_exponential_delays(self):
        self.assertEqual(retry_schedule(2, 4, 6), [2, 4, 6, 6])


if __name__ == "__main__":
    unittest.main()
