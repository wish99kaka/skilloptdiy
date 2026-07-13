import unittest
from app.backoff import backoff_delays


class BackoffTests(unittest.TestCase):
    def test_caps_exponential_delays(self):
        self.assertEqual(backoff_delays(1, 4, 3), [1, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()
