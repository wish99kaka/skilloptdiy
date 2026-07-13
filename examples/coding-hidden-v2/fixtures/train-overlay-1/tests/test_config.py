import unittest
from app.config import merge_settings


class OverlayTests(unittest.TestCase):
    def test_non_none_overrides_win(self):
        self.assertEqual(
            merge_settings({"host": "a", "port": 80}, {"port": 8080, "host": None}),
            {"host": "a", "port": 8080},
        )


if __name__ == "__main__":
    unittest.main()
