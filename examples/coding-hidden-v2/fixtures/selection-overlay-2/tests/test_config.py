import unittest
from app.config import overlay_config


class OverlayTests(unittest.TestCase):
    def test_non_none_overrides_win(self):
        self.assertEqual(
            overlay_config({"host": "a", "port": 80}, {"port": 8081, "host": None}),
            {"host": "a", "port": 8081},
        )


if __name__ == "__main__":
    unittest.main()
