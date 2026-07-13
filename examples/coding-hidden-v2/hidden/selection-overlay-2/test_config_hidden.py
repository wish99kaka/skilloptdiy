import unittest
from app.config import overlay_config


class HiddenOverlayTests(unittest.TestCase):
    def test_keeps_falsey_values_except_none(self):
        self.assertEqual(
            overlay_config({"enabled": True, "count": 3}, {"enabled": False, "count": 0, "extra": None}),
            {"enabled": False, "count": 0},
        )

    def test_does_not_mutate_inputs(self):
        base = {"a": 1}
        overrides = {"a": 2, "b": 3}
        result = overlay_config(base, overrides)
        self.assertEqual(base, {"a": 1})
        self.assertEqual(overrides, {"a": 2, "b": 3})
        self.assertIsNot(result, base)
