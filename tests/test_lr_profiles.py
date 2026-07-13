import os
import unittest
from unittest.mock import patch

from textskill_optimizer.cli import lr_defaults, resolve_lr_value
from textskill_optimizer.lr_profiles import apply_lr_profile, get_lr_profile, profile_names
from textskill_optimizer.optimizer import OptimizerConfig


class LearningRateProfileTests(unittest.TestCase):
    def test_known_profiles_include_real_editor(self) -> None:
        self.assertIn("strict", profile_names())
        self.assertIn("real-editor", profile_names())
        self.assertIn("loose-diagnostic", profile_names())

    def test_real_editor_profile_values(self) -> None:
        profile = get_lr_profile("real-editor")

        self.assertEqual(profile.max_skill_chars, 600)
        self.assertEqual(profile.max_skill_delta_chars, 520)
        self.assertEqual(profile.max_added_bullet_lines, 1)

    def test_apply_lr_profile_replaces_budget_fields(self) -> None:
        config = OptimizerConfig(
            epochs=2,
            max_skill_chars=10,
            max_skill_delta_chars=10,
            max_added_bullet_lines=10,
            rejected_buffer_limit=5,
        )

        updated = apply_lr_profile(config, "loose-diagnostic")

        self.assertEqual(updated.epochs, 2)
        self.assertEqual(updated.rejected_buffer_limit, 5)
        self.assertEqual(updated.max_skill_chars, 750)
        self.assertEqual(updated.max_skill_delta_chars, 700)
        self.assertEqual(updated.max_added_bullet_lines, 3)

    def test_cli_lr_defaults_use_profile_or_legacy_defaults(self) -> None:
        self.assertEqual(lr_defaults(None)["max_skill_delta_chars"], 1800)
        self.assertEqual(lr_defaults("strict")["max_skill_delta_chars"], 260)

    def test_cli_lr_value_resolution_prefers_arg_then_env_then_default(self) -> None:
        with patch.dict(os.environ, {"TEXTSKILL_MAX_SKILL_CHARS": "123"}, clear=True):
            self.assertEqual(resolve_lr_value(456, "TEXTSKILL_MAX_SKILL_CHARS", 789), 456)
            self.assertEqual(resolve_lr_value(None, "TEXTSKILL_MAX_SKILL_CHARS", 789), 123)

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_lr_value(None, "TEXTSKILL_MAX_SKILL_CHARS", 789), 789)


if __name__ == "__main__":
    unittest.main()
