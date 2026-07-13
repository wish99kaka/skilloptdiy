import os
import unittest
from unittest.mock import patch

from work.run_coding_hidden_v2_experiment import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_external_editor_env,
    read_api_key,
)


class CodingHiddenV2ExperimentTests(unittest.TestCase):
    def test_builds_external_editor_env_without_coco_model_override(self) -> None:
        env = build_external_editor_env({}, "secret")

        self.assertEqual(env["EXTERNAL_LLM_BASE_URL"], DEFAULT_BASE_URL)
        self.assertEqual(env["EXTERNAL_LLM_MODEL"], DEFAULT_MODEL)
        self.assertEqual(env["EXTERNAL_LLM_API_KEY"], "secret")
        self.assertNotIn("COCO_AGENT_MODEL", env)
        self.assertNotIn("COCO_EDITOR_MODEL", env)

    def test_respects_external_optimizer_overrides(self) -> None:
        env = build_external_editor_env(
            {
                "BYTEDANCE_MODEL_BASE_URL": "https://example.test/v1",
                "BYTEDANCE_MODEL_ID": "editor-model",
            },
            "secret",
        )

        self.assertEqual(env["EXTERNAL_LLM_BASE_URL"], "https://example.test/v1")
        self.assertEqual(env["EXTERNAL_LLM_MODEL"], "editor-model")

    def test_reads_existing_key_without_prompting(self) -> None:
        with patch.dict(os.environ, {"BYTEDANCE_API_KEY": "from-env"}, clear=True):
            self.assertEqual(read_api_key(), "from-env")


if __name__ == "__main__":
    unittest.main()
