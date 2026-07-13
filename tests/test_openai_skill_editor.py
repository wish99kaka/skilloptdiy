import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "examples/coding/openai_skill_editor.py"


def load_editor_module():
    if str(MODULE_PATH.parent) not in sys.path:
        sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("openai_skill_editor", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OpenAISkillEditorTests(unittest.TestCase):
    def test_build_request_payload_uses_structured_outputs(self) -> None:
        module = load_editor_module()
        payload = module.build_request_payload(
            {
                "epoch": 1,
                "skill_text": "# Skill",
                "train_results": [
                    {
                        "task": {"id": "t1"},
                        "output": {"metadata": {"post_test": {"stderr": "failed"}}},
                        "score": {"success": False, "value": 0.0},
                    }
                ],
            },
            model="test-model",
        )

        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertEqual(payload["text"]["format"]["schema"]["required"], ["proposals"])
        user_text = payload["input"][1]["content"][0]["text"]
        self.assertIn("failed_training_results", user_text)
        self.assertIn("successful_training_results", user_text)

    def test_extract_output_text_supports_direct_field(self) -> None:
        module = load_editor_module()
        result = module.extract_output_text({"output_text": "{\"proposals\": []}"})

        self.assertEqual(json.loads(result), {"proposals": []})

    def test_extract_output_text_supports_output_array(self) -> None:
        module = load_editor_module()
        result = module.extract_output_text(
            {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": "{\"proposals\": []}",
                            }
                        ]
                    }
                ]
            }
        )

        self.assertEqual(json.loads(result), {"proposals": []})


if __name__ == "__main__":
    unittest.main()
