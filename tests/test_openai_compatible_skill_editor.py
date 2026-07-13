import importlib.util
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "examples/coding/openai_compatible_skill_editor.py"


def load_editor_module():
    if str(MODULE_PATH.parent) not in sys.path:
        sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("openai_compatible_skill_editor", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OpenAICompatibleSkillEditorTests(unittest.TestCase):
    def test_build_chat_request_payload_uses_json_mode_by_default(self) -> None:
        module = load_editor_module()
        with patch.dict(os.environ, {}, clear=True):
            payload = module.build_chat_request_payload(
                {
                    "epoch": 1,
                    "skill_text": "# Skill",
                    "train_results": [
                        {"task": {"id": "t1"}, "score": {"success": False}},
                    ],
                },
                model="external-model",
            )

        self.assertEqual(payload["model"], "external-model")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["messages"][0]["role"], "system")
        user_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user_payload["failed_training_results"][0]["task"]["id"], "t1")

    def test_build_chat_request_payload_can_disable_json_mode(self) -> None:
        module = load_editor_module()
        with patch.dict(os.environ, {"EXTERNAL_LLM_JSON_MODE": "0"}, clear=True):
            payload = module.build_chat_request_payload({}, model="external-model")

        self.assertNotIn("response_format", payload)

    def test_system_prompt_is_evidence_driven_and_benchmark_agnostic(self) -> None:
        module = load_editor_module()

        self.assertIn("justified by evidence", module.SYSTEM_PROMPT)
        self.assertIn("Compare failures with successes", module.SYSTEM_PROMPT)
        self.assertIn("Do not inject domain knowledge", module.SYSTEM_PROMPT)
        self.assertIn("add | delete | replace", module.SYSTEM_PROMPT)
        self.assertIn("atomic_edit_budget", module.SYSTEM_PROMPT)
        self.assertIn("no more than 120 words", module.SYSTEM_PROMPT)
        self.assertIn("contract_rejection_evidence", module.SYSTEM_PROMPT)
        self.assertIn("metadata.targeted_contracts", module.SYSTEM_PROMPT)
        self.assertIn("metadata.protected_contracts", module.SYSTEM_PROMPT)
        self.assertIn("metadata.cooldown_override", module.SYSTEM_PROMPT)
        self.assertIn("anti_regression_contracts", module.SYSTEM_PROMPT)
        self.assertIn("protected_priority_contracts", module.SYSTEM_PROMPT)
        self.assertIn("cooldown_contracts", module.SYSTEM_PROMPT)
        self.assertIn("target agent reads only edit.content", module.SYSTEM_PROMPT)
        self.assertIn("metadata alone does not protect behavior", module.SYSTEM_PROMPT)
        self.assertIn("invalid-input guard", module.SYSTEM_PROMPT)
        self.assertIn("preserve invalid input rejection rules", module.SYSTEM_PROMPT)
        self.assertIn("raising for negative totals", module.SYSTEM_PROMPT)
        self.assertIn("largest-remainder proportional allocation evidence", module.SYSTEM_PROMPT)
        self.assertIn("zero-sum weights", module.SYSTEM_PROMPT)
        self.assertIn("break ties by original index", module.SYSTEM_PROMPT)
        self.assertIn("only ask the agent to run public tests", module.SYSTEM_PROMPT)
        self.assertIn('exactly the literal string "contract_rejection_evidence"', module.SYSTEM_PROMPT)
        for leaked_rule in ("casefold", "ROUND_HALF_UP", "nested path", "reversed bounds"):
            self.assertNotIn(leaked_rule, module.SYSTEM_PROMPT)

    def test_reflection_payload_surfaces_contract_rejection_evidence(self) -> None:
        module = load_editor_module()
        with patch.dict(os.environ, {}, clear=True):
            payload = module.build_chat_request_payload(
                {
                    "epoch": 2,
                    "skill_text": "# Skill",
                    "rejected_buffer": [
                        {
                            "candidate": "bad",
                            "reason": "validation_gate_rejected",
                            "validation_score": 0.5,
                            "metadata": {
                                "validation_gate": {
                                    "current_mean": 0.8,
                                    "candidate_mean": 0.5,
                                    "contract_evidence": {
                                        "top_negative_contracts": [
                                            {
                                                "contract": "stable_order",
                                                "current_accuracy": 1.0,
                                                "candidate_accuracy": 0.0,
                                                "delta": -1.0,
                                            }
                                        ],
                                        "top_no_improvement_contracts": [
                                            {
                                                "contract": "input_validation",
                                                "current_accuracy": 0.0,
                                                "candidate_accuracy": 0.0,
                                                "delta": 0.0,
                                            }
                                        ],
                                    },
                                }
                            },
                        }
                    ],
                    "train_results": [],
                },
                model="external-model",
            )

        user_payload = json.loads(payload["messages"][1]["content"])
        evidence = user_payload["contract_rejection_evidence"]
        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["priority_contracts"][0]["contract"], "stable_order")
        self.assertEqual(
            evidence["proposal_policy"]["anti_regression_contracts"][0]["contract"],
            "stable_order",
        )
        self.assertEqual(
            evidence["recent_rejections"][0]["blocking_contracts"][0]["kind"],
            "negative_delta",
        )
        self.assertNotIn("candidate_report", json.dumps(evidence))

    def test_contract_rejection_evidence_is_unavailable_for_old_rejections(self) -> None:
        module = load_editor_module()

        evidence = module.build_contract_rejection_evidence(
            [{"candidate": "old", "reason": "selection_not_improved"}]
        )

        self.assertFalse(evidence["available"])
        self.assertEqual(evidence["priority_contracts"], [])
        self.assertEqual(evidence["recent_rejections"], [])

    def test_main_returns_external_proposal_without_wrapper_hardening(self) -> None:
        module = load_editor_module()
        proposal = {
            "proposals": [
                {
                    "name": "evidence-based",
                    "skill_text": "# Skill\nInspect the documented contract before changing code.",
                    "rationale": "The failed trajectories stopped after satisfying one example.",
                }
            ]
        }
        output = io.StringIO()
        with (
            patch.object(module, "load_optimizer_payload_from_stdin", return_value={}),
            patch.object(module, "build_chat_request_from_env", return_value=({}, "url", "key", 1)),
            patch.object(module, "call_chat_completions", return_value={}),
            patch.object(module, "extract_chat_message_content", return_value=json.dumps(proposal)),
            redirect_stdout(output),
        ):
            returncode = module.main()

        self.assertEqual(returncode, 0)
        self.assertEqual(json.loads(output.getvalue()), proposal)
        self.assertFalse(hasattr(module, "harden_proposals_payload"))

    def test_main_enforces_contract_evidence_source_when_evidence_is_available(self) -> None:
        module = load_editor_module()
        optimizer_payload = {
            "skill_text": "# Skill",
            "train_results": [],
            "rejected_buffer": [
                {
                    "candidate": "bad",
                    "reason": "validation_gate_rejected",
                    "metadata": {
                        "validation_gate": {
                            "current_mean": 0.5,
                            "candidate_mean": 0.5,
                            "contract_evidence": {
                                "top_no_improvement_contracts": [
                                    {
                                        "contract": "stable_order",
                                        "current_accuracy": 0.5,
                                        "candidate_accuracy": 0.5,
                                        "delta": 0.0,
                                    }
                                ]
                            },
                        }
                    },
                }
            ],
        }
        proposal = {
            "proposals": [
                {
                    "name": "target-stable-order",
                    "metadata": {
                        "targeted_contracts": ["stable_order"],
                        "evidence_source": "trajectory_comparison",
                        "expected_behavior_change": "preserve ordering",
                    },
                    "edits": [
                        {
                            "operation": "add",
                            "target": "__end__",
                            "content": "Preserve documented ordering when repairing code.",
                        }
                    ],
                }
            ]
        }
        output = io.StringIO()
        with (
            patch.object(module, "load_optimizer_payload_from_stdin", return_value=optimizer_payload),
            patch.object(module, "build_chat_request_from_env", return_value=({}, "url", "key", 1)),
            patch.object(module, "call_chat_completions", return_value={}),
            patch.object(module, "extract_chat_message_content", return_value=json.dumps(proposal)),
            redirect_stdout(output),
        ):
            returncode = module.main()

        self.assertEqual(returncode, 0)
        parsed = json.loads(output.getvalue())
        self.assertEqual(
            parsed["proposals"][0]["metadata"]["evidence_source"],
            "contract_rejection_evidence",
        )
        self.assertEqual(
            parsed["proposals"][0]["metadata"]["protected_contracts"],
            ["stable_order"],
        )

    def test_reflect_invalid_model_json_returns_empty_proposals(self) -> None:
        module = load_editor_module()
        output = io.StringIO()
        with (
            patch.object(module, "load_optimizer_payload_from_stdin", return_value={"operation": "reflect"}),
            patch.object(module, "build_chat_request_from_env", return_value=({}, "url", "key", 1)),
            patch.object(module, "call_chat_completions", return_value={}),
            patch.object(
                module,
                "extract_chat_message_content",
                return_value='{"proposals":[{"name":"bad","rationale": missing quotes}]}',
            ),
            redirect_stdout(output),
        ):
            returncode = module.main()

        self.assertEqual(returncode, 0)
        self.assertEqual(json.loads(output.getvalue()), {"proposals": []})

    def test_slow_meta_update_uses_longitudinal_prompt(self) -> None:
        module = load_editor_module()
        with patch.dict(os.environ, {}, clear=True):
            payload = module.build_chat_request_payload(
                {
                    "operation": "slow_meta_update",
                    "epoch": 2,
                    "current_skill_text": "# Skill",
                    "comparison": {"counts": {"regression": 1}},
                },
                model="external-model",
            )

        self.assertEqual(payload["messages"][0]["content"], module.SLOW_META_SYSTEM_PROMPT)
        user_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user_payload["operation"], "slow_meta_update")
        self.assertEqual(user_payload["comparison"]["counts"]["regression"], 1)
        self.assertIn("contract_rejection_evidence", user_payload)

    def test_one_shot_uses_development_context_prompt(self) -> None:
        module = load_editor_module()
        with patch.dict(os.environ, {}, clear=True):
            payload = module.build_chat_request_payload(
                {
                    "operation": "one_shot_skill",
                    "seed_label": "seed-a",
                    "development_context": [{"task_input": "Fix the contract"}],
                },
                model="external-model",
            )

        self.assertEqual(payload["messages"][0]["content"], module.ONE_SHOT_SYSTEM_PROMPT)
        user_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user_payload["operation"], "one_shot_skill")
        self.assertEqual(user_payload["seed_label"], "seed-a")

    def test_build_chat_request_from_env_requires_base_url_and_model(self) -> None:
        module = load_editor_module()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                module.build_chat_request_from_env({})

    def test_normalize_chat_completions_url_accepts_full_endpoint(self) -> None:
        module = load_editor_module()
        url = module.normalize_chat_completions_url(
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        )

        self.assertEqual(
            url,
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        )

    def test_normalize_chat_completions_url_accepts_base_url(self) -> None:
        module = load_editor_module()
        url = module.normalize_chat_completions_url(
            "https://ark.cn-beijing.volces.com/api/v3"
        )

        self.assertEqual(
            url,
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        )

    def test_extract_chat_message_content(self) -> None:
        module = load_editor_module()
        content = module.extract_chat_message_content(
            {"choices": [{"message": {"content": "{\"proposals\": []}"}}]}
        )

        self.assertEqual(json.loads(content), {"proposals": []})

    def test_extract_json_text_from_markdown_fence(self) -> None:
        module = load_editor_module()
        extracted = module.extract_json_text(
            "```json\n{\"proposals\": []}\n```"
        )

        self.assertEqual(json.loads(extracted), {"proposals": []})

    def test_extract_json_text_from_surrounding_text(self) -> None:
        module = load_editor_module()
        extracted = module.extract_json_text(
            "Here is the JSON:\n{\"proposals\": []}\nDone."
        )

        self.assertEqual(json.loads(extracted), {"proposals": []})

    def test_parse_model_json_repairs_raw_newlines_inside_strings(self) -> None:
        module = load_editor_module()

        parsed = module.parse_model_json(
            '{"skill_text":"first line\r\nsecond line\tindented","rationale":"ok"}'
        )

        self.assertEqual(parsed["skill_text"], "first line\r\nsecond line\tindented")

    def test_parse_model_json_preserves_already_escaped_newlines(self) -> None:
        module = load_editor_module()

        parsed = module.parse_model_json('{"skill_text":"first line\\nsecond line"}')

        self.assertEqual(parsed["skill_text"], "first line\nsecond line")

    def test_empty_stdin_returns_actionable_error(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH)],
            text=True,
            input="",
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Expected optimizer JSON payload", completed.stderr)

    def test_dry_run_does_not_call_api(self) -> None:
        env = {
            "EXTERNAL_LLM_BASE_URL": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            "EXTERNAL_LLM_MODEL": "ep-test",
            "EXTERNAL_LLM_DRY_RUN": "1",
        }
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH)],
            text=True,
            input="{\"epoch\": 1, \"skill_text\": \"# Skill\", \"train_results\": []}",
            capture_output=True,
            check=False,
            env=env,
        )

        self.assertEqual(completed.returncode, 0)
        payload = json.loads(completed.stdout)
        self.assertEqual(
            payload["url"],
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        )
        self.assertEqual(payload["model"], "ep-test")


if __name__ == "__main__":
    unittest.main()
