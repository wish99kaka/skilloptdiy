import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from textskill_optimizer.command_editor import (
    CommandEditorConfig,
    CommandSkillEditor,
    parse_proposals,
    truncate_value,
)
from textskill_optimizer.models import Score, Task, TaskOutput, TaskResult


class CommandEditorTests(unittest.TestCase):
    def test_parse_single_proposal(self) -> None:
        proposals = parse_proposals(
            {
                "name": "one",
                "skill_text": "# Skill\n",
                "rationale": "Because tests failed.",
            }
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].name, "one")

    def test_parse_atomic_proposal_materializes_current_skill(self) -> None:
        proposals = parse_proposals(
            {
                "name": "atomic",
                "rationale": "Repeated evidence.",
                "edits": [
                    {
                        "operation": "add",
                        "target": "__end__",
                        "content": "Run the verifier.",
                        "priority": 0.8,
                    }
                ],
            },
            current_skill="# Skill\n",
        )

        self.assertEqual(len(proposals[0].edits), 1)
        self.assertIn("Run the verifier.", proposals[0].skill_text)

    def test_truncates_large_payload_strings(self) -> None:
        payload = truncate_value({"diff": "x" * 20}, 5)

        self.assertEqual(payload, {"diff": "xxxxx\n...[truncated]"})

    def test_command_skill_editor_reads_stdout_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "editor.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys

                    payload = json.load(sys.stdin)
                    assert payload["meta_skill"] == "Meta guidance"
                    assert payload["rejected_buffer"][0]["reason"] == "validation_not_improved"
                    assert payload["optimizer_controls"]["max_added_bullet_lines"] == 2
                    print(json.dumps({
                        "proposals": [{
                            "name": "external",
                            "skill_text": payload["skill_text"] + "\\nmore",
                            "rationale": "External edit"
                        }]
                    }))
                    """
                ),
                encoding="utf-8",
            )
            editor = CommandSkillEditor(
                CommandEditorConfig(command=f"{sys.executable} {script}")
            )

            proposals = editor.propose(
                "# Skill",
                [
                    TaskResult(
                        task=Task(id="t1", input="fix"),
                        output=TaskOutput(value={"tests_passed": False}),
                        score=Score(0.0, False, "failed"),
                    )
                ],
                epoch=1,
                rejected_buffer=[{"reason": "validation_not_improved"}],
                meta_skill="Meta guidance",
                optimizer_controls={"max_added_bullet_lines": 2},
            )

        self.assertEqual(proposals[0].skill_text, "# Skill\nmore")

    def test_command_skill_editor_writes_proposal_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "editor.py"
            log_path = Path(tmp) / "proposal_logs" / "external.jsonl"
            script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys

                    payload = json.load(sys.stdin)
                    print(json.dumps({
                        "proposals": [{
                            "name": "external",
                            "skill_text": payload["skill_text"] + "\\nmore",
                            "rationale": "External edit",
                            "metadata": {"source": "test"}
                        }]
                    }))
                    """
                ),
                encoding="utf-8",
            )
            editor = CommandSkillEditor(
                CommandEditorConfig(
                    command=f"{sys.executable} {script}",
                    proposal_log_path=log_path,
                    proposal_log_seed="seed-a",
                    proposal_log_case="gate_lr",
                )
            )

            editor.propose(
                "# Skill",
                [
                    TaskResult(
                        task=Task(id="t1", input="fix"),
                        output=TaskOutput(value={"tests_passed": False}),
                        score=Score(0.0, False, "failed"),
                    )
                ],
                epoch=2,
                rejected_buffer=[{"reason": "learning_rate_exceeded"}],
                meta_skill="Meta guidance",
                optimizer_controls={"max_added_bullet_lines": 1},
            )

            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["seed"], "seed-a")
        self.assertEqual(records[0]["case"], "gate_lr")
        self.assertEqual(records[0]["epoch"], 2)
        self.assertEqual(records[0]["failed_train_task_ids"], ["t1"])
        self.assertEqual(records[0]["rejected_count"], 1)
        self.assertFalse(records[0]["contract_rejection_evidence"]["available"])
        self.assertFalse(records[0]["proposal_targeting_audit"]["required"])
        self.assertTrue(records[0]["meta_skill_present"])
        self.assertEqual(records[0]["proposals"][0]["metadata"], {"source": "test"})

    def test_proposal_log_audits_contract_targeting_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "editor.py"
            log_path = Path(tmp) / "proposal_logs" / "external.jsonl"
            script.write_text(
                textwrap.dedent(
                    """
                    import json

                    print(json.dumps({
                        "proposals": [{
                            "name": "generic",
                            "skill_text": "# Skill\\nCheck every contract.",
                            "rationale": "Generic edit"
                        }]
                    }))
                    """
                ),
                encoding="utf-8",
            )
            editor = CommandSkillEditor(
                CommandEditorConfig(
                    command=f"{sys.executable} {script}",
                    proposal_log_path=log_path,
                )
            )

            editor.propose(
                "# Skill",
                [],
                epoch=1,
                rejected_buffer=[contract_rejected_buffer_item()],
            )
            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertTrue(record["contract_rejection_evidence"]["available"])
        self.assertTrue(record["proposal_targeting_audit"]["required"])
        self.assertEqual(record["proposal_targeting_audit"]["missing_targeted_contract_count"], 1)
        self.assertIn(
            "missing_targeted_contracts",
            record["proposal_targeting_audit"]["proposals"][0]["issues"],
        )

    def test_reflection_timeout_returns_empty_proposals_and_logs_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "slow.py"
            log_path = Path(tmp) / "proposal_logs" / "external.jsonl"
            ledger_path = Path(tmp) / "usage.jsonl"
            script.write_text(
                textwrap.dedent(
                    """
                    import time

                    time.sleep(1)
                    """
                ),
                encoding="utf-8",
            )
            editor = CommandSkillEditor(
                CommandEditorConfig(
                    command=f"{sys.executable} {script}",
                    timeout_seconds=0.01,
                    proposal_log_path=log_path,
                    usage_ledger_path=ledger_path,
                )
            )

            proposals = editor.propose("# Skill", [], epoch=1)
            proposal_record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
            usage_record = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(proposals, [])
        self.assertEqual(proposal_record["proposals"], [])
        self.assertEqual(usage_record["returncode"], 124)
        self.assertTrue(usage_record["timed_out"])

    def test_command_skill_editor_requests_slow_meta_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "editor.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys

                    payload = json.load(sys.stdin)
                    assert payload["operation"] == "slow_meta_update"
                    assert payload["comparison"]["counts"]["improvement"] == 2
                    print(json.dumps({
                        "meta_skill": "Keep verified directions.",
                        "slow_update": "Verify the full contract.",
                        "rationale": "Two tasks improved."
                    }))
                    """
                ),
                encoding="utf-8",
            )
            editor = CommandSkillEditor(CommandEditorConfig(command=f"{sys.executable} {script}"))

            update = editor.update_state(
                epoch=1,
                current_skill="# Skill",
                meta_skill="",
                comparison={"counts": {"improvement": 2}},
                rejected_buffer=[],
                optimizer_controls={"phase": "slow_meta_update"},
            )

        self.assertEqual(update.meta_skill, "Keep verified directions.")
        self.assertEqual(update.slow_update, "Verify the full contract.")

    def test_rejects_non_json_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "bad.py"
            script.write_text("print('not json')\n", encoding="utf-8")
            editor = CommandSkillEditor(
                CommandEditorConfig(command=f"{sys.executable} {script}")
            )

            with self.assertRaises(ValueError):
                editor.propose("# Skill", [], epoch=1)

def contract_rejected_buffer_item() -> dict:
    return {
        "candidate": "bad",
        "reason": "validation_gate_rejected",
        "metadata": {
            "validation_gate": {
                "current_mean": 1.0,
                "candidate_mean": 0.0,
                "contract_evidence": {
                    "top_negative_contracts": [
                        {
                            "contract": "stable_order",
                            "current_accuracy": 1.0,
                            "candidate_accuracy": 0.0,
                            "delta": -1.0,
                        }
                    ],
                    "top_no_improvement_contracts": [],
                },
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
