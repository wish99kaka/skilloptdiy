import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from textskill_optimizer.command_editor import CommandEditorConfig, CommandSkillEditor
from textskill_optimizer.models import Score, Task, TaskOutput, TaskResult
from textskill_optimizer.plugins.coding import CodingRunner
from textskill_optimizer.usage_ledger import (
    append_usage_event,
    read_usage_events,
    summarize_usage_file,
)


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples/coding"))
import openai_compatible_skill_editor  # noqa: E402


class UsageLedgerTests(unittest.TestCase):
    def test_summarizes_actual_and_estimated_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "usage.jsonl"

            append_usage_event(
                ledger,
                {
                    "kind": "optimizer_model_api",
                    "operation": "reflect",
                    "duration_seconds": 1.5,
                    "actual_prompt_tokens": 10,
                    "actual_completion_tokens": 5,
                    "actual_total_tokens": 15,
                },
            )
            append_usage_event(
                ledger,
                {
                    "kind": "target_agent_cli",
                    "operation": "run_task",
                    "estimated_prompt_tokens": 20,
                    "estimated_completion_tokens": 3,
                },
            )

            summary = summarize_usage_file(ledger)

        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["actual_total_tokens"], 15)
        self.assertEqual(summary["estimated_total_tokens"], 23)
        self.assertEqual(summary["by_kind"]["optimizer_model_api"]["calls"], 1)
        self.assertEqual(summary["by_kind"]["target_agent_cli"]["calls"], 1)

    def test_summary_can_filter_by_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "usage.jsonl"

            append_usage_event(
                ledger,
                {
                    "kind": "optimizer_model_api",
                    "operation": "reflect",
                    "actual_total_tokens": 10,
                },
            )
            append_usage_event(
                ledger,
                {
                    "kind": "target_agent_cli",
                    "operation": "run_task",
                    "estimated_prompt_tokens": 40,
                },
            )

            summary = summarize_usage_file(
                ledger,
                include_kinds=("optimizer_model_api",),
            )

        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["actual_total_tokens"], 10)
        self.assertEqual(summary["estimated_total_tokens"], 0)
        self.assertNotIn("target_agent_cli", summary["by_kind"])

    def test_command_editor_writes_usage_ledger_and_passes_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "editor.py"
            ledger = Path(tmp) / "usage.jsonl"
            script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import os
                    import sys

                    payload = json.load(sys.stdin)
                    assert os.environ["TEXTSKILL_USAGE_LEDGER_PATH"].endswith("usage.jsonl")
                    context = json.loads(os.environ["TEXTSKILL_USAGE_CONTEXT_JSON"])
                    assert context["seed"] == "seed-a"
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
                CommandEditorConfig(
                    command=f"{sys.executable} {script}",
                    usage_ledger_path=ledger,
                    usage_context={"seed": "seed-a", "condition": "executive"},
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
                epoch=1,
            )
            events = read_usage_events(ledger)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "optimizer_command")
        self.assertEqual(events[0]["operation"], "reflect")
        self.assertEqual(events[0]["context"]["seed"], "seed-a")
        self.assertGreater(events[0]["estimated_prompt_tokens"], 0)

    def test_external_editor_records_actual_api_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "usage.jsonl"
            with patch.dict(
                os.environ,
                {
                    "TEXTSKILL_USAGE_LEDGER_PATH": str(ledger),
                    "TEXTSKILL_USAGE_CONTEXT_JSON": json.dumps({"seed": "seed-a"}),
                },
            ):
                openai_compatible_skill_editor.record_model_api_usage(
                    {
                        "model": "editor-model",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    {
                        "usage": {
                            "prompt_tokens": 7,
                            "completion_tokens": 3,
                            "total_tokens": 10,
                        }
                    },
                    operation="reflect",
                    content='{"proposals":[]}',
                    url="https://example.test/v1/chat/completions",
                    duration_seconds=0.5,
                )
            events = read_usage_events(ledger)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "optimizer_model_api")
        self.assertEqual(events[0]["actual_total_tokens"], 10)
        self.assertEqual(events[0]["context"]["seed"], "seed-a")

    def test_coding_runner_records_estimated_target_agent_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "usage.jsonl"
            runner = CodingRunner(
                usage_ledger_path=ledger,
                usage_context={"seed": "seed-a", "condition": "executive"},
            )

            runner.record_agent_usage(
                task=Task(
                    id="task-1",
                    input="Fix",
                    metadata={"benchmark_family": "headers"},
                ),
                command="agent",
                agent_result={
                    "returncode": 0,
                    "timed_out": False,
                    "duration_seconds": 2.0,
                    "stdout": "done",
                    "stderr": "",
                },
                skill_text="# Skill",
                task_payload_text='{"id":"task-1"}',
                repo_snapshot_chars=100,
                diff="patch",
            )
            events = read_usage_events(ledger)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "target_agent_cli")
        self.assertEqual(events[0]["benchmark_family"], "headers")
        self.assertGreater(events[0]["estimated_prompt_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
