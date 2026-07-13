import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class ExecutiveCliTests(unittest.TestCase):
    def test_cli_fails_fast_for_builtin_full_replacement_editor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "skill.md"
            train = root / "train.jsonl"
            selection = root / "selection.jsonl"
            out = root / "run"
            skill.write_text("# Skill\n", encoding="utf-8")
            task = {"id": "case", "input": "", "expected": {}}
            train.write_text(json.dumps(task) + "\n", encoding="utf-8")
            selection.write_text(json.dumps(task) + "\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "textskill_optimizer.cli",
                    "optimize",
                    "--protocol",
                    "executive",
                    "--plugin",
                    "extraction",
                    "--skill",
                    str(skill),
                    "--train",
                    str(train),
                    "--valid",
                    str(selection),
                    "--out",
                    str(out),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("atomic_edits", completed.stderr)
            self.assertFalse(out.exists())

    def test_cli_probes_command_capability_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "skill.md"
            train = root / "train.jsonl"
            selection = root / "selection.jsonl"
            editor = root / "legacy_editor.py"
            reflect_marker = root / "reflect-called"
            out = root / "run"
            skill.write_text("# Skill\n", encoding="utf-8")
            task = {"id": "case", "input": "", "expected": {}}
            train.write_text(json.dumps(task) + "\n", encoding="utf-8")
            selection.write_text(json.dumps(task) + "\n", encoding="utf-8")
            editor.write_text(
                textwrap.dedent(
                    f"""
                    import json
                    import pathlib
                    import sys

                    payload = json.load(sys.stdin)
                    if payload.get("operation") == "capabilities":
                        print(json.dumps({{"capabilities": ["full_skill_replacement"]}}))
                    else:
                        pathlib.Path({str(reflect_marker)!r}).write_text("called", encoding="utf-8")
                        print(json.dumps({{
                            "proposals": [{{
                                "name": "whole-document",
                                "skill_text": "# Replacement\\n"
                            }}]
                        }}))
                    """
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "textskill_optimizer.cli",
                    "optimize",
                    "--protocol",
                    "executive",
                    "--plugin",
                    "extraction",
                    "--skill",
                    str(skill),
                    "--train",
                    str(train),
                    "--valid",
                    str(selection),
                    "--editor-command",
                    f"{sys.executable} {editor}",
                    "--out",
                    str(out),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("atomic_edits", completed.stderr)
            self.assertFalse(reflect_marker.exists())
            self.assertFalse(out.exists())

    def test_cli_runs_atomic_external_editor_through_selection_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "skill.md"
            train = root / "train.jsonl"
            selection = root / "selection.jsonl"
            editor = root / "editor.py"
            out = root / "run"
            skill.write_text(
                "# Skill\n- email: aliases=email\n",
                encoding="utf-8",
            )
            task = {
                "id": "email-alias",
                "input": "E-mail: ada@example.com",
                "expected": {"email": "ada@example.com"},
            }
            train.write_text(json.dumps({**task, "id": "train"}) + "\n", encoding="utf-8")
            selection.write_text(json.dumps({**task, "id": "selection"}) + "\n", encoding="utf-8")
            editor.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys

                    payload = json.load(sys.stdin)
                    if payload.get("operation") == "capabilities":
                        print(json.dumps({"capabilities": ["atomic_edits"]}))
                    else:
                        print(json.dumps({
                            "proposals": [{
                                "name": "email-alias",
                                "rationale": "The failed label is absent from the current aliases.",
                                "edits": [{
                                    "operation": "replace",
                                    "target": "- email: aliases=email",
                                    "content": "- email: aliases=email, e-mail",
                                    "priority": 1.0
                                }]
                            }]
                        }))
                    """
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "textskill_optimizer.cli",
                    "optimize",
                    "--protocol",
                    "executive",
                    "--plugin",
                    "extraction",
                    "--skill",
                    str(skill),
                    "--train",
                    str(train),
                    "--valid",
                    str(selection),
                    "--epochs",
                    "1",
                    "--rollout-batch-size",
                    "1",
                    "--reflection-minibatch-size",
                    "1",
                    "--text-learning-rate",
                    "1",
                    "--text-learning-rate-floor",
                    "1",
                    "--disable-slow-update",
                    "--validation-confirmation-rounds",
                    "2",
                    "--validation-required-wins",
                    "2",
                    "--validation-mean-delta",
                    "0.05",
                    "--editor-command",
                    f"{sys.executable} {editor}",
                    "--out",
                    str(out),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("average_score=1.0000", completed.stdout)
            self.assertIn("e-mail", (out / "best_skill.md").read_text(encoding="utf-8"))
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["accepted_steps"], 1)
            gate = result["history"][1]["metadata"]["validation_gate"]
            self.assertEqual(gate["wins"], 3)
            self.assertEqual(gate["total_rounds"], 3)


if __name__ == "__main__":
    unittest.main()
