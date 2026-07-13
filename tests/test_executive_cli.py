import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class ExecutiveCliTests(unittest.TestCase):
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
