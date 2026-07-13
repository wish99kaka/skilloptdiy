import unittest
import tempfile
from pathlib import Path

from textskill_optimizer.models import EditProposal, Score, Task, TaskOutput, TaskResult
from textskill_optimizer.optimizer import OptimizerConfig, SkillOptimizer
from textskill_optimizer.plugins.extraction import (
    AliasMiningEditor,
    ExtractionRunner,
    JsonFieldScorer,
    parse_aliases,
)


INITIAL_SKILL = """# Contact Extraction Skill

## Field Aliases
- name: aliases=name, full name
- email: aliases=email
- company: aliases=company
"""


class OptimizerTests(unittest.TestCase):
    def test_accepts_validation_improving_skill_edit(self) -> None:
        train_tasks = [
            Task(
                id="train-1",
                input="Name: Ada Lovelace; E-mail: ada@example.com; Company: Engines",
                expected={
                    "name": "Ada Lovelace",
                    "email": "ada@example.com",
                    "company": "Engines",
                },
            ),
            Task(
                id="train-2",
                input="Full name: Grace Hopper; Email: grace@example.com; Org: Navy",
                expected={
                    "name": "Grace Hopper",
                    "email": "grace@example.com",
                    "company": "Navy",
                },
            ),
        ]
        valid_tasks = [
            Task(
                id="valid-1",
                input="Name: Alan Turing; E-mail: alan@example.com; Company: Bletchley",
                expected={
                    "name": "Alan Turing",
                    "email": "alan@example.com",
                    "company": "Bletchley",
                },
            ),
            Task(
                id="valid-2",
                input="Full name: Katherine Johnson; Email: kj@example.com; Org: NASA",
                expected={
                    "name": "Katherine Johnson",
                    "email": "kj@example.com",
                    "company": "NASA",
                },
            ),
        ]

        optimizer = SkillOptimizer(
            runner=ExtractionRunner(),
            scorer=JsonFieldScorer(),
            editor=AliasMiningEditor(),
            config=OptimizerConfig(epochs=2),
        )
        result = optimizer.optimize(INITIAL_SKILL, train_tasks, valid_tasks)

        self.assertEqual(result.best_validation_score, 1.0)
        aliases = parse_aliases(result.best_skill_text)
        self.assertIn("e-mail", aliases["email"])
        self.assertIn("org", aliases["company"])
        self.assertTrue(any(item.accepted for item in result.history[1:]))

    def test_rejects_non_improving_skill_edit(self) -> None:
        class StaticRunner:
            def run(self, skill_text: str, task: Task) -> TaskOutput:
                return TaskOutput({"answer": skill_text.strip()})

        class ExactScorer:
            def score(self, task: Task, output: TaskOutput) -> Score:
                success = output.value == task.expected
                return Score(1.0 if success else 0.0, success)

        class BadEditor:
            def propose(
                self,
                skill_text: str,
                train_results: list[TaskResult],
                *,
                epoch: int,
            ) -> list[EditProposal]:
                return [
                    EditProposal(
                        name="worse",
                        skill_text="wrong",
                        rationale="Deliberately worse edit.",
                    )
                ]

        tasks = [
            Task(id="one", input="", expected={"answer": "right"}),
        ]
        optimizer = SkillOptimizer(
            runner=StaticRunner(),
            scorer=ExactScorer(),
            editor=BadEditor(),
            config=OptimizerConfig(epochs=1),
        )
        result = optimizer.optimize("right", tasks, tasks)

        self.assertEqual(result.best_skill_text, "right")
        self.assertEqual(result.best_validation_score, 1.0)
        self.assertFalse(result.history[-1].accepted)

    def test_marks_only_selected_candidate_as_accepted(self) -> None:
        class StaticRunner:
            def run(self, skill_text: str, task: Task) -> TaskOutput:
                return TaskOutput(skill_text.strip())

        class MappedScorer:
            def score(self, task: Task, output: TaskOutput) -> Score:
                scores = {"start": 0.1, "candidate-a": 0.8, "candidate-b": 1.0}
                value = scores[str(output.value)]
                return Score(value, value == 1.0)

        class TwoCandidateEditor:
            def propose(
                self,
                skill_text: str,
                train_results: list[TaskResult],
                *,
                epoch: int,
            ) -> list[EditProposal]:
                return [
                    EditProposal("candidate-a", "candidate-a", "Better than baseline."),
                    EditProposal("candidate-b", "candidate-b", "Best candidate."),
                ]

        tasks = [Task(id="one", input="")]
        optimizer = SkillOptimizer(
            runner=StaticRunner(),
            scorer=MappedScorer(),
            editor=TwoCandidateEditor(),
            config=OptimizerConfig(epochs=1),
        )

        result = optimizer.optimize("start", tasks, tasks)
        epoch_items = [item for item in result.history if item.epoch == 1]

        self.assertEqual(result.best_skill_text, "candidate-b")
        self.assertEqual([item.accepted for item in epoch_items], [False, True])

    def test_rejects_candidate_that_exceeds_learning_rate_budget(self) -> None:
        class StaticRunner:
            def run(self, skill_text: str, task: Task) -> TaskOutput:
                return TaskOutput(skill_text)

        class AlwaysPassScorer:
            def score(self, task: Task, output: TaskOutput) -> Score:
                return Score(1.0, True)

        class LargeEditor:
            def propose(
                self,
                skill_text: str,
                train_results: list[TaskResult],
                *,
                epoch: int,
                rejected_buffer=None,
                meta_skill: str = "",
                optimizer_controls=None,
            ) -> list[EditProposal]:
                return [
                    EditProposal(
                        name="too-large",
                        skill_text=skill_text + "\n" + ("x" * 50),
                        rationale="Oversized rewrite.",
                    )
                ]

        tasks = [Task(id="one", input="")]
        optimizer = SkillOptimizer(
            runner=StaticRunner(),
            scorer=AlwaysPassScorer(),
            editor=LargeEditor(),
            config=OptimizerConfig(epochs=1, max_skill_delta_chars=10),
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = optimizer.optimize("start", tasks, tasks, run_dir=tmp)
            rejected_path = Path(tmp) / "rejected_buffer.jsonl"

            self.assertTrue(rejected_path.exists())
            self.assertIn("learning_rate_exceeded", rejected_path.read_text(encoding="utf-8"))

        self.assertEqual(result.best_skill_text, "start")
        self.assertEqual(result.rejected_buffer[0].reason, "learning_rate_exceeded")
        self.assertIsNone(result.history[-1].validation_score)

    def test_passes_rejected_buffer_and_meta_skill_to_next_epoch(self) -> None:
        class StaticRunner:
            def run(self, skill_text: str, task: Task) -> TaskOutput:
                return TaskOutput(skill_text.strip())

        class ExactScorer:
            def score(self, task: Task, output: TaskOutput) -> Score:
                success = output.value == task.expected
                return Score(1.0 if success else 0.0, success)

        class CapturingEditor:
            def __init__(self) -> None:
                self.calls = []

            def propose(
                self,
                skill_text: str,
                train_results: list[TaskResult],
                *,
                epoch: int,
                rejected_buffer=None,
                meta_skill: str = "",
                optimizer_controls=None,
            ) -> list[EditProposal]:
                self.calls.append(
                    {
                        "epoch": epoch,
                        "rejected_buffer": list(rejected_buffer or []),
                        "meta_skill": meta_skill,
                        "optimizer_controls": dict(optimizer_controls or {}),
                    }
                )
                return [EditProposal("bad", "wrong", "Not enough evidence.")]

        tasks = [Task(id="one", input="", expected="right")]
        editor = CapturingEditor()
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "meta.md"
            meta_path.write_text("Write executable rules.", encoding="utf-8")
            optimizer = SkillOptimizer(
                runner=StaticRunner(),
                scorer=ExactScorer(),
                editor=editor,
                config=OptimizerConfig(epochs=2, meta_skill_path=meta_path),
            )

            optimizer.optimize("right", tasks, tasks, run_dir=tmp)

        self.assertEqual(editor.calls[0]["rejected_buffer"], [])
        self.assertEqual(editor.calls[1]["rejected_buffer"][0]["reason"], "validation_not_improved")
        self.assertEqual(editor.calls[1]["meta_skill"], "Write executable rules.")
        self.assertIn("max_skill_delta_chars", editor.calls[1]["optimizer_controls"])


if __name__ == "__main__":
    unittest.main()
