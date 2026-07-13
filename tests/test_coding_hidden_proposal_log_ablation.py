import importlib.util
import tempfile
import unittest
from pathlib import Path

from textskill_optimizer.models import Score, Task, TaskOutput, TaskResult


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "work/run_coding_hidden_proposal_log_ablation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_coding_hidden_proposal_log_ablation", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CodingHiddenProposalLogAblationTests(unittest.TestCase):
    def test_sample_log_can_be_loaded(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            module.write_sample_proposal_log(path, ["seed-a"])

            proposal_log = module.load_proposal_log(path)

        self.assertIn(("seed-a", "gate_lr_rejected", 2), proposal_log)
        self.assertEqual(
            proposal_log[("seed-a", "gate_lr_rejected", 2)][0].name,
            "proposal-log-rejected-buffer-guided-compact-edit-epoch-2",
        )

    def test_replay_editor_returns_epoch_proposals(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            module.write_sample_proposal_log(path, ["seed-a"])
            proposal_log = module.load_proposal_log(path)

        editor = module.ProposalLogEditor(
            proposal_log,
            seed="seed-a",
            case_name="gate_lr_rejected",
        )

        proposals = editor.propose(
            "skill",
            [TaskResult(
                task=Task(id="t1", input="fix"),
                output=TaskOutput(value={}),
                score=Score(0.0, False),
            )],
            epoch=2,
        )

        self.assertEqual(len(proposals), 1)
        self.assertIn("FULL_CODING_HIDDEN_RULES", proposals[0].skill_text)

    def test_replay_editor_replays_even_when_train_has_no_failures(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            module.write_sample_proposal_log(path, ["seed-a"])
            proposal_log = module.load_proposal_log(path)

        editor = module.ProposalLogEditor(
            proposal_log,
            seed="seed-a",
            case_name="gate_lr_rejected",
        )

        proposals = editor.propose(
            "skill",
            [TaskResult(
                task=Task(id="t1", input="fix"),
                output=TaskOutput(value={}),
                score=Score(1.0, True),
            )],
            epoch=2,
        )

        self.assertEqual(len(proposals), 1)
        self.assertIn("FULL_CODING_HIDDEN_RULES", proposals[0].skill_text)

    def test_aggregate_rows_summarizes_multiple_seeds(self) -> None:
        module = load_module()
        rows = [
            {
                "name": "gate_lr_rejected",
                "label": "+LR+Rejected Buffer",
                "validation_score": 1.0,
                "holdout_score": 1.0,
                "first_success_epoch": 2,
                "lr_rejections": 1,
            },
            {
                "name": "gate_lr_rejected",
                "label": "+LR+Rejected Buffer",
                "validation_score": 0.0,
                "holdout_score": 0.0,
                "first_success_epoch": None,
                "lr_rejections": 2,
            },
        ]

        aggregates = module.aggregate_rows(rows)

        self.assertEqual(aggregates[0]["mean_validation_score"], 0.5)
        self.assertEqual(aggregates[0]["successes"], 1)
        self.assertEqual(aggregates[0]["mean_first_success_epoch"], 2.0)


if __name__ == "__main__":
    unittest.main()
