import unittest

from textskill_optimizer.edits import (
    END_TARGET,
    SLOW_START,
    apply_atomic_edits,
    merge_and_rank_atomic_edits,
    set_slow_update,
)
from textskill_optimizer.models import AtomicEdit, EditProposal


class AtomicEditTests(unittest.TestCase):
    def test_applies_add_replace_and_delete(self) -> None:
        skill = "# Skill\n\nKeep this.\nRemove this.\n"
        edits = [
            AtomicEdit("replace", "Keep this.", "Preserve verified behavior."),
            AtomicEdit("delete", "Remove this."),
            AtomicEdit("add", END_TARGET, "Run the verifier."),
        ]

        updated = apply_atomic_edits(skill, edits)

        self.assertIn("Preserve verified behavior.", updated)
        self.assertNotIn("Remove this.", updated)
        self.assertTrue(updated.endswith("Run the verifier.\n"))

    def test_rejects_ambiguous_target_and_protected_slow_update(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly once"):
            apply_atomic_edits("same\nsame\n", [AtomicEdit("delete", "same")])

        skill = set_slow_update("# Skill\n", "Longitudinal guidance")
        with self.assertRaisesRegex(ValueError, "protected"):
            apply_atomic_edits(skill, [AtomicEdit("replace", "Longitudinal guidance", "Changed")])
        self.assertIn(SLOW_START, skill)

    def test_merge_ranks_support_then_priority_and_clips_budget(self) -> None:
        shared = AtomicEdit("add", END_TARGET, "Shared rule", priority=0.2)
        proposals = [
            EditProposal("one", edits=(shared,), rationale="one"),
            EditProposal("two", edits=(shared,), rationale="two"),
            EditProposal(
                "three",
                edits=(AtomicEdit("add", END_TARGET, "High priority", priority=1.0),),
                rationale="three",
            ),
        ]

        merged = merge_and_rank_atomic_edits(proposals, budget=1)

        self.assertEqual([edit.content for edit in merged.selected], ["Shared rule"])
        self.assertEqual(merged.ranked[0].support, 2)
        self.assertEqual(merged.duplicate_count, 1)

    def test_merge_drops_conflicting_target_edits(self) -> None:
        proposals = [
            EditProposal(
                "replace",
                edits=(AtomicEdit("replace", "old", "new", priority=2),),
                rationale="replace",
            ),
            EditProposal(
                "delete",
                edits=(AtomicEdit("delete", "old", priority=1),),
                rationale="delete",
            ),
        ]

        merged = merge_and_rank_atomic_edits(proposals, budget=4)

        self.assertEqual(len(merged.selected), 1)
        self.assertEqual(merged.selected[0].operation, "replace")
        self.assertEqual(merged.conflict_count, 1)


if __name__ == "__main__":
    unittest.main()
