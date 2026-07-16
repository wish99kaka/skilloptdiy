import unittest

from textskill_optimizer.paper import PaperEdit, PaperEditOperation
from textskill_optimizer.paper.patches import (
    SLOW_UPDATE_END,
    SLOW_UPDATE_START,
    apply_paper_patch,
    read_slow_update_field,
    write_slow_update_field,
)


class PaperPatchApplicationTests(unittest.TestCase):
    def test_epoch_writer_is_the_only_path_that_replaces_protected_guidance(self) -> None:
        initial = write_slow_update_field("# Skill\n", "first guidance")
        updated = write_slow_update_field(initial, "second guidance")

        self.assertEqual(updated.count(SLOW_UPDATE_START), 1)
        self.assertEqual(updated.count(SLOW_UPDATE_END), 1)
        self.assertEqual(read_slow_update_field(updated), "second guidance")
        self.assertNotIn("first guidance", updated)
        with self.assertRaisesRegex(ValueError, "exactly one complete"):
            write_slow_update_field(updated + SLOW_UPDATE_START, "bad")

    def test_all_four_operations_apply_sequentially_with_one_report_each(self) -> None:
        initial = "# Skill\n\n## Rules\n\n- old\n- remove me\n"
        edits = (
            PaperEdit("replace", PaperEditOperation.REPLACE, "- old", "- improved"),
            PaperEdit("insert", PaperEditOperation.INSERT_AFTER, "## Rules", "Intro"),
            PaperEdit("delete", PaperEditOperation.DELETE, "- remove me"),
            PaperEdit("append", PaperEditOperation.APPEND, content="## Tail"),
        )

        result = apply_paper_patch(initial, edits)

        self.assertEqual(
            result.output_skill,
            "# Skill\n\n## Rules\n\nIntro\n\n- improved\n\n## Tail\n",
        )
        self.assertEqual(
            [report.status for report in result.reports],
            [
                "applied_replace",
                "applied_insert_after",
                "applied_delete",
                "applied_append",
            ],
        )
        self.assertEqual(len(result.reports), len(edits))
        self.assertEqual(result.reports[0].before_sha256, result.input_sha256)
        self.assertEqual(result.reports[-1].after_sha256, result.output_sha256)
        self.assertTrue(
            all(
                left.after_sha256 == right.before_sha256
                for left, right in zip(result.reports, result.reports[1:])
            )
        )

    def test_step_edits_preserve_the_protected_slow_update_region(self) -> None:
        protected = (
            f"{SLOW_UPDATE_START}\nprotected guidance\n{SLOW_UPDATE_END}\n"
        )
        initial = "# Skill\n\n## Rules\n\n- safe\n\n" + protected
        edits = (
            PaperEdit(
                "replace-protected",
                PaperEditOperation.REPLACE,
                "protected guidance",
                "changed",
            ),
            PaperEdit(
                "append-markers",
                PaperEditOperation.APPEND,
                content=f"{SLOW_UPDATE_START}\n- general rule\n{SLOW_UPDATE_END}",
            ),
            PaperEdit(
                "insert-fallback",
                PaperEditOperation.INSERT_AFTER,
                "missing target",
                "- fallback rule",
            ),
            PaperEdit(
                "delete-marker",
                PaperEditOperation.DELETE,
                SLOW_UPDATE_END,
            ),
        )

        result = apply_paper_patch(initial, edits)

        self.assertEqual(result.output_skill.count(SLOW_UPDATE_START), 1)
        self.assertEqual(result.output_skill.count(SLOW_UPDATE_END), 1)
        self.assertTrue(result.output_skill.endswith(protected))
        self.assertIn(
            "- general rule\n\n- fallback rule\n\n" + protected,
            result.output_skill,
        )
        self.assertEqual(
            [report.status for report in result.reports],
            [
                "skipped_protected_region",
                "applied_append_before_protected_region",
                "applied_insert_after_fallback_before_protected_region",
                "skipped_protected_region",
            ],
        )

    def test_target_spanning_into_protected_region_is_rejected(self) -> None:
        initial = (
            "# Skill\n\n- safe\n\n"
            f"{SLOW_UPDATE_START}\nprotected guidance\n{SLOW_UPDATE_END}\n"
        )
        crossing_target = f"- safe\n\n{SLOW_UPDATE_START}\nprotected guidance"

        result = apply_paper_patch(
            initial,
            (
                PaperEdit(
                    "crossing-delete",
                    PaperEditOperation.DELETE,
                    crossing_target,
                ),
            ),
        )

        self.assertEqual(result.output_skill, initial)
        self.assertEqual(result.reports[0].status, "skipped_protected_region")

    def test_insert_after_newline_anchor_stays_before_protected_region(self) -> None:
        protected = (
            f"{SLOW_UPDATE_START}\nprotected guidance\n{SLOW_UPDATE_END}\n"
        )
        initial = "# Skill\n" + protected

        result = apply_paper_patch(
            initial,
            (
                PaperEdit(
                    "newline-anchor",
                    PaperEditOperation.INSERT_AFTER,
                    "# Skill\n",
                    "outside guidance",
                ),
            ),
        )

        self.assertEqual(
            result.output_skill,
            "# Skill\n\noutside guidance\n" + protected,
        )
        self.assertEqual(result.reports[0].status, "applied_insert_after")


if __name__ == "__main__":
    unittest.main()
