import unittest

from textskill_optimizer.paper import PaperEpochPlan, load_paper_profile


class PaperEpochPlanTests(unittest.TestCase):
    def test_plan_is_deterministic_and_drives_the_frozen_cosine_budget(self) -> None:
        profile = load_paper_profile()
        kwargs = {
            "profile": profile,
            "train_split_id": "train-v1",
            "train_split_manifest_sha256": "a" * 64,
            "steps_per_epoch": 2,
        }

        plan = PaperEpochPlan.build(**kwargs)
        rebuilt = PaperEpochPlan.build(**kwargs)
        cursors = [
            plan.cursor(epoch=epoch, step=step)
            for epoch in range(1, profile.epochs + 1)
            for step in range(1, plan.steps_per_epoch + 1)
        ]

        self.assertEqual(plan, rebuilt)
        self.assertEqual(
            [cursor.edit_budget for cursor in cursors],
            [4, 4, 3, 3, 3, 2, 2, 2],
        )
        self.assertEqual({cursor.batch_size for cursor in cursors}, {40})
        self.assertEqual(len({cursor.batch_seed for cursor in cursors}), 8)
        self.assertEqual([cursor.global_step for cursor in cursors], list(range(1, 9)))
        self.assertEqual(len({cursor.batch_id for cursor in cursors}), 8)
        self.assertEqual(PaperEpochPlan.from_mapping(plan.to_dict()), plan)

        with self.assertRaisesRegex(ValueError, "outside epoch plan"):
            plan.cursor(epoch=5, step=1)


if __name__ == "__main__":
    unittest.main()
