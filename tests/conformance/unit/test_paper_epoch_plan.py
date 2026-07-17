import unittest

from textskill_optimizer.paper import (
    PaperEpochPlan,
    PaperMechanismSpec,
    load_paper_profile,
)


class PaperEpochPlanTests(unittest.TestCase):
    def test_mechanism_smoke_can_bound_epochs_after_slow_meta_become_visible(self) -> None:
        profile = load_paper_profile()
        mechanisms = PaperMechanismSpec.for_mechanism_test(
            profile,
            analyst_workers=1,
        )

        plan = PaperEpochPlan.build(
            profile=profile,
            train_split_id="searchqa-smoke-train-v1",
            train_split_manifest_sha256="a" * 64,
            steps_per_epoch=1,
            mechanisms=mechanisms,
            epochs_override=2,
        )

        self.assertEqual(plan.epochs, 2)
        self.assertFalse(plan.paper_claim_eligible)
        plan.require_profile(profile)

    def test_default_scope_cannot_shorten_the_frozen_profile(self) -> None:
        profile = load_paper_profile()

        with self.assertRaisesRegex(ValueError, "mechanism-test"):
            PaperEpochPlan.build(
                profile=profile,
                train_split_id="searchqa-smoke-train-v1",
                train_split_manifest_sha256="a" * 64,
                steps_per_epoch=1,
                epochs_override=2,
            )

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

    def test_mechanism_test_plan_is_explicit_and_cannot_claim_default(self) -> None:
        profile = load_paper_profile()
        mechanisms = PaperMechanismSpec.for_mechanism_test(
            profile,
            accumulation=2,
            analyst_workers=2,
            learning_rate_schedule="autonomous",
            update_mode="rewrite_from_suggestions",
        )
        plan = PaperEpochPlan.build(
            profile=profile,
            train_split_id="train-v1",
            train_split_manifest_sha256="a" * 64,
            steps_per_epoch=1,
            mechanisms=mechanisms,
        )
        cursor = plan.cursor(epoch=1, step=1)

        self.assertFalse(plan.paper_claim_eligible)
        self.assertEqual(cursor.edit_budget, None)
        self.assertEqual(cursor.analysis_budget, profile.learning_rate)
        self.assertEqual(len(cursor.batches), 2)
        self.assertEqual(
            [batch.accumulation_index for batch in cursor.batches],
            [1, 2],
        )
        self.assertEqual(len({batch.batch_id for batch in cursor.batches}), 2)
        self.assertEqual(PaperEpochPlan.from_mapping(plan.to_dict()), plan)
        plan.require_profile(profile)

        with self.assertRaisesRegex(ValueError, "explicit deviation"):
            PaperMechanismSpec.for_mechanism_test(profile)

        with self.assertRaisesRegex(ValueError, "default mechanism"):
            PaperMechanismSpec(
                claim_scope="paper_faithful_default",
                accumulation=2,
                analyst_workers=profile.analyst_workers,
                learning_rate_schedule=profile.learning_rate_schedule,
                update_mode=profile.update_mode,
            )


if __name__ == "__main__":
    unittest.main()
