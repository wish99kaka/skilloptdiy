import unittest

from textskill_optimizer.paper import assess_paper_profile, load_paper_profile


class PaperProfileTests(unittest.TestCase):
    def test_loads_the_frozen_machine_readable_default(self) -> None:
        profile = load_paper_profile()

        self.assertEqual(profile.profile, "paper-faithful-v1")
        self.assertEqual(profile.protocol_id, "paper-faithful-v1")
        self.assertEqual(profile.epochs, 4)
        self.assertEqual(profile.selection_gate.comparator, "strict_greater")
        self.assertEqual(profile.max_analyst_rounds, 3)

    def test_rejects_extension_controls_instead_of_disabling_them(self) -> None:
        for path, value in (
            (("validation_confirmation_rounds",), 2),
            (("selection_gate", "contract_guard"), True),
            (("selection_gate", "force_accept"), True),
        ):
            with self.subTest(path=path):
                payload = load_paper_profile().to_dict()
                target = payload
                for segment in path[:-1]:
                    target = target[segment]
                target[path[-1]] = value

                assessment = assess_paper_profile(payload)

                self.assertFalse(assessment.compliant)
                self.assertTrue(
                    any(
                        item.code == "forbidden_extension_control"
                        and item.path == ".".join(path)
                        for item in assessment.violations
                    )
                )

    def test_rejects_values_outside_the_paper_contract(self) -> None:
        payload = load_paper_profile().to_dict()
        payload["max_analyst_rounds"] = 4

        assessment = assess_paper_profile(payload)

        self.assertFalse(assessment.compliant)
        self.assertTrue(
            any(item.path.endswith("max_analyst_rounds") for item in assessment.violations)
        )

    def test_rejects_unregistered_deviations_from_the_frozen_profile(self) -> None:
        for path, value in (
            (("epochs",), 1),
            (("rejected_buffer", "enabled"), False),
            (("slow_update", "enabled"), False),
            (("meta_skill", "enabled"), False),
        ):
            with self.subTest(path=path):
                payload = load_paper_profile().to_dict()
                target = payload
                for segment in path[:-1]:
                    target = target[segment]
                target[path[-1]] = value

                assessment = assess_paper_profile(payload)

                self.assertFalse(assessment.compliant)
                self.assertTrue(
                    any(
                        item.code == "unregistered_profile_override"
                        and item.path == ".".join(path)
                        for item in assessment.violations
                    )
                )


if __name__ == "__main__":
    unittest.main()
