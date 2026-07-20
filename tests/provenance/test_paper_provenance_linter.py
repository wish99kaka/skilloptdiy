import json
import unittest
from copy import deepcopy
from pathlib import Path

from textskill_optimizer.paper import (
    PaperProvenanceViolation,
    assess_paper_provenance,
)


ROOT = Path(__file__).parents[2]


class PaperProvenanceLinterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_lock = json.loads(
            (ROOT / "docs/papers/source-lock.json").read_text(encoding="utf-8")
        )
        self.prompt_snapshot = json.loads(
            (ROOT / "docs/papers/prompt-snapshot-v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.paper_bytes = (
            ROOT / "docs/papers/skillopt-2605.23904.pdf"
        ).read_bytes()

    def test_locked_sources_snapshot_and_bundled_prompts_are_consistent(self) -> None:
        assessment = assess_paper_provenance(
            source_lock=self.source_lock,
            prompt_snapshot=self.prompt_snapshot,
            paper_bytes=self.paper_bytes,
        )

        self.assertTrue(assessment.compliant)
        self.assertEqual(assessment.violations, ())
        self.assertEqual(assessment.prompt_count, 18)

    def test_prompt_snapshot_byte_drift_fails_closed(self) -> None:
        prompt_snapshot = deepcopy(self.prompt_snapshot)
        prompt_snapshot["prompts"][0]["sha256"] = "0" * 64

        assessment = assess_paper_provenance(
            source_lock=self.source_lock,
            prompt_snapshot=prompt_snapshot,
            paper_bytes=self.paper_bytes,
        )

        self.assertFalse(assessment.compliant)
        self.assertTrue(
            any(
                item.code == "prompt_snapshot_mismatch"
                for item in assessment.violations
            )
        )
        with self.assertRaises(PaperProvenanceViolation):
            assessment.require()

    def test_unregistered_local_resolution_fails_closed(self) -> None:
        source_lock = deepcopy(self.source_lock)
        refinement = next(
            item
            for item in source_lock["known_upstream_deviations"]
            if item["id"] == "analyst-refinement-loop"
        )
        refinement["local_resolution_files"] = []

        assessment = assess_paper_provenance(
            source_lock=source_lock,
            prompt_snapshot=self.prompt_snapshot,
            paper_bytes=self.paper_bytes,
        )

        self.assertFalse(assessment.compliant)
        self.assertTrue(
            any(
                item.code == "unregistered_local_resolution"
                for item in assessment.violations
            )
        )

    def test_tracked_paper_byte_drift_fails_closed(self) -> None:
        assessment = assess_paper_provenance(
            source_lock=self.source_lock,
            prompt_snapshot=self.prompt_snapshot,
            paper_bytes=self.paper_bytes + b"drift",
        )

        self.assertFalse(assessment.compliant)
        self.assertTrue(
            any(
                item.code == "paper_bytes_drift"
                for item in assessment.violations
            )
        )

    def test_searchqa_benchmark_reference_drift_fails_closed(self) -> None:
        source_lock = deepcopy(self.source_lock)
        source_lock["benchmark_references"][0]["dataset"]["revision"] = "0" * 40

        assessment = assess_paper_provenance(
            source_lock=source_lock,
            prompt_snapshot=self.prompt_snapshot,
            paper_bytes=self.paper_bytes,
        )

        self.assertFalse(assessment.compliant)
        self.assertTrue(
            any(
                item.code == "benchmark_reference_drift"
                for item in assessment.violations
            )
        )

    def test_invalidated_searchqa_smoke_cannot_be_mistaken_for_evidence(self) -> None:
        invalidation = json.loads(
            (
                ROOT
                / "docs/provenance/"
                "paper-searchqa-mechanism-smoke-c7c8a3e-v3-invalidation.json"
            ).read_text(encoding="utf-8")
        )
        sample_hashes = self.source_lock["benchmark_references"][0][
            "development_sample_sha256"
        ]

        self.assertEqual(invalidation["status"], "invalidated")
        self.assertIn(
            "initial_selection_score",
            invalidation["cause"]["affected_fields"],
        )
        self.assertEqual(invalidation["preserved_facts"]["test_access_attempt"], 0)
        self.assertEqual(
            invalidation["remediation"]["replacement_selection_sha256"],
            sample_hashes["selection_20_seed_43"],
        )


if __name__ == "__main__":
    unittest.main()
