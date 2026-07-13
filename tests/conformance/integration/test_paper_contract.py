import unittest

from textskill_optimizer.paper import (
    ClaimClass,
    EvidenceLevel,
    assess_paper_run,
    canonical_json_sha256,
    load_paper_profile,
)


HASH = "a" * 64


def valid_lineage(*, split_id: str = "searchqa-fresh-v1") -> dict:
    profile = load_paper_profile().to_dict()
    return {
        "schema_version": "paper-lineage-v1",
        "claim_class": "paper_faithful_development",
        "evidence_level": None,
        "protocol_id": "paper-faithful-v1",
        "sources": {
            "paper_version": "2605.23904v2",
            "official_reference_commit": "e4ea6a6771e797ef820cdd8bfea64c57e0481065",
            "local_code_commit": "b" * 40,
            "upstream_deviation_manifest_sha256": HASH,
        },
        "artifacts": {
            "profile_sha256": canonical_json_sha256(profile),
            "prompt_sha256": HASH,
            "skill_sha256": HASH,
        },
        "data": {
            "split_id": split_id,
            "split_manifest_sha256": HASH,
            "scorer_id": "exact-match-v1",
            "scorer_sha256": HASH,
            "runner_id": "direct-chat-v1",
            "runner_sha256": HASH,
            "harness_sha256": HASH,
            "environment_sha256": HASH,
        },
        "models": {
            "student_model": "scripted-student",
            "student_reasoning": "none",
            "optimizer_model": "scripted-optimizer",
            "optimizer_reasoning": "none",
        },
        "execution": {
            "seeds": [42],
            "retry_policy": "none",
            "schedule": "cosine",
            "provider_versions": {},
            "runtime_versions": {"python": "3.10"},
            "optimizer_calls": 0,
            "optimizer_tokens": 0,
            "target_calls": 0,
            "target_tokens": 0,
            "wall_time_seconds": 0.0,
            "cost_scope": "zero-cost scripted contract test",
        },
        "test_exposure": {
            "status": "untouched",
            "archive_commit": "c" * 40,
            "attempt": 0,
            "receipt_sha256": None,
            "history": [],
        },
    }


class PaperContractTests(unittest.TestCase):
    def test_classifies_a_valid_development_run_without_a_backend(self) -> None:
        assessment = assess_paper_run(
            profile=load_paper_profile().to_dict(),
            lineage=valid_lineage(),
        )

        self.assertTrue(assessment.eligible)
        self.assertEqual(assessment.claim_class, ClaimClass.PAPER_FAITHFUL_DEVELOPMENT)
        self.assertEqual(assessment.violations, ())

    def test_rejects_lineage_bound_to_a_different_profile(self) -> None:
        lineage = valid_lineage()
        lineage["artifacts"]["profile_sha256"] = "0" * 64

        assessment = assess_paper_run(
            profile=load_paper_profile().to_dict(),
            lineage=lineage,
        )

        self.assertFalse(assessment.eligible)
        self.assertTrue(
            any(item.code == "profile_hash_mismatch" for item in assessment.violations)
        )

    def test_rejects_a_paper_heldout_claim_on_a_consumed_split(self) -> None:
        lineage = valid_lineage(split_id="coding-hidden-v2")
        lineage["claim_class"] = "paper_faithful_heldout"

        assessment = assess_paper_run(
            profile=load_paper_profile().to_dict(),
            lineage=lineage,
        )

        self.assertFalse(assessment.eligible)
        self.assertTrue(
            any(item.code == "consumed_split" for item in assessment.violations)
        )

    def test_paper_protocol_cannot_emit_an_extension_claim(self) -> None:
        lineage = valid_lineage()
        lineage["claim_class"] = "contract_aware_extension"

        assessment = assess_paper_run(
            profile=load_paper_profile().to_dict(),
            lineage=lineage,
        )

        self.assertFalse(assessment.eligible)
        self.assertTrue(
            any(item.code == "claim_protocol_mismatch" for item in assessment.violations)
        )

    def test_development_artifact_cannot_claim_heldout_evidence(self) -> None:
        lineage = valid_lineage()
        lineage["evidence_level"] = EvidenceLevel.FRESH_LOCAL_EFFICACY.value

        assessment = assess_paper_run(
            profile=load_paper_profile().to_dict(),
            lineage=lineage,
        )

        self.assertFalse(assessment.eligible)
        self.assertTrue(
            any(item.code == "claim_evidence_mismatch" for item in assessment.violations)
        )

    def test_non_null_evidence_level_requires_a_future_measured_gate(self) -> None:
        lineage = valid_lineage()
        lineage["claim_class"] = "mechanism_test"
        lineage["evidence_level"] = EvidenceLevel.PAPER_MECHANISM_CONFORMANT.value

        assessment = assess_paper_run(
            profile=load_paper_profile().to_dict(),
            lineage=lineage,
        )

        self.assertFalse(assessment.eligible)
        self.assertTrue(
            any(item.code == "unverified_evidence_level" for item in assessment.violations)
        )

    def test_heldout_claim_requires_an_untouched_split(self) -> None:
        lineage = valid_lineage()
        lineage["claim_class"] = "paper_faithful_heldout"
        lineage["test_exposure"] = {
            "status": "consumed",
            "archive_commit": "c" * 40,
            "attempt": 1,
            "receipt_sha256": HASH,
            "history": [
                {
                    "attempt": 1,
                    "protocol_id": "paper-faithful-v1",
                    "consumed_at": "2026-07-13T00:00:00+00:00",
                    "receipt_sha256": HASH,
                }
            ],
        }

        assessment = assess_paper_run(
            profile=load_paper_profile().to_dict(),
            lineage=lineage,
        )

        self.assertFalse(assessment.eligible)
        self.assertTrue(
            any(item.code == "heldout_not_untouched" for item in assessment.violations)
        )

    def test_heldout_claim_rejects_hidden_exposure_history(self) -> None:
        lineage = valid_lineage()
        lineage["claim_class"] = "paper_faithful_heldout"
        lineage["test_exposure"]["history"] = [
            {
                "attempt": 1,
                "protocol_id": "another-protocol",
                "consumed_at": "2026-07-12T00:00:00+00:00",
                "receipt_sha256": HASH,
            }
        ]

        assessment = assess_paper_run(
            profile=load_paper_profile().to_dict(),
            lineage=lineage,
        )

        self.assertFalse(assessment.eligible)
        self.assertTrue(
            any(item.code == "heldout_not_untouched" for item in assessment.violations)
        )


if __name__ == "__main__":
    unittest.main()
