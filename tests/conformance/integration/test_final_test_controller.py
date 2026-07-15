import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from textskill_optimizer.paper import (
    ControllerArtifact,
    ControllerRegistration,
    ControllerRegistry,
    ControllerRole,
    DataFirewallViolation,
)
from textskill_optimizer.paper.final_evaluation import (
    FinalEvaluationPlan,
    FinalEvaluationPolicy,
    FinalTestController,
    FrozenCandidate,
)
from textskill_optimizer.paper.controller_process import (
    invoke_optimization_controller,
)


HASH = "a" * 64


def _test_registry(
    root: Path,
    marker: Path,
    *,
    rich_response: bool = False,
) -> tuple[ControllerRegistry, str]:
    private_key = Ed25519PrivateKey.generate()
    private_hex = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    ).hex()
    public_hex = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    diagnostic = ', "diagnostics": "TEST_SECRET_SENTINEL"' if rich_response else ""
    script = root / "test_controller.py"
    script.write_text(
        f"""
import hashlib, json, pathlib, sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

request = json.load(sys.stdin)
pathlib.Path(sys.argv[1]).write_text("invoked", encoding="utf-8")
candidate = request["candidates"][0]
payload = {{
    "plan_sha256": request["plan_sha256"],
    "scores": [{{
        "candidate_id": candidate["candidate_id"],
        "skill_sha256": candidate["skill_sha256"],
        "score": 0.9{diagnostic},
    }}],
}}
signed = {{
    "controller_id": "test-owner",
    "request_sha256": hashlib.sha256(json.dumps(request, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    "payload": payload,
}}
signature = Ed25519PrivateKey.from_private_bytes(bytes.fromhex({private_hex!r})).sign(
    json.dumps(signed, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
).hex()
print(json.dumps({{**signed, "signature": signature}}, ensure_ascii=False))
""",
        encoding="utf-8",
    )
    artifact_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    split_manifest = root / "test-split.json"
    split_manifest.write_text(
        '{"split_id":"fresh-test-v1"}\n', encoding="utf-8"
    )
    registration = ControllerRegistration(
        controller_id="test-owner",
        role=ControllerRole.TEST,
        split_id="fresh-test-v1",
        argv=(sys.executable, str(script), str(marker)),
        launch_artifact_ids=("executable", "runner"),
        response_public_key=public_hex,
        artifacts=(
            ControllerArtifact(
                "executable",
                sys.executable,
                hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            ),
            *tuple(
                ControllerArtifact(name, str(script), artifact_sha)
                for name in ("runner", "scorer", "harness")
            ),
            ControllerArtifact(
                "split_manifest",
                str(split_manifest),
                hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
            ),
        ),
    )
    return ControllerRegistry(registrations=(registration,)), artifact_sha


def _policy(registry: ControllerRegistry, artifact_sha: str) -> FinalEvaluationPolicy:
    return FinalEvaluationPolicy(
        test_split_id="fresh-test-v1",
        split_manifest_sha256=registry.require(
            "test-owner", role=ControllerRole.TEST
        ).artifact("split_manifest").sha256,
        controller_registry_sha256=registry.sha256,
        runner_sha256=artifact_sha,
        scorer_sha256=artifact_sha,
        harness_sha256=artifact_sha,
        environment_sha256=HASH,
        profile_sha256=HASH,
        local_code_commit="b" * 40,
    )


def _plan(
    registry: ControllerRegistry,
    artifact_sha: str,
) -> FinalEvaluationPlan:
    return FinalEvaluationPlan.freeze(
        candidates=(FrozenCandidate.from_skill("skillopt", "# Skill\n"),),
        policy=_policy(registry, artifact_sha),
    )


class ForgedPlan(FinalEvaluationPlan):
    pass


class FinalTestControllerTests(unittest.TestCase):
    def test_optimization_process_api_cannot_invoke_registered_test_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "test-invoked"
            registry, _ = _test_registry(root, marker)

            with self.assertRaisesRegex(DataFirewallViolation, "cannot target final-test"):
                invoke_optimization_controller(
                    registry=registry,
                    controller_id="test-owner",
                    role=ControllerRole.TEST,
                    request={"operation": "final_test"},
                )
            self.assertFalse(marker.exists())

    def test_test_process_runs_only_for_an_exact_revalidated_frozen_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "test-invoked"
            registry, artifact_sha = _test_registry(root, marker)
            controller = FinalTestController(registry, "test-owner")

            with self.assertRaisesRegex(DataFirewallViolation, "FinalEvaluationPlan"):
                controller.evaluate(object())
            self.assertFalse(marker.exists())

            valid = _plan(registry, artifact_sha)
            forged = ForgedPlan(candidates=valid.candidates, policy=valid.policy)
            with self.assertRaisesRegex(DataFirewallViolation, "exact FinalEvaluationPlan"):
                controller.evaluate(forged)
            self.assertFalse(marker.exists())

            tampered = _plan(registry, artifact_sha)
            object.__setattr__(tampered.candidates[0], "skill_sha256", "0" * 64)
            with self.assertRaisesRegex(DataFirewallViolation, "hash"):
                controller.evaluate(tampered)
            self.assertFalse(marker.exists())

    def test_policy_hashes_are_bound_to_registered_controller_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "test-invoked"
            registry, artifact_sha = _test_registry(root, marker)
            plan = _plan(registry, artifact_sha)
            object.__setattr__(plan.policy, "runner_sha256", "0" * 64)

            with self.assertRaisesRegex(DataFirewallViolation, "runner hash"):
                FinalTestController(registry, "test-owner").evaluate(plan)
            self.assertFalse(marker.exists())

    def test_registered_test_artifact_cannot_change_after_plan_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "test-invoked"
            registry, artifact_sha = _test_registry(root, marker)
            controller = FinalTestController(registry, "test-owner")
            plan = _plan(registry, artifact_sha)
            script = root / "test_controller.py"
            script.write_text(
                script.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DataFirewallViolation, "hash does not match"):
                controller.evaluate(plan)
            self.assertFalse(marker.exists())

    def test_valid_frozen_plan_receives_only_scalar_test_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "test-invoked"
            registry, artifact_sha = _test_registry(root, marker)
            controller = FinalTestController(registry, "test-owner")
            plan = _plan(registry, artifact_sha)

            report = controller.evaluate(plan)

            self.assertTrue(marker.exists())
            self.assertEqual(report.plan_sha256, plan.sha256)
            self.assertEqual(report.scores[0].score, 0.9)

    def test_final_test_process_cannot_return_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "test-invoked"
            registry, artifact_sha = _test_registry(
                root, marker, rich_response=True
            )
            controller = FinalTestController(registry, "test-owner")

            with self.assertRaisesRegex(DataFirewallViolation, "exactly"):
                controller.evaluate(_plan(registry, artifact_sha))


if __name__ == "__main__":
    unittest.main()
