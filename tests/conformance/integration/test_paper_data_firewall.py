import hashlib
import json
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
    OptimizerPayload,
    OptimizerRequest,
    OptimizerResponse,
    OptimizerStage,
    PaperOptimizationController,
    SelectionController,
    SelectionScore,
    TrainController,
    TrainEvidenceBatch,
)


SELECTION_SENTINEL = "SELECTION_SECRET_SENTINEL"
TEST_SENTINEL = "TEST_SECRET_SENTINEL"


class CapturingBackend:
    def __init__(self) -> None:
        self.requests: list[OptimizerRequest] = []

    def complete(self, request: OptimizerRequest) -> OptimizerResponse:
        self.requests.append(request)
        return OptimizerResponse(
            call_id=request.call_id,
            payload={"suggestions": []},
            model_id="scripted-optimizer",
        )


def _write_signed_controller(
    path: Path,
    *,
    controller_id: str,
    payload_source: str,
) -> tuple[str, str]:
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
    path.write_text(
        f"""
import hashlib, json, sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

request = json.load(sys.stdin)
{payload_source}
signed = {{
    "controller_id": {controller_id!r},
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
    return public_hex, hashlib.sha256(path.read_bytes()).hexdigest()


def _registration(
    path: Path,
    *,
    controller_id: str,
    role: ControllerRole,
    split_id: str,
    public_key: str,
    runner_sha256: str,
    split_manifest_path: Path | None = None,
) -> ControllerRegistration:
    manifest_path = split_manifest_path or path.with_name(
        f"{controller_id}-split.json"
    )
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps({"split_id": split_id, "owner": controller_id}),
            encoding="utf-8",
        )
    return ControllerRegistration(
        controller_id=controller_id,
        role=role,
        split_id=split_id,
        argv=(sys.executable, str(path)),
        launch_artifact_ids=("executable", "runner"),
        response_public_key=public_key,
        artifacts=(
            ControllerArtifact(
                "executable",
                sys.executable,
                hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            ),
            ControllerArtifact("runner", str(path), runner_sha256),
            ControllerArtifact(
                "split_manifest",
                str(manifest_path),
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            ),
        ),
    )


def _controllers(root: Path, *, rich_selection: bool = False):
    selection_path = root / "selection.py"
    selection_payload = (
        f'payload = {{"score": 0.6, "diagnostics": {SELECTION_SENTINEL!r}}}'
        if rich_selection
        else 'payload = {"score": 0.6}'
    )
    selection_key, selection_sha = _write_signed_controller(
        selection_path,
        controller_id="selection-owner",
        payload_source=(
            f"private_selection_task = {SELECTION_SENTINEL!r}\n" + selection_payload
        ),
    )
    train_path = root / "train.py"
    train_key, train_sha = _write_signed_controller(
        train_path,
        controller_id="train-owner",
        payload_source="""
payload = {
    "split_id": request["split_id"],
    "split_manifest_sha256": request["split_manifest_sha256"],
    "trajectories": [{
        "task_id": "train-1",
        "task_input": "train input",
        "output": "train output",
        "score": 0.0,
        "success": False,
        "trace": ["train trace"],
    }],
}
""",
    )
    registry = ControllerRegistry(
        registrations=(
            _registration(
                train_path,
                controller_id="train-owner",
                role=ControllerRole.TRAIN,
                split_id="train-v1",
                public_key=train_key,
                runner_sha256=train_sha,
            ),
            _registration(
                selection_path,
                controller_id="selection-owner",
                role=ControllerRole.SELECTION,
                split_id="selection-v1",
                public_key=selection_key,
                runner_sha256=selection_sha,
            ),
        )
    )
    return (
        registry,
        TrainController(registry=registry, controller_id="train-owner"),
        SelectionController(registry=registry, controller_id="selection-owner"),
    )


class PaperDataFirewallTests(unittest.TestCase):
    def test_selection_process_returns_only_scalar_and_never_reaches_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, train, selection = _controllers(Path(tmp))
            backend = CapturingBackend()
            controller = PaperOptimizationController(
                optimizer_backend=backend,
                selection=selection,
                train=train,
            )
            train_evidence = train.collect("current skill")

            decision = controller.score_candidate(
                current=SelectionScore(0.5),
                candidate_skill="candidate skill",
            )
            controller.request_optimizer(
                call_id="reflect-1",
                stage=OptimizerStage.REFLECT_FAILURE,
                payload=OptimizerPayload(
                    current_skill="current skill",
                    train_evidence=train_evidence,
                ),
            )

        self.assertTrue(decision.accepted)
        request = backend.requests[0]
        request_payload = json.loads(request.prompt)
        self.assertEqual(set(request_payload), {"current_skill", "train_trajectories"})
        self.assertEqual(request.metadata["data_sources"], ["train"])
        self.assertEqual(request.metadata["controller_registry_sha256"], registry.sha256)
        self.assertEqual(request.metadata["train_split_id"], "train-v1")
        self.assertEqual(
            request.metadata["train_split_manifest_sha256"],
            train_evidence.split_manifest_sha256,
        )
        self.assertNotIn(SELECTION_SENTINEL, request.prompt)
        self.assertNotIn(TEST_SENTINEL, request.prompt)

    def test_selection_process_rejects_every_non_scalar_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, selection = _controllers(Path(tmp), rich_selection=True)
            with self.assertRaisesRegex(DataFirewallViolation, "exactly one scalar"):
                selection.score("candidate skill")

    def test_registry_rejects_one_runner_or_key_registered_across_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed-owner.py"
            public_key, runner_sha = _write_signed_controller(
                path,
                controller_id="mixed-owner",
                payload_source='payload = {"score": 0.5}',
            )
            train = _registration(
                path,
                controller_id="train-owner",
                role=ControllerRole.TRAIN,
                split_id="train-v1",
                public_key=public_key,
                runner_sha256=runner_sha,
            )
            selection = _registration(
                path,
                controller_id="selection-owner",
                role=ControllerRole.SELECTION,
                split_id="selection-v1",
                public_key=public_key,
                runner_sha256=runner_sha,
            )

            with self.assertRaisesRegex(DataFirewallViolation, "cannot cross data roles"):
                ControllerRegistry(registrations=(train, selection))

    def test_registration_rejects_an_unhashed_actual_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registered = root / "registered.py"
            actual = root / "actual.py"
            public_key, runner_sha = _write_signed_controller(
                registered,
                controller_id="train-owner",
                payload_source='payload = {"split_id": "train-v1", "trajectories": []}',
            )
            actual.write_text("raise SystemExit(0)\n", encoding="utf-8")

            with self.assertRaisesRegex(DataFirewallViolation, "argv prefix"):
                ControllerRegistration(
                    controller_id="train-owner",
                    role=ControllerRole.TRAIN,
                    split_id="train-v1",
                    argv=(sys.executable, str(actual)),
                    launch_artifact_ids=("executable", "runner"),
                    response_public_key=public_key,
                    artifacts=(
                        ControllerArtifact(
                            "executable",
                            sys.executable,
                            hashlib.sha256(
                                Path(sys.executable).read_bytes()
                            ).hexdigest(),
                        ),
                        ControllerArtifact("runner", str(registered), runner_sha),
                    ),
                )

    def test_registry_rejects_one_split_registered_across_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.py"
            train_key, train_sha = _write_signed_controller(
                train_path,
                controller_id="train-owner",
                payload_source='payload = {"score": 0.0}',
            )
            selection_path = root / "selection.py"
            selection_key, selection_sha = _write_signed_controller(
                selection_path,
                controller_id="selection-owner",
                payload_source='payload = {"score": 0.0}',
            )
            train = _registration(
                train_path,
                controller_id="train-owner",
                role=ControllerRole.TRAIN,
                split_id="shared-split-v1",
                public_key=train_key,
                runner_sha256=train_sha,
            )
            selection = _registration(
                selection_path,
                controller_id="selection-owner",
                role=ControllerRole.SELECTION,
                split_id="shared-split-v1",
                public_key=selection_key,
                runner_sha256=selection_sha,
            )

            with self.assertRaisesRegex(DataFirewallViolation, "only one registered owner"):
                ControllerRegistry(registrations=(train, selection))

    def test_registry_rejects_a_consumed_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.py"
            public_key, runner_sha = _write_signed_controller(
                path,
                controller_id="train-owner",
                payload_source='payload = {"score": 0.0}',
            )
            registration = _registration(
                path,
                controller_id="train-owner",
                role=ControllerRole.TRAIN,
                split_id="coding-hidden-v2",
                public_key=public_key,
                runner_sha256=runner_sha,
            )

            with self.assertRaisesRegex(DataFirewallViolation, "consumed split"):
                ControllerRegistry(registrations=(registration,))

    def test_train_owner_cannot_spoof_its_registered_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.py"
            public_key, runner_sha = _write_signed_controller(
                path,
                controller_id="train-owner",
                payload_source="""
payload = {
    "split_id": "selection-v1",
    "split_manifest_sha256": request["split_manifest_sha256"],
    "trajectories": [{
        "task_id": "x", "task_input": "x", "output": "x",
        "score": 0.0, "success": False, "trace": [],
    }],
}
""",
            )
            registry = ControllerRegistry(
                registrations=(
                    _registration(
                        path,
                        controller_id="train-owner",
                        role=ControllerRole.TRAIN,
                        split_id="train-v1",
                        public_key=public_key,
                        runner_sha256=runner_sha,
                    ),
                )
            )

            with self.assertRaisesRegex(DataFirewallViolation, "split_id"):
                TrainController(registry, "train-owner").collect("current skill")

    def test_optimizer_rejects_forged_or_mutated_train_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, train, selection = _controllers(Path(tmp))
            controller = PaperOptimizationController(
                optimizer_backend=CapturingBackend(),
                selection=selection,
                train=train,
            )
            evidence = train.collect("current skill")
            object.__setattr__(
                evidence,
                "canonical_payload",
                json.dumps(
                    {
                        "split_id": "train-v1",
                        "split_manifest_sha256": evidence.split_manifest_sha256,
                        "trajectories": [{"task_input": SELECTION_SENTINEL}],
                    }
                ),
            )

            with self.assertRaisesRegex(DataFirewallViolation, "signature"):
                controller.request_optimizer(
                    call_id="forged",
                    stage=OptimizerStage.REFLECT_FAILURE,
                    payload=OptimizerPayload("current skill", evidence),
                )

            forged = TrainEvidenceBatch(
                controller_id="train-owner",
                registry_sha256=registry.sha256,
                split_id="train-v1",
                split_manifest_sha256=evidence.split_manifest_sha256,
                canonical_request=json.dumps({}),
                canonical_payload=json.dumps({}),
                signature="0" * 128,
            )
            with self.assertRaises(DataFirewallViolation):
                controller.request_optimizer(
                    call_id="untrusted",
                    stage=OptimizerStage.REFLECT_FAILURE,
                    payload=OptimizerPayload("current skill", forged),
                )


if __name__ == "__main__":
    unittest.main()
