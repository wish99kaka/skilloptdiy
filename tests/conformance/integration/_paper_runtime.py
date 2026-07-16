from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from textskill_optimizer.paper import (
    ControllerArtifact,
    ControllerRegistration,
    ControllerRegistry,
    ControllerRole,
    PaperOptimizationController,
    SelectionController,
    TrainController,
)


def build_runtime(
    root: Path,
    backend,
    *,
    invalid_selection: bool = False,
    invalid_selection_after_first: bool = False,
    failure_count: int = 1,
    success_count: int = 1,
    longitudinal_fixture: bool = False,
    slow_selection_accept: bool = False,
    truncate_scheduled_batch: bool = False,
):
    trajectories = [
        {
            "task_id": f"failure-{index}",
            "task_input": f"question f{index}",
            "output": "wrong",
            "score": 0.0,
            "success": False,
            "trace": ["trusted an unverified result"],
        }
        for index in range(1, failure_count + 1)
    ] + [
        {
            "task_id": f"success-{index}",
            "task_input": f"question s{index}",
            "output": "right",
            "score": 1.0,
            "success": True,
            "trace": ["verified the result"],
        }
        for index in range(1, success_count + 1)
    ]
    train_path = root / "train.py"
    if longitudinal_fixture:
        train_payload_source = """
has_update = "accepted rule" in request["skill_text"]
trajectories = []
per_category = request.get("batch_size", 20) // 4
for category in ("improvement", "regression", "persistent", "stable"):
    for index in range(1, per_category + 1):
        if category == "improvement":
            success = has_update
        elif category == "regression":
            success = not has_update
        elif category == "persistent":
            success = False
        else:
            success = True
        trajectories.append({
            "task_id": f"{category}-{index}",
            "task_input": f"question {category} {index}",
            "output": "right" if success else "wrong",
            "score": 1.0 if success else 0.0,
            "success": success,
            "trace": [f"{category} outcome"],
        })
payload = {
    "split_id": request["split_id"],
    "split_manifest_sha256": request["split_manifest_sha256"],
    "trajectories": trajectories,
}
if "batch_id" in request:
    payload["batch_id"] = request["batch_id"]
    payload["batch_seed"] = request["batch_seed"]
    payload["batch_size"] = request["batch_size"]
"""
    else:
        train_payload_source = f"""
payload = {{
    "split_id": request["split_id"],
    "split_manifest_sha256": request["split_manifest_sha256"],
    "trajectories": {trajectories!r},
}}
if "batch_id" in request:
    payload["batch_id"] = request["batch_id"]
    payload["batch_seed"] = request["batch_seed"]
    payload["batch_size"] = request["batch_size"]
"""
    if truncate_scheduled_batch:
        train_payload_source += """
if "batch_id" in request:
    payload["trajectories"] = payload["trajectories"][:-1]
"""
    train_key, train_sha = _write_signed_controller(
        train_path,
        controller_id="train-owner",
        payload_source=train_payload_source,
    )
    selection_path = root / "selection.py"
    if invalid_selection_after_first:
        counter_path = root / "selection-count.txt"
        selection_payload = f"""
try:
    count = int(open({str(counter_path)!r}, encoding="utf-8").read())
except FileNotFoundError:
    count = 0
open({str(counter_path)!r}, "w", encoding="utf-8").write(str(count + 1))
payload = (
    {{"score": 0.5}}
    if count == 0
    else {{"score": 0.8, "forbidden": "diagnostics"}}
)
"""
    elif invalid_selection:
        selection_payload = 'payload = {"score": 0.8, "forbidden": "diagnostics"}'
    elif slow_selection_accept:
        selection_payload = (
            'payload = {"score": 0.9 if "durable but tied guidance" '
            'in request["skill_text"] else (0.8 if "accepted rule" '
            'in request["skill_text"] else 0.5)}'
        )
    else:
        selection_payload = (
            'payload = {"score": 0.8 if "accepted rule" '
            'in request["skill_text"] else 0.5}'
        )
    selection_key, selection_sha = _write_signed_controller(
        selection_path,
        controller_id="selection-owner",
        payload_source=selection_payload,
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
    train = TrainController(registry=registry, controller_id="train-owner")
    selection = SelectionController(
        registry=registry,
        controller_id="selection-owner",
    )
    return (
        PaperOptimizationController(
            optimizer_backend=backend,
            selection=selection,
            train=train,
        ),
        train,
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
    "request_sha256": hashlib.sha256(
        json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest(),
    "payload": payload,
}}
signature = Ed25519PrivateKey.from_private_bytes(bytes.fromhex({private_hex!r})).sign(
    json.dumps(
        signed,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
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
) -> ControllerRegistration:
    manifest_path = path.with_name(f"{controller_id}-split.json")
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
