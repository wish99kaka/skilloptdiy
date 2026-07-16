"""Fail-closed provenance audit for the paper prompt and source bundle."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping

from .backend import OptimizerStage
from .prompts import (
    OptimizerPromptRoute,
    load_optimizer_prompt,
    optimizer_prompt_routes,
)
from .provenance import canonical_json_sha256


_PINNED_PAPER = {
    "arxiv_id": "2605.23904v2",
    "source_url": "https://arxiv.org/pdf/2605.23904v2",
    "tracked_path": "docs/papers/skillopt-2605.23904.pdf",
    "sha256": "87f7f0f323b1671e9202b3ebb1596e909e507c71ecd1b360b0075a5ee1727fe3",
}
_PINNED_REFERENCE = {
    "repository": "https://github.com/microsoft/SkillOpt.git",
    "release": "v0.2.0",
    "tag_object_sha": "51d0a4d96e88558c84dee637f98e24e3fb2d1547",
    "commit_sha": "e4ea6a6771e797ef820cdd8bfea64c57e0481065",
    "tree_sha": "5a603e937a20f1078059f94039a50028c022487a",
    "license": "MIT",
}
_PINNED_LOCAL_BASELINE = {
    "tag": "contract-aware-extension-v1",
    "commit_sha": "91c00b9c582e48b077c9282f4ccc80db26341653",
}
_PINNED_DEVIATIONS = {
    "slow-update-selection-gate": {
        "upstream_evidence": {
            "path": "configs/_base_/default.yaml",
            "setting": "optimizer.slow_update_gate_with_selection",
            "value": False,
        },
        "paper_resolution": (
            "Paper mode must apply the strict selection gate to the "
            "slow-update candidate."
        ),
    },
    "analyst-refinement-loop": {
        "upstream_evidence": {
            "path": "skillopt/engine/trainer.py",
            "observation": (
                "max_analyst_rounds is loaded and logged but is not used to "
                "execute refinement rounds"
            ),
        },
        "paper_resolution": (
            "Paper mode must execute real semantic teacher refinement, "
            "capped at three rounds."
        ),
    },
}
_PINNED_REUSE_POLICY = (
    "Before copying any official source or prompt, append its "
    "repository-relative path and SHA256 to official_reference.reused_files. "
    "Upstream main is never a valid reference."
)
_EXPECTED_DEVIATIONS = frozenset(
    {"slow-update-selection-gate", "analyst-refinement-loop"}
)


_ROUTES = optimizer_prompt_routes()


@dataclass(frozen=True)
class ProvenanceLintViolation:
    code: str
    path: str
    message: str


class PaperProvenanceViolation(ValueError):
    """Raised when the locked zero-cost provenance bundle is inconsistent."""


@dataclass(frozen=True)
class PaperProvenanceAssessment:
    prompt_count: int
    prompt_snapshot_sha256: str | None
    violations: tuple[ProvenanceLintViolation, ...]

    @property
    def compliant(self) -> bool:
        return not self.violations

    def require(self) -> "PaperProvenanceAssessment":
        if self.violations:
            detail = "; ".join(
                f"{item.path}: {item.message}" for item in self.violations
            )
            raise PaperProvenanceViolation(
                f"paper provenance is not compliant: {detail}"
            )
        return self


def assess_paper_provenance(
    *,
    source_lock: Mapping[str, Any],
    prompt_snapshot: Mapping[str, Any],
    paper_bytes: bytes,
) -> PaperProvenanceAssessment:
    """Cross-check pinned sources, prompt snapshot, and bundled prompt bytes."""

    violations: list[ProvenanceLintViolation] = []
    if (
        type(source_lock) is not dict
        or type(prompt_snapshot) is not dict
        or type(paper_bytes) is not bytes
    ):
        return PaperProvenanceAssessment(
            prompt_count=0,
            prompt_snapshot_sha256=None,
            violations=(
                ProvenanceLintViolation(
                    "invalid_manifest",
                    "$",
                    "source lock, prompt snapshot, and paper bytes are required",
                ),
            ),
        )
    _check_pins(source_lock, prompt_snapshot, paper_bytes, violations)
    snapshot_items = _indexed_snapshot(prompt_snapshot, violations)
    official_files = _official_files(source_lock, violations)
    local_files = _local_resolution_files(source_lock, violations)
    route_names = {route.route for route in _ROUTES}
    if set(snapshot_items) != route_names:
        _add(
            violations,
            "prompt_route_drift",
            "prompt_snapshot.prompts",
            "snapshot routes do not exactly match executable prompt routes",
        )
    bundled_names = {
        item.name
        for item in files("textskill_optimizer.paper")
        .joinpath("prompts")
        .iterdir()
        if item.name.endswith(".md")
    }
    if bundled_names != {route.bundled_name for route in _ROUTES}:
        _add(
            violations,
            "unregistered_bundled_prompt",
            "textskill_optimizer.paper.prompts",
            "bundled prompt files do not exactly match registered routes",
        )
    for route in _ROUTES:
        item = snapshot_items.get(route.route)
        if item is None:
            continue
        bundled_path = _bundled_path(route)
        source_path = _source_path(route)
        deviation_id = _deviation_id(route)
        mode = "patch" if route.update_mode == "common" else route.update_mode
        bundled_bytes = (
            files("textskill_optimizer.paper")
            .joinpath("prompts", route.bundled_name)
            .read_bytes()
        )
        actual_sha256 = hashlib.sha256(bundled_bytes).hexdigest()
        if (
            load_optimizer_prompt(route.stage, update_mode=mode).encode("utf-8")
            != bundled_bytes
        ):
            _add(
                violations,
                "prompt_loader_byte_drift",
                f"textskill_optimizer.paper.prompts.{route.route}",
                "runtime prompt loader does not preserve bundled bytes",
            )
        expected = {
            "route": route.route,
            "stage": route.stage.value,
            "update_mode": route.update_mode,
            "bundled_path": bundled_path,
            "origin": "official" if source_path is not None else "local_resolution",
            "source_path": source_path,
            "deviation_id": deviation_id,
            "sha256": actual_sha256,
        }
        if item != expected:
            _add(
                violations,
                "prompt_snapshot_mismatch",
                f"prompt_snapshot.prompts.{route.route}",
                "snapshot metadata or hash does not match bundled prompt bytes",
            )
        if source_path is not None:
            if official_files.get(source_path) != actual_sha256:
                _add(
                    violations,
                    "official_prompt_mismatch",
                    f"source_lock.official_reference.reused_files.{source_path}",
                    "official prompt lock does not match bundled bytes",
                )
        elif local_files.get(bundled_path) != (
            route.route,
            actual_sha256,
            deviation_id,
        ):
            _add(
                violations,
                "unregistered_local_resolution",
                f"source_lock.known_upstream_deviations.{deviation_id}",
                "local prompt resolution is absent or hash-mismatched",
            )
    expected_official = {
        source_path
        for route in _ROUTES
        if (source_path := _source_path(route)) is not None
    }
    if set(official_files) != expected_official:
        _add(
            violations,
            "official_prompt_set_drift",
            "source_lock.official_reference.reused_files",
            "official reused-file set does not match official prompt routes",
        )
    expected_local = {
        _bundled_path(route) for route in _ROUTES if _source_path(route) is None
    }
    if set(local_files) != expected_local:
        _add(
            violations,
            "local_resolution_set_drift",
            "source_lock.known_upstream_deviations",
            "local resolution files do not exactly match local prompt routes",
        )
    return PaperProvenanceAssessment(
        prompt_count=len(snapshot_items),
        prompt_snapshot_sha256=canonical_json_sha256(prompt_snapshot),
        violations=tuple(violations),
    )


def _check_pins(
    source_lock: Mapping[str, Any],
    prompt_snapshot: Mapping[str, Any],
    paper_bytes: bytes,
    violations: list[ProvenanceLintViolation],
) -> None:
    if set(source_lock) != {
        "schema_version",
        "locked_on",
        "paper",
        "official_reference",
        "local_baseline",
        "known_upstream_deviations",
        "reuse_policy",
    } or source_lock.get("schema_version") != 1:
        _add(
            violations,
            "invalid_source_lock",
            "source_lock",
            "source lock fields or schema version are not exact",
        )
    if set(prompt_snapshot) != {
        "schema_version",
        "official_reference_commit",
        "prompts",
    }:
        _add(
            violations,
            "invalid_prompt_snapshot",
            "prompt_snapshot",
            "prompt snapshot fields are not exact",
        )
    if source_lock.get("paper") != _PINNED_PAPER:
        _add(violations, "paper_pin_drift", "source_lock.paper", "paper pin changed")
    if hashlib.sha256(paper_bytes).hexdigest() != _PINNED_PAPER["sha256"]:
        _add(
            violations,
            "paper_bytes_drift",
            "source_lock.paper.tracked_path",
            "tracked paper bytes do not match the pinned SHA256",
        )
    reference = source_lock.get("official_reference")
    if type(reference) is not dict or any(
        reference.get(key) != value for key, value in _PINNED_REFERENCE.items()
    ) or set(reference) != {*_PINNED_REFERENCE, "reused_files"}:
        _add(
            violations,
            "reference_pin_drift",
            "source_lock.official_reference",
            "official v0.2.0 reference identity changed",
        )
    if source_lock.get("locked_on") != "2026-07-13":
        _add(
            violations,
            "source_lock_date_drift",
            "source_lock.locked_on",
            "source lock date changed without an explicit re-pin",
        )
    if source_lock.get("local_baseline") != _PINNED_LOCAL_BASELINE:
        _add(
            violations,
            "baseline_pin_drift",
            "source_lock.local_baseline",
            "contract-aware baseline identity changed",
        )
    if source_lock.get("reuse_policy") != _PINNED_REUSE_POLICY:
        _add(
            violations,
            "reuse_policy_drift",
            "source_lock.reuse_policy",
            "official source reuse policy changed",
        )
    if prompt_snapshot.get("schema_version") != "paper-prompt-snapshot-v1":
        _add(
            violations,
            "invalid_snapshot_schema",
            "prompt_snapshot.schema_version",
            "unsupported prompt snapshot schema",
        )
    if prompt_snapshot.get("official_reference_commit") != _PINNED_REFERENCE[
        "commit_sha"
    ]:
        _add(
            violations,
            "snapshot_reference_drift",
            "prompt_snapshot.official_reference_commit",
            "prompt snapshot is not bound to the pinned official commit",
        )
    deviations = source_lock.get("known_upstream_deviations")
    deviation_ids = (
        {
            item.get("id")
            for item in deviations
            if type(item) is dict and type(item.get("id")) is str
        }
        if type(deviations) is list
        else set()
    )
    if deviation_ids != _EXPECTED_DEVIATIONS:
        _add(
            violations,
            "deviation_manifest_drift",
            "source_lock.known_upstream_deviations",
            "known upstream deviations are missing or unregistered",
        )
    if type(deviations) is list and len(deviations) != len(deviation_ids):
        _add(
            violations,
            "duplicate_deviation",
            "source_lock.known_upstream_deviations",
            "upstream deviation IDs must be unique",
        )
    if type(deviations) is list:
        for index, deviation in enumerate(deviations):
            if type(deviation) is not dict:
                continue
            deviation_id = deviation.get("id")
            pinned = _PINNED_DEVIATIONS.get(deviation_id)
            if pinned is None:
                continue
            expected_fields = {"id", *pinned}
            if deviation_id == "analyst-refinement-loop":
                expected_fields.add("local_resolution_files")
            if set(deviation) != expected_fields or any(
                deviation.get(key) != value for key, value in pinned.items()
            ):
                _add(
                    violations,
                    "deviation_detail_drift",
                    f"source_lock.known_upstream_deviations[{index}]",
                    "upstream evidence or paper resolution changed",
                )


def _indexed_snapshot(
    prompt_snapshot: Mapping[str, Any],
    violations: list[ProvenanceLintViolation],
) -> dict[str, dict[str, Any]]:
    prompts = prompt_snapshot.get("prompts")
    if type(prompts) is not list:
        _add(
            violations,
            "invalid_prompt_snapshot",
            "prompt_snapshot.prompts",
            "prompts must be a list",
        )
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    expected_fields = {
        "route",
        "stage",
        "update_mode",
        "bundled_path",
        "origin",
        "source_path",
        "deviation_id",
        "sha256",
    }
    for index, item in enumerate(prompts):
        if type(item) is not dict or set(item) != expected_fields:
            _add(
                violations,
                "invalid_prompt_snapshot",
                f"prompt_snapshot.prompts[{index}]",
                "prompt entry fields are not exact",
            )
            continue
        route = item["route"]
        if type(route) is not str or route in indexed:
            _add(
                violations,
                "duplicate_prompt_route",
                f"prompt_snapshot.prompts[{index}].route",
                "prompt route must be a unique string",
            )
            continue
        indexed[route] = item
    return indexed


def _official_files(
    source_lock: Mapping[str, Any],
    violations: list[ProvenanceLintViolation],
) -> dict[str, str]:
    reference = source_lock.get("official_reference")
    items = reference.get("reused_files") if type(reference) is dict else None
    return _path_hash_map(items, "source_lock.official_reference.reused_files", violations)


def _local_resolution_files(
    source_lock: Mapping[str, Any],
    violations: list[ProvenanceLintViolation],
) -> dict[str, tuple[str, str, str]]:
    deviations = source_lock.get("known_upstream_deviations")
    result: dict[str, tuple[str, str, str]] = {}
    if type(deviations) is not list:
        return result
    for deviation in deviations:
        if type(deviation) is not dict:
            _add(
                violations,
                "invalid_local_resolution",
                "source_lock.known_upstream_deviations",
                "deviation entries must be exact objects",
            )
            continue
        deviation_id = deviation.get("id")
        local_files = deviation.get("local_resolution_files", [])
        if type(deviation_id) is not str or type(local_files) is not list:
            _add(
                violations,
                "invalid_local_resolution",
                "source_lock.known_upstream_deviations",
                "local resolution file list is malformed",
            )
            continue
        for index, item in enumerate(local_files):
            if type(item) is not dict or set(item) != {"path", "route", "sha256"}:
                _add(
                    violations,
                    "invalid_local_resolution",
                    (
                        "source_lock.known_upstream_deviations."
                        f"{deviation_id}.local_resolution_files[{index}]"
                    ),
                    "local resolution fields are not exact",
                )
                continue
            path = item["path"]
            if (
                type(path) is not str
                or type(item["route"]) is not str
                or type(item["sha256"]) is not str
                or path in result
            ):
                _add(
                    violations,
                    "duplicate_or_invalid_local_resolution",
                    (
                        "source_lock.known_upstream_deviations."
                        f"{deviation_id}.local_resolution_files[{index}]"
                    ),
                    "local resolution path and values must be unique strings",
                )
                continue
            result[path] = (item["route"], item["sha256"], deviation_id)
    return result


def _path_hash_map(
    items: object,
    path: str,
    violations: list[ProvenanceLintViolation],
) -> dict[str, str]:
    if type(items) is not list:
        _add(violations, "invalid_source_lock", path, "expected a file list")
        return {}
    result: dict[str, str] = {}
    for index, item in enumerate(items):
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            _add(
                violations,
                "invalid_source_lock",
                f"{path}[{index}]",
                "locked file fields are not exact",
            )
            continue
        file_path = item["path"]
        if type(file_path) is not str or file_path in result:
            _add(
                violations,
                "duplicate_locked_path",
                f"{path}[{index}].path",
                "locked path must be a unique string",
            )
            continue
        result[file_path] = item["sha256"]
    return result


def _bundled_path(route: OptimizerPromptRoute) -> str:
    return f"textskill_optimizer/paper/prompts/{route.bundled_name}"


def _source_path(route: OptimizerPromptRoute) -> str | None:
    if route.stage is OptimizerStage.REFINE:
        return None
    return f"skillopt/prompts/{route.bundled_name}"


def _deviation_id(route: OptimizerPromptRoute) -> str | None:
    return (
        "analyst-refinement-loop"
        if route.stage is OptimizerStage.REFINE
        else None
    )


def _add(
    violations: list[ProvenanceLintViolation],
    code: str,
    path: str,
    message: str,
) -> None:
    violations.append(ProvenanceLintViolation(code, path, message))
