"""Command-backed skill editor integration."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract_rejection_evidence import (
    audit_proposal_targeting,
    build_contract_rejection_evidence,
)
from .edits import apply_atomic_edits
from .models import AtomicEdit, EditProposal, OptimizerStateUpdate, TaskResult
from .usage_ledger import append_usage_event, estimate_tokens_from_chars


@dataclass(frozen=True)
class CommandEditorConfig:
    """Runtime controls for command-backed skill editing."""

    command: str
    timeout_seconds: int = 120
    max_text_chars: int = 12000
    proposal_log_path: str | Path | None = None
    proposal_log_seed: str = "default"
    proposal_log_case: str = "default"
    usage_ledger_path: str | Path | None = None
    usage_context: dict[str, Any] | None = None


class CommandSkillEditor:
    """Calls an external process to propose skill edits.

    The process receives JSON on stdin and must write JSON on stdout.
    Accepted output shapes:
      - {"proposals": [{"name": ..., "edits": [...], "rationale": ...}]}
      - Legacy full replacements using skill_text remain supported.
    """

    def __init__(self, config: CommandEditorConfig) -> None:
        if not config.command.strip():
            raise ValueError("CommandEditorConfig.command must not be empty")
        self.config = config

    def propose(
        self,
        skill_text: str,
        train_results: list[TaskResult],
        *,
        epoch: int,
        rejected_buffer: list[dict[str, Any]] | None = None,
        meta_skill: str = "",
        optimizer_controls: dict[str, Any] | None = None,
    ) -> list[EditProposal]:
        payload = {
            "operation": "reflect",
            "epoch": epoch,
            "skill_text": skill_text,
            "train_results": [
                truncate_value(result.to_dict(), self.config.max_text_chars)
                for result in train_results
            ],
            "rejected_buffer": [
                truncate_value(item, self.config.max_text_chars)
                for item in (rejected_buffer or [])
            ],
            "meta_skill": meta_skill,
            "optimizer_controls": optimizer_controls or {},
        }
        try:
            raw = self._invoke(payload)
        except subprocess.TimeoutExpired:
            proposals = []
        else:
            proposals = parse_proposals(raw, current_skill=skill_text)
        if self.config.proposal_log_path is not None:
            append_proposal_log(
                Path(self.config.proposal_log_path),
                seed=self.config.proposal_log_seed,
                case=self.config.proposal_log_case,
                epoch=epoch,
                train_results=train_results,
                rejected_buffer=rejected_buffer or [],
                meta_skill=meta_skill,
                optimizer_controls=optimizer_controls or {},
                proposals=proposals,
            )
        return proposals

    def update_state(
        self,
        *,
        epoch: int,
        current_skill: str,
        meta_skill: str,
        comparison: dict[str, Any],
        rejected_buffer: list[dict[str, Any]],
        optimizer_controls: dict[str, Any],
    ) -> OptimizerStateUpdate:
        payload = {
            "operation": "slow_meta_update",
            "epoch": epoch,
            "current_skill_text": current_skill,
            "meta_skill": meta_skill,
            "comparison": truncate_value(comparison, self.config.max_text_chars),
            "rejected_buffer": truncate_value(rejected_buffer, self.config.max_text_chars),
            "optimizer_controls": optimizer_controls,
        }
        raw = self._invoke(payload)
        if not isinstance(raw, dict):
            raise ValueError("slow_meta_update output must be a JSON object")
        return OptimizerStateUpdate(
            meta_skill=str(raw.get("meta_skill", "")),
            slow_update=str(raw.get("slow_update", "")),
            rationale=str(raw.get("rationale", "")),
        )

    def _invoke(self, payload: dict[str, Any]) -> Any:
        payload_text = json.dumps(payload)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                shlex.split(self.config.command),
                text=True,
                input=payload_text,
                capture_output=True,
                timeout=self.config.timeout_seconds,
                check=False,
                env=self._subprocess_env(),
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            stdout = _safe_text(exc.stdout)
            stderr = _safe_text(exc.stderr) or f"Command timed out after {self.config.timeout_seconds}s"
            self._record_invocation(
                payload,
                payload_text=payload_text,
                stdout=stdout,
                stderr=stderr,
                returncode=124,
                duration_seconds=duration,
                timed_out=True,
            )
            raise
        duration = time.monotonic() - started
        self._record_invocation(
            payload,
            payload_text=payload_text,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            duration_seconds=duration,
            timed_out=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Editor command failed "
                f"with code {completed.returncode}: {completed.stderr.strip()}"
            )
        try:
            raw = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Editor command must print JSON to stdout. "
                f"Got: {completed.stdout[:500]!r}"
            ) from exc
        return raw

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.config.usage_ledger_path is not None:
            env["TEXTSKILL_USAGE_LEDGER_PATH"] = str(self.config.usage_ledger_path)
        if self.config.usage_context:
            env["TEXTSKILL_USAGE_CONTEXT_JSON"] = json.dumps(self.config.usage_context)
        return env

    def _record_invocation(
        self,
        payload: dict[str, Any],
        *,
        payload_text: str,
        stdout: str,
        stderr: str,
        returncode: int,
        duration_seconds: float,
        timed_out: bool,
    ) -> None:
        output_chars = len(stdout) + len(stderr)
        append_usage_event(
            self.config.usage_ledger_path,
            {
                "kind": "optimizer_command",
                "operation": str(payload.get("operation") or "reflect"),
                "context": self.config.usage_context or {},
                "command": self.config.command,
                "returncode": returncode,
                "timed_out": timed_out,
                "duration_seconds": duration_seconds,
                "input_chars": len(payload_text),
                "output_chars": output_chars,
                "estimated_prompt_tokens": estimate_tokens_from_chars(len(payload_text)),
                "estimated_completion_tokens": estimate_tokens_from_chars(output_chars),
            },
        )


def parse_proposals(payload: Any, *, current_skill: str | None = None) -> list[EditProposal]:
    if isinstance(payload, dict) and "proposals" in payload:
        payload = payload["proposals"]
    elif isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Editor command output must be a proposal or proposal list")

    proposals: list[EditProposal] = []
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Proposal {index} must be a JSON object")
        name = str(item.get("name") or f"command-proposal-{index}")
        raw_edits = item.get("edits") or []
        if not isinstance(raw_edits, list):
            raise ValueError(f"Proposal {name!r} edits must be a list")
        edits = tuple(parse_atomic_edit(edit, name=name, index=edit_index) for edit_index, edit in enumerate(raw_edits, 1))
        skill_text = item.get("skill_text")
        if edits and not isinstance(skill_text, str):
            if current_skill is None:
                skill_text = ""
            else:
                skill_text = apply_atomic_edits(current_skill, edits)
        if not edits and (not isinstance(skill_text, str) or not skill_text.strip()):
            raise ValueError(f"Proposal {name!r} is missing skill_text or atomic edits")
        rationale = str(item.get("rationale") or "External editor proposal.")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        proposals.append(
            EditProposal(
                name=name,
                skill_text=skill_text,
                rationale=rationale,
                metadata=metadata,
                edits=edits,
            )
        )
    return proposals


def parse_atomic_edit(payload: Any, *, name: str, index: int) -> AtomicEdit:
    if not isinstance(payload, dict):
        raise ValueError(f"Proposal {name!r} edit {index} must be a JSON object")
    return AtomicEdit(
        operation=str(payload.get("operation", "")),
        target=str(payload.get("target", "")),
        content=str(payload.get("content", "")),
        rationale=str(payload.get("rationale", "")),
        priority=float(payload.get("priority", 0.0)),
    )


def append_proposal_log(
    path: Path,
    *,
    seed: str,
    case: str,
    epoch: int,
    train_results: list[TaskResult],
    rejected_buffer: list[dict[str, Any]],
    meta_skill: str,
    optimizer_controls: dict[str, Any],
    proposals: list[EditProposal],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contract_rejection_evidence = build_contract_rejection_evidence(rejected_buffer)
    proposal_targeting_audit = audit_proposal_targeting(proposals, contract_rejection_evidence)
    record = {
        "seed": seed or "default",
        "case": case or "default",
        "epoch": epoch,
        "train_task_ids": [result.task.id for result in train_results],
        "failed_train_task_ids": [
            result.task.id for result in train_results if not result.score.success
        ],
        "rejected_count": len(rejected_buffer),
        "contract_rejection_evidence": contract_rejection_evidence,
        "proposal_targeting_audit": proposal_targeting_audit,
        "meta_skill_present": bool(meta_skill.strip()),
        "optimizer_controls": optimizer_controls,
        "proposals": [proposal.to_dict() for proposal in proposals],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def truncate_value(value: Any, max_text_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= max_text_chars:
            return value
        return value[:max_text_chars] + "\n...[truncated]"
    if isinstance(value, list):
        return [truncate_value(item, max_text_chars) for item in value]
    if isinstance(value, dict):
        return {
            str(key): truncate_value(item, max_text_chars)
            for key, item in value.items()
        }
    return value


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
