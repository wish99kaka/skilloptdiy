"""Optimization-side controller whose model seam accepts train evidence only."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .backend import (
    OptimizerBackend,
    OptimizerRequest,
    OptimizerResponse,
    OptimizerStage,
)
from .data import (
    OptimizerPayload,
    SelectionController,
    SelectionDecision,
    SelectionScore,
    TrainController,
    strict_selection_decision,
)
from .errors import DataFirewallViolation


@dataclass(frozen=True)
class PaperOptimizationController:
    """Route train evidence to the optimizer and scalar scores to the gate."""

    optimizer_backend: OptimizerBackend
    selection: SelectionController
    train: TrainController

    def __post_init__(self) -> None:
        if type(self.selection) is not SelectionController:
            raise DataFirewallViolation("optimizer requires exact SelectionController")
        if type(self.train) is not TrainController:
            raise DataFirewallViolation("optimizer requires exact TrainController")
        self.selection.__post_init__()
        self.train.__post_init__()
        if self.selection.registry.sha256 != self.train.registry.sha256:
            raise DataFirewallViolation(
                "train and selection must share one controller registry"
            )

    def score_candidate(
        self,
        *,
        current: SelectionScore,
        candidate_skill: str,
    ) -> SelectionDecision:
        self.__post_init__()
        candidate = self.selection.score(candidate_skill)
        return strict_selection_decision(current=current, candidate=candidate)

    def request_optimizer(
        self,
        *,
        call_id: str,
        stage: OptimizerStage,
        payload: OptimizerPayload,
    ) -> OptimizerResponse:
        self.__post_init__()
        if type(payload) is not OptimizerPayload:
            raise DataFirewallViolation(
                "optimizer seam accepts only the sealed OptimizerPayload type"
            )
        payload.__post_init__()
        train_trajectories = self.train.verify(
            payload.train_evidence,
            current_skill=payload.current_skill,
        )
        try:
            prompt = json.dumps(
                {
                    "current_skill": payload.current_skill,
                    "train_trajectories": list(train_trajectories),
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise DataFirewallViolation(
                f"train-only optimizer payload is not JSON-safe: {error}"
            ) from error
        request = OptimizerRequest(
            call_id=call_id,
            stage=stage,
            prompt=prompt,
            response_schema={"type": "object"},
            metadata={
                "protocol_id": "paper-faithful-v1",
                "data_sources": ["train"],
                "controller_registry_sha256": self.train.registry.sha256,
                "train_controller_id": self.train.controller_id,
                "train_split_id": payload.train_evidence.split_id,
                "train_split_manifest_sha256": (
                    payload.train_evidence.split_manifest_sha256
                ),
                "train_trajectory_count": len(train_trajectories),
            },
        )
        return self.optimizer_backend.complete(request)
