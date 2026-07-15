"""Immutable optimizer prompts copied from the locked official reference."""

from __future__ import annotations

from importlib.resources import files

from .backend import OptimizerStage


_PROMPT_NAMES = {
    OptimizerStage.REFLECT_FAILURE: "analyst_error.md",
    OptimizerStage.REFLECT_SUCCESS: "analyst_success.md",
    OptimizerStage.REFINE: "refine.md",
    OptimizerStage.MERGE_FAILURE: "merge_failure.md",
    OptimizerStage.MERGE_SUCCESS: "merge_success.md",
    OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED: "merge_final.md",
    OptimizerStage.RANK_TOP_L: "ranking.md",
}


def load_optimizer_prompt(stage: OptimizerStage) -> str:
    """Load the locked prompt for one fast-loop stage.

    Six stage prompts are byte-identical v0.2.0 resources. ``refine.md`` is
    the explicit local resolution for the refinement loop missing upstream.
    """

    if type(stage) is not OptimizerStage or stage not in _PROMPT_NAMES:
        raise ValueError(f"no paper prompt registered for stage: {stage!r}")
    return (
        files("textskill_optimizer.paper")
        .joinpath("prompts", _PROMPT_NAMES[stage])
        .read_text(encoding="utf-8")
    )
