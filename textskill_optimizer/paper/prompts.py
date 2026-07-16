"""Immutable optimizer prompts copied from the locked official reference."""

from __future__ import annotations

from importlib.resources import files

from .backend import OptimizerStage


_PATCH_PROMPTS = {
    OptimizerStage.REFLECT_FAILURE: "analyst_error.md",
    OptimizerStage.REFLECT_SUCCESS: "analyst_success.md",
    OptimizerStage.REFINE: "refine.md",
    OptimizerStage.MERGE_FAILURE: "merge_failure.md",
    OptimizerStage.MERGE_SUCCESS: "merge_success.md",
    OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED: "merge_final.md",
    OptimizerStage.RANK_TOP_L: "ranking.md",
}

_REWRITE_PROMPTS = {
    OptimizerStage.REFLECT_FAILURE: "analyst_error_rewrite.md",
    OptimizerStage.REFLECT_SUCCESS: "analyst_success_rewrite.md",
    OptimizerStage.REFINE: "refine_rewrite.md",
    OptimizerStage.MERGE_FAILURE: "merge_failure_rewrite.md",
    OptimizerStage.MERGE_SUCCESS: "merge_success_rewrite.md",
    OptimizerStage.MERGE_FINAL_FAILURE_PRIORITIZED: "merge_final_rewrite.md",
    OptimizerStage.RANK_TOP_L: "ranking_rewrite.md",
    OptimizerStage.REWRITE_SKILL: "rewrite_skill.md",
}

_COMMON_PROMPTS = {
    OptimizerStage.DECIDE_LEARNING_RATE: "lr_autonomous.md",
    OptimizerStage.PROPOSE_SLOW_UPDATE: "slow_update.md",
    OptimizerStage.UPDATE_META_SKILL: "meta_skill.md",
}


def load_optimizer_prompt(
    stage: OptimizerStage,
    *,
    update_mode: str = "patch",
) -> str:
    """Load one locked prompt or the explicit local refinement resolution."""

    if type(stage) is not OptimizerStage:
        raise ValueError(f"no paper prompt registered for stage: {stage!r}")
    if stage in _COMMON_PROMPTS:
        prompt_name = _COMMON_PROMPTS[stage]
    elif update_mode == "patch":
        prompt_name = _PATCH_PROMPTS.get(stage)
    elif update_mode == "rewrite_from_suggestions":
        prompt_name = _REWRITE_PROMPTS.get(stage)
    else:
        raise ValueError(f"unsupported paper update mode: {update_mode!r}")
    if prompt_name is None:
        raise ValueError(f"no paper prompt registered for stage: {stage!r}")
    return (
        files("textskill_optimizer.paper")
        .joinpath("prompts", prompt_name)
        .read_text(encoding="utf-8")
    )
