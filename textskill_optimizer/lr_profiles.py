"""Named textual learning-rate budget profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .optimizer import OptimizerConfig


@dataclass(frozen=True)
class LearningRateProfile:
    name: str
    label: str
    max_skill_chars: int
    max_skill_delta_chars: int
    max_added_bullet_lines: int


LR_PROFILES: dict[str, LearningRateProfile] = {
    "strict": LearningRateProfile(
        name="strict",
        label="strict 600/260/1",
        max_skill_chars=600,
        max_skill_delta_chars=260,
        max_added_bullet_lines=1,
    ),
    "real-editor": LearningRateProfile(
        name="real-editor",
        label="real-editor 600/520/1",
        max_skill_chars=600,
        max_skill_delta_chars=520,
        max_added_bullet_lines=1,
    ),
    "loose-diagnostic": LearningRateProfile(
        name="loose-diagnostic",
        label="loose-diagnostic 750/700/3",
        max_skill_chars=750,
        max_skill_delta_chars=700,
        max_added_bullet_lines=3,
    ),
}


def profile_names() -> list[str]:
    return sorted(LR_PROFILES)


def get_lr_profile(name: str) -> LearningRateProfile:
    try:
        return LR_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(profile_names())
        raise ValueError(f"Unknown LR profile {name!r}; expected one of: {known}") from exc


def apply_lr_profile(config: OptimizerConfig, profile_name: str) -> OptimizerConfig:
    profile = get_lr_profile(profile_name)
    return replace(
        config,
        max_skill_chars=profile.max_skill_chars,
        max_skill_delta_chars=profile.max_skill_delta_chars,
        max_added_bullet_lines=profile.max_added_bullet_lines,
    )


def profile_budget_dict(profile_name: str) -> dict[str, int | str]:
    profile = get_lr_profile(profile_name)
    return {
        "name": profile.name,
        "label": profile.label,
        "max_skill_chars": profile.max_skill_chars,
        "max_skill_delta_chars": profile.max_skill_delta_chars,
        "max_added_bullet_lines": profile.max_added_bullet_lines,
    }
